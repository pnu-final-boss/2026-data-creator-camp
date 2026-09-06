# -*- coding: utf-8 -*-
"""Mission 3 베이스라인: 대화 텍스트 → 증상 9클래스 multi-label.

선행연구 대응
  - 권수정 외(2020) 정보처리학회 9(10):317-322 : 119 신고 전사문 + TF-IDF + SVM/SGD/RF
  - Ridnik et al. ICCV'21 (ASL) 의 문제의식(양/음 불균형) 은 클래스별 임계값 튜닝으로 대체
    (선형모델이라 손실함수 교체 대신 threshold 최적화가 같은 효과를 낸다)

규정 준수
  - 입력은 utterances[].text 와 speaker 뿐. symptom 이외의 라벨 필드(disasterMedium,
    urgencyLevel, triage, sentiment)는 일절 읽지 않는다.
  - Validation 은 학습에 쓰지 않는다. 임계값 튜닝은 Training 에서 떼어낸 dev split 에서만 한다.
"""
import argparse
import json
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from ..common.labels import load_split
from ..common.metrics import macro_f1_pdf, tune_thresholds
from ..common.paths import CACHE, CKPT, SYMPTOM_9

SPK = {0: "[대원]", 1: "[신고자]"}


def build_text(rec, tagged=True):
    if tagged:
        return " ".join(f"{SPK.get(u['speaker'], '[?]')} {u['text']}" for u in rec["utterances"])
    return " ".join(u["text"] for u in rec["utterances"])


def build_labels(recs):
    idx = {s: i for i, s in enumerate(SYMPTOM_9)}
    Y = np.zeros((len(recs), len(SYMPTOM_9)), dtype=np.int8)
    for i, r in enumerate(recs):
        for s in r["symptom"]:
            if s in idx:            # 9개 밖의 라벨은 '샘플이 아니라 라벨만' 제거 (PDF 9쪽)
                Y[i, idx[s]] = 1
    return Y


def make_vectorizer():
    """char n-gram + word n-gram 결합 TF-IDF.

    한국어는 교착어라 "아파요/아프고/아픈데"가 word 단위로는 전부 다른 토큰이 된다.
    char n-gram 은 공통 어간("아프")을 잡아내므로 형태소 분석기(konlpy/JDK) 없이도 강하다.
    Windows 에서 konlpy 설치 리스크를 피하려는 의도도 있다.

    char_wb: 단어 경계 안에서만 n-gram 을 만든다 (어절을 넘나드는 잡음 조합 방지).
    sublinear_tf: tf 대신 1+log(tf). "아파요"가 10번 나왔다고 10배 중요하진 않다.
    min_df=3: 3개 미만 문서에 나온 희귀 표현은 버려 과적합과 차원을 줄인다.
    """
    return FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                 min_df=3, max_features=300_000, sublinear_tf=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                 min_df=3, max_features=100_000, sublinear_tf=True)),
    ])


def make_clf(name):
    """OneVsRest: 9개 라벨마다 독립 이진 분류기를 세운다.

    multi-label 이라 "9개 중 하나"가 아니라 "각 라벨이 있나 없나"를 따로 판정해야 한다.
    n_jobs=9 로 9개 분류기를 동시에 학습한다 (16코어 환경 기준).

    class_weight='balanced': 양성 비율이 10~23% 로 낮아 그대로 두면 전부 음성으로 예측한다.
    다만 이게 결정경계를 이미 옮겨놓기 때문에 뒤의 임계값 튜닝 효과가 거의 사라진다
    (실측 +0.0001). 두 기법은 더해지지 않고 같은 자리를 다툰다 — ablate.py 에서 검증.
    """
    if name == "logreg":
        return OneVsRestClassifier(
            LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"), n_jobs=9)
    if name == "svc":
        return OneVsRestClassifier(
            CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3), n_jobs=9)
    if name == "sgd":
        return OneVsRestClassifier(
            SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30,
                          class_weight="balanced", random_state=0), n_jobs=9)
    raise ValueError(name)


def scores_of(clf, X):
    """OvR 점수를 (n, 9) 확률 행렬로 반환.

    multi-label 에서 OneVsRestClassifier.predict_proba 는 라벨별 확률을 그대로 준다
    (다중클래스와 달리 합이 1 이 되도록 정규화하지 않는다) — 임계값 튜닝에 그대로 쓸 수 있다.
    predict_proba 가 없는 분류기(LinearSVC 등)는 decision_function 을 시그모이드로 눌러
    같은 [0, 1] 범위로 맞춘다. 보정된 확률은 아니지만 임계값 탐색에는 충분하다.
    """
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    d = clf.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))


def run(model="logreg", tagged=True, dev_frac=0.1, seed=0, save=True):
    tr = load_split("train")
    va = load_split("val")

    Xtr_txt = [build_text(r, tagged) for r in tr]
    Xva_txt = [build_text(r, tagged) for r in va]
    Ytr_all = build_labels(tr)
    Yva = build_labels(va)

    # Training 내부 dev split — 임계값 튜닝 전용 (Validation 은 절대 사용 안 함).
    # Validation 으로 임계값을 고르면 그 점수는 낙관적으로 부풀려져 실제 채점과 어긋난다.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr))
    n_dev = int(len(tr) * dev_frac)
    dev_i, fit_i = perm[:n_dev], perm[n_dev:]

    vec = make_vectorizer()
    # fit_transform 은 fit split 에서만. dev/val 에 fit 하면 어휘 정보가 새어 들어간다.
    Xfit = vec.fit_transform([Xtr_txt[i] for i in fit_i])
    Xdev = vec.transform([Xtr_txt[i] for i in dev_i])
    Xva = vec.transform(Xva_txt)
    print(f"[m3] tfidf features = {Xfit.shape[1]:,}  fit={Xfit.shape[0]}  dev={Xdev.shape[0]}  val={Xva.shape[0]}")

    clf = make_clf(model)
    clf.fit(Xfit, Ytr_all[fit_i])

    dev_s = scores_of(clf, Xdev)
    va_s = scores_of(clf, Xva)

    # 클래스별 임계값을 dev 에서 결정. 희소 클래스는 0.5 보다 낮은 임계값이 F1 에 유리하다.
    th = tune_thresholds(Ytr_all[dev_i], dev_s)

    # 0.5 고정과 튜닝본을 둘 다 보고해 임계값의 실제 기여를 드러낸다.
    # th[None, :] 는 (9,) 를 (1, 9) 로 만들어 (n, 9) 점수 행렬에 열별로 브로드캐스팅한다.
    pred_05 = (va_s >= 0.5).astype(int)
    pred_th = (va_s >= th[None, :]).astype(int)
    m05, _ = macro_f1_pdf(Yva, pred_05)
    mth, per_class = macro_f1_pdf(Yva, pred_th)

    print(f"[m3] {model:6s} tagged={tagged}  macro-F1 @0.5 = {m05:.4f}   @tuned = {mth:.4f}")

    if save:
        with open(CKPT / "m3_model.pkl", "wb") as f:
            pickle.dump({"vec": vec, "clf": clf, "thresholds": th,
                         "classes": SYMPTOM_9, "tagged": tagged}, f)
        np.savez_compressed(CACHE / "m3_val_pred.npz",
                            y_true=Yva, y_score=va_s, y_pred=pred_th,
                            thresholds=th, classes=np.array(SYMPTOM_9))
        with open(CACHE / "m3_report.json", "w", encoding="utf-8") as f:
            json.dump({"model": model, "tagged": tagged,
                       "macro_f1_05": m05, "macro_f1_tuned": mth,
                       "thresholds": th.tolist(),
                       "per_class": {c: per_class[i] for i, c in enumerate(SYMPTOM_9)}},
                      f, ensure_ascii=False, indent=2)
    return dict(model=model, tagged=tagged, macro_f1_05=m05, macro_f1_tuned=mth,
                per_class=per_class, thresholds=th)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logreg", choices=["logreg", "svc", "sgd"])
    ap.add_argument("--plain", action="store_true", help="화자 태그 없이 텍스트만 사용")
    args = ap.parse_args()
    run(model=args.model, tagged=not args.plain)
