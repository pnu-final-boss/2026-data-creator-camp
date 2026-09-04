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
    """한국어는 형태소 분석기 없이도 char n-gram 이 강하다. word n-gram 과 결합."""
    return FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                 min_df=3, max_features=300_000, sublinear_tf=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                 min_df=3, max_features=100_000, sublinear_tf=True)),
    ])


def make_clf(name):
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
    """OvR 확률/결정값을 (n, 9) 로 정규화해 반환."""
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

    # Training 내부 dev split — 임계값 튜닝 전용 (Validation 은 절대 사용 안 함)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr))
    n_dev = int(len(tr) * dev_frac)
    dev_i, fit_i = perm[:n_dev], perm[n_dev:]

    vec = make_vectorizer()
    Xfit = vec.fit_transform([Xtr_txt[i] for i in fit_i])
    Xdev = vec.transform([Xtr_txt[i] for i in dev_i])
    Xva = vec.transform(Xva_txt)
    print(f"[m3] tfidf features = {Xfit.shape[1]:,}  fit={Xfit.shape[0]}  dev={Xdev.shape[0]}  val={Xva.shape[0]}")

    clf = make_clf(model)
    clf.fit(Xfit, Ytr_all[fit_i])

    dev_s = scores_of(clf, Xdev)
    va_s = scores_of(clf, Xva)

    th = tune_thresholds(Ytr_all[dev_i], dev_s)

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
