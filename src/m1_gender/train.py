# -*- coding: utf-8 -*-
"""Mission 1 베이스라인: 신고자 음성 → 성별(M/F).

선행연구 대응
  - griko/voice-gender-classification : 화자 임베딩 + 얕은 분류기. GPU 가 없으므로 여기서는
    ECAPA-TDNN 임베딩 대신 MFCC/F0/스펙트럼 통계로 대체한 동일 구조(고정 특징 + 얕은 분류기).
  - Kaushik et al. ICASSP'21 : F0 가 성별의 1차 단서라는 점을 특징 설계에 반영.

규정 준수
  - 라벨링데이터에서 startAt/endAt/speaker 만 사용해 신고자(speaker==1) 구간을 잘라낸다.
    text, symptom 등 다른 필드는 읽지 않는다.
"""
import argparse
import json
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..common.metrics import binary_confusion
from ..common.paths import CACHE, CKPT

MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(),
                                    LogisticRegression(C=1.0, max_iter=3000)),
    "rf": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                         n_jobs=12, random_state=0),
    "hgb": lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                                  random_state=0),
}


def load():
    tr = np.load(CACHE / "audio_train.npz", allow_pickle=True)
    va = np.load(CACHE / "audio_val.npz", allow_pickle=True)
    return tr, va


def run(models=("logreg", "rf", "hgb"), save=True):
    tr, va = load()
    Xtr, ytr = tr["call_X"], tr["call_g"]
    Xva, yva = va["call_X"], va["call_g"]
    names = list(tr["feat_names"])
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xva = np.nan_to_num(Xva, nan=0.0, posinf=0.0, neginf=0.0)

    majority = max(np.mean(yva == 0), np.mean(yva == 1))
    print(f"[m1] train={Xtr.shape} val={Xva.shape}  다수클래스 baseline acc={majority:.4f}")

    results, best, best_acc = {}, None, -1.0
    for m in models:
        clf = MODELS[m]()
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xva)[:, 1]
        pred = (p >= 0.5).astype(int)
        acc = accuracy_score(yva, pred)
        auc = roc_auc_score(yva, p)
        cm = binary_confusion(yva, pred)
        results[m] = dict(acc=float(acc), auc=float(auc), cm=cm.tolist())
        print(f"[m1] {m:7s} acc={acc:.4f}  auc={auc:.4f}  cm={cm.tolist()}")
        if acc > best_acc:
            best_acc, best = acc, (m, clf, p, pred)

    m, clf, p, pred = best
    print(f"[m1] best = {m} (acc={best_acc:.4f})")

    # 특징 중요도 (F0 가 실제로 1등인지 확인)
    imp = None
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    if imp is not None:
        order = np.argsort(imp)[::-1][:15]
        print("[m1] top features:", [(names[i], round(float(imp[i]), 4)) for i in order[:8]])

    if save:
        with open(CKPT / "m1_model.pkl", "wb") as f:
            pickle.dump({"model": clf, "name": m, "feat_names": names}, f)
        np.savez_compressed(CACHE / "m1_val_pred.npz",
                            y_true=yva, y_score=p, y_pred=pred,
                            caller_sec=va["call_sec"],
                            importance=(imp if imp is not None else np.zeros(len(names))),
                            feat_names=np.array(names))
        with open(CACHE / "m1_report.json", "w", encoding="utf-8") as f:
            json.dump({"majority": float(majority), "best": m, "results": results},
                      f, ensure_ascii=False, indent=2)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["logreg", "rf", "hgb"])
    a = ap.parse_args()
    run(tuple(a.models))
