# -*- coding: utf-8 -*-
"""Mission 2 베이스라인: 발화 조각 오디오 → 신고자(1) / 119대원(0).

선행연구 대응
  - Ozan, arXiv:2106.02422 (콜센터 상담원/고객 구간 CRNN) : 이 미션과 동형. 그 논문의 핵심
    관찰은 "모델이 언어가 아니라 채널·음향 특성을 학습한다"는 것. GPU 가 없어 CRNN 대신
    동일 입력(log-mel/MFCC 통계)에 얕은 분류기를 쓴다.
  - 통화 단위 정규화: 대원은 고정 헤드셋/상황실, 신고자는 휴대폰/야외. 절대적 음색이 아니라
    "같은 통화 안에서의 상대적 채널 차이"가 진짜 신호이므로 통화별 평균을 빼준다.

규정 준수
  - text 와 발화 순서(index/홀짝)는 특징에 일절 넣지 않는다. startAt/endAt 은 구간을 자르는
    용도와 overlap 판정에만 쓴다 (PDF 가 허용한 필드).
"""
import argparse
import json
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..common.metrics import binary_confusion
from ..common.paths import CACHE, CKPT


def call_normalize(X, call_id):
    """통화별 평균을 빼서 채널 차이를 상대화한다. (원본과 잔차를 함께 반환)"""
    Xn = X.copy()
    order = np.argsort(call_id, kind="stable")
    cid = call_id[order]
    bounds = np.flatnonzero(np.diff(cid)) + 1
    for sl in np.split(order, bounds):
        Xn[sl] = X[sl] - X[sl].mean(axis=0, keepdims=True)
    return Xn


def build_matrix(d, mode):
    X = np.nan_to_num(d["utt_X"], nan=0.0, posinf=0.0, neginf=0.0)
    cid = d["utt_call"]
    if mode == "raw":
        return X
    Xn = call_normalize(X, cid)
    if mode == "norm":
        return Xn
    if mode == "both":
        return np.hstack([X, Xn])
    raise ValueError(mode)


MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(),
                                    LogisticRegression(C=1.0, max_iter=3000, n_jobs=12)),
    "hgb": lambda: HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08,
                                                  random_state=0),
}


def run(mode="both", model="hgb", train_on="all", save=True):
    tr = np.load(CACHE / "audio_train.npz", allow_pickle=True)
    va = np.load(CACHE / "audio_val.npz", allow_pickle=True)

    Xtr, ytr, ovtr = build_matrix(tr, mode), tr["utt_y"], tr["utt_ov"]
    Xva, yva, ovva = build_matrix(va, mode), va["utt_y"], va["utt_ov"]

    if train_on == "clean":          # 겹치지 않는 구간만으로 학습 (라벨 노이즈 제거 실험)
        keep = ovtr == 0
        Xtr, ytr = Xtr[keep], ytr[keep]

    majority = max(np.mean(yva == 0), np.mean(yva == 1))
    print(f"[m2] mode={mode} model={model} train_on={train_on} "
          f"train={Xtr.shape} val={Xva.shape} 다수클래스={majority:.4f}")

    clf = MODELS[model]()
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    pred = (p >= 0.5).astype(int)

    acc = accuracy_score(yva, pred)
    auc = roc_auc_score(yva, p)
    cm = binary_confusion(yva, pred)
    acc_clean = accuracy_score(yva[ovva == 0], pred[ovva == 0])
    acc_ov = accuracy_score(yva[ovva == 1], pred[ovva == 1])
    print(f"[m2] acc={acc:.4f} auc={auc:.4f} | 비중첩 {acc_clean:.4f} / 중첩 {acc_ov:.4f} | cm={cm.tolist()}")

    res = dict(mode=mode, model=model, train_on=train_on, majority=float(majority),
               acc=float(acc), auc=float(auc), cm=cm.tolist(),
               acc_nonoverlap=float(acc_clean), acc_overlap=float(acc_ov),
               n_val=int(len(yva)), n_overlap=int((ovva == 1).sum()))

    if save:
        with open(CKPT / "m2_model.pkl", "wb") as f:
            pickle.dump({"model": clf, "mode": mode, "name": model}, f)
        np.savez_compressed(CACHE / "m2_val_pred.npz",
                            y_true=yva, y_score=p, y_pred=pred, overlap=ovva,
                            dur_ms=(va["utt_end"] - va["utt_start"]))
        with open(CACHE / "m2_report.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["raw", "norm", "both"])
    ap.add_argument("--model", default="hgb", choices=["logreg", "hgb"])
    ap.add_argument("--train-on", default="all", choices=["all", "clean"])
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    run(a.mode, a.model, a.train_on, save=not a.no_save)
