# -*- coding: utf-8 -*-
"""혼동행렬의 각 칸에서 실제 데이터로 내려가는 오류 분석 도구.

혼동행렬은 "몇 개 틀렸나"만 알려준다. 고치려면 "무엇이 틀렸나"를 봐야 한다.
이 스크립트는 FP/FN 칸에 실제로 들어 있는 샘플을 꺼내 원문·음향 통계와 함께 보여준다.

사용 예
  python -m src.viz.inspect_errors m3 --cls 오심 --n 5     # 오심 FP/FN 통화 원문
  python -m src.viz.inspect_errors m1 --n 10               # 성별 오분류 통화 (F0 포함)
  python -m src.viz.inspect_errors m2                      # 화자 오분류의 분포 요약
"""
import argparse
import sys

import numpy as np

from ..common.labels import load_split
from ..common.paths import CACHE, SYMPTOM_9

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LINE = "─" * 78


# ------------------------------------------------------------------ M3
def m3(cls=None, n=5):
    d = np.load(CACHE / "m3_val_pred.npz", allow_pickle=True)
    yt, yp, sc = d["y_true"], d["y_pred"], d["y_score"]
    va = load_split("val")            # 예측 행 순서 == load_split('val') 순서
    classes = list(SYMPTOM_9)
    targets = [cls] if cls else classes

    for c in targets:
        k = classes.index(c)
        fp = np.flatnonzero((yt[:, k] == 0) & (yp[:, k] == 1))   # 없는데 있다고 함
        fn = np.flatnonzero((yt[:, k] == 1) & (yp[:, k] == 0))   # 있는데 놓침
        print(f"\n{LINE}\n[{c}]  FP {len(fp)}건 / FN {len(fn)}건")
        print(LINE)

        for tag, idx in (("FP (오탐)", fp), ("FN (누락)", fn)):
            if len(idx) == 0:
                continue
            print(f"\n  ── {tag} 상위 {min(n, len(idx))}건 " + ("(확신도 높은 순)" if tag.startswith("FP") else "(확신도 낮은 순)"))
            order = idx[np.argsort(-sc[idx, k])] if tag.startswith("FP") else idx[np.argsort(sc[idx, k])]
            for i in order[:n]:
                r = va[i]
                truth = [classes[j] for j in range(9) if yt[i, j] == 1]
                pred = [classes[j] for j in range(9) if yp[i, j] == 1]
                # 신고자 발화만 이어붙여 보여준다 (증상 진술은 대부분 신고자 쪽에 있다)
                txt = " ".join(u["text"] for u in r["utterances"] if u["speaker"] == 1)
                print(f"\n    [{r['stem'][:14]}] p={sc[i,k]:.3f}")
                print(f"      정답: {truth}")
                print(f"      예측: {pred}")
                print(f"      신고자 발화: {txt[:220]}{'…' if len(txt) > 220 else ''}")


# ------------------------------------------------------------------ M1
def m1(n=10):
    d = np.load(CACHE / "m1_val_pred.npz", allow_pickle=True)
    a = np.load(CACHE / "audio_val.npz", allow_pickle=True)
    yt, yp, s = d["y_true"], d["y_pred"], d["y_score"]
    X, names, stems = a["call_X"], list(a["feat_names"]), a["call_stem"]
    f0 = X[:, names.index("f0_p50")]
    sec = d["caller_sec"]

    wrong = np.flatnonzero(yt != yp)
    print(f"\n{LINE}\n[M1 성별] 오분류 {len(wrong)} / {len(yt)}  ({len(wrong)/len(yt)*100:.1f}%)")
    print(LINE)
    print(f"  정답별 F0 중앙값 — 여성 {np.median(f0[yt==0]):.1f} Hz / 남성 {np.median(f0[yt==1]):.1f} Hz")
    print(f"  오분류 F0 중앙값 — {np.median(f0[wrong]):.1f} Hz  (정분류 {np.median(f0[yt==yp]):.1f} Hz)")
    print(f"  오분류 신고자 음성 길이 중앙값 — {np.median(sec[wrong]):.1f}초 (정분류 {np.median(sec[yt==yp]):.1f}초)")

    for lab, name in ((0, "여성→남성 오분류"), (1, "남성→여성 오분류")):
        w = wrong[yt[wrong] == lab]
        print(f"\n  ── {name}: {len(w)}건, F0 중앙값 {np.median(f0[w]):.1f} Hz")
        # 확신을 갖고 틀린 것부터 = 가장 병리적인 사례
        order = w[np.argsort(-np.abs(s[w] - 0.5))]
        for i in order[:n]:
            print(f"    [{stems[i][:14]}] F0={f0[i]:6.1f}Hz  음성={sec[i]:5.1f}s  "
                  f"p(남)={s[i]:.3f}")


# ------------------------------------------------------------------ M2
def m2(n=10):
    d = np.load(CACHE / "m2_val_pred.npz", allow_pickle=True)
    yt, yp, ov, dur = d["y_true"], d["y_pred"], d["overlap"], d["dur_ms"] / 1000.0
    wrong = yt != yp
    print(f"\n{LINE}\n[M2 화자] 오분류 {wrong.sum():,} / {len(yt):,}  ({wrong.mean()*100:.1f}%)")
    print(LINE)

    print("\n  ── 중첩 여부별")
    for v, name in ((0, "비중첩"), (1, "중첩  ")):
        m = ov == v
        print(f"    {name}: {m.sum():6,}건 중 오분류 {wrong[m].sum():5,} ({wrong[m].mean()*100:.1f}%)")

    print("\n  ── 발화 길이별 (오분류율)")
    edges = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]
    lab = ["<0.5s", "0.5-1s", "1-1.5s", "1.5-2s", "2-3s", "3-5s", ">5s"]
    for i in range(len(edges) - 1):
        m = (dur >= edges[i]) & (dur < edges[i + 1])
        if m.sum() < 30:
            continue
        print(f"    {lab[i]:>7s}: {m.sum():6,}건  오분류 {wrong[m].mean()*100:5.1f}%  "
              f"(중첩비율 {ov[m].mean()*100:4.1f}%)")

    print("\n  ── 오류 방향")
    a = ((yt == 0) & (yp == 1)).sum()
    b = ((yt == 1) & (yp == 0)).sum()
    print(f"    대원을 신고자로: {a:6,}건")
    print(f"    신고자를 대원으로: {b:6,}건")
    print(f"    → 모델이 {'신고자' if a > b else '대원'} 쪽으로 {abs(a-b):,}건 치우쳐 예측한다")

    print("\n  ── 가장 나쁜 조합 (짧고 + 중첩)")
    m = (dur < 1.0) & (ov == 1)
    print(f"    1초 미만 & 중첩: {m.sum():,}건, 오분류 {wrong[m].mean()*100:.1f}%  "
          f"← 전체 오류의 {wrong[m].sum()/wrong.sum()*100:.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mission", choices=["m1", "m2", "m3"])
    ap.add_argument("--cls", default=None, help="M3 전용: 특정 증상만 (예: 오심)")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    {"m1": lambda: m1(a.n), "m2": lambda: m2(a.n), "m3": lambda: m3(a.cls, a.n)}[a.mission]()
