# -*- coding: utf-8 -*-
"""결과 시각화: 혼동행렬(confusion matrix) 중심.

Windows 한글 폰트(Malgun Gothic)를 명시적으로 지정한다 — 지정하지 않으면 축 라벨이 두부(□)로 깨진다.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams

from ..common.metrics import binary_confusion, macro_f1_pdf
from ..common.paths import CACHE, FIGS, SYMPTOM_9

# ---- 한글 폰트 ----
for cand in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    if any(cand == f.name for f in font_manager.fontManager.ttflist):
        rcParams["font.family"] = cand
        break
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
rcParams["savefig.bbox"] = "tight"
rcParams["savefig.facecolor"] = "white"

INK = "#16202B"
MUTED = "#5D6C7B"
ACCENT = "#1B5E96"
GRID = "#DCE3EA"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def draw_cm(ax, cm, labels, title, normalize=True, xlab=True, ylab=True):
    """혼동행렬 한 장. 셀에는 건수와 행-정규화 비율을 함께 적는다."""
    cm = np.asarray(cm, dtype=float)
    row = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, np.where(row == 0, 1, row))
    im = ax.imshow(pct if normalize else cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    if xlab:
        ax.set_xlabel("예측", fontsize=9)
    if ylab:
        ax.set_ylabel("정답", fontsize=9)
    ax.set_title(title, fontsize=10.5, pad=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c = "white" if pct[i, j] > 0.55 else INK
            ax.text(j, i, f"{int(cm[i,j]):,}\n{pct[i,j]*100:.1f}%",
                    ha="center", va="center", color=c, fontsize=9.5, linespacing=1.35)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)
    return im


# ------------------------------------------------------------------ M1
def m1():
    d = np.load(CACHE / "m1_val_pred.npz", allow_pickle=True)
    rep = json.load(open(CACHE / "m1_report.json", encoding="utf-8"))
    y, s, p = d["y_true"], d["y_score"], d["y_pred"]
    names = list(d["feat_names"])
    imp = d["importance"]

    fig = plt.figure(figsize=(13, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1.5], wspace=0.42)

    ax = fig.add_subplot(gs[0])
    draw_cm(ax, binary_confusion(y, p), ["여성 F", "남성 M"],
            f"Mission 1 · 성별 (acc={rep['results'][rep['best']]['acc']:.3f})")

    # ROC
    ax = fig.add_subplot(gs[1]); _style(ax)
    order = np.argsort(-s)
    tp = np.cumsum(y[order] == 1) / max((y == 1).sum(), 1)
    fp = np.cumsum(y[order] == 0) / max((y == 0).sum(), 1)
    ax.plot([0, 1], [0, 1], "--", color=GRID, lw=1)
    ax.plot(fp, tp, color=ACCENT, lw=1.8)
    ax.set_xlabel("거짓 양성률"); ax.set_ylabel("참 양성률")
    ax.set_title(f"ROC (AUC={rep['results'][rep['best']]['auc']:.3f})", fontsize=10.5)

    # 특징 중요도
    ax = fig.add_subplot(gs[2]); _style(ax)
    if imp.sum() > 0:
        top = np.argsort(imp)[::-1][:14][::-1]
        colors = [ACCENT if ("f0" in names[i].lower()) else "#9FB6C9" for i in top]
        ax.barh(range(len(top)), imp[top], color=colors, height=0.72)
        ax.set_yticks(range(len(top)), [names[i] for i in top], fontsize=8)
        ax.set_xlabel("중요도")
        ax.set_title("상위 특징 (파랑 = F0 계열)", fontsize=10.5)
    fig.savefig(FIGS / "m1_gender.png")
    plt.close(fig)
    print("saved figs/m1_gender.png")


# ------------------------------------------------------------------ M2
def m2():
    d = np.load(CACHE / "m2_val_pred.npz", allow_pickle=True)
    rep = json.load(open(CACHE / "m2_report.json", encoding="utf-8"))
    y, p, ov, dur = d["y_true"], d["y_pred"], d["overlap"], d["dur_ms"]

    fig = plt.figure(figsize=(13.5, 4.1))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.35], wspace=0.5)

    draw_cm(fig.add_subplot(gs[0]), binary_confusion(y, p), ["119대원", "신고자"],
            f"전체 (acc={rep['acc']:.3f})")
    draw_cm(fig.add_subplot(gs[1]), binary_confusion(y[ov == 0], p[ov == 0]),
            ["119대원", "신고자"], f"비중첩 구간 (acc={rep['acc_nonoverlap']:.3f})")
    draw_cm(fig.add_subplot(gs[2]), binary_confusion(y[ov == 1], p[ov == 1]),
            ["119대원", "신고자"], f"중첩 구간 (acc={rep['acc_overlap']:.3f})")

    # 구간 길이별 정확도
    ax = fig.add_subplot(gs[3]); _style(ax)
    edges = np.array([0, 500, 1000, 1500, 2000, 3000, 5000, 100000])
    xs, ys, ns = [], [], []
    for i in range(len(edges) - 1):
        m = (dur >= edges[i]) & (dur < edges[i + 1])
        if m.sum() > 30:
            xs.append(i); ys.append((y[m] == p[m]).mean()); ns.append(int(m.sum()))
    ax.bar(xs, ys, color=ACCENT, width=0.68)
    ax.axhline(rep["majority"], color="#B3261E", ls="--", lw=1.2, label="다수클래스")
    lab = ["<0.5s", "0.5-1s", "1-1.5s", "1.5-2s", "2-3s", "3-5s", ">5s"]
    ax.set_xticks(xs, [lab[i] for i in xs], fontsize=8, rotation=30)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("정확도")
    ax.set_title("발화 길이별 정확도", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(FIGS / "m2_speaker.png")
    plt.close(fig)
    print("saved figs/m2_speaker.png")


# ------------------------------------------------------------------ M3
def m3():
    d = np.load(CACHE / "m3_val_pred.npz", allow_pickle=True)
    yt, yp, sc = d["y_true"], d["y_pred"], d["y_score"]
    macro, per = macro_f1_pdf(yt, yp)

    # (1) 클래스별 2x2 혼동행렬 3x3 격자
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10))
    for k, ax in enumerate(axes.ravel()):
        draw_cm(ax, binary_confusion(yt[:, k], yp[:, k]), ["없음", "있음"],
                f"{SYMPTOM_9[k]}  F1={per[k]['f1']:.2f}",
                xlab=(k // 3 == 2), ylab=(k % 3 == 0))
    fig.suptitle(f"Mission 3 · 클래스별 혼동행렬 (macro-F1 = {macro:.4f})",
                 fontsize=13, color=INK, y=0.995)
    fig.tight_layout()
    fig.savefig(FIGS / "m3_confusion_per_class.png")
    plt.close(fig)

    # (2) 클래스별 P/R/F1 + 지지도
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    _style(ax1); _style(ax2)
    x = np.arange(9); w = 0.27
    ax1.bar(x - w, [p["precision"] for p in per], w, label="Precision", color="#9FB6C9")
    ax1.bar(x, [p["recall"] for p in per], w, label="Recall", color="#5B8DB8")
    ax1.bar(x + w, [p["f1"] for p in per], w, label="F1", color=ACCENT)
    ax1.axhline(macro, color="#B3261E", ls="--", lw=1.2, label=f"macro-F1 {macro:.3f}")
    ax1.set_xticks(x, SYMPTOM_9, rotation=30, ha="right", fontsize=9)
    ax1.set_ylim(0, 1); ax1.legend(fontsize=8, frameon=False, ncol=2)
    ax1.set_title("클래스별 성능", fontsize=11)

    sup = yt.sum(axis=0)
    o = np.argsort(sup)
    ax2.barh(range(9), sup[o], color="#9FB6C9", height=0.7)
    ax2.set_yticks(range(9), [SYMPTOM_9[i] for i in o], fontsize=9)
    ax2.set_xlabel("Validation 내 양성 샘플 수")
    ax2.set_title("클래스 빈도", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "m3_per_class.png")
    plt.close(fig)

    # (3) 라벨 간 혼동 히트맵: 정답 i 인데 j 를 예측한 비율
    M = np.zeros((9, 9))
    for i in range(9):
        rows = yt[:, i] == 1
        if rows.sum():
            M[i] = yp[rows].mean(axis=0)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(9), SYMPTOM_9, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(9), SYMPTOM_9, fontsize=9)
    ax.set_xlabel("예측된 증상"); ax.set_ylabel("실제 증상")
    ax.set_title("라벨 간 혼동 — 정답이 i일 때 j를 예측한 비율", fontsize=10.5, pad=10)
    for i in range(9):
        for j in range(9):
            if M[i, j] > 0.04:
                ax.text(j, i, f"{M[i,j]*100:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white" if M[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.045)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(FIGS / "m3_label_confusion.png")
    plt.close(fig)
    print("saved figs/m3_confusion_per_class.png, m3_per_class.png, m3_label_confusion.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["m1", "m2", "m3"]
    for w in which:
        try:
            globals()[w]()
        except FileNotFoundError as e:
            print(f"skip {w}: {e}")
