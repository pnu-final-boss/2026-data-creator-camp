# -*- coding: utf-8 -*-
"""Mission 3 라벨 간 상관 분석 — 이진 변수 쌍에 맞는 통계로.

두 라벨은 모두 0/1 이진 변수다. 이런 쌍에 피어슨 상관을 그냥 쓰면 안 되고
(정확히는 이진×이진에서 피어슨은 phi 계수와 같아지므로 결과는 같지만 해석이 달라진다),
2x2 분할표를 만들어 다음 세 가지를 함께 봐야 한다.

  1) phi 계수      : 효과크기. -1 ~ +1. "얼마나 강한 관계인가"
  2) 카이제곱 p값  : 유의성. "우연일 확률"
  3) lift          : 실용적 해석. 독립일 때 대비 몇 배로 함께 나오는가

중요 — 표본이 크면 p값은 거의 항상 유의해진다. n=29,200 에서는 phi=0.02 같은
사실상 무의미한 관계도 p<0.001 이 나온다. 그래서 p값만 보고 판단하면 안 되고
반드시 phi(효과크기)를 함께 봐야 한다. 이 스크립트는 둘 다 출력한다.

또 9개 클래스에서 쌍은 36개다. 유의수준 0.05 로 36번 검정하면 우연히 유의한 게
평균 1.8개 나온다(다중검정 문제). Bonferroni 와 Benjamini-Hochberg 보정을 함께 적용한다.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams
from scipy.stats import chi2, fisher_exact

from ..common.labels import load_split
from ..common.paths import CACHE, FIGS, SYMPTOM_9
from ..m3_symptom.train import build_labels

for _c in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    if any(_c == f.name for f in font_manager.fontManager.ttflist):
        rcParams["font.family"] = _c
        break
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
rcParams["savefig.bbox"] = "tight"
rcParams["savefig.facecolor"] = "white"

INK, MUTED, GRID = "#16202B", "#5D6C7B", "#DCE3EA"


def pair_stats(a, b):
    """두 이진 벡터의 2x2 분할표와 통계량.

    분할표
                b=0    b=1
        a=0     n00    n01
        a=1     n10    n11
    """
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    n = n11 + n10 + n01 + n00

    # phi 계수 = 이진 변수용 상관계수. 분모는 네 주변합의 곱의 제곱근.
    num = n11 * n00 - n10 * n01
    den = np.sqrt(float(n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = num / den if den > 0 else 0.0

    # 카이제곱 통계량은 n * phi^2 와 같다 (2x2 에서). 자유도 1.
    chi2_stat = n * phi * phi
    p = float(chi2.sf(chi2_stat, df=1))

    # 기대빈도가 5 미만인 칸이 있으면 카이제곱 근사가 깨지므로 피셔 정확검정을 쓴다.
    exp_min = min((n11 + n10) * (n11 + n01), (n11 + n10) * (n10 + n00),
                  (n01 + n00) * (n11 + n01), (n01 + n00) * (n10 + n00)) / n
    if exp_min < 5:
        _, p = fisher_exact([[n00, n01], [n10, n11]])

    # lift: 실제 동시출현 확률 / 독립이라면 기대되는 확률
    p_a, p_b = (n11 + n10) / n, (n11 + n01) / n
    lift = (n11 / n) / (p_a * p_b) if p_a * p_b > 0 else np.nan
    return dict(n11=n11, n10=n10, n01=n01, n00=n00, phi=phi,
                chi2=chi2_stat, p=p, lift=lift, exp_min=exp_min)


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR 보정. 기각된 것들의 불리언 배열을 반환."""
    p = np.asarray(pvals)
    m = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    k = np.flatnonzero(passed)
    out = np.zeros(m, bool)
    if len(k):
        out[order[: k[-1] + 1]] = True
    return out


def analyze(split="train"):
    recs = load_split(split)
    Y = build_labels(recs)
    n, K = Y.shape
    print(f"[label-corr] {split}: {n:,}통 x {K}클래스\n")

    PHI = np.zeros((K, K))
    LIFT = np.ones((K, K))
    pairs = []
    for i in range(K):
        PHI[i, i] = 1.0
        for j in range(i + 1, K):
            s = pair_stats(Y[:, i], Y[:, j])
            PHI[i, j] = PHI[j, i] = s["phi"]
            LIFT[i, j] = LIFT[j, i] = s["lift"]
            pairs.append((i, j, s))

    pvals = np.array([s["p"] for _, _, s in pairs])
    m = len(pairs)
    bonf = pvals < (0.05 / m)
    fdr = bh_fdr(pvals, 0.05)

    print(f"검정한 쌍의 수: {m}개  (9개 클래스에서 나오는 모든 조합)")
    print(f"  p < 0.05 그냥       : {int((pvals < 0.05).sum()):2d}개")
    print(f"  Bonferroni (p<{0.05/m:.5f}) : {int(bonf.sum()):2d}개")
    print(f"  BH-FDR 5%           : {int(fdr.sum()):2d}개")
    print(f"  그런데 |phi| >= 0.10 인 쌍 : {int((np.abs([s['phi'] for _,_,s in pairs]) >= 0.10).sum()):2d}개")
    print("  -> 표본이 크면 p값은 거의 다 유의해진다. 효과크기(phi)를 함께 봐야 한다.\n")

    order = sorted(range(m), key=lambda k: -abs(pairs[k][2]["phi"]))
    print("=== 효과크기 상위 12쌍 ===")
    print(f'{"쌍":24s} {"phi":>7s} {"lift":>6s} {"동시":>6s} {"p값":>10s} {"Bonf":>5s}')
    for k in order[:12]:
        i, j, s = pairs[k]
        nm = f"{SYMPTOM_9[i]} ↔ {SYMPTOM_9[j]}"
        pv = "<1e-300" if s["p"] < 1e-300 else f"{s['p']:.2e}"
        print(f'{nm:24s} {s["phi"]:+7.3f} {s["lift"]:6.2f} {s["n11"]:6,} {pv:>10s} {"O" if bonf[k] else "X":>5s}')

    print("\n=== 효과크기 하위 5쌍 (유의하지만 무의미한 예) ===")
    for k in order[-5:]:
        i, j, s = pairs[k]
        nm = f"{SYMPTOM_9[i]} ↔ {SYMPTOM_9[j]}"
        pv = "<1e-300" if s["p"] < 1e-300 else f"{s['p']:.2e}"
        print(f'{nm:24s} {s["phi"]:+7.3f} {s["lift"]:6.2f} {s["n11"]:6,} {pv:>10s} {"O" if bonf[k] else "X":>5s}')

    np.savez(CACHE / "m3_label_corr.npz", phi=PHI, lift=LIFT,
             classes=np.array(SYMPTOM_9), pvals=pvals,
             pair_idx=np.array([(i, j) for i, j, _ in pairs]))
    return PHI, LIFT, pairs, bonf


def draw(PHI, LIFT, pairs, bonf):
    K = len(SYMPTOM_9)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))

    # ---- 왼쪽: phi 상관행렬 ----
    ax = axes[0]
    M = PHI.copy()
    np.fill_diagonal(M, np.nan)
    v = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(K), SYMPTOM_9, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(K), SYMPTOM_9, fontsize=9)
    ax.set_title("phi 계수 (이진 변수용 상관)\n빨강 = 함께 나타남 · 파랑 = 서로 배타",
                 fontsize=11, color=INK, pad=10)
    # 유의한 쌍에 별표
    sig = {(i, j) for (i, j, _), b in zip(pairs, bonf) if b}
    for i in range(K):
        for j in range(K):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color=MUTED, fontsize=9)
                continue
            mark = "*" if (min(i, j), max(i, j)) in sig else ""
            ax.text(j, i, f"{M[i,j]:+.2f}{mark}", ha="center", va="center",
                    fontsize=7.4, color="white" if abs(M[i, j]) > v * 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.045, label="phi")

    # ---- 오른쪽: lift ----
    ax = axes[1]
    L = LIFT.copy()
    np.fill_diagonal(L, np.nan)
    im = ax.imshow(L, cmap="PuOr_r", vmin=0, vmax=np.nanmax(L))
    ax.set_xticks(range(K), SYMPTOM_9, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(K), SYMPTOM_9, fontsize=9)
    ax.set_title("lift (독립이면 1.0)\n2.0 = 독립일 때보다 2배 자주 함께 나타남",
                 fontsize=11, color=INK, pad=10)
    for i in range(K):
        for j in range(K):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color=MUTED, fontsize=9)
                continue
            ax.text(j, i, f"{L[i,j]:.2f}", ha="center", va="center", fontsize=7.4,
                    color="white" if abs(L[i, j] - 1) > np.nanmax(L) * 0.35 else INK)
    fig.colorbar(im, ax=ax, fraction=0.045, label="lift")

    for a in axes:
        a.tick_params(colors=MUTED)
        for s in a.spines.values():
            s.set_color(GRID)
    fig.suptitle("Mission 3 · 라벨 간 상관 (Training 29,200통)   * = Bonferroni 보정 후 유의",
                 fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "m3_label_correlation.png")
    plt.close(fig)
    print(f"\nsaved figs/m3_label_correlation.png")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    PHI, LIFT, pairs, bonf = analyze("train")
    draw(PHI, LIFT, pairs, bonf)
