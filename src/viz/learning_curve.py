# -*- coding: utf-8 -*-
"""학습 곡선: train vs validation 오차가 학습이 진행되며 어떻게 갈라지는지 본다.

주의 — 현재 CPU 베이스라인 중 '에폭'이 있는 모델은 일부뿐이다. 모델 성격에 맞는 x축을 쓴다.

  M1 (LogisticRegression) : 에폭 개념이 없다(lbfgs 는 수렴할 때까지 돈다).
                            대신 '학습 데이터 양'을 x축으로 두는 learning curve 를 그린다.
                            과소적합/과대적합과 '데이터를 더 넣으면 오를까'를 진단하는 표준 도구.
  M2 (HistGradientBoosting): 부스팅 반복(iteration)이 곧 에폭에 해당한다.
                            early_stopping=True 로 두면 sklearn 이 반복마다 train/val 점수를
                            train_score_ / validation_score_ 에 기록해준다.
  M3 (SGDClassifier)      : partial_fit 으로 진짜 에폭 루프를 돌려 매 에폭 train/val macro-F1 을 잰다.
                            (본 학습의 LogisticRegression 과는 다른 모델이지만, 같은 특징·같은 손실이라
                             수렴 양상 진단에는 유효하다. 권수정 외(2020)도 SGD 를 비교군으로 썼다.)

Colab 노트북(KLUE-RoBERTa / wav2vec2)에는 진짜 에폭이 있으므로 그쪽은 노트북 안에서 직접 기록한다.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..common.paths import CACHE, FIGS, SYMPTOM_9

for _c in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    if any(_c == f.name for f in font_manager.fontManager.ttflist):
        rcParams["font.family"] = _c
        break
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
rcParams["savefig.bbox"] = "tight"
rcParams["savefig.facecolor"] = "white"

INK, MUTED, GRID = "#16202B", "#5D6C7B", "#DCE3EA"
TRAIN_C, VAL_C = "#9FB6C9", "#1B5E96"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ M1
def m1(ax):
    """학습 데이터 양에 따른 train/val 정확도."""
    tr = np.load(CACHE / "audio_train.npz", allow_pickle=True)
    X = np.nan_to_num(tr["call_X"], nan=0.0, posinf=0.0, neginf=0.0)
    y = tr["call_g"]
    sizes, tr_s, va_s = learning_curve(
        make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
        X, y, train_sizes=np.linspace(0.1, 1.0, 8), cv=4, scoring="accuracy", n_jobs=8)
    _style(ax)
    ax.plot(sizes, tr_s.mean(1), "o-", color=TRAIN_C, lw=1.8, ms=4, label="Training")
    ax.plot(sizes, va_s.mean(1), "o-", color=VAL_C, lw=1.8, ms=4, label="Validation (4-fold CV)")
    ax.fill_between(sizes, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1),
                    color=VAL_C, alpha=0.13)
    ax.set_xlabel("학습 통화 수"); ax.set_ylabel("정확도")
    ax.set_title("M1 성별 · 데이터 양에 따른 곡선", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    return dict(sizes=sizes.tolist(), train=tr_s.mean(1).tolist(), val=va_s.mean(1).tolist())


# ------------------------------------------------------------------ M2
def m2(ax):
    """부스팅 반복(에폭)에 따른 train/val 정확도 — sklearn 이 자동 기록.

    특징은 실제 배포 모델과 동일하게 mode='both'(원본 143 + 통화별 평균차감 143 = 286)를 쓴다.
    여기서 utt_X 만 쓰면 raw 143차원 모델의 곡선이 되어 본 모델과 어긋난다.
    """
    from ..m2_speaker.train import build_matrix
    tr = np.load(CACHE / "audio_train.npz", allow_pickle=True)
    X = build_matrix(tr, "both")
    y = tr["utt_y"]
    clf = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.08, random_state=0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=500,
        scoring="accuracy")
    clf.fit(X, y)
    it = np.arange(1, len(clf.train_score_) + 1)
    _style(ax)
    ax.plot(it, clf.train_score_, color=TRAIN_C, lw=1.6, label="Training")
    ax.plot(it, clf.validation_score_, color=VAL_C, lw=1.8, label="Validation (15% holdout)")
    best = int(np.argmax(clf.validation_score_))
    ax.axvline(best + 1, color="#9E2B22", ls="--", lw=1.1)
    ax.annotate(f"val 최고 {clf.validation_score_[best]:.4f}\n@ {best+1}회",
                xy=(best + 1, clf.validation_score_[best]), xytext=(8, -28),
                textcoords="offset points", fontsize=8, color="#9E2B22")
    ax.set_xlabel("부스팅 반복 (= 에폭)"); ax.set_ylabel("정확도")
    ax.set_title("M2 화자역할 · 반복에 따른 곡선", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    return dict(iters=it.tolist(), train=list(map(float, clf.train_score_)),
                val=list(map(float, clf.validation_score_)), best_iter=best + 1)


# ------------------------------------------------------------------ M3
def m3(ax, epochs=12):
    """에폭에 따른 train/val macro-F1 — SGD partial_fit 으로 진짜 에폭 루프."""
    import pickle
    from ..common.labels import load_split
    from ..m3_symptom.train import build_labels, build_text, make_vectorizer

    tr, va = load_split("train"), load_split("val")
    Xtr_txt = [build_text(r, True) for r in tr]
    Xva_txt = [build_text(r, True) for r in va]
    Ytr, Yva = build_labels(tr), build_labels(va)

    vec = make_vectorizer()
    Xtr = vec.fit_transform(Xtr_txt)
    Xva = vec.transform(Xva_txt)

    # partial_fit 은 class_weight='balanced' 를 받지 않는다 (에폭마다 부분 데이터만 보므로
    # sklearn 이 전체 분포를 추정할 수 없기 때문). 전체 train 에서 미리 계산해 dict 로 넘긴다.
    from sklearn.utils.class_weight import compute_class_weight
    cw = []
    for k in range(9):
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=Ytr[:, k])
        cw.append({0: float(w[0]), 1: float(w[1])})

    clfs = [SGDClassifier(loss="log_loss", alpha=1e-5, class_weight=cw[k],
                          random_state=0) for k in range(9)]

    # train 쪽 평가는 val 과 같은 크기로 서브샘플링한다. 29k 전체를 매 에폭 9번 예측하면
    # 곡선 하나 그리는 데 학습보다 오래 걸린다. 곡선의 목적은 train-val 격차 추세다.
    rng = np.random.default_rng(0)
    sub = rng.choice(Xtr.shape[0], size=min(len(Yva), Xtr.shape[0]), replace=False)
    Xtr_ev, Ytr_ev = Xtr[sub], Ytr[sub]

    hist_tr, hist_va = [], []
    for ep in range(epochs):
        order = rng.permutation(Xtr.shape[0])          # 에폭마다 셔플
        Ptr = np.zeros_like(Ytr_ev, dtype=np.int8)
        Pva = np.zeros_like(Yva, dtype=np.int8)
        for k in range(9):
            clfs[k].partial_fit(Xtr[order], Ytr[order, k], classes=np.array([0, 1]))
            Ptr[:, k] = clfs[k].predict(Xtr_ev)
            Pva[:, k] = clfs[k].predict(Xva)
        f_tr = np.mean([f1_score(Ytr_ev[:, k], Ptr[:, k], zero_division=0) for k in range(9)])
        f_va = np.mean([f1_score(Yva[:, k], Pva[:, k], zero_division=0) for k in range(9)])
        hist_tr.append(f_tr); hist_va.append(f_va)
        print(f"  [m3-curve] epoch {ep+1:2d}  train macro-F1 {f_tr:.4f}  val {f_va:.4f}", flush=True)

    ep_ax = np.arange(1, epochs + 1)
    _style(ax)
    ax.plot(ep_ax, hist_tr, "o-", color=TRAIN_C, lw=1.8, ms=3.5, label="Training")
    ax.plot(ep_ax, hist_va, "o-", color=VAL_C, lw=1.8, ms=3.5, label="Validation")
    ax.axhline(0.5848, color="#2A6B48", ls=":", lw=1.3, label="LogReg 본 모델 0.5848")
    best = int(np.argmax(hist_va))
    ax.axvline(best + 1, color="#9E2B22", ls="--", lw=1.1)
    ax.set_xlabel("에폭"); ax.set_ylabel("macro-F1")
    ax.set_title("M3 증상 · 에폭에 따른 곡선 (SGD)", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    return dict(epochs=ep_ax.tolist(), train=hist_tr, val=hist_va, best_epoch=best + 1)


def _replot(name, ax, d):
    """이미 계산해 둔 곡선 데이터를 JSON 에서 읽어 다시 그린다 (재계산 없음)."""
    _style(ax)
    if name == "m1":
        ax.plot(d["sizes"], d["train"], "o-", color=TRAIN_C, lw=1.8, ms=4, label="Training")
        ax.plot(d["sizes"], d["val"], "o-", color=VAL_C, lw=1.8, ms=4, label="Validation (4-fold CV)")
        ax.set_xlabel("학습 통화 수"); ax.set_ylabel("정확도")
        ax.set_title("M1 성별 · 데이터 양에 따른 곡선", fontsize=11)
    elif name == "m2":
        ax.plot(d["iters"], d["train"], color=TRAIN_C, lw=1.6, label="Training")
        ax.plot(d["iters"], d["val"], color=VAL_C, lw=1.8, label="Validation (15% holdout)")
        b = d["best_iter"]
        ax.axvline(b, color="#9E2B22", ls="--", lw=1.1)
        ax.annotate(f"val 최고 {max(d['val']):.4f}\n@ {b}회", xy=(b, max(d["val"])),
                    xytext=(-64, -30), textcoords="offset points", fontsize=8, color="#9E2B22")
        ax.set_xlabel("부스팅 반복 (= 에폭)"); ax.set_ylabel("정확도")
        ax.set_title("M2 화자역할 · 반복에 따른 곡선", fontsize=11)
    else:
        ax.plot(d["epochs"], d["train"], "o-", color=TRAIN_C, lw=1.8, ms=3.5, label="Training")
        ax.plot(d["epochs"], d["val"], "o-", color=VAL_C, lw=1.8, ms=3.5, label="Validation")
        ax.axhline(0.5848, color="#2A6B48", ls=":", lw=1.3, label="LogReg 본 모델 0.5848")
        ax.axvline(d["best_epoch"], color="#9E2B22", ls="--", lw=1.1)
        ax.set_xlabel("에폭"); ax.set_ylabel("macro-F1")
        ax.set_title("M3 증상 · 에폭에 따른 곡선 (SGD)", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")


def main(only=None):
    """세 곡선을 한 장에 그린다.

    only 로 특정 미션만 재계산하고 나머지는 캐시에서 다시 그린다. M3 가 가장 비싸서
    M1/M2 를 매번 다시 돌리면 M3 에 쓸 시간이 남지 않는다.
    """
    path = CACHE / "learning_curves.json"
    out = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    todo = set(only) if only else {"m1", "m2", "m3"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, fn, ax in (("m1", m1, axes[0]), ("m2", m2, axes[1]), ("m3", m3, axes[2])):
        if name in todo:
            print(f"[curve] {name.upper()} 계산 ...", flush=True)
            try:
                out[name] = fn(ax)
            except Exception as e:                 # noqa: BLE001 - 하나 실패로 전체를 잃지 않는다
                print(f"[curve] {name} 실패: {type(e).__name__}: {e}", flush=True)
                _style(ax)
                ax.text(0.5, 0.5, f"{name.upper()} 생성 실패", ha="center", va="center",
                        color=MUTED, transform=ax.transAxes)
        elif name in out:
            print(f"[curve] {name.upper()} 캐시 사용", flush=True)
            _replot(name, ax, out[name])
        else:
            _style(ax)
            ax.text(0.5, 0.5, f"{name.upper()} 미생성", ha="center", va="center",
                    color=MUTED, transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(FIGS / "learning_curves.png")   # 진행분을 매번 저장
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved figs/learning_curves.png  (보유: {', '.join(sorted(out))})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=["m1", "m2", "m3"],
                    help="이 미션만 재계산. 나머지는 cache/learning_curves.json 에서 다시 그린다.")
    main(ap.parse_args().only)
