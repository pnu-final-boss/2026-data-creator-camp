# -*- coding: utf-8 -*-
"""Mission 3 ablation: class_weight 와 화자 태그가 macro-F1 에 실제로 기여하는지 검증.

본 학습(train.py)에서 class_weight='balanced' 를 쓰자 임계값 튜닝 효과가 0 에 가까웠다.
balanced 가 이미 결정경계를 옮겨놓았기 때문이라는 가설을 여기서 확인한다.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from ..common.labels import load_split
from ..common.metrics import macro_f1_pdf, tune_thresholds
from ..common.paths import CACHE, SYMPTOM_9
from .train import build_labels, build_text, make_vectorizer, scores_of


def one(tag, tagged, balanced, Xtr_txt, Xva_txt, Ytr, Yva, fit_i, dev_i):
    vec = make_vectorizer()
    Xfit = vec.fit_transform([Xtr_txt[i] for i in fit_i])
    Xdev = vec.transform([Xtr_txt[i] for i in dev_i])
    Xva = vec.transform(Xva_txt)
    clf = OneVsRestClassifier(
        LogisticRegression(C=4.0, max_iter=2000,
                           class_weight="balanced" if balanced else None), n_jobs=9)
    clf.fit(Xfit, Ytr[fit_i])
    th = tune_thresholds(Ytr[dev_i], scores_of(clf, Xdev))
    s = scores_of(clf, Xva)
    m05, _ = macro_f1_pdf(Yva, (s >= 0.5).astype(int))
    mth, per = macro_f1_pdf(Yva, (s >= th[None, :]).astype(int))
    print(f"[ablate] {tag:28s} macro-F1 @0.5={m05:.4f}  @tuned={mth:.4f}  (+{mth-m05:.4f})",
          flush=True)
    return dict(tag=tag, tagged=tagged, balanced=balanced,
                macro_f1_05=float(m05), macro_f1_tuned=float(mth),
                thresholds=th.tolist(),
                per_class={c: per[i]["f1"] for i, c in enumerate(SYMPTOM_9)})


def main():
    tr, va = load_split("train"), load_split("val")
    Ytr, Yva = build_labels(tr), build_labels(va)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tr))
    n_dev = int(len(tr) * 0.1)
    dev_i, fit_i = perm[:n_dev], perm[n_dev:]

    out = []
    for tagged in (True, False):
        for balanced in (True, False):
            txt_tr = [build_text(r, tagged) for r in tr]
            txt_va = [build_text(r, tagged) for r in va]
            tag = f"tagged={tagged} balanced={balanced}"
            out.append(one(tag, tagged, balanced, txt_tr, txt_va,
                           Ytr, Yva, fit_i, dev_i))
            with open(CACHE / "m3_ablation.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
    print("[ablate] done")


if __name__ == "__main__":
    main()
