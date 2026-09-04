# -*- coding: utf-8 -*-
"""세 미션의 결과 리포트를 하나의 JSON 으로 모은다 (문서/PPT 작성용)."""
import json

import numpy as np

from ..common.metrics import macro_f1_pdf
from ..common.paths import CACHE, OUTPUTS, SYMPTOM_9


def _load(name):
    p = CACHE / name
    if not p.exists():
        return None
    return json.load(open(p, encoding="utf-8"))


def main():
    out = {}

    m1 = _load("m1_report.json")
    if m1:
        best = m1["best"]
        out["m1"] = {
            "task": "신고자 성별 (M/F)", "metric": "Accuracy",
            "majority_baseline": m1["majority"],
            "best_model": best,
            "accuracy": m1["results"][best]["acc"],
            "auc": m1["results"][best]["auc"],
            "confusion_matrix": m1["results"][best]["cm"],
            "all_models": {k: {"acc": v["acc"], "auc": v["auc"]} for k, v in m1["results"].items()},
        }

    m2 = _load("m2_report.json")
    if m2:
        out["m2"] = {
            "task": "신고자(1)/119대원(0)", "metric": "Accuracy",
            "majority_baseline": m2["majority"],
            "accuracy": m2["acc"], "auc": m2["auc"],
            "confusion_matrix": m2["cm"],
            "accuracy_nonoverlap": m2["acc_nonoverlap"],
            "accuracy_overlap": m2["acc_overlap"],
            "n_val_segments": m2["n_val"], "n_overlap": m2["n_overlap"],
            "feature_mode": m2["mode"], "model": m2["model"],
        }

    m3 = _load("m3_report.json")
    if m3:
        d = np.load(CACHE / "m3_val_pred.npz", allow_pickle=True)
        macro, per = macro_f1_pdf(d["y_true"], d["y_pred"])
        out["m3"] = {
            "task": "증상 9클래스 multi-label", "metric": "Macro-F1",
            "macro_f1": macro,
            "macro_f1_at_0.5": m3["macro_f1_05"],
            "model": m3["model"],
            "per_class": {c: {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in per[i].items()}
                          for i, c in enumerate(SYMPTOM_9)},
            "thresholds": {c: round(t, 3) for c, t in zip(SYMPTOM_9, m3["thresholds"])},
        }

    ab = _load("m3_ablation.json")
    if ab:
        out["m3_ablation"] = ab

    p = OUTPUTS / "results_summary.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved {p}")
    for k, v in out.items():
        if k == "m3_ablation":
            continue
        head = v.get("accuracy", v.get("macro_f1"))
        print(f"  {k}: {v['task']}  {v['metric']}={head:.4f}  "
              f"(baseline {v.get('majority_baseline', 0):.4f})")


if __name__ == "__main__":
    main()
