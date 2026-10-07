"""
Mission 3 앙상블 (임계값 0.5 고정)
  mean = 확률 평균 >= 0.5 / vote = 각자 0.5 판정 다수결, 동점은 확률 평균
  공통 val(모든 모델 미학습)에서 최고 구성 채택 -> Validation 1회
"""
import argparse, csv, json
from itertools import combinations
from pathlib import Path
import numpy as np
from m3_train_qwen import SYMPTOMS, DEFAULTS as TD, f1_report
from m3_final_eval import load_docs, split_like_training, VAL_LABEL_DIR

IN = Path("outputs/m3_team")


def read(path, names):
    rows = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows[r["라벨파일명"]] = [float(r[s]) for s in SYMPTOMS]
    miss = [n for n in names if n not in rows]
    assert not miss, f"{path}: 누락 {len(miss)}건"
    return np.array([rows[n] for n in names], dtype=np.float32)


def combine(Ps, method):
    if len(Ps) == 1:
        return (Ps[0] >= 0.5).astype(int)
    mean = np.mean(Ps, 0)
    if method == "mean":
        return (mean >= 0.5).astype(int)
    v = np.sum([(p >= 0.5) for p in Ps], 0); k = len(Ps)
    return np.where(v * 2 > k, 1, np.where(v * 2 < k, 0, (mean >= 0.5).astype(int)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    args = ap.parse_args()

    names, texts, Y = load_docs(TD["label_dir"])
    _, va = split_like_training(texts, 42, 0.1)
    int_names, Yi = [names[i] for i in va], Y[va]
    val_names, _, Yv = load_docs(VAL_LABEL_DIR)

    Pi = {m: read(IN / f"{m}_internal.csv", int_names) for m in args.members}
    Pv = {m: read(IN / f"{m}_validation.csv", val_names) for m in args.members}

    cands = []
    for k in range(1, len(args.members) + 1):
        for c in combinations(args.members, k):
            for meth in (["single"] if k == 1 else ["mean", "vote"]):
                f = f1_report(Yi, combine([Pi[x] for x in c], meth).astype(np.float32))["macro_f1"]
                cands.append((f, meth, c))
    cands.sort(key=lambda x: -x[0])
    f_best, meth, combo = cands[0]

    print("[공통 val] 상위 10개")
    for f, m, c in cands[:10]:
        print(f"  {f:.4f}  {m:<6} {'+'.join(c)}")
    print("  --- 단독 ---")
    for f, m, c in cands:
        if m == "single":
            print(f"  {f:.4f}  {c[0]}")
    print(f"\n>>> 채택: {meth} / {'+'.join(combo)} (공통 val {f_best:.4f})")

    pred = combine([Pv[x] for x in combo], meth)
    rep = f1_report(Yv, pred.astype(np.float32))
    print(f"\n[Validation {len(val_names):,}건 — 최종] macro F1 {rep['macro_f1']:.4f}")
    for s in SYMPTOMS:
        print(f"  {s:<6} F1 {rep[s]['f1']:.3f}  P {rep[s]['precision']:.3f}  R {rep[s]['recall']:.3f}")

    out = Path("outputs/m3_team_final"); out.mkdir(parents=True, exist_ok=True)
    json.dump({"chosen": {"members": list(combo), "method": meth}, "internal_macro_f1": f_best,
               "validation": rep, "threshold": 0.5,
               "candidates": [(f, m, list(c)) for f, m, c in cands]},
              open(out / "decision.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(out / "mission3.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["라벨파일명", "symptom"])
        for n, row in zip(val_names, pred):
            w.writerow([n, str([s for s, v in zip(SYMPTOMS, row) if v])])


if __name__ == "__main__":
    main()
