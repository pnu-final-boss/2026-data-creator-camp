"""
Mission 2 — 3모델 앙상블 조합 비교 (재학습 없음, 임계값 0.5, logit 단순 평균)
  --split internal   : 내부 val 에서 조합 비교 → 최고 조합을 decision.json 에 기록
  --split validation : decision.json 의 조합으로 Validation 1회 평가 (보고용)
  각 발화는 독립 추론
"""
import argparse, csv, json, time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import m2_preprocess as P
from m2_train_wavlm_v2 import SR
from m2_eval_wavlm import ValDS, collate, DEFAULTS as ED
from m2_ensemble import internal_val_records, load, summarize

MODELS = {
    "base":  ("checkpoints/m2_wavlm_v2/best.pt",    "pretrained/wavlm-base-plus"),
    "large": ("checkpoints/m2_wavlm_large/best.pt", "pretrained/wavlm-large"),
    "w2v2":  ("checkpoints/m2_w2v2ko/best.pt",      "pretrained/w2v2-ko"),
}
COMBOS = [("base", "large"), ("large", "w2v2"), ("base", "large", "w2v2")]
DEFAULTS = dict(split="internal", out_dir="outputs/m2_ensemble3", seed=42, val_ratio=0.1,
                stride_sec=1.5, max_windows=8, batch_utts=32, chunk=64, num_workers=8)


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.out_dir); out = root / args.split; out.mkdir(parents=True, exist_ok=True)

    recs = (internal_val_records(args.seed, args.val_ratio) if args.split == "internal"
            else P.build_index(ED["label_dir"], ED["audio_dir"]))
    print(f"[data] split={args.split} utterances={len(recs):,}", flush=True)

    models, win = {}, None
    for k, (ck, pre) in MODELS.items():
        models[k], w = load(ck, pre, device)
        assert win in (None, w); win = w

    ld = DataLoader(ValDS(recs, win, int(args.stride_sec * SR), args.max_windows),
                    batch_size=args.batch_utts, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate)
    logits = {k: np.zeros(len(recs), np.float32) for k in MODELS}
    done = 0
    with torch.no_grad():
        for ws, counts, idxs in ld:
            for k, m in models.items():
                parts = []
                for i in range(0, ws.size(0), args.chunk):
                    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                        _, g = m(ws[i:i + args.chunk].to(device))
                    parts.append(g.float().cpu())
                g = torch.cat(parts); pos = 0
                for c, idx in zip(counts, idxs):
                    logits[k][idx] = g[pos:pos + c].mean().item(); pos += c
            done += len(idxs)
            if done % (args.batch_utts * 300) < args.batch_utts:
                print(f"  {done:,}/{len(recs):,}", flush=True)

    base_rows = [dict(file=Path(r["wav"]).name, start=int(r["raw_s"]), end=int(r["raw_e"]),
                      true=int(r["speaker"]), dur=(int(r["raw_e"]) - int(r["raw_s"])) / 1000,
                      ov=int(r["ov"])) for r in recs]
    cands = {k: logits[k] for k in MODELS}
    for c in COMBOS:
        cands["+".join(c)] = np.mean([logits[k] for k in c], 0)
    res = {}
    for name, lg in cands.items():
        rows = [dict(r, s=float(v)) for r, v in zip(base_rows, lg)]
        res[name] = summarize(name, rows, "s")

    dec_path = root / "decision.json"
    if args.split == "internal":
        chosen = max(["+".join(c) for c in COMBOS], key=lambda n: res[n]["accuracy"])
        json.dump({"chosen": chosen, "internal_acc": {n: res[n]["accuracy"] for n in res}},
                  open(dec_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    else:
        chosen = json.load(open(dec_path, encoding="utf-8"))["chosen"]
        with open(out / "mission2.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["오디오파일명", "start", "end", "label"])
            for r, v in zip(base_rows, cands[chosen]):
                w.writerow([r["file"], r["start"], r["end"], int(v > 0)])
    np.savez(out / "logits.npz", **logits, true=np.array([r["true"] for r in base_rows]))
    json.dump({"split": args.split, "chosen": chosen, "threshold": 0.5, "results": res},
              open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n" + "=" * 62)
    print(f"Mission 2 3모델 앙상블 — {args.split} (재학습 없음, 임계값 0.5)")
    print("=" * 62)
    print(f"발화 {len(recs):,}건 / 소요 {time.time()-t0:.0f}초")
    print(f"  {'구성':<18}{'Accuracy':>10}{'Macro-F1':>10}{'ROC-AUC':>10}")
    for n, m in res.items():
        mark = "  <== 채택" if n == chosen else ""
        print(f"  {n:<18}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}{m['roc_auc']:>10.4f}{mark}")
    m = res[chosen]
    print(f"\n  채택 Confusion: {m['confusion']}  (행=정답 0대원/1신고자)")
    print(f"  채택 길이별  : {m['by_duration']}")
    print(f"  채택 겹침별  : {m['by_overlap']}")
    print("=" * 62)


if __name__ == "__main__":
    main()
