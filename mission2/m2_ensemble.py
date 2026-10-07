"""
Mission 2 — WavLM base + large 앙상블 (재학습 없음)
  결합: 발화별 두 모델 logit 단순 평균 (가중치 튜닝 없음), 임계값 0.5 고정
  --split internal   : Training 내부 val 로 채택 여부 결정
  --split validation : 채택 후 Validation 1회 평가 (보고용)
  각 발화는 독립 추론 (다른 발화 정보 미참조)
"""
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import m2_preprocess as P
from m2_train_wavlm_v2 import WavLMRole, SR, DEFAULTS as TD
from m2_eval_wavlm import ValDS, collate, DEFAULTS as ED
from m2_eval_validation import metrics, bucket_acc

DEFAULTS = dict(
    split="internal",
    base_ckpt="checkpoints/m2_wavlm_v2/best.pt",
    base_pretrained="pretrained/wavlm-base-plus",
    large_ckpt="checkpoints/m2_wavlm_large/best.pt",
    large_pretrained="pretrained/wavlm-large",
    out_dir="outputs/m2_ensemble",
    seed=42, val_ratio=0.1,
    stride_sec=1.5, max_windows=8, batch_utts=32, chunk=64, num_workers=8,
)


def internal_val_records(seed, val_ratio):
    """학습 스크립트와 동일한 방식으로 내부 val 통화 재현."""
    records = P.build_index(TD["label_dir"], TD["audio_dir"])
    by_call = {}
    for r in records:
        by_call.setdefault(r["call_id"], []).append(r)
    calls = []
    for cid, rs in by_call.items():
        rs2 = [r for r in rs if int(r["raw_e"]) > int(r["raw_s"])]
        if len(rs2) >= 4 and len({int(r["speaker"]) for r in rs2}) == 2:
            calls.append(rs2)
    perm = np.random.RandomState(seed).permutation(len(calls))
    n_val = max(1, int(len(calls) * val_ratio))
    return [r for i in perm[:n_val] for r in calls[i]]


def load(ckpt, pretrained, device):
    ck = torch.load(ckpt, map_location="cpu")
    m = WavLMRole(pretrained, ck["emb_dim"])
    m.load_state_dict(ck["state_dict"]); m.to(device).eval()
    print(f"[ckpt] {ckpt} epoch={ck.get('epoch')} val_acc={ck.get('val_acc')}", flush=True)
    return m, int(ck.get("win_sec", 3.0) * SR)


def summarize(name, rows, key):
    rr = [dict(r, prob=float(1 / (1 + np.exp(-r[key])))) for r in rows]
    m = metrics([r["true"] for r in rr], [r["prob"] for r in rr])
    m["by_duration"] = bucket_acc(rr, lambda r: r["dur"], [0, 0.5, 1.0, 3.0, 1e9],
                                  ["<0.5s", "0.5-1s", "1-3s", ">3s"])
    m["by_overlap"] = bucket_acc(rr, lambda r: r["ov"], [0, 1, 1e9], ["겹침없음", "겹침있음"])
    return m


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir) / args.split; out.mkdir(parents=True, exist_ok=True)

    if args.split == "internal":
        recs = internal_val_records(args.seed, args.val_ratio)
    else:
        recs = P.build_index(ED["label_dir"], ED["audio_dir"])
    print(f"[data] split={args.split} utterances={len(recs):,}", flush=True)

    mb, win_b = load(args.base_ckpt, args.base_pretrained, device)
    ml, win_l = load(args.large_ckpt, args.large_pretrained, device)
    assert win_b == win_l, "두 모델 입력 길이가 다름"

    ld = DataLoader(ValDS(recs, win_b, int(args.stride_sec * SR), args.max_windows),
                    batch_size=args.batch_utts, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate)
    lb = np.zeros(len(recs), np.float32); ll = np.zeros(len(recs), np.float32)
    done = 0
    with torch.no_grad():
        for ws, counts, idxs in ld:
            outs = {}
            for tag, model in (("b", mb), ("l", ml)):
                parts = []
                for i in range(0, ws.size(0), args.chunk):
                    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                        _, g = model(ws[i:i + args.chunk].to(device))
                    parts.append(g.float().cpu())
                outs[tag] = torch.cat(parts)
            pos = 0
            for c, idx in zip(counts, idxs):
                lb[idx] = outs["b"][pos:pos + c].mean().item()
                ll[idx] = outs["l"][pos:pos + c].mean().item()
                pos += c
            done += len(idxs)
            if done % (args.batch_utts * 300) < args.batch_utts:
                print(f"  {done:,}/{len(recs):,}", flush=True)

    rows = [dict(file=Path(r["wav"]).name, start=int(r["raw_s"]), end=int(r["raw_e"]),
                 true=int(r["speaker"]), lb=float(a), ll=float(b), le=float((a + b) / 2),
                 dur=(int(r["raw_e"]) - int(r["raw_s"])) / 1000, ov=int(r["ov"]))
            for r, a, b in zip(recs, lb, ll)]
    res = {k: summarize(k, rows, key) for k, key in
           (("base", "lb"), ("large", "ll"), ("ensemble", "le"))}

    # 두 모델 오답 겹침 (앙상블 효과의 근거)
    eb = np.array([(r["lb"] > 0) != r["true"] for r in rows])
    el = np.array([(r["ll"] > 0) != r["true"] for r in rows])
    overlap = {"base만 오답": int((eb & ~el).sum()), "large만 오답": int((~eb & el).sum()),
               "둘다 오답": int((eb & el).sum())}

    np.savez(out / "logits.npz", lb=lb, ll=ll, true=np.array([r["true"] for r in rows]))
    json.dump({"split": args.split, "retrained": False, "combine": "mean_logit",
               "threshold": 0.5, "results": res, "error_overlap": overlap,
               "elapsed_sec": round(time.time() - t0, 1)},
              open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if args.split == "validation":
        with open(out / "mission2.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["오디오파일명", "start", "end", "label"])
            for r in rows:
                w.writerow([r["file"], r["start"], r["end"], int(r["le"] > 0)])

    print("\n" + "=" * 62)
    print(f"Mission 2 앙상블 평가 — {args.split} (재학습 없음, 임계값 0.5)")
    print("=" * 62)
    print(f"발화 {len(rows):,}건 / 소요 {time.time()-t0:.0f}초 / 저장 {out}")
    print(f"  {'':<10}{'Accuracy':>10}{'Macro-F1':>10}{'ROC-AUC':>10}")
    for k in ("base", "large", "ensemble"):
        m = res[k]
        print(f"  {k:<10}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}{m['roc_auc']:>10.4f}")
    print(f"\n  앙상블 Confusion: {res['ensemble']['confusion']}  (행=정답 0대원/1신고자)")
    print(f"  앙상블 길이별  : {res['ensemble']['by_duration']}")
    print(f"  앙상블 겹침별  : {res['ensemble']['by_overlap']}")
    print(f"  오답 겹침      : {overlap}")
    d = res["ensemble"]["accuracy"] - res["large"]["accuracy"]
    print(f"\n  앙상블 - large = {d*100:+.2f}%p", end="")
    if args.split == "internal":
        print("  →  " + ("앙상블 채택" if d > 0 else "large 단독 유지"))
    else:
        print()
    print("=" * 62)


if __name__ == "__main__":
    main()
