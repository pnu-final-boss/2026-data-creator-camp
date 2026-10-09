"""
Mission 2 — WavLM v2 Validation 최종 평가 (재학습 없음)
  짧은 발화: 반복 패딩 / 긴 발화: sliding window + stride 후 logit 평균
  이 결과로 체크포인트나 임계값을 다시 고르지 말 것.
"""
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import m2_preprocess as P
from m2_train_wavlm_v2 import WavLMRole, load_segment, fit_length, normalize, SR
from m2_eval_validation import metrics, bucket_acc

VAL_ROOT = "/home/ldw2003/2026-data-creator-camp/data/raw/Validation"
DEFAULTS = dict(
    label_dir=f"{VAL_ROOT}/2.라벨링데이터/VL_서울_구급",
    audio_dir=f"{VAL_ROOT}/1.원천데이터/VS_서울_구급",
    ckpt="checkpoints/m2_wavlm_v2/best.pt",
    pretrained="pretrained/wavlm-base-plus",
    out_dir="outputs/m2_wavlm_v2_validation",
    stride_sec=1.5,
    max_windows=8,
    batch_utts=32,
    chunk=64,
    num_workers=8,
)


def make_windows(seg, win, stride, max_w):
    n = seg.numel()
    if n <= win:
        return fit_length(seg, win, train=False)[None]
    starts = list(range(0, n - win + 1, stride))
    if starts[-1] != n - win:
        starts.append(n - win)
    if len(starts) > max_w:
        idx = np.linspace(0, len(starts) - 1, max_w).round().astype(int)
        starts = [starts[i] for i in idx]
    return torch.stack([seg[s:s + win] for s in starts])


class ValDS(Dataset):
    def __init__(self, recs, win, stride, max_w):
        self.recs, self.win, self.stride, self.max_w = recs, win, stride, max_w

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        seg = load_segment(r["wav"], r["raw_s"], r["raw_e"])
        ws = make_windows(seg, self.win, self.stride, self.max_w)
        return torch.stack([normalize(w) for w in ws]), i


def collate(batch):
    ws = [b[0] for b in batch]
    return torch.cat(ws), [w.size(0) for w in ws], [b[1] for b in batch]


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    recs = P.build_index(args.label_dir, args.audio_dir)
    print(f"[data] calls={len({r['call_id'] for r in recs}):,} utterances={len(recs):,}", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    model = WavLMRole(args.pretrained, ck["emb_dim"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    win = int(ck.get("win_sec", 3.0) * SR)
    print(f"[ckpt] {args.ckpt} epoch={ck.get('epoch')} val_acc={ck.get('val_acc')}", flush=True)

    ld = DataLoader(ValDS(recs, win, int(args.stride_sec * SR), args.max_windows),
                    batch_size=args.batch_utts, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate)
    probs = np.zeros(len(recs), dtype=np.float32)
    done = 0
    with torch.no_grad():
        for ws, counts, idxs in ld:
            logits = []
            for i in range(0, ws.size(0), args.chunk):
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    _, lg = model(ws[i:i + args.chunk].to(device))
                logits.append(lg.float().cpu())
            logits = torch.cat(logits)
            pos = 0
            for c, idx in zip(counts, idxs):
                probs[idx] = torch.sigmoid(logits[pos:pos + c].mean()).item()
                pos += c
            done += len(idxs)
            if done % (args.batch_utts * 200) < args.batch_utts:
                print(f"  {done:,}/{len(recs):,}", flush=True)

    rows = [dict(file=Path(r["wav"]).name, start=int(r["raw_s"]), end=int(r["raw_e"]),
                 prob=float(p), true=int(r["speaker"]),
                 dur=(r["raw_e"] - r["raw_s"]) / 1000.0, ov=int(r["ov"]))
            for r, p in zip(recs, probs)]
    met = metrics([r["true"] for r in rows], [r["prob"] for r in rows])
    met["by_duration"] = bucket_acc(rows, lambda r: r["dur"], [0, 0.5, 1.0, 3.0, 1e9],
                                    ["<0.5s", "0.5-1s", "1-3s", ">3s"])
    met["by_overlap"] = bucket_acc(rows, lambda r: r["ov"], [0, 1, 1e9], ["겹침없음", "겹침있음"])
    elapsed = time.time() - t0

    with open(out_dir / "mission2.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["오디오파일명", "start", "end", "label", "prob", "true"])
        for r in rows:
            w.writerow([r["file"], r["start"], r["end"], int(r["prob"] >= 0.5),
                        round(r["prob"], 6), r["true"]])
    json.dump({"ckpt": args.ckpt, "retrained": False, "elapsed_sec": round(elapsed, 1),
               "utterances": len(rows), **met},
              open(out_dir / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Mission 2 WavLM v2 Validation 평가 (재학습 없음, strict)")
    print("=" * 60)
    print(f"발화 {len(rows):,}건 / 소요 {elapsed:.0f}초 / 저장 {out_dir}")
    print(f"  Accuracy : {met['accuracy']:.6f}")
    print(f"  Macro-F1 : {met['macro_f1']:.6f}")
    print(f"  ROC-AUC  : {met['roc_auc']:.6f}")
    print(f"  Confusion: {met['confusion']}  (행=정답 0대원/1신고자)")
    print(f"  길이별   : {met['by_duration']}")
    print(f"  겹침별   : {met['by_overlap']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
