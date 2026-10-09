"""
Mission 2 — Validation 최종 평가 (재학습 없음)
  strict : 원본 startAt~endAt 그대로 (규정상 안전)
  trim   : 이웃 발화 startAt/endAt 기반 겹침 트리밍 (학습 조건, 규정 회색지대)
이 결과로 체크포인트/임계값을 다시 고르지 말 것.
"""
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import m2_preprocess as P
from m2_train_contrastive import Encoder

VAL_ROOT = "/home/ldw2003/2026-data-creator-camp/data/raw/Validation"
DEFAULTS = dict(
    label_dir=f"{VAL_ROOT}/2.라벨링데이터/VL_서울_구급",
    audio_dir=f"{VAL_ROOT}/1.원천데이터/VS_서울_구급",
    ckpt="checkpoints/m2/m2_encoder_best.pt",
    out_dir="outputs/m2_validation_full",
    batch=512,
    num_workers=4,
)
MODES = {"strict": ("raw_s", "raw_e"), "trim": ("s_ms", "e_ms")}


class CallDS(Dataset):
    """통화 1건 = 오디오 1회 로드 -> 모든 발화 특징을 두 모드로 생성."""
    def __init__(self, calls):
        self.calls = calls

    def __len__(self):
        return len(self.calls)

    def __getitem__(self, i):
        cid, wav, rs = self.calls[i]
        try:
            w = P.load_call_audio(wav)
            mu, sg = P.call_stats(w)
        except Exception as e:
            return cid, rs, None, str(e)
        out = {}
        for mode, (ks, ke) in MODES.items():
            fs, ms = [], []
            for r in rs:
                f, m = P.segment_feature(w, r[ks], r[ke], mu, sg, train=False)
                fs.append(f); ms.append(m)
            out[mode] = (torch.stack(fs), torch.stack(ms))
        return cid, rs, out, None


def rankdata(a):
    """scipy.stats.rankdata(method='average') 대체 — 동점은 평균 순위."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def metrics(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    f1 = lambda a, b, c: 2 * a / max(1, 2 * a + b + c)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    r = rankdata(p)
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0)
    return dict(n=len(y), accuracy=(tp + tn) / max(1, len(y)),
                macro_f1=(f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
                roc_auc=float(auc),
                confusion=[[tn, fp], [fn, tp]])   # 행=정답(0 대원,1 신고자)


def bucket_acc(rows, key, edges, labels):
    out = {}
    for lo, hi, lb in zip(edges[:-1], edges[1:], labels):
        sel = [r for r in rows if lo <= key(r) < hi]
        if sel:
            acc = sum(int(r["prob"] >= 0.5) == r["true"] for r in sel) / len(sel)
            out[lb] = (len(sel), round(acc, 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    records = P.build_index(args.label_dir, args.audio_dir)
    by_call = {}
    for r in records:
        by_call.setdefault(r["call_id"], []).append(r)
    calls = [(c, rs[0]["wav"], rs) for c, rs in by_call.items()]
    print(f"[data] calls={len(calls):,} utterances={len(records):,}", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    model = Encoder(ck.get("emb_dim", 192))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    print(f"[ckpt] {args.ckpt} epoch={ck.get('epoch')} val_acc={ck.get('val_acc')}", flush=True)

    loader = DataLoader(CallDS(calls), batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=lambda b: b[0])
    shape = None
    rows = {m: [] for m in MODES}
    failed = []

    @torch.no_grad()
    def infer(feats, masks):
        nonlocal shape
        probs = []
        for i in range(0, feats.size(0), args.batch):
            x = feats[i:i + args.batch].to(device)
            m = masks[i:i + args.batch].to(device)
            if shape is None:                       # 입력 차원 자동 판별
                try:
                    model(x.unsqueeze(1), m); shape = "4d"
                except Exception:
                    model(x, m); shape = "3d"
                print(f"[model] input shape = {shape}", flush=True)
            _, logit = model(x.unsqueeze(1) if shape == "4d" else x, m)
            probs.append(torch.sigmoid(logit).float().cpu())
        return torch.cat(probs).numpy()

    for cid, rs, out, err in tqdm(loader, desc="[eval]"):
        if out is None:
            failed.append((cid, err)); continue
        for mode, (feats, masks) in out.items():
            p = infer(feats, masks)
            ks, ke = MODES[mode]
            for r, pi in zip(rs, p):
                rows[mode].append(dict(file=Path(r["wav"]).name,
                                       start=int(r["raw_s"]), end=int(r["raw_e"]),
                                       prob=float(pi), true=int(r["speaker"]),
                                       dur=(r["raw_e"] - r["raw_s"]) / 1000.0,
                                       ov=int(r["ov"])))

    elapsed = time.time() - t0
    summary = {"calls": len(calls), "utterances": len(records),
               "failed_calls": len(failed), "elapsed_sec": round(elapsed, 1),
               "ckpt": args.ckpt, "retrained": False}

    for mode, rr in rows.items():
        met = metrics([r["true"] for r in rr], [r["prob"] for r in rr])
        met["by_duration"] = bucket_acc(rr, lambda r: r["dur"],
                                        [0, 0.5, 1.0, 3.0, 1e9],
                                        ["<0.5s", "0.5-1s", "1-3s", ">3s"])
        met["by_overlap"] = bucket_acc(rr, lambda r: r["ov"],
                                       [0, 1, 1e9], ["겹침없음", "겹침있음"])
        summary[mode] = met
        with open(out_dir / f"mission2_{mode}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["오디오파일명", "start", "end", "label", "prob", "true"])
            for r in rr:
                w.writerow([r["file"], r["start"], r["end"],
                            int(r["prob"] >= 0.5), round(r["prob"], 6), r["true"]])

    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    if failed:
        with open(out_dir / "failed.txt", "w") as f:
            for c, e in failed:
                f.write(f"{c}\t{e}\n")

    print("\n" + "=" * 60)
    print("Mission 2 Validation 평가 (재학습 없음)")
    print("=" * 60)
    print(f"통화 {len(calls):,}건 / 발화 {len(records):,}건 / 실패 통화 {len(failed)}건")
    print(f"소요 시간 {elapsed:.0f}초 / 저장 {out_dir}")
    for mode in MODES:
        m = summary[mode]
        print(f"\n[{mode}]")
        print(f"  Accuracy : {m['accuracy']:.6f}")
        print(f"  Macro-F1 : {m['macro_f1']:.6f}")
        print(f"  ROC-AUC  : {m['roc_auc']:.6f}")
        print(f"  Confusion: {m['confusion']}  (행=정답 0대원/1신고자)")
        print(f"  길이별   : {m['by_duration']}")
        print(f"  겹침별   : {m['by_overlap']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
