"""
Mission 3 — 한국어 인코더(RoBERTa 계열) 다중라벨 증상 분류
  긴 전사: 512토큰 창(stride 384, 최대 6창)으로 나눠 창별 logit -> 문서 단위 max-pool
  입력  : 전사 텍스트만 (m3_train_qwen.build_text 와 동일 전처리)
  분할  : m3_final_eval.split_like_training (Qwen 과 동일 = 팀 공통 val)
  손실  : BCE + pos_weight sqrt(음성/양성) (학습 전 공식 고정), 임계값 0.5
  종료 후 best 로 outputs/m3_team/{name}_internal.csv, {name}_validation.csv 저장
"""
import argparse, csv, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from m3_train_qwen import SYMPTOMS, DEFAULTS as TD, f1_report
from m3_final_eval import load_docs, split_like_training, VAL_LABEL_DIR

DEFAULTS = dict(
    pretrained="pretrained/klue-roberta-large",
    name="klue_large",
    out_dir="checkpoints/m3_klue_large",
    win=512, stride=384, max_chunks=6,
    batch=4, accum=4, epochs=3,
    lr=2e-5, head_lr=1e-4, warmup_ratio=0.06, weight_decay=0.01,
    pos_weight=1, seed=42, eval_batch=16, log_every=200,
)


def to_chunks(ids, win, stride, max_chunks, cls_id, sep_id):
    body = win - 2
    out, i = [], 0
    while True:
        out.append([cls_id] + ids[i:i + body] + [sep_id])
        if i + body >= len(ids) or len(out) >= max_chunks:
            break
        i += stride
    return out


def collate(docs, pad_id):
    flat, owner = [], []
    for b, chs in enumerate(docs):
        flat += chs; owner += [b] * len(chs)
    L = max(len(c) for c in flat)
    x = torch.full((len(flat), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(flat), L), dtype=torch.long)
    for i, c in enumerate(flat):
        x[i, :len(c)] = torch.tensor(c); m[i, :len(c)] = 1
    return x, m, torch.tensor(owner)


def doc_logits(model, x, m, owner, n_docs, device):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        lg = model(input_ids=x.to(device), attention_mask=m.to(device)).logits.float()
    owner = owner.to(device)
    return torch.stack([lg[owner == b].max(0).values for b in range(n_docs)])


@torch.no_grad()
def predict(model, chunks, pad_id, bs, device):
    model.eval()
    order = np.argsort([sum(len(c) for c in d) for d in chunks])
    P = np.zeros((len(chunks), len(SYMPTOMS)), dtype=np.float32)
    for i in range(0, len(order), bs):
        b = order[i:i + bs]
        x, m, o = collate([chunks[j] for j in b], pad_id)
        P[b] = torch.sigmoid(doc_logits(model, x, m, o, len(b), device)).cpu().numpy()
    return P


def write_csv(path, names, P):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["라벨파일명"] + SYMPTOMS)
        for n, p in zip(names, P):
            w.writerow([n] + [f"{v:.6f}" for v in p])


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    team = Path("outputs/m3_team"); team.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    names, texts, Y = load_docs(TD["label_dir"])
    tr, va = split_like_training(texts, 42, 0.1)
    print(f"[data] train={len(tr):,} val={len(va):,}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.pretrained)
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    chunks = [to_chunks(e, args.win, args.stride, args.max_chunks,
                        tok.cls_token_id, tok.sep_token_id) for e in enc]
    nch = np.array([len(c) for c in chunks])
    full = np.array([len(e) for e in enc]) <= (args.win - 2) + args.stride * (args.max_chunks - 1)
    print(f"[chunk] 창 수 p50={int(np.median(nch))} p90={int(np.percentile(nch,90))} "
          f"| 전체가 창에 들어오는 문서 {full.mean()*100:.1f}%", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained, num_labels=len(SYMPTOMS),
        problem_type="multi_label_classification").to(device)
    head = [p for n, p in model.named_parameters() if "classifier" in n]
    body = [p for n, p in model.named_parameters() if "classifier" not in n]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}],
                            weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(tr) / args.batch / args.accum)
    total = steps_per_epoch * args.epochs; warm = int(total * args.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    pw = None
    if args.pos_weight:
        pos = Y[tr].sum(0).clip(min=1)
        pw = np.sqrt((len(tr) - pos) / pos).astype(np.float32)
        print("[pos_weight] " + " | ".join(f"{s} {w:.2f}" for s, w in zip(SYMPTOMS, pw)), flush=True)
    bce = nn.BCEWithLogitsLoss(pos_weight=None if pw is None else torch.tensor(pw, device=device))

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        rng = np.random.RandomState(args.seed + ep)
        idx = rng.permutation(tr)
        te = time.time(); run = 0.0; n = 0
        opt.zero_grad(set_to_none=True)
        nb = math.ceil(len(idx) / args.batch)
        for bi in range(nb):
            b = idx[bi * args.batch:(bi + 1) * args.batch]
            x, m, o = collate([chunks[j] for j in b], tok.pad_token_id)
            lg = doc_logits(model, x, m, o, len(b), device)
            loss = bce(lg, torch.tensor(Y[b], device=device)) / args.accum
            loss.backward(); run += loss.item() * args.accum; n += 1
            if (bi + 1) % args.accum == 0 or bi + 1 == nb:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if (bi + 1) % (args.log_every * args.accum) == 0:
                eta = (time.time() - te) / (bi + 1) * (nb - bi - 1) / 60
                print(f"  [ep {ep} batch {bi+1}/{nb}] loss={run/n:.4f} eta={eta:.0f}min", flush=True)
                run = 0.0; n = 0

        pv = predict(model, [chunks[i] for i in va], tok.pad_token_id, args.eval_batch, device)
        rep = f1_report(Y[va], pv)
        print(f"[epoch {ep}] val macro_F1={rep['macro_f1']:.4f} ({(time.time()-te)/60:.0f}min)", flush=True)
        print("   " + " | ".join(f"{s} {rep[s]['f1']:.3f}" for s in SYMPTOMS), flush=True)
        if rep["macro_f1"] > best:
            best = rep["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "pretrained": args.pretrained,
                        "win": args.win, "stride": args.stride, "max_chunks": args.max_chunks,
                        "epoch": ep, "val_macro_f1": best, "threshold": 0.5,
                        "pos_weight": None if pw is None else pw.tolist()}, out / "best.pt")
            print(f"  -> best 저장 (macro_F1={best:.4f})", flush=True)

    # best 로 팀 공통 형식 확률 저장
    ck = torch.load(out / "best.pt", map_location="cpu")
    model.load_state_dict(ck["state_dict"])
    pv = predict(model, [chunks[i] for i in va], tok.pad_token_id, args.eval_batch, device)
    write_csv(team / f"{args.name}_internal.csv", [names[i] for i in va], pv)
    vnames, vtexts, _ = load_docs(VAL_LABEL_DIR)
    venc = tok(vtexts, add_special_tokens=False)["input_ids"]
    vch = [to_chunks(e, args.win, args.stride, args.max_chunks,
                     tok.cls_token_id, tok.sep_token_id) for e in venc]
    write_csv(team / f"{args.name}_validation.csv", vnames,
              predict(model, vch, tok.pad_token_id, args.eval_batch, device))
    print(f"[done] best val macro_F1={best:.4f} / 확률 CSV 저장: {team}", flush=True)


if __name__ == "__main__":
    main()
