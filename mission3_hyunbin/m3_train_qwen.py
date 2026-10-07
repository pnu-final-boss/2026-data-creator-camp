"""
Mission 3 — Qwen2.5 다중라벨 증상 분류 (LoRA)

입력 : 전사 텍스트만. utterances[].text 를 JSON 순서대로 이어붙임.
       speaker, startAt/endAt 등 다른 라벨링 정보 미사용.
출력 : 9개 증상 독립 sigmoid, 임계값 0.5 고정
데이터: Training 서울만. Validation 미사용.
분할 : 동일 전사문은 같은 쪽에 배치 (중복 누수 방지)
"""
import argparse, hashlib, json, math, re, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

SYMPTOMS = ['고열', '구토', '두통', '복통', '어지러움', '열상', '오심', '전신쇠약', '호흡곤란']
HOME = str(Path.home())
DEFAULTS = dict(
    label_dir=f"{HOME}/대학부 데이터/2.라벨링데이터/TL_서울_구급",
    pretrained="pretrained/qwen2.5-1.5b",
    out_dir="checkpoints/m3_qwen15",
    max_len=2048,
    batch=4, accum=4,
    epochs=3,
    lr=2e-4, warmup_ratio=0.05,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    val_ratio=0.1, seed=42,
    grad_ckpt=1, eval_batch=8,
    max_files=0,          # 0 = 전체 (테스트용 축소 가능)
    log_every=100,
    pos_weight=0,         # 1 = 클래스별 sqrt(음성/양성) 가중 (학습 전 공식으로 고정)
)


# ---------------- 전처리 (추론 코드도 이 함수를 import 해서 사용) ----------------
def build_text(d):
    """전사 텍스트만 JSON 순서대로 연결. 다른 필드는 쓰지 않는다."""
    parts = [(u.get("text") or "").strip() for u in d.get("utterances", [])]
    return " ".join(p for p in parts if p)


def build_labels(d):
    s = d.get("symptom") or []
    if isinstance(s, str):
        s = [s]
    s = {str(x).strip() for x in s}
    return [1.0 if k in s else 0.0 for k in SYMPTOMS]   # 9개 밖 라벨은 무시(샘플은 유지)


# ---------------- 데이터 ----------------
class TextDS(Dataset):
    def __init__(self, ids, labels):
        self.ids, self.labels = ids, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def make_collate(pad_id):
    def collate(batch):
        L = max(len(b[0]) for b in batch)
        x = torch.full((len(batch), L), pad_id, dtype=torch.long)
        m = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (ids, _) in enumerate(batch):
            x[i, :len(ids)] = torch.tensor(ids); m[i, :len(ids)] = 1
        y = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        return x, m, y
    return collate


def length_batches(lengths, bs, seed, epoch):
    """길이가 비슷한 샘플끼리 묶어 패딩 낭비를 줄인다."""
    rng = np.random.RandomState(seed + epoch)
    idx = rng.permutation(len(lengths))
    chunk = bs * 50
    batches = []
    for i in range(0, len(idx), chunk):
        c = sorted(idx[i:i + chunk], key=lambda j: lengths[j])
        batches += [c[k:k + bs] for k in range(0, len(c), bs)]
    rng.shuffle(batches)
    return batches


def f1_report(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    out = {}
    for j, s in enumerate(SYMPTOMS):
        tp = int(((pred[:, j] == 1) & (y[:, j] == 1)).sum())
        fp = int(((pred[:, j] == 1) & (y[:, j] == 0)).sum())
        fn = int(((pred[:, j] == 0) & (y[:, j] == 1)).sum())
        pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        out[s] = dict(f1=2 * tp / max(1, 2 * tp + fp + fn), precision=pr, recall=rc)
    out["macro_f1"] = float(np.mean([out[s]["f1"] for s in SYMPTOMS]))
    return out


@torch.no_grad()
def predict(model, ids, bs, pad_id, device):
    model.eval()
    order = np.argsort([len(x) for x in ids])
    probs = np.zeros((len(ids), len(SYMPTOMS)), dtype=np.float32)
    col = make_collate(pad_id)
    for i in range(0, len(order), bs):
        b = order[i:i + bs]
        x, m, _ = col([(ids[j], [0.0] * len(SYMPTOMS)) for j in b])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            lg = model(input_ids=x.to(device), attention_mask=m.to(device)).logits
        probs[b] = torch.sigmoid(lg.float()).cpu().numpy()
    return probs


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- 데이터 로드 ----
    files = sorted(Path(args.label_dir).rglob("*.json"))
    if args.max_files:
        files = files[:args.max_files]
    names, texts, labels = [], [], []
    for jp in files:
        try:
            d = json.load(open(jp, encoding="utf-8"))
        except Exception:
            continue
        t = build_text(d)
        if not t:
            continue
        names.append(jp.name); texts.append(t); labels.append(build_labels(d))
    Y = np.array(labels, dtype=np.float32)
    print(f"[data] 문서 {len(texts):,}건 | 클래스별 양성: "
          + ", ".join(f"{s}{int(Y[:, j].sum())}" for j, s in enumerate(SYMPTOMS)), flush=True)

    # ---- 동일 전사문 그룹 단위 분할 ----
    keys = [hashlib.md5(re.sub(r"\s+", "", t).encode()).hexdigest() for t in texts]
    ug = sorted(set(keys))
    rng = np.random.RandomState(args.seed); rng.shuffle(ug)
    val_g = set(ug[:max(1, int(len(ug) * args.val_ratio))])
    va = [i for i, k in enumerate(keys) if k in val_g]
    tr = [i for i, k in enumerate(keys) if k not in val_g]
    print(f"[split] train={len(tr):,} val={len(va):,} (중복 전사 그룹 {len(keys)-len(ug):,}건 반영)",
          flush=True)

    # ---- 토크나이즈 ----
    tok = AutoTokenizer.from_pretrained(args.pretrained)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(texts, add_special_tokens=True, truncation=True, max_length=args.max_len)["input_ids"]
    lens = np.array([len(e) for e in enc])
    raw = tok(texts[:2000], add_special_tokens=True)["input_ids"]
    trunc = np.mean([len(r) > args.max_len for r in raw]) * 100
    print(f"[token] p50={int(np.percentile(lens,50))} p90={int(np.percentile(lens,90))} "
          f"max={lens.max()} | {args.max_len} 초과 비율(표본) {trunc:.1f}%", flush=True)

    # ---- 모델 ----
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained, num_labels=len(SYMPTOMS),
        problem_type="multi_label_classification", torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type="SEQ_CLS", r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(tr) / args.batch / args.accum)
    total = steps_per_epoch * args.epochs
    warm = int(total * args.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    pw = None
    if args.pos_weight:
        ytr = Y[tr]
        pos = ytr.sum(0).clip(min=1)
        pw = np.sqrt((len(ytr) - pos) / pos).astype(np.float32)
        print("[pos_weight] " + " | ".join(f"{s} {w:.2f}" for s, w in zip(SYMPTOMS, pw)),
              flush=True)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    else:
        bce = nn.BCEWithLogitsLoss()
    col = make_collate(tok.pad_token_id)
    tr_lens = lens[tr]
    best, hist = -1.0, []
    print(f"[plan] optimizer steps/epoch={steps_per_epoch:,} total={total:,} "
          f"effective_batch={args.batch*args.accum}", flush=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        batches = length_batches(tr_lens, args.batch, args.seed, ep)
        te = time.time(); run = 0.0; n = 0
        opt.zero_grad(set_to_none=True)
        for bi, b in enumerate(batches, 1):
            idx = [tr[j] for j in b]
            x, m, y = col([(enc[i], labels[i]) for i in idx])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                lg = model(input_ids=x.to(device), attention_mask=m.to(device)).logits
            loss = bce(lg.float(), y.to(device)) / args.accum
            loss.backward()
            run += loss.item() * args.accum; n += 1
            if bi % args.accum == 0 or bi == len(batches):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if bi % (args.log_every * args.accum) == 0:
                spb = (time.time() - te) / bi
                print(f"  [ep {ep} batch {bi}/{len(batches)}] loss={run/n:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} eta={spb*(len(batches)-bi)/60:.0f}min",
                      flush=True)
                run = 0.0; n = 0

        pv = predict(model, [enc[i] for i in va], args.eval_batch, tok.pad_token_id, device)
        rep = f1_report(Y[va], pv)
        hist.append({"epoch": ep, "macro_f1": rep["macro_f1"]})
        print(f"[epoch {ep}] val macro_F1={rep['macro_f1']:.4f} ({(time.time()-te)/60:.0f}min)",
              flush=True)
        print("   " + " | ".join(f"{s} {rep[s]['f1']:.3f}" for s in SYMPTOMS), flush=True)
        if rep["macro_f1"] > best:
            best = rep["macro_f1"]
            sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
                  if "lora_" in k or "score" in k}
            torch.save({"trainable_state_dict": sd, "symptoms": SYMPTOMS,
                        "pretrained": args.pretrained, "max_len": args.max_len,
                        "lora": dict(r=args.lora_r, alpha=args.lora_alpha,
                                     dropout=args.lora_dropout),
                        "epoch": ep, "val_macro_f1": best, "threshold": 0.5,
                        "pos_weight": None if pw is None else pw.tolist()},
                       out / "best.pt")
            np.savez(out / "val_probs.npz", probs=pv, y=Y[va],
                     names=np.array([names[i] for i in va]))
            json.dump(rep, open(out / "val_report.json", "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print(f"  -> best 저장 (macro_F1={best:.4f})", flush=True)

    json.dump({"best_macro_f1": best, "history": hist,
               "minutes": round((time.time() - t0) / 60, 1)},
              open(out / "summary.json", "w"), indent=2)
    print(f"[done] best val macro_F1={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
