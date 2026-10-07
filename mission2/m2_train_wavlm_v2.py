"""
Mission 2 — WavLM-base-plus v2 (제출 후보)

v1 대비 변경점
  1. 13개 hidden layer 학습 가능 가중합 (화자 정보가 많은 중간층 활용)
  2. CNN feature encoder 영구 동결
  3. warmup + cosine 학습률 스케줄
  4. Training 서울 전체 통화 사용, 매 epoch 통화마다 다른 발화를 샘플링
  5. 발화 조각 단위 정규화 (통화 전체 음성을 참조하지 않음)
  6. 원본 startAt~endAt 만 사용 (이웃 발화 기반 겹침 트리밍 없음)

규정
  Training 서울 데이터만 사용 / Validation 폴더 미사용
  입력은 발화 구간 음성뿐 (text, speaker, 발화 순서 미사용)
  speaker 는 학습 라벨로만 사용
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoModel

import m2_preprocess as P   # 라벨 JSON 파싱(build_index)만 사용

SR = 16000
HOME = str(Path.home())

DEFAULTS = dict(
    label_dir=f"{HOME}/대학부 데이터/2.라벨링데이터/TL_서울_구급",
    audio_dir=f"{HOME}/대학부 데이터/1.원천데이터/TS_서울_구급",
    pretrained="pretrained/wavlm-base-plus",
    out_dir="checkpoints/m2_wavlm_v2",
    win_sec=3.0,
    emb_dim=256,
    calls_per_batch=8,
    utts_per_call=4,
    epochs=15,
    warmup_ratio=0.05,
    lr_head=3e-4,
    lr_backbone=3e-5,
    weight_decay=1e-2,
    freeze_epochs=1,
    temperature=0.07,
    lambda_con=0.5,
    contaminated_ratio=0.15,
    val_ratio=0.1,
    val_utts_per_call=10,
    seed=42,
    num_workers=8,
    max_calls=0,          # 0 = 전체
    amp=1,
    log_every=200,
    resume=1,
)


# ----------------------------------------------------------------------
# 전처리 (학습/추론 공통 — 추론 코드는 이 함수들을 import 해서 쓴다)
# ----------------------------------------------------------------------
_SR_CACHE = {}


def load_segment(wav_path, s_ms, e_ms):
    """wav 에서 [s_ms, e_ms] 구간만 읽어 16kHz mono 1D 텐서로 반환."""
    sr = _SR_CACHE.get(wav_path)
    if sr is None:
        sr = torchaudio.info(wav_path).sample_rate
        _SR_CACHE[wav_path] = sr
    a = max(0, int(s_ms * sr / 1000))
    n = max(1, int((e_ms - s_ms) * sr / 1000))
    try:
        w, _ = torchaudio.load(wav_path, frame_offset=a, num_frames=n)
    except Exception:
        return torch.zeros(1)
    if w.numel() == 0:
        return torch.zeros(1)
    if w.size(0) > 1:
        w = w.mean(dim=0, keepdim=True)
    w = w.reshape(-1).float()
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w


def fit_length(seg, win, train=False):
    """길면 크롭(학습=랜덤, 평가=중앙), 짧으면 반복 패딩."""
    n = seg.numel()
    if n < 2:
        return torch.zeros(win)
    if n >= win:
        off = np.random.randint(0, n - win + 1) if train else (n - win) // 2
        return seg[off:off + win].clone()
    rep = int(math.ceil(win / n))
    return seg.repeat(rep)[:win].clone()


def normalize(x):
    """발화 조각 단위 zero-mean / unit-variance."""
    sd = x.std()
    if sd < 1e-5:
        return torch.zeros_like(x)
    return (x - x.mean()) / sd


# ----------------------------------------------------------------------
# 데이터셋 / 샘플러
# ----------------------------------------------------------------------
class UttDataset(Dataset):
    def __init__(self, calls, win_sec, train, cratio):
        self.win = int(win_sec * SR)
        self.train = train
        self.cratio = cratio
        self.calls = calls
        self.items = [(ci, ui) for ci, c in enumerate(calls) for ui in range(len(c["utts"]))]

    def __len__(self):
        return len(self.items)

    def _seg(self, c, ui):
        s, e, _ = c["utts"][ui]
        return fit_length(load_segment(c["wav"], s, e), self.win, self.train)

    def __getitem__(self, i):
        ci, ui = self.items[i]
        c = self.calls[ci]
        sp = c["utts"][ui][2]
        x = self._seg(c, ui)

        # 오염 증강: 실제 데이터의 발화 겹침을 흉내내 상대 화자 음성을 일부 혼입
        # (라벨과 무관하게 모든 샘플에 동일 확률로 적용)
        if self.train and np.random.rand() < self.cratio:
            others = [j for j, u in enumerate(c["utts"]) if u[2] != sp]
            if others:
                o = normalize(self._seg(c, others[np.random.randint(len(others))]))
                x = normalize(x)
                g = 0.15 + 0.25 * np.random.rand()
                L = int(self.win * (0.10 + 0.15 * np.random.rand()))
                if np.random.rand() < 0.5:
                    x[:L] = x[:L] + g * o[:L]
                else:
                    x[-L:] = x[-L:] + g * o[-L:]

        return normalize(x), torch.tensor(float(sp)), torch.tensor(ci), torch.tensor(int(sp))


class CallBatchSampler(Sampler):
    """배치 = 통화 여러 개 x 통화당 발화 여러 개. 매 epoch 다른 발화를 뽑는다."""

    def __init__(self, ds, calls_per_batch, utts_per_call, seed):
        self.by_call = {}
        for k, (ci, _) in enumerate(ds.items):
            self.by_call.setdefault(ci, []).append(k)
        self.cpb, self.upc, self.seed = calls_per_batch, utts_per_call, seed
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        cids = list(self.by_call.keys())
        rng.shuffle(cids)
        for i in range(0, len(cids) - self.cpb + 1, self.cpb):
            batch = []
            for c in cids[i:i + self.cpb]:
                pool = self.by_call[c]
                batch += list(rng.choice(pool, size=min(self.upc, len(pool)), replace=False))
            yield batch

    def __len__(self):
        return len(self.by_call) // self.cpb


def worker_init(_):
    np.random.seed(torch.initial_seed() % (2 ** 32))


# ----------------------------------------------------------------------
# 모델
# ----------------------------------------------------------------------
class WavLMRole(nn.Module):
    def __init__(self, pretrained, emb_dim):
        super().__init__()
        # LayerDrop 끄기: 켜져 있으면 학습 중 hidden_states 개수가 바뀌어 층 가중합이 깨짐
        self.backbone = AutoModel.from_pretrained(pretrained, layerdrop=0.0)
        for p in self.backbone.feature_extractor.parameters():   # CNN 앞단 영구 동결
            p.requires_grad = False
        # hidden_states 개수는 transformers 버전마다 다름(12 또는 13) -> 실제로 확인
        with torch.no_grad():
            n_layers = len(self.backbone(torch.zeros(1, 16000),
                                         output_hidden_states=True).hidden_states)
        self.layer_w = nn.Parameter(torch.zeros(n_layers))       # 층 가중합 가중치
        h = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(h * 2, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, emb_dim))
        self.head = nn.Linear(emb_dim, 1)

    def transformer_params(self):
        return [p for n, p in self.backbone.named_parameters()
                if not n.startswith("feature_extractor")]

    def set_backbone_grad(self, flag):
        for p in self.transformer_params():
            p.requires_grad = flag

    def layer_weights(self):
        return torch.softmax(self.layer_w.detach(), 0).cpu().numpy()

    def forward(self, x):
        out = self.backbone(x, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0).float()           # (L, B, T, H)
        w = torch.softmax(self.layer_w, 0)
        o = (w[:, None, None, None] * hs).sum(0)                 # (B, T, H)
        stat = torch.cat([o.mean(1), o.std(1)], -1)              # 통계 풀링
        emb = F.normalize(self.proj(stat), dim=-1)
        return emb, self.head(emb).squeeze(-1)


def supcon_loss(emb, call_ids, spk, temperature):
    """같은 통화 + 같은 화자 = positive. 다른 통화와는 비교하지 않음."""
    B = emb.size(0)
    sim = emb @ emb.t() / temperature
    eye = torch.eye(B, dtype=torch.bool, device=emb.device)
    same_call = call_ids[:, None] == call_ids[None, :]
    pos = same_call & (spk[:, None] == spk[None, :]) & ~eye
    valid = same_call & ~eye
    sim = sim.masked_fill(~valid, -1e4)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    cnt = pos.sum(1)
    ok = cnt > 0
    if ok.sum() == 0:
        return emb.sum() * 0.0
    return -((logp * pos).sum(1)[ok] / cnt[ok]).mean()


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    correct = total = 0
    for x, y, _, _ in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=amp):
            _, logit = model(x)
        correct += ((logit.float() > 0).float() == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


# ----------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(args.amp) and device == "cuda"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[env] device={device} torch={torch.__version__}", flush=True)

    # 라벨 인덱스 (원본 startAt/endAt 사용)
    records = P.build_index(args.label_dir, args.audio_dir)
    by_call = {}
    for r in records:
        by_call.setdefault(r["call_id"], []).append(r)
    calls = []
    for cid, rs in by_call.items():
        utts = [(int(r["raw_s"]), int(r["raw_e"]), int(r["speaker"])) for r in rs]
        utts = [u for u in utts if u[1] > u[0]]
        if len(utts) >= 4 and len({u[2] for u in utts}) == 2:
            calls.append({"call_id": cid, "wav": rs[0]["wav"], "utts": utts})

    rng = np.random.RandomState(args.seed)
    if args.max_calls and len(calls) > args.max_calls:
        calls = [calls[i] for i in rng.choice(len(calls), args.max_calls, replace=False)]

    perm = rng.permutation(len(calls))                          # 통화 단위 분할
    n_val = max(1, int(len(calls) * args.val_ratio))
    val_calls = [dict(calls[i]) for i in perm[:n_val]]
    tr_calls = [calls[i] for i in perm[n_val:]]
    for c in val_calls:                                          # 내부 val 은 통화당 일부만
        if len(c["utts"]) > args.val_utts_per_call:
            idx = sorted(rng.choice(len(c["utts"]), args.val_utts_per_call, replace=False))
            c["utts"] = [c["utts"][i] for i in idx]

    tr_ds = UttDataset(tr_calls, args.win_sec, True, args.contaminated_ratio)
    va_ds = UttDataset(val_calls, args.win_sec, False, 0.0)
    sampler = CallBatchSampler(tr_ds, args.calls_per_batch, args.utts_per_call, args.seed)
    tr_ld = DataLoader(tr_ds, batch_sampler=sampler, num_workers=args.num_workers,
                       pin_memory=True, worker_init_fn=worker_init)
    va_ld = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=args.num_workers,
                       pin_memory=True, worker_init_fn=worker_init)
    steps_per_epoch = len(sampler)
    total_steps = steps_per_epoch * args.epochs
    print(f"[data] train_calls={len(tr_calls):,} val_calls={len(val_calls):,} "
          f"train_utts_pool={len(tr_ds):,} val_utts={len(va_ds):,}", flush=True)
    print(f"[plan] steps/epoch={steps_per_epoch:,} total_steps={total_steps:,} "
          f"batch={args.calls_per_batch * args.utts_per_call}", flush=True)

    model = WavLMRole(args.pretrained, args.emb_dim).to(device)
    head_params = list(model.proj.parameters()) + list(model.head.parameters()) + [model.layer_w]
    opt = torch.optim.AdamW([
        {"params": model.transformer_params(), "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    warm = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    bce = nn.BCEWithLogitsLoss()

    start_ep, best, history = 1, 0.0, []
    last_path = out_dir / "last.pt"
    if args.resume and last_path.exists():                     # 중단 후 이어하기
        ck = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_ep, best, history = ck["epoch"] + 1, ck["best"], ck.get("history", [])
        print(f"[resume] epoch {ck['epoch']} 부터 이어서 (best={best:.4f})", flush=True)

    for ep in range(start_ep, args.epochs + 1):
        sampler.set_epoch(ep)
        model.set_backbone_grad(ep > args.freeze_epochs)
        model.train()
        t0 = time.time(); run = 0.0; n = 0
        for step, (x, y, cid, spk) in enumerate(tr_ld, 1):
            x, y = x.to(device, non_blocking=True), y.to(device)
            cid, spk = cid.to(device), spk.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                emb, logit = model(x)
                loss = bce(logit.float(), y) + args.lambda_con * supcon_loss(
                    emb.float(), cid, spk, args.temperature)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); n += 1
            if step % args.log_every == 0:
                sps = (time.time() - t0) / step
                eta = sps * (steps_per_epoch - step) / 60
                print(f"  [ep {ep} step {step}/{steps_per_epoch}] loss={run/n:.4f} "
                      f"lr_bb={opt.param_groups[0]['lr']:.2e} {sps:.2f}s/step "
                      f"epoch_eta={eta:.0f}min", flush=True)
                run = 0.0; n = 0

        acc = evaluate(model, va_ld, device, amp)
        lw = model.layer_weights()
        history.append({"epoch": ep, "val_acc": acc, "minutes": (time.time() - t0) / 60})
        print(f"[epoch {ep}] val_acc={acc:.4f} time={(time.time()-t0)/60:.0f}min "
              f"top_layers={list(np.argsort(-lw)[:3])}", flush=True)

        ck = {"state_dict": model.state_dict(), "emb_dim": args.emb_dim,
              "win_sec": args.win_sec, "sr": SR, "epoch": ep, "val_acc": acc,
              "layer_weights": lw.tolist()}
        if acc > best:
            best = acc
            torch.save(ck, out_dir / "best.pt")
            print(f"  -> best 저장 (val_acc={acc:.4f})", flush=True)
        ck.update({"opt": opt.state_dict(), "sched": sched.state_dict(),
                   "scaler": scaler.state_dict(), "best": best, "history": history})
        torch.save(ck, last_path)

    json.dump({"best_val_acc": best, "history": history},
              open(out_dir / "summary.json", "w"), indent=2)
    print(f"[done] best_val_acc={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
