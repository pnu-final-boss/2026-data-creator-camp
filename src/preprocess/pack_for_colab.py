# -*- coding: utf-8 -*-
"""Colab 런타임으로 올릴 압축 번들 생성.

Colab 은 원격 런타임이라 로컬 33.7 GB 원본을 볼 수 없다. GPU 가 실제로 필요한 것만
골라 담는다.

  m3_text.json.gz   : 전사 텍스트 + 9클래스 라벨. KLUE-RoBERTa / Kc-ELECTRA 파인튜닝용.
  m1_audio.npz      : 통화별 신고자 음성(최대 CALLER_SEC 초) int16. wav2vec2 성별 파인튜닝용.
  m2_audio.npz      : 발화 조각 int16. AST / wav2vec2 화자역할 파인튜닝용.

오디오는 8 kHz int16 원본 그대로 담는다. 16 kHz 업샘플은 Colab 쪽에서 하면 되고,
여기서 미리 올리면 용량이 2배가 된다.
"""
import argparse
import gzip
import json

import numpy as np

from ..common.audio import SR
from ..common.labels import load_split
from ..common.paths import CACHE, TRAIN_AUDIO, VAL_AUDIO
from ..m3_symptom.train import build_labels, build_text

BUNDLE = CACHE / "colab"
BUNDLE.mkdir(exist_ok=True)


def _read_all(path):
    import wave
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2")


def _slice(x, s_ms, e_ms, max_sec=None):
    i0 = max(0, int(s_ms * SR / 1000))
    i1 = min(len(x), int(e_ms * SR / 1000))
    if i1 <= i0:
        return np.zeros(0, dtype=np.int16)
    seg = x[i0:i1]
    if max_sec is not None and len(seg) > int(max_sec * SR):
        seg = seg[: int(max_sec * SR)]
    return seg


# ------------------------------------------------------------------ M3
def pack_m3():
    out = {}
    for split in ("train", "val"):
        recs = load_split(split)
        Y = build_labels(recs)
        out[split] = [
            {"stem": r["stem"], "text": build_text(r, tagged=True),
             "plain": build_text(r, tagged=False), "y": Y[i].tolist()}
            for i, r in enumerate(recs)
        ]
    p = BUNDLE / "m3_text.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    mb = p.stat().st_size / 1e6
    print(f"[pack] {p.name}  train={len(out['train'])} val={len(out['val'])}  {mb:.1f} MB")


# ------------------------------------------------------------------ M1 / M2
def pack_audio(split, n_calls, caller_sec, max_utt_per_call, max_utt_sec, seed=0):
    recs = load_split(split)
    audio_dir = TRAIN_AUDIO if split == "train" else VAL_AUDIO
    rng = np.random.default_rng(seed)
    if n_calls < len(recs):
        idx = sorted(rng.choice(len(recs), size=n_calls, replace=False))
        recs = [recs[i] for i in idx]

    m1_buf, m1_off, m1_y, m1_stem = [], [0], [], []
    m2_buf, m2_off, m2_y, m2_ov, m2_call = [], [0], [], [], []

    for ci, r in enumerate(recs):
        try:
            x = _read_all(audio_dir / (r["stem"] + ".wav"))
        except Exception:
            continue
        utts = r["utterances"]

        # --- M1: 신고자 구간 이어붙이기 ---
        chunks, tot = [], 0
        for u in utts:
            if u["speaker"] != 1:
                continue
            seg = _slice(x, u["startAt"], u["endAt"])
            if len(seg) < 400:
                continue
            chunks.append(seg)
            tot += len(seg)
            if tot >= caller_sec * SR:
                break
        if chunks:
            cat = np.concatenate(chunks)[: int(caller_sec * SR)]
            m1_buf.append(cat)
            m1_off.append(m1_off[-1] + len(cat))
            m1_y.append(1 if r["gender"] == "M" else 0)
            m1_stem.append(r["stem"])

        # --- M2: 발화 조각 (통화당 균등 샘플) ---
        cand = []
        for j, u in enumerate(utts):
            seg = _slice(x, u["startAt"], u["endAt"], max_sec=max_utt_sec)
            if len(seg) < 400:
                continue
            ov = any(b["startAt"] < u["endAt"] and b["endAt"] > u["startAt"]
                     and b["speaker"] != u["speaker"] for k, b in enumerate(utts) if k != j)
            cand.append((seg, int(u["speaker"]), int(ov)))
        if len(cand) > max_utt_per_call:
            pick = rng.choice(len(cand), size=max_utt_per_call, replace=False)
            cand = [cand[i] for i in sorted(pick)]
        for seg, spk, ov in cand:
            m2_buf.append(seg)
            m2_off.append(m2_off[-1] + len(seg))
            m2_y.append(spk)
            m2_ov.append(ov)
            m2_call.append(ci)

        if (ci + 1) % 500 == 0:
            print(f"  {split} {ci+1}/{len(recs)}", flush=True)

    p1 = BUNDLE / f"m1_audio_{split}.npz"
    np.savez(p1, audio=np.concatenate(m1_buf), offsets=np.array(m1_off, dtype=np.int64),
             y=np.array(m1_y, dtype=np.int8), stem=np.array(m1_stem), sr=SR)
    p2 = BUNDLE / f"m2_audio_{split}.npz"
    np.savez(p2, audio=np.concatenate(m2_buf), offsets=np.array(m2_off, dtype=np.int64),
             y=np.array(m2_y, dtype=np.int8), overlap=np.array(m2_ov, dtype=np.int8),
             call=np.array(m2_call, dtype=np.int32), sr=SR)
    print(f"[pack] {p1.name}  n={len(m1_y):,}  {p1.stat().st_size/1e6:.0f} MB")
    print(f"[pack] {p2.name}  n={len(m2_y):,}  {p2.stat().st_size/1e6:.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all", choices=["all", "m3", "audio"])
    ap.add_argument("--train-calls", type=int, default=4000)
    ap.add_argument("--val-calls", type=int, default=1200)
    ap.add_argument("--caller-sec", type=float, default=10.0,
                    help="M1: 통화당 신고자 음성 최대 길이(초)")
    ap.add_argument("--utt-per-call", type=int, default=12,
                    help="M2: 통화당 담을 발화 조각 수")
    ap.add_argument("--utt-sec", type=float, default=4.0,
                    help="M2: 발화 조각 최대 길이(초)")
    a = ap.parse_args()

    if a.what in ("all", "m3"):
        pack_m3()
    if a.what in ("all", "audio"):
        pack_audio("val", a.val_calls, a.caller_sec, a.utt_per_call, a.utt_sec)
        pack_audio("train", a.train_calls, a.caller_sec, a.utt_per_call, a.utt_sec)
    print(f"[pack] 번들 위치: {BUNDLE}")
