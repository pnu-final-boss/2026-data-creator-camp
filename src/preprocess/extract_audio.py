# -*- coding: utf-8 -*-
"""wav → 특징 행렬 캐시. M1(통화 단위)과 M2(발화 단위)를 한 번의 디스크 읽기로 함께 만든다.

33.7 GB 를 매 epoch 디코딩하면 CPU 환경에서 실험이 불가능하므로, 특징을 한 번만 뽑아
npz 로 굽고 이후 모든 학습은 그 행렬만 쓴다.

M2 의 overlap 플래그는 startAt/endAt 만으로 계산한다 — 규정상 추론 시에도 허용된 필드다.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..common.audio import SR, feature_names, feature_vector
from ..common.labels import load_split
from ..common.paths import CACHE, TRAIN_AUDIO, VAL_AUDIO

MAX_CALLER_SEC = 20.0     # M1: 통화당 신고자 음성 최대 20초만 사용 (길이 편차 억제)


def _read_all(path):
    import wave
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def _slice(x, s_ms, e_ms):
    i0 = max(0, int(s_ms * SR / 1000))
    i1 = min(len(x), int(e_ms * SR / 1000))
    return x[i0:i1] if i1 > i0 else np.zeros(0, dtype=np.float32)


def _overlap_flags(utts):
    """다른 화자의 발화와 시간적으로 겹치는가. startAt/endAt 만 사용."""
    n = len(utts)
    flags = np.zeros(n, dtype=np.int8)
    for i, a in enumerate(utts):
        for j, b in enumerate(utts):
            if i == j:
                continue
            if b["startAt"] < a["endAt"] and b["endAt"] > a["startAt"] and b["speaker"] != a["speaker"]:
                flags[i] = 1
                break
    return flags


def _one_call(args):
    rec, audio_dir = args
    path = audio_dir / (rec["stem"] + ".wav")
    try:
        x = _read_all(path)
    except Exception:
        return None
    utts = rec["utterances"]
    if not utts:
        return None
    ov = _overlap_flags(utts)

    u_feat, u_spk, u_ov, u_start, u_end = [], [], [], [], []
    caller_chunks, total = [], 0.0
    for u, o in zip(utts, ov):
        seg = _slice(x, u["startAt"], u["endAt"])
        if len(seg) < 400:                       # 50 ms 미만은 버림
            continue
        u_feat.append(feature_vector(seg))
        u_spk.append(int(u["speaker"]))
        u_ov.append(int(o))
        u_start.append(int(u["startAt"]))
        u_end.append(int(u["endAt"]))
        # M1 용: 신고자(speaker==1) 구간만 모은다
        if u["speaker"] == 1 and total < MAX_CALLER_SEC:
            caller_chunks.append(seg)
            total += len(seg) / SR

    if not u_feat:
        return None
    caller = np.concatenate(caller_chunks) if caller_chunks else np.zeros(SR, dtype=np.float32)
    call_feat = feature_vector(caller)

    return dict(
        stem=rec["stem"],
        gender=rec["gender"],
        call_feat=call_feat,
        caller_sec=total,
        u_feat=np.stack(u_feat),
        u_spk=np.array(u_spk, dtype=np.int8),
        u_ov=np.array(u_ov, dtype=np.int8),
        u_start=np.array(u_start, dtype=np.int32),
        u_end=np.array(u_end, dtype=np.int32),
    )


def build(split, n_calls=None, workers=12, seed=0):
    recs = load_split(split)
    audio_dir = TRAIN_AUDIO if split == "train" else VAL_AUDIO
    if n_calls is not None and n_calls < len(recs):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(recs), size=n_calls, replace=False)
        recs = [recs[i] for i in sorted(idx)]

    print(f"[extract] {split}: {len(recs)} calls, workers={workers}")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_one_call, [(r, audio_dir) for r in recs], chunksize=8)):
            if r is not None:
                results.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(recs)}", flush=True)

    # 통화 단위 (M1)
    call_X = np.stack([r["call_feat"] for r in results])
    call_g = np.array([1 if r["gender"] == "M" else 0 for r in results], dtype=np.int8)
    call_stem = np.array([r["stem"] for r in results])
    call_sec = np.array([r["caller_sec"] for r in results], dtype=np.float32)

    # 발화 단위 (M2)
    utt_X = np.concatenate([r["u_feat"] for r in results])
    utt_y = np.concatenate([r["u_spk"] for r in results])
    utt_ov = np.concatenate([r["u_ov"] for r in results])
    utt_s = np.concatenate([r["u_start"] for r in results])
    utt_e = np.concatenate([r["u_end"] for r in results])
    utt_call = np.concatenate([np.full(len(r["u_spk"]), i, dtype=np.int32)
                               for i, r in enumerate(results)])

    out = CACHE / f"audio_{split}.npz"
    np.savez_compressed(
        out,
        call_X=call_X, call_g=call_g, call_stem=call_stem, call_sec=call_sec,
        utt_X=utt_X, utt_y=utt_y, utt_ov=utt_ov, utt_start=utt_s, utt_end=utt_e,
        utt_call=utt_call, feat_names=np.array(feature_names()),
    )
    print(f"[extract] saved {out}  calls={call_X.shape}  utts={utt_X.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    build(a.split, a.n, a.workers)
