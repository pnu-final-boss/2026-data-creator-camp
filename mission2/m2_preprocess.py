import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

SR = 16000
N_FFT = 400
HOP = 160
WIN = 400
N_MELS = 64
FMIN = 20
FMAX = 8000
POWER = 2.0
LOG_EPS = 1e-6

FIXED_SEC = 3.0
FIXED_FRAMES = int(FIXED_SEC * SR / HOP) + 1

TRIM_GUARD_MS = 50
MAX_TRIM_RATIO = 0.2
MIN_SEG_MS = 120

ALLOW_NEIGHBOR_TRIM = True

_MEL_TF = None


def _mel_transform():
    global _MEL_TF
    if _MEL_TF is None:
        _MEL_TF = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
            f_min=FMIN, f_max=FMAX, n_mels=N_MELS, power=POWER, center=True,
        )
    return _MEL_TF


def annotate_overlaps(utterances):
    utts = sorted([dict(u) for u in utterances],
                  key=lambda u: (int(u["startAt"]), int(u["endAt"])))
    n = len(utts)
    for i, u in enumerate(utts):
        s, e = int(u["startAt"]), int(u["endAt"])
        head = int(utts[i - 1]["endAt"]) - s if i > 0 else 0
        tail = e - int(utts[i + 1]["startAt"]) if i < n - 1 else 0
        u["ov_head"] = max(0, head)
        u["ov_tail"] = max(0, tail)
    return utts


def trim_bounds(start_ms, end_ms, ov_head=0, ov_tail=0, allow_neighbor=None):
    if allow_neighbor is None:
        allow_neighbor = ALLOW_NEIGHBOR_TRIM
    s, e = int(start_ms), int(end_ms)
    if e <= s:
        return s, s + MIN_SEG_MS
    if not allow_neighbor:
        return s, e
    cap = (e - s) * MAX_TRIM_RATIO
    cut_h = min(ov_head + TRIM_GUARD_MS, cap) if ov_head > 0 else 0.0
    cut_t = min(ov_tail + TRIM_GUARD_MS, cap) if ov_tail > 0 else 0.0
    ns, ne = int(s + cut_h), int(e - cut_t)
    if ne - ns < MIN_SEG_MS:
        return s, e
    return ns, ne


def load_call_audio(wav_path):
    wav, sr = torchaudio.load(str(wav_path))
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.reshape(-1).float()


def logmel(wave_1d):
    if wave_1d.numel() < N_FFT:
        wave_1d = torch.nn.functional.pad(wave_1d, (0, N_FFT - wave_1d.numel()))
    m = _mel_transform()(wave_1d.unsqueeze(0)).squeeze(0)
    return torch.log(m + LOG_EPS)


def call_stats(wave_1d):
    lm = logmel(wave_1d)
    return lm.mean(dim=1, keepdim=True), lm.std(dim=1, keepdim=True).clamp_min(1e-5)


def segment_feature(wave_1d, s_ms, e_ms, mu, sigma, train=False, rng=None):
    i0 = max(0, int(s_ms * SR / 1000))
    i1 = min(int(wave_1d.numel()), int(e_ms * SR / 1000))
    if i1 - i0 < int(MIN_SEG_MS * SR / 1000):
        i1 = min(int(wave_1d.numel()), i0 + int(MIN_SEG_MS * SR / 1000))
    seg = wave_1d[i0:i1]
    if seg.numel() == 0:
        seg = torch.zeros(int(MIN_SEG_MS * SR / 1000))

    lm = (logmel(seg) - mu) / sigma
    t = lm.size(1)
    feat = torch.zeros(N_MELS, FIXED_FRAMES)
    mask = torch.zeros(FIXED_FRAMES)

    if t <= FIXED_FRAMES:
        feat[:, :t] = lm
        mask[:t] = 1.0
    else:
        if train:
            off = int((rng or np.random).integers(0, t - FIXED_FRAMES + 1))
        else:
            off = (t - FIXED_FRAMES) // 2
        feat = lm[:, off:off + FIXED_FRAMES]
        mask[:] = 1.0
    return feat, mask


def build_index(label_dir, audio_dir, verbose=True):
    label_dir, audio_dir = Path(label_dir), Path(audio_dir)
    records, n_skip_json, n_skip_wav = [], 0, 0
    wav_map = {p.stem: p for p in audio_dir.rglob("*.wav")}

    for jp in sorted(label_dir.rglob("*.json")):
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            n_skip_json += 1
            continue

        wav = wav_map.get(jp.stem)
        if wav is None:
            rel = data.get("audioPath") or ""
            wav = wav_map.get(Path(rel).stem) if rel else None
        if wav is None:
            n_skip_wav += 1
            continue

        for u in annotate_overlaps(data.get("utterances", [])):
            if "speaker" not in u:
                continue
            s, e = trim_bounds(u["startAt"], u["endAt"], u["ov_head"], u["ov_tail"])
            records.append({
                "call_id": jp.stem, "wav": str(wav),
                "s_ms": s, "e_ms": e,
                "raw_s": int(u["startAt"]), "raw_e": int(u["endAt"]),
                "ov": int(u["ov_head"]) + int(u["ov_tail"]),
                "speaker": int(u["speaker"]),
            })

    if verbose:
        print(f"[index] utterances={len(records):,} "
              f"calls={len({r['call_id'] for r in records}):,} "
              f"json_skip={n_skip_json} wav_missing={n_skip_wav}", flush=True)
    return records
