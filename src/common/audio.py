# -*- coding: utf-8 -*-
"""8 kHz 전화 음성용 특징 추출기 — numpy/scipy 만 사용 (torch/librosa 불필요).

이 PC 에는 NVIDIA GPU 가 없으므로 wav2vec2/AST 파인튜닝은 불가능하다.
대신 출제 PDF 가 예시로 든 경로(MFCC / Mel-spectrogram / RandomForest)를 CPU 로 구현한다.

설계 근거
  - 8 kHz 협대역이므로 mel 필터뱅크 상한을 4000 Hz(Nyquist)로 둔다. 광대역 사전학습 모델을
    쓰지 않으므로 리샘플링에 따른 대역 불일치 문제(Sivaraman & Khoury, Odyssey'20)가 없다.
  - M1(성별)의 핵심 단서는 F0 이므로 자기상관 기반 피치 추정을 직접 넣는다.
  - M2(화자 역할)의 핵심 단서는 목소리가 아니라 채널(헤드셋/상황실 vs 휴대폰/야외)이므로
    log-mel 평균(스펙트럼 포락)과 잡음/에너지 통계를 함께 담는다.
"""
import wave

import numpy as np
from scipy.fftpack import dct

SR = 8000
WIN = 256          # 32 ms
HOP = 80           # 10 ms
NMEL = 40
NMFCC = 20
FMIN, FMAX = 20.0, 3900.0
F0_MIN, F0_MAX = 60.0, 400.0     # 전화 대역 성인 화자


# ---------------------------------------------------------------- wav I/O
def read_wav(path, start_ms=None, end_ms=None):
    """16-bit PCM mono wav 를 float32 [-1,1) 로 읽는다. ms 구간 지정 시 그 구간만 읽는다."""
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, "16-bit mono 가정"
        sr = w.getframerate()
        n = w.getnframes()
        if start_ms is None:
            i0, i1 = 0, n
        else:
            i0 = max(0, int(start_ms * sr / 1000))
            i1 = min(n, int(end_ms * sr / 1000))
            if i1 <= i0:
                return np.zeros(0, dtype=np.float32), sr
        w.setpos(i0)
        raw = w.readframes(i1 - i0)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return x, sr


# ---------------------------------------------------------------- mel
def _hz2mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=WIN, n_mel=NMEL, fmin=FMIN, fmax=FMAX):
    pts = _mel2hz(np.linspace(_hz2mel(fmin), _hz2mel(fmax), n_mel + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mel, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mel):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        r = min(r, n_fft // 2)
        if c >= r or l >= c:
            continue
        fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


_FB = mel_filterbank()
_WINDOW = np.hanning(WIN).astype(np.float32)


def frame_signal(x, win=WIN, hop=HOP):
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def spectrogram(x):
    fr = frame_signal(x) * _WINDOW
    return np.abs(np.fft.rfft(fr, n=WIN, axis=1)) ** 2      # (T, 129)


def logmel_mfcc(x):
    P = spectrogram(x)
    mel = P @ _FB.T                                          # (T, 40)
    logmel = np.log(mel + 1e-10)
    mfcc = dct(logmel, type=2, axis=1, norm="ortho")[:, :NMFCC]
    return logmel, mfcc, P


# ---------------------------------------------------------------- F0
def estimate_f0(x, sr=SR):
    """프레임 단위 자기상관 F0. (f0_hz, voiced_flag) 반환. 무성 프레임은 nan."""
    win = 400                                                # 50 ms — 60 Hz 도 2주기 확보
    hop = 160                                                # 20 ms
    if len(x) < win:
        return np.array([np.nan]), np.array([False])
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx].astype(np.float64)
    fr = fr - fr.mean(axis=1, keepdims=True)

    nfft = 1024
    S = np.fft.rfft(fr, n=nfft, axis=1)
    ac = np.fft.irfft(S * np.conj(S), n=nfft, axis=1)[:, :win]
    e0 = ac[:, :1].copy()
    e0[e0 <= 0] = 1e-12
    ac = ac / e0                                             # 정규화 자기상관

    lag_lo = max(2, int(sr / F0_MAX))                        # 20
    lag_hi = min(win - 1, int(sr / F0_MIN))                  # 133
    seg = ac[:, lag_lo:lag_hi]
    k = np.argmax(seg, axis=1)
    peak = seg[np.arange(len(k)), k]
    lag = k + lag_lo

    # 포물선 보간으로 lag 정밀화
    kk = np.clip(lag, 1, win - 2)
    y0 = ac[np.arange(len(kk)), kk - 1]
    y1 = ac[np.arange(len(kk)), kk]
    y2 = ac[np.arange(len(kk)), kk + 1]
    denom = (y0 - 2 * y1 + y2)
    shift = np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / np.where(denom == 0, 1e-12, denom), 0.0)
    lag_ref = lag + np.clip(shift, -1, 1)

    rms = np.sqrt((fr ** 2).mean(axis=1))
    voiced = (peak > 0.35) & (rms > 1e-4)
    f0 = np.where(voiced, sr / np.maximum(lag_ref, 1e-9), np.nan)
    return f0, voiced


# ---------------------------------------------------------------- 통계 요약
def _stats(v, prefix, out):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        for s in ("mean", "std", "p10", "p50", "p90"):
            out[f"{prefix}_{s}"] = 0.0
        return
    out[f"{prefix}_mean"] = float(v.mean())
    out[f"{prefix}_std"] = float(v.std())
    out[f"{prefix}_p10"] = float(np.percentile(v, 10))
    out[f"{prefix}_p50"] = float(np.percentile(v, 50))
    out[f"{prefix}_p90"] = float(np.percentile(v, 90))


FEATURE_NAMES = None       # 최초 호출 시 확정


def segment_features(x, sr=SR):
    """한 오디오 조각 → 고정 길이 특징 벡터(dict)."""
    out = {}
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))

    logmel, mfcc, P = logmel_mfcc(x)

    # MFCC 평균/표준편차 + 1차 차분 표준편차
    for i in range(NMFCC):
        out[f"mfcc{i:02d}_mean"] = float(mfcc[:, i].mean())
        out[f"mfcc{i:02d}_std"] = float(mfcc[:, i].std())
    d = np.diff(mfcc, axis=0) if mfcc.shape[0] > 1 else np.zeros((1, NMFCC))
    for i in range(NMFCC):
        out[f"dmfcc{i:02d}_std"] = float(d[:, i].std())

    # log-mel 평균 = 스펙트럼 포락 (채널 특성)
    lm = logmel.mean(axis=0)
    for i in range(NMEL):
        out[f"logmel{i:02d}"] = float(lm[i])

    # F0
    f0, voiced = estimate_f0(x, sr)
    _stats(f0, "f0", out)
    lf0 = np.log(f0[np.isfinite(f0)]) if np.isfinite(f0).any() else np.array([])
    _stats(lf0, "logf0", out)
    out["voiced_frac"] = float(np.mean(voiced)) if voiced.size else 0.0

    # 스펙트럼 형상
    freqs = np.linspace(0, sr / 2, P.shape[1])
    Ps = P + 1e-12
    tot = Ps.sum(axis=1)
    centroid = (Ps * freqs).sum(axis=1) / tot
    spread = np.sqrt((Ps * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / tot)
    cum = np.cumsum(Ps, axis=1) / tot[:, None]
    rolloff = freqs[np.argmax(cum >= 0.85, axis=1)]
    flatness = np.exp(np.log(Ps).mean(axis=1)) / (Ps.mean(axis=1))
    _stats(centroid, "centroid", out)
    _stats(spread, "spread", out)
    _stats(rolloff, "rolloff", out)
    _stats(flatness, "flatness", out)

    # 시간 영역
    fr = frame_signal(x)
    zcr = (np.diff(np.sign(fr), axis=1) != 0).mean(axis=1)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    _stats(zcr, "zcr", out)
    _stats(20 * np.log10(rms), "rmsdb", out)
    out["dyn_range_db"] = float(np.percentile(20 * np.log10(rms), 95)
                                - np.percentile(20 * np.log10(rms), 5))
    out["log_dur"] = float(np.log(max(len(x) / sr, 1e-3)))
    return out


def feature_vector(x, sr=SR):
    global FEATURE_NAMES
    d = segment_features(x, sr)
    if FEATURE_NAMES is None:
        FEATURE_NAMES = sorted(d.keys())
    return np.array([d[k] for k in FEATURE_NAMES], dtype=np.float32)


def feature_names():
    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        feature_vector(np.zeros(SR, dtype=np.float32))
    return list(FEATURE_NAMES)
