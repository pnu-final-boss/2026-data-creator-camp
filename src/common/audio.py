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
    """O'Shaughnessy mel 스케일. 사람 귀는 저주파에서 주파수 차이를 더 잘 분간하므로
    저역을 넓게, 고역을 좁게 펴는 로그 척도를 쓴다."""
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel2hz(m):
    """_hz2mel 의 역함수."""
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=WIN, n_mel=NMEL, fmin=FMIN, fmax=FMAX):
    """삼각 mel 필터뱅크 (n_mel, n_fft//2+1) 를 만든다.

    mel 축에서 등간격인 n_mel+2 개의 점을 잡고, 연속한 세 점 (left, center, right) 마다
    center 에서 1, 양 끝에서 0 이 되는 삼각형을 하나씩 세운다. 그래서 점이 n_mel 이 아니라
    n_mel+2 개 필요하다.
    """
    # mel 축 등간격 -> Hz 로 되돌림 (Hz 축에서는 저역이 촘촘, 고역이 성김)
    pts = _mel2hz(np.linspace(_hz2mel(fmin), _hz2mel(fmax), n_mel + 2))
    # Hz -> FFT bin 인덱스
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mel, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mel):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        # WIN=256 은 bin 폭이 31.25 Hz 라 저역에서 서로 다른 mel 점이 같은 bin 으로 뭉갠다.
        # 그러면 폭 0 짜리 삼각형이 생겨 0 나누기가 나므로 최소 1 bin 을 확보한다.
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        r = min(r, n_fft // 2)
        if c >= r or l >= c:
            continue                      # 나이퀴스트에 밀려 자리가 없으면 그 필터는 비워 둔다
        fb[i, l:c] = (np.arange(l, c) - l) / (c - l)      # 상승 사면 0 -> 1
        fb[i, c:r] = (r - np.arange(c, r)) / (r - c)      # 하강 사면 1 -> 0
    return fb


_FB = mel_filterbank()
_WINDOW = np.hanning(WIN).astype(np.float32)


def frame_signal(x, win=WIN, hop=HOP):
    """신호를 겹치는 프레임 (T, win) 으로 자른다.

    음성은 짧은 구간에서만 정상(stationary)이라 25~32 ms 단위로 잘라 분석한다.
    루프 대신 브로드캐스팅 인덱스 행렬로 한 번에 잘라낸다 (n x win 크기의 정수 인덱스).
    """
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def spectrogram(x):
    """프레임별 파워 스펙트럼 (T, 129).

    한 프레임을 그냥 잘라내면 양 끝의 불연속이 스펙트럼 누설을 일으키므로
    해닝 창을 곱해 끝을 0 으로 부드럽게 만든다.
    """
    fr = frame_signal(x) * _WINDOW
    return np.abs(np.fft.rfft(fr, n=WIN, axis=1)) ** 2      # rfft: 실수 입력이라 절반만 계산


def logmel_mfcc(x):
    """log-mel 스펙트로그램과 MFCC 를 함께 반환.

    파워 -> mel 필터뱅크 -> log -> DCT 순서다. log 는 사람의 음량 지각이 로그 스케일인 것과
    맞추는 동시에, 채널 특성(곱셈)을 덧셈으로 바꿔 뒤의 평균차감으로 제거 가능하게 만든다.
    DCT 는 mel 밴드 간 상관을 없애 앞쪽 계수에 정보를 몰아준다 — 그래서 20개만 남겨도 충분하다.
    """
    P = spectrogram(x)
    mel = P @ _FB.T                                          # (T, 129) x (129, 40) -> (T, 40)
    logmel = np.log(mel + 1e-10)                             # 1e-10: log(0) 방지
    mfcc = dct(logmel, type=2, axis=1, norm="ortho")[:, :NMFCC]
    return logmel, mfcc, P


# ---------------------------------------------------------------- F0
def estimate_f0(x, sr=SR):
    """프레임 단위 자기상관 F0(기본주파수). (f0_hz, voiced_flag) 반환. 무성 프레임은 nan.

    성별 분류의 1차 단서가 F0 이다 (성인 남성 대략 85~180 Hz, 여성 165~255 Hz).
    실측에서도 상위 특징 4개가 전부 F0 계열이었다.

    원리: 유성음은 주기 신호라 자기상관이 기본주기 T 의 배수 지점에서 봉우리를 만든다.
    그 첫 봉우리의 lag 를 찾으면 F0 = sr / lag 이다.
    """
    win = 400            # 50 ms. 최저 60 Hz(주기 16.7 ms)도 두 주기 이상 담아야 봉우리가 뚜렷하다.
    hop = 160            # 20 ms. F0 는 스펙트럼보다 천천히 변하므로 10 ms 까지 촘촘할 필요가 없다.
    if len(x) < win:
        return np.array([np.nan]), np.array([False])
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx].astype(np.float64)
    fr = fr - fr.mean(axis=1, keepdims=True)      # DC 제거: 안 하면 lag 0 쪽으로 편향된다

    # Wiener-Khinchin: 자기상관 = |FFT|^2 의 역변환. O(n^2) 직접 계산보다 훨씬 빠르다.
    # nfft 를 win 의 2배 이상(1024 >= 800)으로 잡아야 원형 겹침(circular wrap)이 안 생긴다.
    nfft = 1024
    S = np.fft.rfft(fr, n=nfft, axis=1)
    ac = np.fft.irfft(S * np.conj(S), n=nfft, axis=1)[:, :win]

    # lag 0 값(= 프레임 에너지)으로 나눠 [-1, 1] 로 정규화. 음량과 무관하게 임계값을 쓸 수 있다.
    e0 = ac[:, :1].copy()
    e0[e0 <= 0] = 1e-12
    ac = ac / e0

    # 탐색 범위를 F0_MIN~F0_MAX 에 해당하는 lag 로 제한한다.
    # 이 제한이 없으면 lag 0 근처의 자명한 최대값(=1)을 잡아버린다.
    lag_lo = max(2, int(sr / F0_MAX))                        # 400 Hz -> lag 20
    lag_hi = min(win - 1, int(sr / F0_MIN))                  # 60 Hz  -> lag 133
    seg = ac[:, lag_lo:lag_hi]
    k = np.argmax(seg, axis=1)
    peak = seg[np.arange(len(k)), k]                         # 봉우리 높이 = 주기성의 강도
    lag = k + lag_lo

    # 포물선 보간: lag 은 정수 샘플 단위라 8 kHz 에서 해상도가 거칠다.
    # 예를 들어 lag 32/33 은 250.0/242.4 Hz 로 8 Hz 나 벌어진다.
    # 봉우리와 좌우 이웃 세 점에 포물선을 맞춰 꼭짓점을 소수점까지 추정한다.
    kk = np.clip(lag, 1, win - 2)
    y0 = ac[np.arange(len(kk)), kk - 1]
    y1 = ac[np.arange(len(kk)), kk]
    y2 = ac[np.arange(len(kk)), kk + 1]
    denom = (y0 - 2 * y1 + y2)
    shift = np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / np.where(denom == 0, 1e-12, denom), 0.0)
    lag_ref = lag + np.clip(shift, -1, 1)                    # 보정은 +-1 샘플로 제한

    # 유/무성 판정. 무성음(ㅅ, ㅎ)과 묵음에 F0 를 부여하면 통계가 오염된다.
    # peak > 0.35: 주기성이 충분한가.  rms > 1e-4: 묵음이 아닌가.
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

    # 스펙트럼 형상 — 음색과 녹음 채널을 요약한다 (M2 의 대원/신고자 구분에 기여)
    freqs = np.linspace(0, sr / 2, P.shape[1])
    Ps = P + 1e-12
    tot = Ps.sum(axis=1)
    # centroid: 스펙트럼의 무게중심 주파수. 높을수록 밝은 소리.
    centroid = (Ps * freqs).sum(axis=1) / tot
    # spread: centroid 기준 표준편차. 에너지가 한 대역에 몰렸는지 퍼졌는지.
    spread = np.sqrt((Ps * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / tot)
    # rolloff: 누적 에너지가 85% 에 도달하는 주파수. 고역 성분의 양을 나타낸다.
    cum = np.cumsum(Ps, axis=1) / tot[:, None]
    rolloff = freqs[np.argmax(cum >= 0.85, axis=1)]
    # flatness: 기하평균/산술평균. 1 에 가까우면 잡음성, 0 에 가까우면 조성음(유성음).
    flatness = np.exp(np.log(Ps).mean(axis=1)) / (Ps.mean(axis=1))
    _stats(centroid, "centroid", out)
    _stats(spread, "spread", out)
    _stats(rolloff, "rolloff", out)
    _stats(flatness, "flatness", out)

    # 시간 영역
    fr = frame_signal(x)
    # zcr: 부호가 바뀌는 비율. 무성 자음에서 높고 유성음에서 낮다 — F0 의 거친 대용치.
    zcr = (np.diff(np.sign(fr), axis=1) != 0).mean(axis=1)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    _stats(zcr, "zcr", out)
    _stats(20 * np.log10(rms), "rmsdb", out)      # dB 로 변환해 분포를 정규분포에 가깝게
    # 동적 범위: 상황실(조용, 압축된 헤드셋)과 야외 휴대폰(잡음, 넓은 변동)을 가르는 단서.
    out["dyn_range_db"] = float(np.percentile(20 * np.log10(rms), 95)
                                - np.percentile(20 * np.log10(rms), 5))
    # 길이 자체도 신호다. 대원 발화는 정형화돼 짧고, 신고자 발화는 길게 늘어지는 경향이 있다.
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
