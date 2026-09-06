# -*- coding: utf-8 -*-
"""Colab 학습 노트북 생성기.

대회 규정상 '학습 로그가 포함된 .ipynb' 를 제출해야 하므로, 노트북을 손으로 관리하지 않고
여기서 생성한다. Colab 에서 실행 → 로그가 찍힌 상태로 저장 → 그대로 제출.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[2] / "notebooks"
NB_DIR.mkdir(exist_ok=True)

REPO = "https://github.com/pnu-final-boss/2026-data-creator-camp.git"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip().split("\n")}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.strip("\n").split("\n")}


def notebook(cells, name):
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    p = NB_DIR / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"wrote {p}")


# ===================================================================== M3
M3 = [
    md("""
# Mission 3 · 증상 9클래스 multi-label (Colab GPU)

로컬 CPU 베이스라인(TF-IDF + LogReg)은 macro-F1 **0.5848**. 이 노트북은 그 위에
KLUE-RoBERTa / Kc-ELECTRA 파인튜닝을 올린다.

**데이터 경로**: 로컬에서 `python -m src.preprocess.pack_for_colab --what m3` 로 만든
`m3_text.json.gz` (14 MB) 를 Google Drive 의 `MyDrive/dcc/` 에 올려둘 것.
전사 텍스트에는 개인정보가 포함되므로 **GitHub 에는 절대 올리지 않는다** — Drive 경유만 사용.
"""),
    code("""
import torch, subprocess
print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv'],
                     capture_output=True, text=True).stdout)
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
DATA = '/content/drive/MyDrive/dcc/m3_text.json.gz'
CKPT = '/content/drive/MyDrive/dcc/ckpt'
import os; os.makedirs(CKPT, exist_ok=True)
"""),
    code("""
!pip -q install "transformers>=4.44" "accelerate>=0.33" scikit-learn
"""),
    md("""
## 저장소 코드 불러오기

지표 계산과 그림 그리기는 **로컬 베이스라인과 똑같은 코드**를 써야 한다.
노트북에 따로 구현하면 GPU 결과와 CPU 기준선이 다른 계산으로 나와 비교가 무의미해진다.
`src.zip`(49 KB)을 Drive 의 `MyDrive/dcc/` 에 함께 올려 두고 여기서 풀어 쓴다.
(GitHub 에 push 했다면 `!git clone` 으로 대체해도 된다.)
"""),
    code("""
import sys, zipfile, os
os.makedirs('/content/repo', exist_ok=True)
with zipfile.ZipFile(f'{DIR}/src.zip') as z:
    z.extractall('/content/repo')
sys.path.insert(0, '/content/repo')

# 로컬과 동일한 지표 구현을 그대로 가져온다 (출제 PDF 10쪽 절차)
from src.common.metrics import macro_f1_pdf, tune_thresholds
from src.common.paths import SYMPTOM_9 as SYM_FROM_SRC
print('불러온 클래스:', SYM_FROM_SRC)
"""),
    code("""
import gzip, json, numpy as np
with gzip.open(DATA, 'rt', encoding='utf-8') as f:
    D = json.load(f)
SYMPTOM_9 = ["고열","구토","두통","복통","어지러움","열상","오심","전신쇠약","호흡곤란"]

tr, va = D['train'], D['val']
Xtr = [r['text'] for r in tr];  Ytr = np.array([r['y'] for r in tr], dtype=np.float32)
Xva = [r['text'] for r in va];  Yva = np.array([r['y'] for r in va], dtype=np.float32)

# Training 안에서 dev 분리 — 임계값 튜닝 전용. Validation 은 학습/튜닝에 절대 사용 금지.
rng = np.random.default_rng(0); perm = rng.permutation(len(tr)); n_dev = len(tr)//10
dev_i, fit_i = perm[:n_dev], perm[n_dev:]
print(f'fit={len(fit_i)}  dev={len(dev_i)}  val={len(Xva)}  labels={Ytr.shape}')
print('클래스별 양성 비율:', dict(zip(SYMPTOM_9, (Ytr.mean(0)*100).round(1))))
"""),
    code("""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# klue/roberta-base: KLUE 벤치마크 기준 한국어 이해 태스크 전반에서 안정적.
# 대안 'beomi/KcELECTRA-base-v2022' 는 구어체 말뭉치로 학습돼 있어 통화 전사문과 결이 맞고,
# AI-Hub 가 이 데이터셋의 공식 베이스라인으로 쓴 Kc-ELECTRA 계열이다. 둘 다 돌려 비교할 것.
MODEL = 'klue/roberta-base'

# 전체 대화 길이는 중앙값 397자 / p95 842자 / 최대 1,292자 (로컬 실측).
# 512 토큰이면 대부분 들어가지만 상위 5% 가량은 뒤가 잘린다.
# 증상은 통화 초반(신고 사유 진술)에 몰리므로 앞을 남기는 기본 truncation 이 유리하다.
MAXLEN = 512

tok = AutoTokenizer.from_pretrained(MODEL)

# problem_type='multi_label_classification' 이 핵심.
# 이 값을 주면 HF 가 손실을 BCEWithLogitsLoss 로 바꾸고 9개 라벨을 각각 독립 판정한다.
# (기본값이면 softmax CrossEntropy 가 걸려 "9개 중 정확히 1개" 문제로 잘못 풀린다)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=9, problem_type='multi_label_classification').cuda()
print(f'{MODEL}  파라미터 {sum(p.numel() for p in model.parameters())/1e6:.0f}M')
"""),
    code("""
import torch
from torch.utils.data import Dataset, DataLoader


class DS(Dataset):
    \"\"\"토큰화는 collate 에서 배치 단위로 한다 — 배치 안의 최대 길이에만 padding 해 낭비를 줄인다.\"\"\"
    def __init__(self, texts, y):
        self.t, self.y = texts, y

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return self.t[i], self.y[i]


def collate(batch):
    txt = [b[0] for b in batch]
    # 라벨은 float 여야 한다. multi_label 모드는 BCEWithLogitsLoss 를 쓰는데 이 손실은
    # 정수 타깃을 받지 않는다 (단일 클래스 CrossEntropy 와 다른 점).
    y = torch.tensor(np.stack([b[1] for b in batch]))
    enc = tok(txt, truncation=True, max_length=MAXLEN, padding=True, return_tensors='pt')
    enc['labels'] = y
    return enc


# fit = 실제 가중치 학습, dev = 임계값 튜닝, val = 최종 보고. 세 집합의 역할을 섞지 않는다.
fit_dl = DataLoader(DS([Xtr[i] for i in fit_i], Ytr[fit_i]), batch_size=16,
                    shuffle=True, collate_fn=collate, num_workers=2)
dev_dl = DataLoader(DS([Xtr[i] for i in dev_i], Ytr[dev_i]), batch_size=32,
                    collate_fn=collate, num_workers=2)
val_dl = DataLoader(DS(Xva, Yva), batch_size=32, collate_fn=collate, num_workers=2)

# 학습 곡선용: 에폭마다 train 쪽 손실/F1 을 '평가 모드'로 다시 재기 위한 로더.
# fit 전체(26k)를 매 에폭 평가하면 비싸므로 val 과 비슷한 크기로 무작위 추출해 쓴다.
# (곡선의 목적은 절대값이 아니라 train-val 격차의 추세를 보는 것이다)
fit_eval_i = fit_i[:len(Xva)]
fit_eval_dl = DataLoader(DS([Xtr[i] for i in fit_eval_i], Ytr[fit_eval_i]), batch_size=32,
                         collate_fn=collate, num_workers=2)
"""),
    code("""
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

EPOCHS = 3      # 26k 샘플 기준 3 epoch 이면 수렴한다. 더 돌리면 희소 클래스에 과적합되기 쉽다.

# lr 2e-5 는 BERT 계열 파인튜닝의 표준값. weight_decay 는 LayerNorm/bias 에도 걸리지만
# 이 규모에서는 영향이 미미해 단순하게 둔다.
opt = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# 선형 감쇠 + 초반 6% warmup. warmup 없이 시작하면 초기 큰 gradient 가 사전학습 표현을 망친다.
steps = len(fit_dl) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.amp.GradScaler('cuda')      # 혼합정밀: T4 에서 메모리 절감 + 속도 향상


@torch.no_grad()
def predict(dl):
    \"\"\"(n, 9) 확률 행렬을 반환. 임계값은 적용하지 않는다 — 튜닝은 나중 셀에서 한다.\"\"\"
    model.eval()
    out = []
    for b in dl:
        b = {k: v.cuda() for k, v in b.items()}
        with torch.amp.autocast('cuda'):
            # labels 를 빼고 넘겨 손실 계산을 건너뛴다.
            logits = model(**{k: v for k, v in b.items() if k != 'labels'}).logits
        # multi-label 이므로 softmax 가 아니라 sigmoid. 각 라벨이 독립 확률이다.
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def eval_loss_and_f1(dl, y_true):
    \"\"\"에폭 끝마다 호출: 해당 split 의 평균 손실과 macro-F1(임계값 0.5)을 함께 잰다.\"\"\"
    from sklearn.metrics import f1_score
    model.eval()
    tot, n, probs = 0.0, 0, []
    for b in dl:
        b = {k: v.cuda() for k, v in b.items()}
        with torch.amp.autocast('cuda'):
            out = model(**b)
        tot += out.loss.item() * b['labels'].size(0)
        n += b['labels'].size(0)
        probs.append(torch.sigmoid(out.logits.float()).cpu().numpy())
    p = (np.concatenate(probs) >= 0.5).astype(int)
    f1 = float(np.mean([f1_score(y_true[:, c], p[:, c], zero_division=0) for c in range(9)]))
    return tot / n, f1


# 에폭별 곡선을 남긴다. train/val 손실이 갈라지기 시작하는 지점이 과적합 시작점이고,
# 거기서 EPOCHS 를 줄이거나 early stopping 을 걸면 된다.
hist = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

for ep in range(EPOCHS):
    model.train()
    tot = 0.0
    for i, b in enumerate(fit_dl):
        b = {k: v.cuda() for k, v in b.items()}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            loss = model(**b).loss      # labels 가 있으면 BCEWithLogitsLoss 를 내부 계산
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot += loss.item()
        if (i+1) % 200 == 0:
            print(f'ep{ep+1} step {i+1}/{len(fit_dl)} loss {tot/(i+1):.4f}', flush=True)

    # train 손실은 학습 중 누적값(드롭아웃 켜진 상태)이라 val 과 직접 비교가 어렵다.
    # 공정한 비교를 위해 eval 모드로 fit split 을 다시 한 번 잰다.
    trl, trf = eval_loss_and_f1(fit_eval_dl, Ytr[fit_eval_i])
    val, vaf = eval_loss_and_f1(val_dl, Yva)
    hist['epoch'].append(ep + 1)
    hist['train_loss'].append(trl); hist['val_loss'].append(val)
    hist['train_f1'].append(trf);   hist['val_f1'].append(vaf)
    print(f'== epoch {ep+1}  train loss {trl:.4f} / F1 {trf:.4f}   '
          f'val loss {val:.4f} / F1 {vaf:.4f}')
"""),
    code("""
# ---- 학습 곡선: train vs validation ----
import matplotlib.pyplot as plt, matplotlib
!apt-get -qq install fonts-nanum > /dev/null
matplotlib.font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
matplotlib.rc('font', family='NanumGothic'); matplotlib.rc('axes', unicode_minus=False)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
e = hist['epoch']

# 왼쪽: 손실. 두 곡선이 벌어지기 시작하면 과적합이 시작된 것이다.
a1.plot(e, hist['train_loss'], 'o-', color='#9FB6C9', label='Training')
a1.plot(e, hist['val_loss'], 'o-', color='#1B5E96', label='Validation')
a1.set_xlabel('에폭'); a1.set_ylabel('BCE 손실'); a1.set_title('손실')
a1.legend(frameon=False); a1.grid(axis='y', alpha=.3)

# 오른쪽: macro-F1. 실제 채점 지표이므로 이쪽이 모델 선택의 기준이다.
a2.plot(e, hist['train_f1'], 'o-', color='#9FB6C9', label='Training')
a2.plot(e, hist['val_f1'], 'o-', color='#1B5E96', label='Validation')
a2.axhline(0.5848, color='#2A6B48', ls=':', label='로컬 CPU 베이스라인')
a2.set_xlabel('에폭'); a2.set_ylabel('macro-F1'); a2.set_title('macro-F1 (임계값 0.5)')
a2.legend(frameon=False); a2.grid(axis='y', alpha=.3)
fig.tight_layout(); plt.show()

best = int(np.argmax(hist['val_f1'])) + 1
print(f"val macro-F1 최고: epoch {best} ({max(hist['val_f1']):.4f})")
if best < len(e):
    print(f"→ 이후 에폭에서 하락했다면 EPOCHS 를 {best} 로 줄이는 편이 낫다.")
"""),
    code("""
from sklearn.metrics import f1_score

# macro_f1_pdf 와 tune_thresholds 는 위에서 src.zip 에서 불러왔다.
# 노트북에 다시 구현하지 않는 이유: 로컬 CPU 베이스라인(0.5848)과 반드시 같은 계산이어야
# 두 숫자를 비교할 수 있다. 구현이 갈리면 비교 자체가 무의미해진다.
def macro_f1(y_true, y_pred):
    return macro_f1_pdf(y_true.astype(int), y_pred.astype(int))[0]

dev_s, val_s = predict(dev_dl), predict(val_dl)

# 클래스별 임계값을 dev 에서만 튜닝한다. Validation 으로 튜닝하면 점수가 낙관적으로 부풀려진다.
th = tune_thresholds(Ytr[dev_i].astype(int), dev_s)

pred05 = (val_s >= 0.5).astype(int)
predth = (val_s >= th[None,:]).astype(int)
print(f'Validation macro-F1  @0.5={macro_f1(Yva, pred05):.4f}   @tuned={macro_f1(Yva, predth):.4f}')
print('로컬 CPU 베이스라인(TF-IDF+LogReg) = 0.5848  <- 이 값을 넘어야 GPU 도입이 정당화된다')

# 로컬 베이스라인의 클래스별 F1. 어느 클래스가 개선됐는지 직접 비교하기 위해 박아둔다.
LOCAL = {'고열':0.657,'구토':0.487,'두통':0.438,'복통':0.768,'어지러움':0.617,
         '열상':0.846,'오심':0.300,'전신쇠약':0.541,'호흡곤란':0.610}
for c, s, t in zip(SYMPTOM_9, [f1_score(Yva[:,i], predth[:,i], zero_division=0) for i in range(9)], th):
    print(f'  {c:6s} F1={s:.3f}  th={t:.2f}   (로컬 {LOCAL[c]:.3f}, 차이 {s-LOCAL[c]:+.3f})')
"""),
    code("""
import matplotlib.pyplot as plt
import matplotlib
!apt-get -qq install fonts-nanum > /dev/null
matplotlib.font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
matplotlib.rc('font', family='NanumGothic'); matplotlib.rc('axes', unicode_minus=False)

fig, axes = plt.subplots(3, 3, figsize=(10.5, 10))
for k, ax in enumerate(axes.ravel()):
    t, p = Yva[:,k].astype(int), predth[:,k]
    cm = np.array([[((t==0)&(p==0)).sum(), ((t==0)&(p==1)).sum()],
                   [((t==1)&(p==0)).sum(), ((t==1)&(p==1)).sum()]], dtype=float)
    pct = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(pct, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0,1], ['없음','있음']); ax.set_yticks([0,1], ['없음','있음'])
    ax.set_title(f"{SYMPTOM_9[k]}  F1={f1_score(t, p, zero_division=0):.2f}", fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{int(cm[i,j]):,}\\n{pct[i,j]*100:.1f}%', ha='center', va='center',
                    fontsize=9, color='white' if pct[i,j] > 0.55 else '#16202B')
fig.suptitle(f'Mission 3 · KLUE-RoBERTa 혼동행렬 (macro-F1={macro_f1(Yva, predth):.4f})', fontsize=13)
fig.tight_layout(); plt.show()
"""),
    code("""
torch.save({'state_dict': model.state_dict(), 'thresholds': th,
            'model_name': MODEL, 'classes': SYMPTOM_9, 'maxlen': MAXLEN},
           f'{CKPT}/m3.pt')
print('저장 완료:', f'{CKPT}/m3.pt')
"""),
    md("""
## 예측을 로컬과 같은 형식으로 저장

`cache/m3_val_pred.npz` 의 스키마(y_true / y_score / y_pred / thresholds / classes)를 그대로 맞춘다.
이 파일을 로컬로 내려받아 `cache/` 에 넣으면, 로컬에서 이미 쓰던
`python -m src.viz.plots m3` 와 `python -m src.viz.inspect_errors m3 --cls 오심` 이
**GPU 결과에 대해서도 그대로 동작한다.** 혼동행렬을 같은 코드로 그려야 CPU 기준선과 비교된다.
"""),
    code("""
np.savez_compressed(f'{DIR}/m3_val_pred_gpu.npz',
                    y_true=Yva.astype('int8'), y_score=val_s,
                    y_pred=predth.astype('int64'), thresholds=th,
                    classes=np.array(SYMPTOM_9))
print('저장:', f'{DIR}/m3_val_pred_gpu.npz')
print('\\n로컬에서 할 일:')
print('  1) Drive 에서 m3_val_pred_gpu.npz 를 내려받아 cache/m3_val_pred.npz 로 저장')
print('  2) python -m src.viz.plots m3')
print('  3) python -m src.viz.inspect_errors m3 --cls 오심 --n 5')
"""),
]

# ===================================================================== M1/M2
AUDIO = [
    md("""
# Mission 1 / 2 · 음성 파인튜닝 (Colab GPU)

로컬 CPU 베이스라인은 MFCC/F0 통계 + 얕은 분류기. 이 노트북은 wav2vec2 를 파인튜닝한다.

**중요**: 원본은 8 kHz 협대역이고 wav2vec2 는 16 kHz 광대역 사전학습이다. 단순 업샘플만으로도
동작하지만 대역 불일치로 성능이 깎인다 (Sivaraman & Khoury, Odyssey'20). 업샘플은 GPU 쪽에서
수행하고, 업로드는 8 kHz 원본으로 해 용량을 절반으로 줄인다.

`python -m src.preprocess.pack_for_colab --what audio` 로 만든 npz 를 `MyDrive/dcc/` 에 올릴 것.
"""),
    code("""
from google.colab import drive; drive.mount('/content/drive')
DIR = '/content/drive/MyDrive/dcc'
!pip -q install "transformers>=4.44" torchaudio scikit-learn
import torch, numpy as np, torchaudio
print('cuda', torch.cuda.is_available())
"""),
    code("""
MISSION = 'm2'      # 'm1' = 성별(통화 단위), 'm2' = 화자 역할(발화 단위). 이 값만 바꾸면 전환된다.


def load(split):
    \"\"\"번들을 읽어 조각 리스트로 되돌린다.

    번들은 모든 오디오를 하나의 긴 int16 배열(audio)로 이어붙이고 경계를 offsets 에 담았다.
    조각을 개별 배열로 저장하면 npz 안에 수만 개의 엔트리가 생겨 로딩이 매우 느려지기 때문이다.
    offsets 는 길이 n+1 이고 i 번째 조각은 audio[offsets[i]:offsets[i+1]] 이다.
    \"\"\"
    d = np.load(f'{DIR}/{MISSION}_audio_{split}.npz')
    a, off, y = d['audio'], d['offsets'], d['y']
    segs = [a[off[i]:off[i+1]] for i in range(len(off)-1)]
    # overlap: 다른 화자와 시간이 겹치는가 (M2 진단용)
    # call: 어느 통화에서 나온 조각인가 (통화 단위 분석용)
    extra = {k: d[k] for k in ('overlap', 'call') if k in d}
    return segs, y.astype(np.int64), extra


tr_x, tr_y, _ = load('train')
va_x, va_y, va_extra = load('val')
print(f'train {len(tr_x):,}  val {len(va_x):,}  양성비율 {tr_y.mean():.3f}')
if 'overlap' in va_extra:
    print(f'val 중첩 비율 {va_extra["overlap"].mean():.3f}')
"""),
    code("""
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL = 'facebook/wav2vec2-base'
SEC = 3.0            # 모든 조각을 이 길이로 맞춘다. 배치로 묶으려면 길이가 같아야 하고,
                     # 3초는 발화 길이 p95(5.4초)와 중앙값(1.4초) 사이의 절충값이다.
SR_IN, SR_OUT = 8000, 16000

# 8 kHz -> 16 kHz 업샘플. wav2vec2 는 16 kHz 로 사전학습돼 있어 입력 샘플레이트를 맞춰야 한다.
# 업샘플이 없는 정보를 만들어내지는 못한다 (원본은 여전히 0.3-3.4 kHz 전화 대역).
# 대역 불일치로 성능이 깎이는 건 알려진 문제이며, 개선하려면 대역확장(BWE)이 필요하다.
resamp = torchaudio.transforms.Resample(SR_IN, SR_OUT)
N = int(SEC * SR_OUT)      # 고정 입력 길이 = 48,000 샘플

fe = AutoFeatureExtractor.from_pretrained(MODEL)
# num_labels=2 -> M1 은 (여성, 남성), M2 는 (119대원, 신고자). 둘 다 이진 분류라 코드를 공유한다.
model = AutoModelForAudioClassification.from_pretrained(MODEL, num_labels=2).cuda()


class ADS(Dataset):
    \"\"\"int16 조각 -> 16 kHz 고정 길이 float 파형.\"\"\"
    def __init__(self, segs, y):
        self.s, self.y = segs, y

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        # int16 (-32768..32767) -> float [-1, 1). 번들을 int16 로 저장해 용량을 절반으로 줄였다.
        w = torch.from_numpy(self.s[i].astype(np.float32) / 32768.0)
        w = resamp(w)
        if len(w) < N:
            # 짧은 발화는 뒤를 0 으로 채운다. 맞장구("예", "네")가 0.5초 미만이라 이 경우가 흔하다.
            w = torch.nn.functional.pad(w, (0, N - len(w)))
        elif len(w) > N:
            # 긴 발화는 가운데를 자른다. 발화 시작/끝은 묵음이나 겹침이 섞이기 쉬워 중앙이 안전하다.
            o = (len(w) - N) // 2
            w = w[o:o+N]
        return w, self.y[i]


def collate(b):
    \"\"\"배치를 쌓고 파형별로 표준화한다.

    wav2vec2 의 feature extractor 는 zero-mean/unit-variance 입력을 가정한다(do_normalize=True).
    여기서는 조각마다 개별 정규화하므로, 통화별 녹음 음량 차이가 모델에 노출되지 않는다.
    M2 에서는 이 점을 유의할 것 — 음량/채널 차이가 신고자 vs 대원의 실제 단서이기 때문에,
    정규화가 오히려 유용한 신호를 지울 수 있다. 정규화를 끈 버전과 비교해볼 가치가 있다.
    \"\"\"
    x = torch.stack([i[0] for i in b])
    x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
    return x, torch.tensor([i[1] for i in b])


# batch_size 는 T4 16GB 기준. OOM 이 나면 16 -> 8 로 줄일 것.
tr_dl = DataLoader(ADS(tr_x, tr_y), batch_size=16, shuffle=True, collate_fn=collate, num_workers=2)
va_dl = DataLoader(ADS(va_x, va_y), batch_size=32, collate_fn=collate, num_workers=2)
print(f'배치당 입력 shape: {next(iter(tr_dl))[0].shape}  (batch, 48000 샘플 = 3초 @16kHz)')
"""),
    code("""
from torch.optim import AdamW

# lr 3e-5 는 wav2vec2 파인튜닝의 관례적 범위(1e-5 ~ 5e-5). 더 크면 사전학습 표현이 무너진다.
opt = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

# 혼합정밀(AMP). T4 에서 메모리를 절반으로 줄이고 속도를 약 2배 올린다.
# GradScaler 는 fp16 언더플로로 gradient 가 0 이 되는 것을 막아준다.
scaler = torch.amp.GradScaler('cuda')


@torch.no_grad()
def evaluate():
    \"\"\"Validation 전체 예측. argmax 이므로 임계값 0.5 고정과 동일하다.\"\"\"
    model.eval()
    P = []
    for x, _ in va_dl:
        with torch.amp.autocast('cuda'):
            P.append(model(input_values=x.cuda()).logits.float().argmax(-1).cpu().numpy())
    return np.concatenate(P)


# Colab 무료 티어는 세션이 끊길 수 있으므로 epoch 마다 평가하고 결과를 남긴다.
# 18만 세그먼트 x 1 epoch 은 T4 에서 1~2시간이 걸린다.
for ep in range(2):
    model.train()
    tot = 0
    for i, (x, y) in enumerate(tr_dl):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            # labels 를 넘기면 HF 모델이 CrossEntropyLoss 를 내부에서 계산한다.
            out = model(input_values=x.cuda(), labels=y.cuda())
        scaler.scale(out.loss).backward()
        scaler.step(opt)
        scaler.update()
        tot += out.loss.item()
        if (i+1) % 200 == 0:
            print(f'ep{ep+1} {i+1}/{len(tr_dl)} loss {tot/(i+1):.4f}', flush=True)

    p = evaluate()
    acc = (p == va_y).mean()
    print(f'== epoch {ep+1}  val acc {acc:.4f}   (CPU 베이스라인: M1 0.9434 / M2 0.8799)')

    # M2 의 핵심 진단: 발화의 37% 가 다른 화자와 시간적으로 겹친다.
    # 로컬 베이스라인에서 비중첩 0.8944 vs 중첩 0.8550 으로 3.9p 차이가 났다.
    # 이 격차가 줄어드는지가 wav2vec2 도입의 실질적 판단 기준이다.
    if 'overlap' in va_extra:
        ov = va_extra['overlap']
        print(f'   비중첩 {(p[ov==0]==va_y[ov==0]).mean():.4f} / 중첩 {(p[ov==1]==va_y[ov==1]).mean():.4f}')
"""),
    code("""
import matplotlib.pyplot as plt, matplotlib

# Colab 기본 이미지에는 한글 폰트가 없어 축 라벨이 두부(□)로 깨진다. 나눔고딕을 설치해 등록한다.
!apt-get -qq install fonts-nanum > /dev/null
matplotlib.font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
matplotlib.rc('font', family='NanumGothic')
matplotlib.rc('axes', unicode_minus=False)      # 한글 폰트 사용 시 음수 기호가 깨지는 것 방지

p = evaluate()
labels = ['여성 F', '남성 M'] if MISSION == 'm1' else ['119대원', '신고자']

# 혼동행렬을 [[TN, FP], [FN, TP]] 순서로 만든다. 행=정답, 열=예측.
cm = np.array([[((va_y == i) & (p == j)).sum() for j in (0, 1)] for i in (0, 1)], dtype=float)
pct = cm / np.maximum(cm.sum(1, keepdims=True), 1)      # 행 정규화 = 클래스별 재현율

fig, ax = plt.subplots(figsize=(4.2, 3.8))
ax.imshow(pct, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks([0, 1], labels)
ax.set_yticks([0, 1], labels)
ax.set_xlabel('예측')
ax.set_ylabel('정답')
ax.set_title(f'{MISSION.upper()} wav2vec2  acc={(p==va_y).mean():.4f}')
for i in range(2):
    for j in range(2):
        # 진한 셀에는 흰 글씨, 옅은 셀에는 검은 글씨로 대비를 확보한다.
        ax.text(j, i, f'{int(cm[i,j]):,}\\n{pct[i,j]*100:.1f}%', ha='center', va='center',
                color='white' if pct[i, j] > 0.55 else '#16202B')
plt.tight_layout(); plt.show()

# 체크포인트는 Drive 에 저장한다. Colab 런타임 디스크는 연결이 끊기면 사라진다.
torch.save({'state_dict': model.state_dict(), 'model_name': MODEL, 'sec': SEC},
           f'{DIR}/ckpt/{MISSION}.pt')
print('저장 완료:', f'{DIR}/ckpt/{MISSION}.pt')
"""),
]

if __name__ == "__main__":
    notebook(M3, "colab_m3_klue.ipynb")
    notebook(AUDIO, "colab_m1m2_wav2vec2.ipynb")
