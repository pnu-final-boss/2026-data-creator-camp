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
MODEL = 'klue/roberta-base'      # 대안: 'beomi/KcELECTRA-base-v2022' (AI-Hub 공식 베이스라인 계열)
MAXLEN = 512                      # 전체 대화 중앙값 397자 / p95 842자

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=9, problem_type='multi_label_classification').cuda()
"""),
    code("""
import torch
from torch.utils.data import Dataset, DataLoader

class DS(Dataset):
    def __init__(self, texts, y):
        self.t, self.y = texts, y
    def __len__(self):
        return len(self.t)
    def __getitem__(self, i):
        return self.t[i], self.y[i]

def collate(batch):
    txt = [b[0] for b in batch]
    y = torch.tensor(np.stack([b[1] for b in batch]))
    enc = tok(txt, truncation=True, max_length=MAXLEN, padding=True, return_tensors='pt')
    enc['labels'] = y
    return enc

fit_dl = DataLoader(DS([Xtr[i] for i in fit_i], Ytr[fit_i]), batch_size=16,
                    shuffle=True, collate_fn=collate, num_workers=2)
dev_dl = DataLoader(DS([Xtr[i] for i in dev_i], Ytr[dev_i]), batch_size=32,
                    collate_fn=collate, num_workers=2)
val_dl = DataLoader(DS(Xva, Yva), batch_size=32, collate_fn=collate, num_workers=2)
"""),
    code("""
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

EPOCHS = 3
opt = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
steps = len(fit_dl) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(0.06*steps), steps)
scaler = torch.amp.GradScaler('cuda')

@torch.no_grad()
def predict(dl):
    model.eval(); out = []
    for b in dl:
        b = {k: v.cuda() for k, v in b.items()}
        with torch.amp.autocast('cuda'):
            logits = model(**{k: v for k, v in b.items() if k != 'labels'}).logits
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)

for ep in range(EPOCHS):
    model.train(); tot = 0.0
    for i, b in enumerate(fit_dl):
        b = {k: v.cuda() for k, v in b.items()}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            loss = model(**b).loss
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item()
        if (i+1) % 200 == 0:
            print(f'ep{ep+1} step {i+1}/{len(fit_dl)} loss {tot/(i+1):.4f}', flush=True)
    print(f'== epoch {ep+1} 평균 loss {tot/len(fit_dl):.4f}')
"""),
    code("""
from sklearn.metrics import f1_score

def macro_f1(y_true, y_pred):
    \"\"\"출제 PDF 10쪽 정의: 클래스별 F1 → 단순 평균.\"\"\"
    return float(np.mean([f1_score(y_true[:,c], y_pred[:,c], zero_division=0)
                          for c in range(y_true.shape[1])]))

dev_s, val_s = predict(dev_dl), predict(val_dl)

# 클래스별 임계값을 dev 에서 튜닝
grid = np.arange(0.05, 0.95, 0.01)
th = np.array([grid[np.argmax([f1_score(Ytr[dev_i][:,c], (dev_s[:,c]>=g).astype(int),
                                        zero_division=0) for g in grid])]
               for c in range(9)])

pred05 = (val_s >= 0.5).astype(int)
predth = (val_s >= th[None,:]).astype(int)
print(f'Validation macro-F1  @0.5={macro_f1(Yva, pred05):.4f}   @tuned={macro_f1(Yva, predth):.4f}')
print('로컬 CPU 베이스라인(TF-IDF+LogReg) = 0.5848')
for c, s, t in zip(SYMPTOM_9, [f1_score(Yva[:,i], predth[:,i], zero_division=0) for i in range(9)], th):
    print(f'  {c:6s} F1={s:.3f}  th={t:.2f}')
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
MISSION = 'm2'      # 'm1' = 성별, 'm2' = 화자 역할

def load(split):
    d = np.load(f'{DIR}/{MISSION}_audio_{split}.npz')
    a, off, y = d['audio'], d['offsets'], d['y']
    segs = [a[off[i]:off[i+1]] for i in range(len(off)-1)]
    extra = {k: d[k] for k in ('overlap','call') if k in d}
    return segs, y.astype(np.int64), extra

tr_x, tr_y, _ = load('train')
va_x, va_y, va_extra = load('val')
print(f'train {len(tr_x):,}  val {len(va_x):,}  양성비율 {tr_y.mean():.3f}')
"""),
    code("""
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL = 'facebook/wav2vec2-base'
SEC = 3.0                      # 고정 길이 창 (짧으면 pad, 길면 center-crop)
SR_IN, SR_OUT = 8000, 16000
resamp = torchaudio.transforms.Resample(SR_IN, SR_OUT)
N = int(SEC * SR_OUT)

fe = AutoFeatureExtractor.from_pretrained(MODEL)
model = AutoModelForAudioClassification.from_pretrained(MODEL, num_labels=2).cuda()

class ADS(Dataset):
    def __init__(self, segs, y): self.s, self.y = segs, y
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        w = torch.from_numpy(self.s[i].astype(np.float32) / 32768.0)
        w = resamp(w)
        if len(w) < N:
            w = torch.nn.functional.pad(w, (0, N - len(w)))
        elif len(w) > N:
            o = (len(w) - N) // 2; w = w[o:o+N]
        return w, self.y[i]

def collate(b):
    x = torch.stack([i[0] for i in b])
    x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
    return x, torch.tensor([i[1] for i in b])

tr_dl = DataLoader(ADS(tr_x, tr_y), batch_size=16, shuffle=True, collate_fn=collate, num_workers=2)
va_dl = DataLoader(ADS(va_x, va_y), batch_size=32, collate_fn=collate, num_workers=2)
"""),
    code("""
from torch.optim import AdamW
opt = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
scaler = torch.amp.GradScaler('cuda')

@torch.no_grad()
def evaluate():
    model.eval(); P = []
    for x, _ in va_dl:
        with torch.amp.autocast('cuda'):
            P.append(model(input_values=x.cuda()).logits.float().argmax(-1).cpu().numpy())
    return np.concatenate(P)

for ep in range(2):
    model.train(); tot = 0
    for i, (x, y) in enumerate(tr_dl):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(input_values=x.cuda(), labels=y.cuda())
        scaler.scale(out.loss).backward(); scaler.step(opt); scaler.update()
        tot += out.loss.item()
        if (i+1) % 200 == 0: print(f'ep{ep+1} {i+1}/{len(tr_dl)} loss {tot/(i+1):.4f}', flush=True)
    p = evaluate()
    acc = (p == va_y).mean()
    print(f'== epoch {ep+1}  val acc {acc:.4f}')
    if 'overlap' in va_extra:
        ov = va_extra['overlap']
        print(f'   비중첩 {(p[ov==0]==va_y[ov==0]).mean():.4f} / 중첩 {(p[ov==1]==va_y[ov==1]).mean():.4f}')
"""),
    code("""
import matplotlib.pyplot as plt, matplotlib
!apt-get -qq install fonts-nanum > /dev/null
matplotlib.font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
matplotlib.rc('font', family='NanumGothic'); matplotlib.rc('axes', unicode_minus=False)

p = evaluate()
labels = ['여성 F','남성 M'] if MISSION=='m1' else ['119대원','신고자']
cm = np.array([[((va_y==i)&(p==j)).sum() for j in (0,1)] for i in (0,1)], dtype=float)
pct = cm/np.maximum(cm.sum(1,keepdims=True),1)
fig, ax = plt.subplots(figsize=(4.2,3.8))
ax.imshow(pct, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks([0,1], labels); ax.set_yticks([0,1], labels)
ax.set_xlabel('예측'); ax.set_ylabel('정답')
ax.set_title(f'{MISSION.upper()} wav2vec2  acc={(p==va_y).mean():.4f}')
for i in range(2):
    for j in range(2):
        ax.text(j,i,f'{int(cm[i,j]):,}\\n{pct[i,j]*100:.1f}%',ha='center',va='center',
                color='white' if pct[i,j]>0.55 else '#16202B')
plt.tight_layout(); plt.show()
torch.save({'state_dict': model.state_dict(), 'model_name': MODEL, 'sec': SEC},
           f'{DIR}/ckpt/{MISSION}.pt')
"""),
]

if __name__ == "__main__":
    notebook(M3, "colab_m3_klue.ipynb")
    notebook(AUDIO, "colab_m1m2_wav2vec2.ipynb")
