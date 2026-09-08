# 인수인계 — 2026 데이터+AI 크리에이터 캠프 대학부 예선

새 세션이 이 문서만 읽고 작업을 이어받을 수 있도록 정리했다.
**먼저 이 문서를 끝까지 읽고, 아래 "하지 말아야 할 것"을 확인한 뒤 시작할 것.**

---

## 1. 과제

119 응급신고 통화(AI-Hub 「119 지능형 신고접수 음성 인식 데이터」, 서울/구급)로 세 가지를 푼다.

| 미션 | 입력 | 출력 | 지표 |
|---|---|---|---|
| M1 | 음성 (+ `startAt`/`endAt`/`speaker` 사용 가능) | 신고자 성별 M/F | Accuracy |
| M2 | `startAt`–`endAt` 발화 조각 **오디오만** | 신고자(1) / 119대원(0) | Accuracy |
| M3 | 전사 텍스트만 | 증상 9클래스 multi-label | Macro-F1 |

**규정상 금지**
- M2: `text` 사용 불가, 발화 순서(인덱스/홀짝) 사용 불가
- M3: 전사 텍스트 외 라벨 필드(`disasterMedium`, `urgencyLevel`, `triage`, `sentiment`) 사용 불가
- 공통: Validation을 학습·튜닝에 사용 금지
- 제출: `python inference.py --audio_dir … --label_dir … --ckpt_path … --output …` **한 줄**로 완료
- 코드 내 주석 필수, 학습 로그가 포함된 `.ipynb` 제출

M3 대상 9클래스: 고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란

---

## 2. 현재 성능 (공식 Validation 3,640통)

| 미션 | 모델 | 결과 | 다수클래스 기준선 |
|---|---|---|---|
| M1 | `StandardScaler → LogisticRegression`, 143차원 | **0.9434** (AUC 0.983) | 0.5382 |
| M2 | `HistGradientBoosting`, 286차원 | **0.8799** (AUC 0.959) | 0.5190 |
| M3 | TF-IDF 266,779 → OvR LogReg × 9 | **0.5848** | 0.0 |

**M3 클래스별 F1**: 열상 0.846 · 복통 0.768 · 고열 0.657 · 어지러움 0.617 · 호흡곤란 0.610 ·
전신쇠약 0.541 · 구토 0.487 · 두통 0.438 · **오심 0.300**

전부 로컬 CPU에서 학습했다. **이 PC에는 NVIDIA GPU가 없다** (Intel Iris Xe 내장만).

---

## 3. 핵심 발견 — 반드시 읽을 것

### 3.1 M3의 라벨은 노이즈가 아니라 접수 프로토콜의 산물이다

이것이 이 프로젝트에서 가장 중요한 발견이다.

소방청 「119구급상황관리센터 상담 매뉴얼」(공개 문서)에 **주증상 33개 범주**가 정의돼 있고,
그중 **#10이 `오심/구토` 한 항목**, **#29가 `출혈/열상` 한 항목**이다.
즉 데이터 스키마가 하나의 범주를 두 라벨로 쪼갠 것이다.

`오심/구토` 프로토콜의 실제 내용:
```
핵심질문 4. 다른 증상이 있습니까?
   - 흉통 → ☞17. 흉통
   - 복통 → ☞6. 복통
   - 두통 → ☞5. 발작/경련
   - 어지러움
중증도 [응급] 판정
   1. 어지러움을 동반한 경우
   2. 전신쇠약감을 동반한 경우
```

측정된 라벨 상관이 이 문서와 정확히 대응한다.

| 관측 | 프로토콜 근거 |
|---|---|
| 오심 ↔ 구토 φ=+0.233, lift 2.53 | 같은 범주 #10 |
| 오심 ↔ 어지러움 φ=+0.236, lift 2.25 | 핵심질문 4 + 응급 기준 1 |
| 오심 ↔ 전신쇠약 | 응급 기준 2 |
| 열상 ↔ 모든 질환 lift 0.00~0.33 | 열상은 **손상 범주**, 나머지는 질환 범주 |

**함의: 맞혀야 하는 것은 "통화에 언급된 증상"이 아니라 "접수요원이 절차에 따라 붙인 범주"다.**
LLM에게 "어떤 증상이 나오나?"라고 물으면 체계적으로 틀린다.

### 3.2 M3의 성능 상한은 라벨에 걸려 있다

- 오심 라벨 3,341건 중 **2,942건(88%)에 관련 표현이 텍스트에 전혀 없다**
- 그 2,942건의 93.3%는 다른 증상 라벨을 함께 갖는다 (어지러움 48.3%, 구토 40.2%)
- **키워드 정밀도가 F1 순위를 그대로 예측한다**: 복통 90.6%→0.768, 구토 70.7%→0.487,
  두통 62.8%→0.438, 오심 52.8%→0.300
- **다른 8개 라벨만으로 오심을 예측하면 F1 0.368**로, 텍스트 모델(0.300)보다 높다

사용자가 별도로 Qwen·RoBERTa·QLoRA를 시도했으나 **TF-IDF 대비 6% 개선에 그쳤다.**
이는 모델 실패가 아니라 위 상한의 결과다. **현실적 목표는 0.65 근처이고 0.75는 어렵다.**

### 3.3 M2의 오류 주범은 중첩이 아니라 길이다

발화의 37.1%가 다른 화자와 시간적으로 겹치지만, **길이가 진짜 원인**이다.

| 발화 길이 | 오분류율 | 중첩 비율 |
|---|---|---|
| <0.5s | 12.3% | 22.7% |
| 0.5–1s | 16.6% | 30.6% |
| 3–5s | 5.3% | 46.8% |
| >5s | **3.2%** | **48.3%** |

**긴 발화일수록 중첩이 많은데 오류는 적다.** 둘은 독립적인 요인이다.
처방은 중첩 제거가 아니라 **짧은 구간에 앞뒤 문맥을 붙이는 것**.

### 3.4 학습곡선 진단 — 셋 다 "더 돌려서는 안 오른다"

| 미션 | 관측 | 진단 | 처방 |
|---|---|---|---|
| M1 | val이 2,764통에서 포화, 격차 0.009 | 과소적합 | 데이터·반복 무의미. **표현을 바꿔야** |
| M2 | val 448회에서 정점, 격차 0.042 | 적정 수렴 | 이 특징의 한계 |
| M3 | 격차 0.35, val 평평 | 암기형 과적합 | 규제 아님. 라벨 상한 |

---

## 4. 검증된 것 / 반증된 것

**효과가 확인된 것**
- M2 통화별 평균차감: 0.8638 → 0.8799 (**+1.6pp**), 이득이 중첩 구간에 집중(+2.3pp)
- M1의 F0 계열이 지배적: F0 4개만으로 0.855 (전체 143개는 0.9434)

**시도했으나 효과 없던 것 — 반복하지 말 것**
- **M1 분류기 교체**: 12종 시도. 최고 SVM RBF 0.9475 vs 현재 LogReg 0.9434.
  **McNemar p=0.21로 우연과 구분 불가.** 커널·MLP는 소폭 상승, 트리 계열은 하락.
- **M3 클래스별 임계값 튜닝**: +0.0001에 그침. `class_weight='balanced'`가 이미 같은 일을 함.
  두 기법은 더해지지 않는다.
- **M2 비중첩 구간만 학습**: 0.8799 → 0.8726 (**-0.7pp**). 중첩 발화는 버릴 노이즈가 아니라
  학습에 필요한 사례다. 추론 시에도 37%가 중첩으로 들어온다.
- **M3 대형 모델**: 사용자 실험에서 TF-IDF 대비 6%.

**해석이 틀렸던 것 — 인용하지 말 것**
- ~~"f0_std 가중치가 커서 억양 변동이 성별 신호"~~ → **다중공선성 때문**.
  `f0_std`와 `logf0_std`의 상관이 0.952이고, 단독으로 쓰면 부호가 뒤집힌다(+1.55 → -1.29).
  개별 가중치의 부호를 해석하면 안 된다.
- ~~"짧은 발화와 중첩은 같은 원인"~~ → 3.3 참조.

---

## 5. 데이터

```
대학부 데이터/            33.7 GB, git 제외
├─ Training/{1.원천데이터/TS_서울_구급, 2.라벨링데이터/TL_서울_구급}    29,200통
└─ Validation/{1.원천데이터/VS_서울_구급, 2.라벨링데이터/VL_서울_구급}   3,640통
```

- 음성: **8 kHz · mono · 16-bit** 전화망 협대역 (300–3400 Hz만 전달됨)
- 통화 길이 중앙값 65.3초, 통화당 발화 30.3개(중앙값 28)
- 발화 길이 중앙값 1.43초 (p5 0.43 / p95 5.42)
- 전체 대화 중앙값 397자, p95 842자, 최대 1,292자
- **라벨이 0개인 통화는 하나도 없다** (평균 1.43개, 최대 5개) — 추론 시 최소 1개는 내야 한다
- 성별 F 53.8% / M 46.2%, 화자 신고자 52.0% / 대원 48.0% — 거의 균형

M3 클래스 빈도(train+val 32,840): 복통 7,609 · 어지러움 7,108 · 전신쇠약 5,961 · 고열 5,864 ·
구토 5,000 · 호흡곤란 4,419 · 열상 3,928 · 오심 3,770 · 두통 3,279

---

## 6. 저장소

**GitHub**: https://github.com/pnu-final-boss/2026-data-creator-camp
**작업 브랜치**: `claude/cpu-baselines` (커밋 `c685c1f`, `28e3b18`)

> ⚠️ 이 브랜치는 `main`과 **공통 조상이 없는 독립 히스토리**다.
> `main`에는 다른 구현(`codex/*` 브랜치들)이 이미 머지돼 있고 구조가 충돌한다.
> - `main`: `src/preprocess.py` (파일), `src/eda.py`, `inference.py`
> - 이 브랜치: `src/preprocess/` (디렉터리), `src/viz/` 6개 모듈
> - `src/preprocess.py` vs `src/preprocess/`는 **git에서 공존 불가**. 통합 시 결정 필요.

### 구조

```
src/common/      paths.py  labels.py  audio.py  metrics.py
src/preprocess/  extract_audio.py  pack_for_colab.py  make_colab_nb.py
src/m1_gender/   train.py
src/m2_speaker/  train.py  ablate.py
src/m3_symptom/  train.py  ablate.py
src/viz/         plots.py  summary.py  make_report.py
                 inspect_errors.py  learning_curve.py  label_corr.py
notebooks/       colab_m3_klue.ipynb  colab_m1m2_wav2vec2.ipynb
```

`src/common/audio.py`는 **torch·librosa 없이** `wave`+numpy+scipy로 log-mel / MFCC /
자기상관 F0를 직접 구현한다. GPU 없는 환경에서 베이스라인을 돌리기 위한 선택이다.

### git 제외 항목

`대학부 데이터/`, `data_orgin/`, `cache/`, `ckpt/`, `outputs/`, `figs/`, `dcc_annotation/`,
`docs/`, `test.ipynb`

`docs/`는 42쪽 학습자료 PDF/TeX가 있으나 **통화 원문이 인용돼 있어 제외**했다.
인용을 제거하면 포함 가능.

---

## 7. 실행

```bash
pip install -r requirements.txt

python -m src.common.labels                                   # 라벨 캐시
python -m src.preprocess.extract_audio --split val            # 오디오 특징
python -m src.preprocess.extract_audio --split train --n 6000

python -m src.m1_gender.train        # 0.9434
python -m src.m2_speaker.train       # 0.8799
python -m src.m3_symptom.train       # 0.5848

python -m src.viz.plots              # 혼동행렬 → figs/
python -m src.viz.learning_curve     # 학습곡선 (--only m3 로 부분 재계산)
python -m src.viz.label_corr         # 라벨 상관 φ/χ²/lift
python -m src.viz.inspect_errors m3 --cls 오심 --n 5   # 혼동행렬 → 원본 통화
python -m src.viz.summary            # outputs/results_summary.json
```

오디오 특징 추출이 가장 오래 걸린다(통화당 약 2.5초, 12코어 병렬).
한 번 뽑으면 `cache/audio_*.npz`에 남아 이후 학습은 수 분.

### Colab

```bash
python -m src.preprocess.pack_for_colab --what src    # src.zip 50 KB
python -m src.preprocess.pack_for_colab --what m3     # m3_text.json.gz 14 MB
python -m src.preprocess.pack_for_colab --what audio  # m1/m2 npz 2.5 GB
```

`cache/colab/`의 파일을 Google Drive `MyDrive/dcc/`에 올린 뒤 노트북 실행.
**전사 텍스트에 개인정보가 있으므로 Drive 경유만 사용하고 GitHub에는 올리지 않는다.**

노트북은 `src.zip`을 풀어 `src.common.metrics`를 import한다 —
GPU 결과와 CPU 기준선이 **같은 코드로 채점**되게 하기 위함이다. 노트북에 지표를 재구현하지 말 것.

---

## 8. 환경 함정 — 시간 낭비 방지

| 항목 | 주의 |
|---|---|
| **로컬 GPU** | 없음 (Intel Iris Xe). torch 미설치. 학습은 Colab으로. |
| **로컬 메모리** | 15.7 GB 중 가용 ~3 GB. **무거운 스크립트 동시 실행 금지** (여러 번 OOM killed). |
| `build_matrix` | float64 복사본을 여러 개 만듦. 대용량은 float32로 직접 할당할 것. |
| **Colab T4** | sm_75(Turing). `torch.cuda.is_bf16_supported()`가 **True를 반환하지만 에뮬레이션**이다. **fp16을 쓸 것.** FlashAttention-2 미지원. |
| **Colab transformers** | 5.16.1 (메이저 버전). 4.x 기준 코드는 API 확인 필요. |
| **TPU (v5e/v6e)** | `bitsandbytes`가 CUDA 전용이라 **QLoRA 불가**. torch_xla 필요, shape 변경 시 재컴파일. 이 과제에는 비권장. |
| 한글 폰트 | 로컬 `Malgun Gothic`, Colab은 `apt install fonts-nanum` 후 등록. 없으면 축이 □로 깨짐. |
| XeLaTeX | `\XeTeXlinebreaklocale "ko"` 를 **켜면 안 됨**. 한글 단어 사이 공백이 사라진다. |
| bash grep | 한글 출력에 `-a` 필요 (바이너리로 오판). Windows 콘솔 인코딩 때문에 mojibake 발생. |
| `SGDClassifier` | `partial_fit`은 `class_weight='balanced'`를 받지 않음. `compute_class_weight`로 미리 계산해 dict로 전달. |
| IDE↔Colab 브리지 | `mcp__ide__executeCode`가 세 번 연속 취소됨. 코드를 사용자에게 주고 실행 결과를 받는 방식이 확실. |

---

## 9. 산출물 링크

| | |
|---|---|
| **선행연구 조사 노트** | https://claude.ai/code/artifact/9236282d-1ddc-4ff5-8702-251de2a292f4 |
| **베이스라인 실측 결과** | https://claude.ai/code/artifact/cdacf26e-aeb4-492b-9da4-017e4189c4e5 |
| 소방청 상담 매뉴얼 | Wikimedia Commons 공개 PDF (63쪽, 주증상 33개 범주) |
| AI-Hub 원본 데이터 | https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71768 |

AI-Hub 페이지에 **공식 베이스라인**이 공개돼 있다: Kc-ELECTRA(텍스트, Acc 89.5),
Kc-ELECTRA + AST(MFCC) 멀티모달(91.1), 신고유형 16종 분류(F1 ≥90), Conformer STT(CER <10).

### 주요 선행연구

- **M1**: griko/voice-gender-classification (ECAPA-TDNN + SVM), Burkhardt et al. arXiv:2306.16962,
  Sivaraman & Khoury Odyssey'20 (협대역 대역확장)
- **M2**: **Ozan, arXiv:2106.02422** — 콜센터 상담원/고객 구간 CRNN 분류. 이 미션과 동형.
  터키어 학습 모델이 독일어·영어에서도 동작 → 언어가 아닌 채널 특성 학습의 증거
- **M3**: 권수정 외(2020) 정보처리학회 9(10):317-322 (119 신고 전사문 TF-IDF+SVM),
  Ridnik et al. ICCV'21 (ASL), Huang et al. EMNLP'21 (MLTC long-tail)

---

## 10. 다음에 할 일 (기대효과 순)

| 순위 | 작업 | GPU | 기대 |
|---|---|---|---|
| 1 | **M3 라벨 0개 방지 후처리** — 데이터에 0개 통화가 없음 | 불필요 | 확실한 소폭 이득 |
| 2 | **M3 범주 배타성 후처리** — 열상↔질환 lift 0.00~0.33 | 불필요 | 소폭 |
| 3 | **M1/M2 mel 하이퍼파라미터 탐색** — `n_mels`, `n_fft`, `hop` 격자 | T4 | 미지수 |
| 4 | M3 라벨 상관 공동 모델링 (9개 독립 → 공유 표현) | 소 | 중간 |
| 5 | M2 Whisper 전사 + 음성·텍스트 융합 | L4 | 중간 |
| 6 | M3 프로토콜 주입 프롬프팅 | 중 | 미지수 |
| 7 | ~~더 큰 모델~~ | — | **거의 없음 (검증됨)** |

**1·2번은 GPU 없이 지금 바로 가능하다.**

미완 항목: `inference.py` 미작성 (제출 필수), M2 통화 내 2-means 구조 후처리 실험이
메모리 부족으로 세 번 중단됨.

---

## 11. 대회 서사 관점

점수만 보면 M3가 약해 보이지만, **왜 안 오르는지를 규명한 것**이 이 프로젝트의 강점이다.
근거가 모두 갖춰져 있다 — 키워드 통계, 라벨 상관 φ/χ²/lift, 학습곡선,
그리고 결정적으로 **소방청 공개 프로토콜 문서**.

"모델을 키웠는데 안 올랐다"가 아니라 **"라벨 생성 절차를 역추적해 상한의 위치를 규명했다"**는
서사가 가능하다. 출제 PDF가 외부 데이터 사용을 허용하므로 프로토콜 문서 인용도 문제없다.
