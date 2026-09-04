# 2026 데이터+AI 크리에이터 캠프 · 대학부 예선

119 응급신고 통화(AI-Hub 「119 지능형 신고접수 음성 인식 데이터」, 서울/구급)로 세 가지 미션을 푼다.

| 미션 | 입력 | 출력 | 지표 |
|---|---|---|---|
| M1 | 음성 (+ startAt/endAt/speaker) | 신고자 성별 M/F | Accuracy |
| M2 | startAt–endAt 발화 조각 **오디오만** | 신고자(1) / 119대원(0) | Accuracy |
| M3 | 전사 텍스트만 | 증상 9클래스 multi-label | Macro-F1 |

**규정상 금지** — M2 는 text 와 발화 순서 사용 불가. M3 는 전사 텍스트 외 라벨 필드
(`disasterMedium`, `urgencyLevel`, `triage`, `sentiment`) 사용 불가. Validation 은 학습 금지.

## 데이터 배치

저장소에 데이터는 포함되지 않는다 (33.7 GB + 개인정보). 아래 구조로 로컬에 두어야 한다.

```
대학부 데이터/
├─ Training/{1.원천데이터/TS_서울_구급, 2.라벨링데이터/TL_서울_구급}    29,200 통화
└─ Validation/{1.원천데이터/VS_서울_구급, 2.라벨링데이터/VL_서울_구급}   3,640 통화
```

원천 음성은 **8 kHz · mono · 16-bit** 전화망 협대역이다.

## 로컬 실행 (CPU)

```bash
pip install -r requirements.txt

python -m src.common.labels                                  # 라벨 캐시 생성
python -m src.preprocess.extract_audio --split val           # 오디오 특징 추출
python -m src.preprocess.extract_audio --split train --n 6000

python -m src.m3_symptom.train        # 텍스트 → 증상
python -m src.m1_gender.train         # 음성 → 성별
python -m src.m2_speaker.train        # 발화 조각 → 화자 역할

python -m src.viz.plots               # 혼동행렬 등 그림 → figs/
python -m src.viz.summary             # 결과 집계 → outputs/results_summary.json
```

## Colab GPU 학습

로컬에 NVIDIA GPU가 없으므로 파인튜닝은 Colab 에서 한다. Colab 런타임은 로컬 디스크를
보지 못하므로 **코드는 이 저장소에서 clone, 데이터는 Drive 에서 mount** 한다.

```bash
# 1) 로컬에서 업로드 번들 생성
python -m src.preprocess.pack_for_colab --what m3      # m3_text.json.gz  (14 MB)
python -m src.preprocess.pack_for_colab --what audio   # m1/m2 오디오 npz

# 2) cache/colab/ 의 파일을 Google Drive 의 MyDrive/dcc/ 에 업로드
# 3) notebooks/*.ipynb 를 Colab(또는 VS Code + google.colab 확장)에서 실행
```

번들은 **Drive 로만** 옮긴다. 전사 텍스트에 개인정보가 있어 공개 저장소에 올리면 안 된다.

## 구조

```
src/common/      paths.py  labels.py  audio.py  metrics.py
src/preprocess/  extract_audio.py  pack_for_colab.py  make_colab_nb.py
src/m1_gender/   train.py
src/m2_speaker/  train.py
src/m3_symptom/  train.py  ablate.py
src/viz/         plots.py  summary.py
notebooks/       colab_m3_klue.ipynb  colab_m1m2_wav2vec2.ipynb
```

`src/common/audio.py` 는 torch/librosa 없이 `wave`+numpy+scipy 로 log-mel / MFCC / F0(자기상관)를
계산한다. GPU 없는 환경에서 베이스라인을 돌리기 위한 선택이다.
