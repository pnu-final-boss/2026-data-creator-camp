# 2026 Data Creator Camp

부산대학교 대학부 예선 미션 1, 2, 3을 위한 공동 작업 저장소입니다.

## 현재 최고 점수

| 미션 | 점수 |
|---|---:|
| Mission 1 | 0.9434 |
| Mission 2 | 0.8850 |
| Mission 3 | 0.6400 |

## Mission 3: 통화 전사문 기반 증상 분류

Mission 3에서는 119 통화의 전사문을 읽고 다음 9개 증상이 있는지 예측합니다.

`고열`, `구토`, `두통`, `복통`, `어지러움`, `열상`, `오심`, `전신쇠약`, `호흡곤란`

한 통화에 여러 증상이 동시에 있을 수 있으므로, 하나만 고르는 분류가 아니라 **다중 레이블 분류** 문제입니다.

### 실험한 방식

| 방식 | 쉬운 설명 |
|---|---|
| 공유 모델 | 하나의 모델이 9개 증상을 함께 학습하고 각각의 유무를 판단 |
| 선택형 조합 | 증상별로 공유 모델과 전담 모델 중 개발 성능이 좋은 쪽을 선택 |
| 전문가 9개 전담 | 증상마다 별도 이진분류 모델을 두고, 9개 결과를 합침 |

공유 모델과 전문가 모델 모두 사전학습된 한국어 언어모델 `klue/roberta-base`를 전사문 분류에 맞게 추가 학습했다. 생성형 LLM이나 외부 LLM API는 사용하지 않았으며, 입력에는 전사문 텍스트만 사용했다.

### 내부 평가 결과

| 방식 | Macro-F1 |
|---|---:|
| **공유 모델** | **0.6365** |
| 선택형 조합 | 0.6341 |
| 전문가 9개 전담 | 0.6327 |

Macro-F1은 9개 증상별 F1 점수를 같은 비중으로 평균한 지표이며, 단순한 통화 정답률과는 다릅니다. 현재 설정에서는 공유 모델이 가장 높은 내부 점수를 기록했습니다. 전문가 모델은 일부 증상에서 소폭 개선됐지만, 전체 평균 성능 향상으로 이어지지는 않았습니다.

실험 설정, 증상별 성능, 에포크별 관찰, 오분류 분석은 [Mission 3 실험 정리](docs/reports/mission3/MISSION3_EXPERIMENT_SUMMARY.md)에서 확인할 수 있습니다. 최신 A100 공유 모델 실행 결과는 [A100 실행 결과](docs/reports/mission3/A100_V1_RUN.md)에 정리했습니다.

> Mission 3 점수는 제공된 Training 데이터에서 분리한 내부 holdout 평가 결과입니다. 공식 대회 점수나 실제 현장 적용 성능을 의미하지 않습니다.

## 저장소 구성

```text
.
├── src/                 # 전처리와 EDA 코드
├── notebooks/           # 실험 노트북
├── docs/reports/        # EDA 및 실험 보고서
├── checkpoints/         # 로컬 모델 체크포인트 경로
├── outputs/             # 예측 결과 경로
└── inference.py         # 대회 제출 형식의 추론 진입점
```

원본 데이터, 전사문, 개인 정보가 포함될 수 있는 산출물, 로컬 체크포인트는 저장소에 올리지 않습니다.

## 로컬 환경 준비

현재 PyTorch 환경과의 호환성을 위해 Python 3.12를 권장합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell에서는 가상환경을 다음과 같이 활성화합니다.

```powershell
.venv\Scripts\Activate.ps1
```

## 전처리와 EDA

원본 WAV·JSON 파일은 수정하지 않습니다. 아래 명령은 기본적으로 각 분할에서 매칭된 통화 100개를 표본으로 뽑아 전처리와 EDA를 수행합니다.

```bash
python -m src.preprocess --max_files_per_split 100 --seed 42
python -m src.eda
```

생성되는 주요 파일은 다음과 같습니다.

- `data/raw/`: 수정하지 않은 Training·Validation 원본 경로
- `data/processed/audio_manifest.csv`: 음성 파일 메타데이터와 매칭 상태
- `data/processed/mission1_segments.csv`: Mission 1용 구간 정보
- `data/processed/mission2_segments.csv`: Mission 2용 구간과 화자 정보
- `data/processed/mission3_calls.csv`: Mission 3용 전사문과 필터링된 증상 정보
- `data/processed/data_issues.csv`: 누락 또는 잘못된 파일 쌍
- `docs/reports/eda/`: EDA 보고서와 그래프

전체 데이터가 필요할 때만 아래처럼 `0`을 사용합니다.

```bash
python -m src.preprocess --max_files_per_split 0 --seed 42
```

## 추론과 제출 파일

대회 제출용 추론 명령은 주최 측이 제공한 인터페이스를 따릅니다.

```bash
python inference.py \
  --audio_dir {wav_folder} \
  --label_dir {json_folder} \
  --ckpt_path {checkpoint_file} \
  --output ./outputs/missionN.csv
```

제출 전에는 주최 측의 최신 샘플 제출 파일을 기준으로 CSV 컬럼명과 파일 ID 규칙을 반드시 확인합니다.
