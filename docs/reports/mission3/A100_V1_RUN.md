# Mission 3 A100 실행 결과: `a100_v1`

## 실행 개요

| 항목 | 값 |
|---|---|
| 모델 | `klue/roberta-base` 단일 공유 모델 |
| 실행 환경 | 서버 A100, 물리 GPU 1번 사용 |
| 학습 에포크 | 4 |
| 실행 이름 | `a100_v1` |

## 결과

| 평가 구간 | Calibrated Macro-F1 |
|---|---:|
| Select | 0.6357 |
| Holdout | **0.6511** |

학습 종료 시 기록된 loss는 `0.28297`이다. 학습 프로세스 종료, 저장 모델 생성, holdout 진단표 생성, 원본 JSON 1건에 대한 GPU 추론 CSV 생성까지 확인했다.

## 생성 산출물

서버 실행 경로 기준 주요 산출물은 다음과 같다.

```text
mission3_a100/runs/a100_v1/
├── metrics.json
├── train.log
├── model/
│   └── mission3.json
├── holdout_non_target_matrix.csv
└── inference_check.csv
```

- `metrics.json`: 학습 및 평가 지표
- `train.log`: 학습 진행 로그
- `model/`: 저장된 공유 모델
- `holdout_non_target_matrix.csv`: holdout 오류 진단표
- `inference_check.csv`: 원본 JSON 1건을 사용한 추론 형식 점검 결과

## 해석 및 주의사항

- 이 실행은 기존 내부 holdout 평가에서 공유 모델 기준선으로 사용한다.
- `0.6511`은 내부 holdout 점수이며, 공식 대회 점수나 실제 현장 적용 성능을 의미하지 않는다.
- 이후 학습률, 에포크, 임계값, 모델 구조 선택은 holdout에 맞춰 조정하지 않고 `train`·`select`·`calibration`에서 결정한다.
- 모델 가중치, 전사문, 개인 정보가 포함될 수 있는 예측 결과와 원본 로그는 저장소에 커밋하지 않는다.

## 다음 단계

1. `a100_v1` 모델로 공식 Validation 예측 CSV를 생성한다.
2. 주최 측 샘플 파일과 CSV 컬럼명 및 파일 ID 규칙을 비교한다.
3. 제출 파일의 ID 누락·중복과 증상 라벨 범위를 점검한다.
4. 추가 실험은 별도 실행 이름으로 수행하고, 확정 후 별도 테스트셋 또는 공식 평가로 확인한다.
