# Mission 3 · 환자 증상 인식 제출물

119 신고 전사 텍스트에서 9개 증상을 멀티라벨로 예측한다. Validation macro F1 **0.6562** (기준표 0.64).

## 구성 (출제문제 12쪽 기준)

| 제출 항목 | 파일 |
|---|---|
| inference.py | `inference.py` |
| 학습 코드 (학습 로그 포함 .ipynb) | `mission3_train.ipynb`, 원본 학습 로그 `logs/` |
| 모델 가중치 | `mission3_ensemble.pt` (4.5 GB, 6개 모델 + 토크나이저 + 임계값) — 용량 때문에 저장소에는 없음, 아래 참고 |
| 모델 코드 | `mission3_model.py` |
| requirements.txt | `requirements.txt` |
| 결과 정리 PPT | 미포함 (팀 PPT에 통합 예정) |

`results/validation_metrics.json`은 아래 명령으로 Validation을 채점한 지표다. 예측 CSV는 원본 데이터 산출물이라 저장소에 올리지 않는다.

**가중치 파일**: GitHub 용량 한도(100 MB)를 넘어 `.gitignore`로 제외했다. 학습 서버의 `mission3/mission3_ensemble.pt`에 있고, `mission3_train.ipynb`를 실행하면 같은 파일이 다시 만들어진다(구성원 체크포인트 6개 필요). 제출할 때는 이 파일을 함께 낸다.

**팀 내 비교**: 기존 Mission 3 최고(klue/roberta-base 공유 모델, Training 내부 holdout macro F1 0.6365)보다 높다. 이 앙상블의 Training 내부 dev 점수는 0.6553, Validation 점수는 0.6562다.

## 실행

```bash
pip install -r requirements.txt
python inference.py --audio_dir {wav folder} --label_dir {json folder} \
    --ckpt_path mission3_ensemble.pt --output ./outputs/mission3.csv
```

- 인터넷 없이 동작한다. 토크나이저와 모델 설정이 체크포인트 안에 들어 있다.
- GPU가 있으면 GPU, 없으면 CPU로 실행한다. 3,640건 기준 A100에서 약 2분, CPU에서 1~2시간.
- `--audio_dir`은 형식상 받기만 한다. Mission 3 입력은 전사 텍스트뿐이다.

출력 형식은 `label file name, symptom`이며, symptom은 출제문제 9쪽 예시처럼 증상 문자열 목록이다.

```
label file name,symptom
651e464d69a4f266f0626837_20220101.json,['전신쇠약']
651e464d69a4f266f0626845_20220101.json,"['구토', '오심']"
```

## 모델

| 구성원 | 사전학습 모델 | 차이점 |
|---|---|---|
| s42 / s43 / s44 | klue/roberta-large | 초기화 seed |
| xlmr | xlm-roberta-large | 아키텍처 |
| pw05 | klue/roberta-large | 희소 증상 가중치 완화 (pos_weight^0.5) |
| asl | klue/roberta-large | Asymmetric Loss |

6개 모델의 확률을 평균한 뒤 증상별 임계값으로 판정한다. 임계값은 Training 내부 dev(10%, 2,920건)에서만 골랐다.

## 결과 (Validation 3,640건)

| 증상 | F1 | 증상 | F1 |
|---|---|---|---|
| 열상 | 0.882 | 구토 | 0.614 |
| 복통 | 0.822 | 전신쇠약 | 0.590 |
| 고열 | 0.694 | 두통 | 0.541 |
| 호흡곤란 | 0.694 | 오심 | 0.403 |
| 어지러움 | 0.667 | **macro** | **0.656** |

## 규정 준수

- Training(서울)만 학습에 사용했다. Validation은 최종 채점에만 썼다.
- 임계값과 앙상블 구성은 Training 내부 dev에서만 정했다.
- 전처리(전사 연결, 512 토큰 절단)는 라벨과 무관하게 모든 통화에 같다.
- 공개 사전학습 모델을 직접 파인튜닝했고 상용 API는 쓰지 않았다.

## 확인 필요

- 전사 앞에 붙인 `[대원]`/`[신고자]` 표시는 utterances의 speaker 값에서 만든다. 입력이 "신고자 및 119대원의 대화 텍스트"라 허용 범위로 판단했지만, "전사 텍스트 외 라벨링데이터 사용 불가" 조항을 좁게 해석하면 문제가 될 수 있다. 주최측 확인이 필요하다.
- `mission3_train.ipynb`는 제출 가중치를 만든 원래 학습 실행의 로그를 보여준다. 노트북 안에서 다시 학습하려면 `RETRAIN = True`로 실행한다 (모델당 A100 약 15분).
