# Mission 1 v2 · WavLM 성별 분류

학교 Linux 서버에서 실행한 WavLM 미세조정 실험이다. 실행 출력이 포함된 노트북은
[`m1_v2_linux_r5.ipynb`](m1_v2_linux_r5.ipynb), 공개 가능한 평가 결과는
[`../../outputs/m1_v2`](../../outputs/m1_v2)에 보관한다.

## 최신 결과 (2026-09-19)

| 평가 데이터 | Accuracy | Macro-F1 | ROC-AUC | 통화 수 |
|---|---:|---:|---:|---:|
| 내부 dev | 0.988356 | 0.988288 | 0.997495 | 5,840 |
| 공식 validation | **0.987363** | **0.987292** | **0.996465** | 3,640 |

- 선택 체크포인트: epoch 4
- 조기 종료: epoch 7 완료 후(3 epoch 연속 개선 없음)
- 공식 validation 혼동행렬: `[[1933, 26], [20, 1661]]` (`F=0`, `M=1`)
- 기존 MFCC/F0 + LogisticRegression Accuracy: 0.943407
- WavLM v2 개선폭: **+4.40%p**

## 결과 저장

노트북은 실행 결과를 `runs/m1_v2/<run_id>/`에 저장한다. 학습과 공식 validation 평가를
완료한 뒤 저장소 루트에서 아래 명령으로 공개 가능한 결과만 복사한다.

```bash
uv run python scripts/export_m1_v2_results.py \
  /home/a202355692/data/runs/m1_v2/<run_id> \
  --output outputs/m1_v2
```

이 명령은 metrics, predictions, 학습 이력과 시각화만 복사한다. `best.pt`, 원본 WAV,
JSON 라벨은 복사하지 않는다. 결과를 확인한 후 다음과 같이 커밋한다.

```bash
git add notebooks/m1_v2 outputs/m1_v2 requirements-m1.txt
git commit -m "feat: update Mission 1 WavLM results"
git push origin 도윤
```

100 MiB를 넘는 `best.pt`와 전체 portable bundle은 GitHub Release
[`m1-v2-20260919`](https://github.com/pnu-final-boss/2026-data-creator-camp/releases/tag/m1-v2-20260919)에 둔다.
원본 데이터와 개인정보가 포함된 파일은 GitHub에 올리지 않는다.

## 결과 구성

- `outputs/m1_v2/dev_metrics.json`: 내부 dev 전체 지표
- `outputs/m1_v2/validation/metrics.json`: 공식 validation 전체 지표
- `outputs/m1_v2/*_predictions.csv`: 통화별 예측값과 확률
- `outputs/m1_v2/figures/`: 혼동행렬, ROC/PR, 길이별 정확도, 오분류 표
- `outputs/m1_v2/history.jsonl`: epoch별 학습 이력
- `outputs/m1_v2/result_summary.png`: 핵심 결과 요약
- `requirements-m1.txt`: 실행 당시 패키지 버전

전체 결과 설명과 체크섬은 [`outputs/m1_v2/README.md`](../../outputs/m1_v2/README.md)를 참고한다.
