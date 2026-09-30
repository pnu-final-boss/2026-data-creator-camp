# Mission 1 제출 묶음 — WavLM v2

이 ZIP의 최상위 폴더에서 Python 3.10 환경으로 실행합니다.

```bash
python -m pip install -r requirements.txt
python inference.py --audio_dir "{wav folder}" --label_dir "{json folder}" \
  --ckpt_path best.pt --output ./outputs/mission1.csv
```

`{wav folder}`와 `{json folder}`에는 같은 파일명 줄기의 WAV와 JSON이 있어야 합니다.
한 번 실행하면 `audio file name,gender` 두 열의 CSV가 생성됩니다. 첫 열에는
`.wav` 확장자를 포함한 파일명, 둘째 열에는 `F` 또는 `M`을 씁니다.

## 파일

- `inference.py`: 지정된 단일 명령 실행과 CSV 작성
- `model.py`: 학습 노트북과 동일한 WavLM 모델 구조 및 가중치 로더
- `best.pt`: 내부 dev에서 선택된 epoch 4 가중치와 전처리 설정
- `m1_v2_linux_r5.ipynb`: 실행 로그가 포함된 학습 코드
- `requirements.txt`: 추론 필수 패키지 버전
- `result_summary.pptx`: 데이터·모델·결과 분석과 사회안전 시사점

## 규정 및 재현 범위

추론은 WAV와 JSON 발화별 `startAt`, `endAt`, `speaker`만 사용합니다.
`gender`가 없는 JSON으로도 실행되며 `text`, `id`, 기타 annotation은 읽지 않습니다.
학습에는 서울/구급 Training 29,200건만 사용했고, 이 중 5,840건을 내부 dev로
분리했습니다. 공식 Validation 3,640건은 학습과 임계값 선택에 쓰지 않았습니다.

기존 실험의 공식 Validation 정확도는 0.9873626373626374입니다. 이는
정답 라벨로 계산한 사후 평가 수치이며, Test 성능을 보장하는 값은 아닙니다.
제출 코드는 정답 없는 JSON 7건에서 실행해 기존 예측과 7건 모두 일치함을
확인했습니다. `best.pt`에는 모델 상태가 모두 있어 추론 시 네트워크에서
사전학습 가중치를 다시 받지 않습니다.

## 제출 양식 확인 사항

대회 PDF에는 Mission 1 CSV의 두 열 `[audio file name], [gender]`만 제시되고
헤더 존재 여부나 구체적인 헤더 문자열은 명시되지 않았습니다. 본 코드는
`audio file name,gender` 헤더를 작성합니다. 주최 측 샘플 CSV가 별도로
배포되어 있다면 제출 전에 해당 헤더와 파일명 표기를 대조해야 합니다.
