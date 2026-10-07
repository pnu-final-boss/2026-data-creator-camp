# Mission 3 선택 모델: 전사 텍스트 RoBERTa 50:50 앙상블

현재 선택된 모델의 두 신경망 가중치·tokenizer·고정 임계값과 독립 추론 코드를 제공합니다. 기존 공식 Validation 3,640통에서 저장된 **Macro-F1은 64.7122%**입니다. 후속 문맥 후보의 64.9390%는 Calibration의 모델 교체 조건을 통과하지 못해 선택하지 않았습니다.

학습된 신경망 두 개의 tensor 이름·shape·dtype·값과 tokenizer 바이트를 원 저장본과 정확히 대조했습니다. 공개 `model.pt`는 신경망만 추출한 파일이므로 원 체크포인트와 파일 해시가 다릅니다. 재학습·새 임계값 적합·새 모델 선택은 없습니다. 원래 묶음의 비신경망 모델, 개별 예측, 분할 ID, 실제 전사 앵커와 로컬 원천 경로는 포함하지 않습니다.

## 모델 받기와 확인

가중치는 Git LFS로 저장합니다. 저장소를 clone한 뒤 실행하세요.

```bash
git checkout 도윤
git lfs install
git lfs pull --include="submission/mission3_roberta_ensemble/model.pt"
cd submission/mission3_roberta_ensemble
uv run --no-project --python 3.10.21 python -B inference.py --ckpt_path model.pt --verify-only
```

`--verify-only`는 PyTorch 설치·GPU·입력 데이터 없이 파일 크기와 SHA256을 확인합니다. GitHub의 코드 ZIP 다운로드에는 LFS 포인터가 들어갈 수 있으므로 실제 모델을 받았는지 확인해야 합니다.

## 추론

원 평가 환경은 Python 3.10.21, torch 2.6.0, transformers 5.17.0, A100 CUDA입니다. CPU fallback은 없습니다. 모델과 tokenizer는 로컬 파일에서 복원하며 Hugging Face 다운로드와 `trust_remote_code`를 사용하지 않습니다.

```bash
uv venv --python 3.10.21 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# nvidia-smi로 확인한 비어 있는 GPU UUID 하나를 설정합니다.
export CUDA_VISIBLE_DEVICES="GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
uv run --no-project --python .venv/bin/python python -B inference.py \
  --label_dir /path/to/annotation_json \
  --ckpt_path model.pt \
  --output /path/to/new_outputs/mission3_internal.csv
```

학교 서버에서는 실행 직전 **메모리 32MiB 이하·compute process 없는 GPU가 최소 두 장**이어야 합니다. 지정 GPU 한 장만 사용하고 매 forward 직전에 다른 GPU 최소 한 장이 비었는지 확인합니다. 조건이 깨지면 종료하며 다른 사용자의 프로세스를 중단하거나 GPU를 reset하지 않습니다.

JSON의 `utterances[].text`만 발화 순서대로 합쳐 입력합니다. 실제 화자, 성별, 원 증상 정답, 시점·중증도 등 annotation 필드는 특징으로 쓰지 않습니다. `--audio_dir`는 호환용 인자이며 음성을 읽지 않습니다. 저장된 설정은 512토큰·stride 128·조각별 최대 logit 집계, chunk batch 16, padding 배수 8, FP32 가중치/BF16 autocast/SDPA입니다. 두 member의 sigmoid 확률을 NumPy float64로 50:50 평균하고 Calibration 2,920통에서 고정한 라벨별 임계값 이상을 양성으로 판정합니다.

라벨 순서는 **고열·구토·두통·복통·어지러움·열상·오심·전신쇠약·호흡곤란**입니다. 임계값의 정확한 값은 `bundle_config.json`에 있습니다.

## 검증과 남은 한계

- `export_receipt.json`: CPU에서 원/공개 신경망 모든 tensor와 tokenizer를 대조한 결과. 모델 forward와 GPU 사용은 0입니다.
- `load_check.json`: 원 평가 환경에서 두 모델 각각 201개 state key를 CPU FP32 모델 구조에 strict 로딩한 결과. config 문자열도 별도로 검토했습니다.
- `SHA256.json`: 공개 파일 전체의 실제 바이트 해시·크기. LFS 포인터의 해시가 아니라 다운로드한 모델 바이트의 해시입니다.
- `test_inference.py`: 입력 필드 불변성·중복 ID 거부·확률 평균·임계값 경계·조각 집계·CSV·파일 변조·GPU 여유 정책의 합성 테스트.

원 backend는 10월 3일 A100에서 16개 고정 개발 앵커를 네 번 불러 총 64개 예측을 실행했고 저장 확률과 차이 0을 확인했습니다. 이번 공개 CLI의 전체 공식 Validation을 다시 실행한 결과는 아닙니다. 학습 가중치 동일성, 원 구현의 과거 검사, 공개 CLI의 합성 검증을 구분합니다.

원 Calibration manifest에는 NumPy와 tokenizers 버전이 기록되지 않았습니다. 따라서 새 환경의 bitwise 재현을 보장하지 않습니다. 10월 4일 별도 tokenizers 0.23.2 환경의 합성 장문에서 overflow 조각이 일부 토큰을 포함하지 못하는 현상이 확인됐으며 실제 대회 통화의 누락 비율·점수 영향은 아직 측정하지 않았습니다. 이 배포에서 원 tokenization 방법을 바꾸거나 점수를 다시 선택하지 않았습니다.

출력 CSV는 `labelfile,symptom`, UTF-8 BOM, CRLF이며 symptom 셀은 한국어 라벨의 JSON 배열(빈 예측 `[]`)인 **내부 후보 형식**입니다. 주최 측의 정확한 헤더·라벨 코드·리스트 직렬화·빈 셀 규칙은 아직 확인되지 않았으므로 공식 제출 형식이 확정됐다고 주장하지 않습니다. 실행 receipt에도 이 상태를 남깁니다. 성능 수치는 정답으로 계산한 기존 사후 평가이며 Test 성능이나 임상 진단의 정확성을 보장하지 않습니다.

## 가중치 추출 재현

원 체크포인트와 동일 tokenizer를 가진 경우 아래 CPU 전용 도구로 신경망만 추출할 수 있습니다. 입력 원 파일의 SHA256을 고정하고, 공개 파일을 다시 읽어 모든 tensor 내용이 같은지 확인합니다.

```bash
uv run --no-project --python /path/to/existing/python python -B export_neural.py \
  --source-dir /path/to/original_assets --output-dir /path/to/fresh_public_assets
```

원 체크포인트는 공개하지 않습니다. 입력 데이터·추론 출력도 이 패키지에 넣지 마세요.
