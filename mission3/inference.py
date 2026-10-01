"""Mission 3 추론: 라벨링 JSON의 전사 텍스트 → 환자 증상 CSV.

실행 (출제문제 11쪽 규칙):
    python inference.py --audio_dir {wav folder} --label_dir {json folder} \
        --ckpt_path mission3_ensemble.pt --output ./outputs/mission3.csv

- Mission 3은 전사 텍스트만 입력으로 쓰므로 --audio_dir은 받기만 하고 사용하지 않는다.
- JSON에서는 utterances의 speaker와 text만 읽는다. symptom 등 다른 필드는 읽지 않는다.
- 출력 CSV: [label file name], [symptom]. symptom은 9개 클래스 중 예측된 증상의 목록이다.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from mission3_model import ensemble_probabilities, load_calls, to_symptom_lists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_dir", type=Path, required=True, help="Mission 3에서는 사용하지 않음")
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # 1) 입력: 라벨 폴더의 JSON 전사 (정답 symptom은 읽지 않음)
    calls = load_calls(args.label_dir, with_labels=False)
    print(f"loaded {len(calls):,} calls from {args.label_dir}")

    # 2) 모델: 6개 모델, 토크나이저, 임계값이 모두 담긴 번들 하나
    bundle = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if bundle.get("format") != "mission3_ensemble_v1" or bundle.get("thresholds") is None:
        raise ValueError(f"not a finished Mission 3 bundle: {args.ckpt_path}")

    # 3) 추론: 구성원 확률 평균 → 증상별 임계값 적용
    probabilities = ensemble_probabilities(bundle, calls, device, args.batch_size)
    symptoms = to_symptom_lists(probabilities, bundle["thresholds"])

    # 4) 출력: 출제문제 예시와 같은 문자열 목록 형식, 예) ['두통', '복통'] / 증상 없음은 []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label file name", "symptom"])
        for call, predicted in zip(calls, symptoms, strict=True):
            writer.writerow([call["file"], str(predicted)])
    print(f"wrote {args.output}: {len(calls):,} rows")


if __name__ == "__main__":
    main()
