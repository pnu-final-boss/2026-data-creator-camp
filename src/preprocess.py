"""Build mission-safe tabular manifests without modifying the source data."""

from __future__ import annotations

import argparse
import json
import random
import wave
from pathlib import Path

import pandas as pd


TARGET_SYMPTOMS = {
    "고열",
    "구토",
    "두통",
    "복통",
    "어지러움",
    "열상",
    "오심",
    "전신쇠약",
    "호흡곤란",
}


def audio_metadata(path: Path) -> dict[str, int | float]:
    """Read PCM WAV metadata without decoding the full recording."""
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        return {
            "sample_rate": sample_rate,
            "channels": wav.getnchannels(),
            "sample_width_bytes": wav.getsampwidth(),
            "frames": frames,
            "duration_seconds": frames / sample_rate if sample_rate else 0.0,
        }


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def build_manifests(
    data_dir: Path,
    output_dir: Path,
    max_files_per_split: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_rows: list[dict] = []
    mission1_rows: list[dict] = []
    mission2_rows: list[dict] = []
    mission3_rows: list[dict] = []
    issue_rows: list[dict] = []

    for split_dir in (data_dir / "Training", data_dir / "Validation"):
        if not split_dir.exists():
            continue

        split = split_dir.name.lower()
        all_wavs = {path.stem: path for path in split_dir.rglob("*.wav")}
        all_labels = {path.stem: path for path in split_dir.rglob("*.json")}
        matched_ids = sorted(set(all_wavs) & set(all_labels))
        if max_files_per_split > 0 and len(matched_ids) > max_files_per_split:
            matched_ids = sorted(
                random.Random(f"{seed}:{split}").sample(
                    matched_ids, max_files_per_split
                )
            )
        wavs = {file_id: all_wavs[file_id] for file_id in matched_ids}
        labels = {file_id: all_labels[file_id] for file_id in matched_ids}

        for file_id, wav_path in sorted(wavs.items()):
            row = {
                "split": split,
                "file_id": file_id,
                "audio_path": str(wav_path.resolve()),
                "label_path": str(labels[file_id].resolve()) if file_id in labels else "",
                "has_label": file_id in labels,
            }
            try:
                row.update(audio_metadata(wav_path))
            except (wave.Error, EOFError) as error:
                row.update(
                    sample_rate=None,
                    channels=None,
                    sample_width_bytes=None,
                    frames=None,
                    duration_seconds=None,
                )
                issue_rows.append(
                    {"split": split, "file_id": file_id, "issue": f"invalid_wav: {error}"}
                )
            audio_rows.append(row)

        for file_id, label_path in sorted(labels.items()):
            try:
                label = load_json(label_path)
            except (json.JSONDecodeError, OSError) as error:
                issue_rows.append(
                    {"split": split, "file_id": file_id, "issue": f"invalid_json: {error}"}
                )
                continue

            audio_path = str(wavs[file_id].resolve())
            gender = label.get("gender")
            utterances = label.get("utterances") or []

            for utterance_index, utterance in enumerate(utterances):
                common = {
                    "split": split,
                    "file_id": file_id,
                    "audio_path": audio_path,
                    "utterance_index": utterance_index,
                    "start_at_ms": utterance.get("startAt"),
                    "end_at_ms": utterance.get("endAt"),
                    "speaker": utterance.get("speaker"),
                }
                mission1_rows.append({**common, "gender": gender})
                mission2_rows.append(common)

            symptoms = [
                symptom
                for symptom in (label.get("symptom") or [])
                if symptom in TARGET_SYMPTOMS
            ]
            mission3_rows.append(
                {
                    "split": split,
                    "file_id": file_id,
                    "label_file_name": label_path.name,
                    "text": " ".join(
                        str(item.get("text", "")).strip()
                        for item in utterances
                        if str(item.get("text", "")).strip()
                    ),
                    "symptom": json.dumps(symptoms, ensure_ascii=False),
                }
            )

        for file_id in sorted(set(all_labels) - set(all_wavs)):
            issue_rows.append(
                {"split": split, "file_id": file_id, "issue": "missing_audio"}
            )
        for file_id in sorted(set(all_wavs) - set(all_labels)):
            issue_rows.append(
                {"split": split, "file_id": file_id, "issue": "missing_label"}
            )

    outputs = {
        "audio_manifest.csv": audio_rows,
        "mission1_segments.csv": mission1_rows,
        "mission2_segments.csv": mission2_rows,
        "mission3_calls.csv": mission3_rows,
        "data_issues.csv": issue_rows,
    }
    for filename, rows in outputs.items():
        pd.DataFrame(rows).to_csv(output_dir / filename, index=False)
        print(f"wrote {output_dir / filename}: {len(rows):,} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument(
        "--max_files_per_split",
        type=int,
        default=100,
        help="Matched calls sampled per split; use 0 to process all matched calls.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_manifests(
        args.data_dir,
        args.output_dir,
        args.max_files_per_split,
        args.seed,
    )
