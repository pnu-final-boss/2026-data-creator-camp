"""One-command Mission 1 inference using WAV and allowed JSON interval fields.

Run:
  python inference.py --audio_dir WAV_DIR --label_dir JSON_DIR \
      --ckpt_path best.pt --output ./outputs/mission1.csv
"""

import argparse
import contextlib
import csv
import json
import math
import wave
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from model import load_checkpoint


SAMPLE_RATE = 16000


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def caller_intervals(utterances, duration_ms):
    """Keep speaker 1 and subtract any overlapping speaker 0 intervals (ms)."""
    caller, other = [], []
    for utterance in utterances:
        start = float(utterance["startAt"])
        end = float(utterance["endAt"])
        speaker = int(utterance["speaker"])
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Invalid utterance timestamps")
        if speaker not in (0, 1):
            raise ValueError(f"Unexpected speaker: {speaker}")
        interval = (max(0.0, start), min(duration_ms, end))
        (caller if speaker == 1 else other).append(interval)
    clean = merge_intervals(caller)
    for left, right in merge_intervals(other):
        next_clean = []
        for start, end in clean:
            if right <= start or left >= end:
                next_clean.append((start, end))
            else:
                if start < left:
                    next_clean.append((start, left))
                if right < end:
                    next_clean.append((right, end))
        clean = next_clean
    return clean


def read_allowed_utterances(json_path):
    with json_path.open(encoding="utf-8-sig") as handle:
        document = json.load(handle)
    # No gender or other annotation field is accessed, including when present.
    return [
        {key: item[key] for key in ("startAt", "endAt", "speaker")}
        for item in document["utterances"]
    ]


def extract_caller(wav_path, utterances):
    with wave.open(str(wav_path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM: {wav_path}")
        rate, channels = wav.getframerate(), wav.getnchannels()
        intervals = caller_intervals(
            utterances, 1000 * wav.getnframes() / rate
        )
        chunks = []
        for start, end in intervals:
            first, last = int(start * rate / 1000), int(end * rate / 1000)
            if last <= first:
                continue
            wav.setpos(first)
            audio = np.frombuffer(
                wav.readframes(last - first), dtype="<i2"
            ).reshape(-1, channels)
            chunks.append(audio.astype(np.float32).mean(1) / 32768.0)
    audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    if len(audio) and rate != SAMPLE_RATE:
        common = math.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(
            audio, SAMPLE_RATE // common, rate // common
        ).astype(np.float32)
    return audio


def window_starts(length, window, max_windows):
    if window <= 0 or max_windows < 1:
        raise ValueError("Window sizes must be positive")
    if length <= window:
        return [0]
    count = min(max_windows, int(np.ceil(length / window)))
    return np.unique(np.linspace(0, length - window, count).astype(int)).tolist()


def iter_windows(files, label_dir, crop_seconds, eval_windows):
    window = int(crop_seconds * SAMPLE_RATE)
    for row_index, wav_path in enumerate(files):
        json_path = label_dir / f"{wav_path.stem}.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing JSON for {wav_path.name}: {json_path}")
        utterances = read_allowed_utterances(json_path)
        audio = extract_caller(wav_path, utterances)
        usable = len(audio) >= SAMPLE_RATE // 4 and float(np.std(audio)) > 1e-5
        if not usable:
            continue
        for start in window_starts(len(audio), window, eval_windows):
            clip = np.array(audio[start:start + window], dtype=np.float32, copy=True)
            clip = (clip - clip.mean()) / np.sqrt(clip.var() + 1e-7)
            yield row_index, clip


def predict(files, label_dir, model, checkpoint, device):
    config = checkpoint["train_config"]
    batch_size = int(config["eval_batch_size"])
    if batch_size < 1:
        raise ValueError("Invalid checkpoint batch size")
    sums = np.zeros(len(files), dtype=float)
    counts = np.zeros(len(files), dtype=int)
    batch = []

    def flush():
        length = max(len(clip) for _, clip in batch)
        values = torch.zeros(len(batch), length)
        mask = torch.zeros(len(batch), length, dtype=torch.long)
        for offset, (_, clip) in enumerate(batch):
            values[offset, :len(clip)] = torch.from_numpy(clip)
            mask[offset, :len(clip)] = 1
        with torch.inference_mode(), amp_context(device):
            logits = model(values.to(device), mask.to(device))
        probs = logits.float().softmax(-1)[:, 1].cpu().numpy()
        for (row_index, _), probability in zip(batch, probs):
            sums[row_index] += float(probability)
            counts[row_index] += 1
        batch.clear()

    for item in iter_windows(
        files, label_dir, float(config["crop_seconds"]), int(config["eval_windows"])
    ):
        batch.append(item)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    return np.divide(
        sums,
        counts,
        out=np.full(len(files), float(checkpoint["prior"]), dtype=float),
        where=counts > 0,
    )


def main():
    parser = argparse.ArgumentParser(description="Mission 1 gender inference")
    parser.add_argument("--audio_dir", required=True, type=Path)
    parser.add_argument("--label_dir", required=True, type=Path)
    parser.add_argument("--ckpt_path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.audio_dir.is_dir():
        raise NotADirectoryError(args.audio_dir)
    if not args.label_dir.is_dir():
        raise NotADirectoryError(args.label_dir)
    files = sorted(args.audio_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files in {args.audio_dir}")
    for wav_path in files:
        if not (args.label_dir / f"{wav_path.stem}.json").is_file():
            raise FileNotFoundError(f"Missing JSON for {wav_path.name}")

    device = choose_device()
    model, checkpoint = load_checkpoint(args.ckpt_path, device)
    probabilities = predict(files, args.label_dir, model, checkpoint, device)
    threshold = float(checkpoint["threshold"])
    inverse_labels = {value: key for key, value in checkpoint["label_map"].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audio file name", "gender"])
        for wav_path, probability in zip(files, probabilities):
            writer.writerow([wav_path.name, inverse_labels[int(probability >= threshold)]])
    print(f"Saved {len(files)} predictions to {args.output}")


if __name__ == "__main__":
    main()
