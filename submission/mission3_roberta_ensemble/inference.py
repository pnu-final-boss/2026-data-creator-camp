"""Private-data-free inference for the selected text-only Mission 3 ensemble.

The CSV is an INTERNAL candidate: organizer serialization/runtime confirmation
is still required. This CLI has not re-run the full Validation evaluation.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
from gpu_guard import GPUChecker

LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
FROZEN_THRESHOLDS = [0.27499999999999997, 0.3749999999999999, 0.3749999999999999,
                     0.5499999999999999, 0.42499999999999993, 0.6249999999999999,
                     0.15, 0.3749999999999999, 0.47499999999999987]
TOKENIZER_FILES = ["special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]
RUNTIME = {"python": "3.10.21", "torch": "2.6.0", "transformers": "5.17.0"}


def require(value, message):
    if not value:
        raise ValueError(message)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON contains duplicate keys.")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=no_duplicate_keys,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON is forbidden.")))


def safe_file(root, name):
    require(isinstance(name, str) and "\\" not in name and ":" not in name and not Path(name).is_absolute()
            and all(part not in ("", ".", "..") for part in name.split("/")), "Unsafe bundle-relative filename.")
    path = (root / name).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Missing file or file outside bundle.")
    return path


def validate_config(config):
    require(isinstance(config, dict) and config.get("labels") == LABELS
            and config.get("weights") == [0.5, 0.5] and config.get("thresholds") == FROZEN_THRESHOLDS,
            "Bundle label order, 50:50 weights, or exact frozen thresholds differ.")
    require(config.get("max_length") == 512 and config.get("stride") == 128
            and config.get("pooling") == "max_logits" and config.get("batch_chunks") == 16
            and config.get("calibration_fit_n") == 2920 and config.get("runtime_versions") == RUNTIME,
            "Bundle inference settings or frozen runtime differ.")
    return config


def verify_bundle(ckpt_path):
    """Verify local bytes without importing torch, reading input, or using CUDA."""
    checkpoint = Path(ckpt_path).resolve()
    require(checkpoint.name == "model.pt" and checkpoint.is_file(), "Expected an existing bundle model.pt.")
    with checkpoint.open("rb") as stream:
        prefix = stream.read(200)
    require(not prefix.startswith(b"version https://git-lfs.github.com/spec/v1"),
            "model.pt is a Git LFS pointer. Install Git LFS and run git lfs pull to fetch the model bytes.")
    root = checkpoint.parent
    manifest = read_json(root / "SHA256.json")
    files = manifest.get("files") if isinstance(manifest, dict) else None
    required = {"model.pt", "bundle_config.json", "inference.py", "gpu_guard.py", "requirements.txt"}
    required.update("tokenizer/" + name for name in TOKENIZER_FILES)
    require(isinstance(files, dict) and required <= files.keys(), "SHA256.json is missing required assets or code.")
    for name, entry in files.items():
        require(isinstance(entry, dict) and set(entry) == {"sha256", "bytes"}
                and isinstance(entry["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                and type(entry["bytes"]) is int and entry["bytes"] >= 0, "Invalid manifest hash/byte record.")
        path = safe_file(root, name)
        require(path.stat().st_size == entry["bytes"] and file_hash(path) == entry["sha256"],
                "Bundle file changed or is incomplete: " + name)
    guard_source = Path(sys.modules[GPUChecker.__module__].__file__)
    require(file_hash(__file__) == files["inference.py"]["sha256"]
            and file_hash(guard_source) == files["gpu_guard.py"]["sha256"],
            "Executing inference/guard code differs from the bundle manifest.")
    config = validate_config(read_json(root / "bundle_config.json"))
    return {"root": root, "checkpoint": checkpoint, "config": config, "files": files,
            "manifest_sha256": file_hash(root / "SHA256.json")}


def clean_text(text):
    return re.sub(r"\[(?:신고자|대원|119대원|\?)\]\s*", "\n", str(text)).strip()


def annotation_row(record, filename):
    require(isinstance(filename, str) and filename.endswith(".json") and
            not any(char in filename for char in ("/", "\\", "\n", "\r", "\0"))
            and Path(filename).name == filename and Path(filename).stem not in ("", ".", ".."),
            "Invalid annotation filename.")
    require(isinstance(record, dict) and isinstance(record.get("utterances"), list)
            and record["utterances"], "Annotation needs a nonempty utterances list.")
    utterances = []
    for utterance in record["utterances"]:
        require(isinstance(utterance, dict) and
                (utterance.get("text") is None or isinstance(utterance.get("text"), str)), "Invalid utterance text.")
        # Actual speaker, targets, gender, timestamps and other annotation fields
        # never enter model input. Preserve original per-utterance strip/join.
        utterances.append((utterance.get("text") or "").strip())
    text = clean_text("\n".join(utterances))
    require(bool(text), "Empty inference text.")
    return {"id": Path(filename).stem, "filename": filename, "text": text}


def load_inputs(label_dir):
    root = Path(label_dir).resolve()
    require(root.is_dir(), "Annotation directory does not exist.")
    paths, rows = {}, []
    for path in sorted(root.rglob("*.json")):
        resolved = path.resolve()
        require(resolved.is_relative_to(root), "Annotation symlink points outside the input directory.")
        paths[resolved] = file_hash(resolved)
        rows.append(annotation_row(read_json(resolved), path.name))
    require(rows, "No annotation JSON files found.")
    require(len({row["id"] for row in rows}) == len(rows) and
            len({row["filename"] for row in rows}) == len(rows), "Duplicate input ID or filename.")
    rows.sort(key=lambda row: (row["filename"], row["id"]))
    return rows, paths


def ensemble_probabilities(first, second):
    import numpy as np
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    require(a.ndim == 2 and a.shape == b.shape and a.shape[1] == 9
            and np.isfinite(a).all() and np.isfinite(b).all()
            and ((a >= 0) & (a <= 1)).all() and ((b >= 0) & (b <= 1)).all(), "Invalid member probability arrays.")
    return 0.5 * a + 0.5 * b


def decisions(probabilities, thresholds):
    import numpy as np
    p, t = np.asarray(probabilities, dtype=np.float64), np.asarray(thresholds, dtype=np.float64)
    require(p.ndim == 2 and p.shape[1] == 9 and t.shape == (9,) and np.isfinite(p).all()
            and ((p >= 0) & (p <= 1)).all() and np.isfinite(t).all(), "Invalid probability/threshold shapes.")
    return (p >= t).astype(int)


def pooled_logits(chunks, tokenizer, model, torch, guard, batch_chunks=16):
    require(chunks, "Tokenizer returned no chunks.")
    pooled = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for pos in range(0, len(chunks), batch_chunks):
            guard.check()
            batch = tokenizer.pad(chunks[pos:pos + batch_chunks], padding=True,
                                  pad_to_multiple_of=8, return_tensors="pt").to("cuda:0")
            logits = model(**batch).logits.float().amax(0)
            pooled = logits if pooled is None else torch.maximum(pooled, logits)
    return pooled


def validate_package(package, torch):
    require(isinstance(package, dict) and set(package) == {"kind", "labels", "members"}
            and package["kind"] == "mission3_neural_ensemble" and package["labels"] == LABELS
            and isinstance(package["members"], list) and len(package["members"]) == 2,
            "Expected two-member neural-only checkpoint schema.")
    keys = {"kind", "labels", "config", "state_dict", "max_length", "stride", "pooling"}
    for saved in package["members"]:
        require(isinstance(saved, dict) and set(saved) == keys and saved["kind"] == "roberta_multilabel"
                and saved["labels"] == LABELS and saved["max_length"] == 512 and saved["stride"] == 128
                and saved["pooling"] == "max_logits" and isinstance(saved["config"], dict)
                and saved["config"].get("model_type") == "roberta", "Invalid neural member metadata.")
        require(isinstance(saved["state_dict"], dict) and saved["state_dict"]
                and all(isinstance(key, str) and torch.is_tensor(value)
                        for key, value in saved["state_dict"].items()), "State dict must contain tensors only.")
    return package


def infer(rows, bundle, guard):
    # Check capacity immediately before importing/initializing the model runtime.
    guard.check(starting=True)
    require({"python": sys.version.split()[0], "torch": importlib.metadata.version("torch"),
             "transformers": importlib.metadata.version("transformers")} == RUNTIME,
            "Use the frozen Python 3.10.21 / torch 2.6.0 / transformers 5.17.0 runtime.")
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and "A100" in torch.cuda.get_device_name(0) and torch.cuda.is_bf16_supported(),
            "Exactly one visible A100 with BF16 is required; CPU fallback is disabled.")
    package = validate_package(torch.load(bundle["checkpoint"], map_location="cpu", weights_only=True), torch)
    tokenizer = AutoTokenizer.from_pretrained(bundle["root"] / "tokenizer", local_files_only=True,
                                              trust_remote_code=False)
    scores = []
    for saved in package["members"]:  # Immutable original member order 0 -> 1.
        guard.check()
        config = dict(saved["config"])
        config = AutoConfig.for_model(config.pop("model_type"), **config)
        model = AutoModelForSequenceClassification.from_config(config, attn_implementation="sdpa").float()
        try:
            model.load_state_dict(saved["state_dict"], strict=True)
            model.to("cuda:0").eval()
            require(all(parameter.dtype == torch.float32 and parameter.device.type == "cuda"
                        for parameter in model.parameters()), "Model must use CUDA FP32 weights.")
            member_scores = []
            for row in rows:
                encoded = tokenizer(row["text"], truncation=True, max_length=512, stride=128,
                                    return_overflowing_tokens=True)
                chunks = [{key: encoded[key][i] for key in tokenizer.model_input_names if key in encoded}
                          for i in range(len(encoded["input_ids"]))]
                logits = pooled_logits(chunks, tokenizer, model, torch, guard, bundle["config"]["batch_chunks"])
                member_scores.append(logits.sigmoid().cpu().tolist())
            scores.append(member_scores)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    guard.check()
    probabilities = ensemble_probabilities(scores[0], scores[1])
    return probabilities, decisions(probabilities, bundle["config"]["thresholds"])


def write_internal_csv(path, rows, predicted):
    require(len(rows) == len(predicted), "CSV row count differs from predictions.")
    for vector in predicted:
        require(len(vector) == 9 and all(value in (0, 1, False, True) for value in vector),
                "CSV needs nine binary decisions per call.")
    path = Path(path)
    require(not path.exists(), "Output already exists; choose a new output file.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(["labelfile", "symptom"])
        for row, vector in zip(rows, predicted):
            selected = [label for label, value in zip(LABELS, vector) if value]
            writer.writerow([row["filename"], json.dumps(selected, ensure_ascii=False)])


def run(ckpt_path, label_dir=None, output=None, *, verify_only=False, audio_dir=None):
    started = time.perf_counter()
    bundle = verify_bundle(ckpt_path)
    if verify_only:
        return {"status": "bundle_bytes_verified", "model_forward_calls": 0,
                "cuda_used": False, "manifest_sha256": bundle["manifest_sha256"]}
    require(label_dir and output, "--label_dir and --output are required for inference.")
    destination = Path(output).resolve()
    companion = destination.with_suffix(".receipt.json")
    input_root = Path(label_dir).resolve()
    require(not destination.exists() and not companion.exists() and
            not destination.is_relative_to(bundle["root"]) and not destination.is_relative_to(input_root),
            "Use fresh output files outside the model and input directories.")
    rows, input_hashes = load_inputs(label_dir)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    probabilities, predicted = infer(rows, bundle, GPUChecker())
    require(all(file_hash(path) == sha for path, sha in input_hashes.items()), "Input files changed during inference.")
    require(verify_bundle(ckpt_path)["manifest_sha256"] == bundle["manifest_sha256"], "Bundle changed during inference.")
    write_internal_csv(destination, rows, predicted)
    receipt = {"status": "completed_internal_csv_candidate", "n": len(rows), "labels": LABELS,
               "model_input_fields": ["text"], "audio_read": False,
               "annotation_features_used_as_input": False, "manifest_sha256": bundle["manifest_sha256"],
               "csv_sha256": file_hash(destination), "elapsed_seconds": time.perf_counter() - started,
               "official_csv_confirmed": False, "new_training_or_threshold_fits": 0,
               "new_full_validation_replay_or_bitwise_equivalence_claimed": False,
               "csv_contract": {"header": ["labelfile", "symptom"], "encoding": "utf-8-sig",
                                "symptom_cell": "JSON string array; [] for empty", "line_ending": "CRLF",
                                "status": "internal_candidate_not_organizer_confirmed"}}
    with companion.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label_dir")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output")
    parser.add_argument("--audio_dir", help="Accepted for command compatibility; never read.")
    parser.add_argument("--verify-only", action="store_true", help="Check bundle bytes with no input read or CUDA use.")
    args = parser.parse_args(argv)
    try:
        result = run(args.ckpt_path, args.label_dir, args.output, verify_only=args.verify_only, audio_dir=args.audio_dir)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
