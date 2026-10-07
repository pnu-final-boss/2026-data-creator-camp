"""Synthetic CPU checks with unittest mocks; no real data/model/GPU is read."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import inference as subject
import gpu_guard

UUID0 = "GPU-00000000-0000-0000-0000-000000000000"
UUID1 = "GPU-11111111-1111-1111-1111-111111111111"
UUID2 = "GPU-22222222-2222-2222-2222-222222222222"


def temporary_directory():
    # Keep Windows sandbox fixtures in this known writable project directory.
    # Verify the cleanup target before TemporaryDirectory can remove it.
    context = tempfile.TemporaryDirectory(prefix="m3_fixture_", dir=ROOT)
    if not Path(context.name).resolve().is_relative_to(ROOT.resolve()):
        raise AssertionError("temporary cleanup target escaped the project")
    return context


def config_fixture():
    return {"labels": subject.LABELS, "weights": [0.5, 0.5],
            "thresholds": subject.FROZEN_THRESHOLDS, "max_length": 512, "stride": 128,
            "pooling": "max_logits", "batch_chunks": 16, "calibration_fit_n": 2920,
            "runtime_versions": subject.RUNTIME}


def make_bundle(folder):
    (folder / "model.pt").write_bytes(b"synthetic-weights-not-loaded")
    (folder / "bundle_config.json").write_text(json.dumps(config_fixture(), ensure_ascii=False), encoding="utf-8")
    for name in ("inference.py", "gpu_guard.py", "requirements.txt"):
        shutil.copyfile(ROOT / name, folder / name)
    tokenizer = folder / "tokenizer"
    tokenizer.mkdir()
    for name in subject.TOKENIZER_FILES:
        (tokenizer / name).write_bytes(b"synthetic-tokenizer-not-loaded")
    files = {path.relative_to(folder).as_posix(): {"sha256": subject.file_hash(path), "bytes": path.stat().st_size}
             for path in folder.rglob("*") if path.is_file()}
    (folder / "SHA256.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return folder / "model.pt"


class FakeVector:
    def __init__(self, values):
        self.values = values


class FakeLogits:
    def __init__(self, rows):
        self.rows = rows

    def float(self):
        return self

    def amax(self, dimension):
        if dimension != 0:
            raise AssertionError("chunk pooling dimension must be zero")
        return FakeVector([max(row[j] for row in self.rows) for j in range(9)])


class FakeTorch:
    bfloat16 = "bf16"

    def __init__(self):
        self.autocast_options = []

    def inference_mode(self):
        return nullcontext()

    def autocast(self, device, dtype):
        self.autocast_options.append((device, dtype))
        return nullcontext()

    def maximum(self, a, b):
        return FakeVector([max(x, y) for x, y in zip(a.values, b.values)])


class FakeBatch(dict):
    def to(self, device):
        if device != "cuda:0":
            raise AssertionError("unexpected device")
        return self


class SyntheticInputTests(unittest.TestCase):
    def test_annotation_fields_invariant_and_original_strip_join(self):
        original = {"utterances": [{"text": "  [대원] 열 있나요?  ", "speaker": 0},
                                     {"text": "  아니요  ", "speaker": 1}], "symptom": ["고열"], "gender": "x"}
        altered = copy.deepcopy(original)
        altered.update(symptom=["복통"], gender="y", severity=9, recordId="synthetic")
        for utterance in altered["utterances"]:
            utterance.update(speaker=99, timestamp="future", role="anything")
        a, b = subject.annotation_row(original, "fixture.json"), subject.annotation_row(altered, "fixture.json")
        self.assertEqual(a, b)
        self.assertEqual(a["text"], "열 있나요?\n아니요")

    def test_none_text_is_empty_line_but_invalid_type_rejected(self):
        self.assertEqual(subject.annotation_row({"utterances": [{"text": None}, {"text": " 응 "}]}, "x.json")["text"], "응")
        with self.assertRaises(ValueError):
            subject.annotation_row({"utterances": [{"text": 2}]}, "x.json")
        with self.assertRaises(ValueError):
            subject.annotation_row({"utterances": [{"text": " "}]}, "x.json")

    def test_basename_duplicate_across_subdirectories_rejected(self):
        with temporary_directory() as temp:
            root = Path(temp)
            for name in ("a", "b"):
                (root / name).mkdir()
                (root / name / "same.json").write_text('{"utterances":[{"text":"synthetic"}]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate input"):
                subject.load_inputs(root)

    def test_input_order_is_sorted_filename(self):
        with temporary_directory() as temp:
            root = Path(temp)
            for name in ("z.json", "a.json"):
                (root / name).write_text('{"utterances":[{"text":"synthetic"}]}', encoding="utf-8")
            rows, hashes = subject.load_inputs(root)
            self.assertEqual([row["filename"] for row in rows], ["a.json", "z.json"])
            self.assertEqual(len(hashes), 2)

    def test_duplicate_json_keys_and_nonfinite_rejected(self):
        with temporary_directory() as temp:
            path = Path(temp) / "x.json"
            for content in ('{"a":1,"a":2}', '{"a":NaN}'):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    subject.read_json(path)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is required for the actual ensemble arithmetic checks")
class MathTests(unittest.TestCase):
    def test_float64_probability_mean_not_float32(self):
        result = subject.ensemble_probabilities([[1.0] * 9], [[1e-8] * 9])
        self.assertEqual(str(result.dtype), "float64")
        self.assertEqual(float(result[0, 0]), 0.500000005)
        self.assertNotEqual(float(result[0, 0]), 0.5)

    def test_threshold_equality_and_nearest_lower(self):
        equal = subject.FROZEN_THRESHOLDS
        below = [math.nextafter(value, -math.inf) for value in equal]
        self.assertEqual(subject.decisions([equal, below], equal).tolist(), [[1] * 9, [0] * 9])

    def test_invalid_member_probabilities_rejected(self):
        for second in ([[1.1] * 9], [[0.5] * 8], [[math.nan] * 9]):
            with self.assertRaises(ValueError):
                subject.ensemble_probabilities([[0.1] * 9], second)


class BundleAndCSVTests(unittest.TestCase):
    def test_verify_only_never_imports_model_or_reads_input(self):
        with temporary_directory() as temp:
            checkpoint = make_bundle(Path(temp))
            with patch.object(subject, "infer", side_effect=AssertionError("model called")), \
                 patch.object(subject, "load_inputs", side_effect=AssertionError("input read")), \
                 patch.object(subject, "GPUChecker", side_effect=AssertionError("GPU queried")):
                # Keep original class for source-byte verification while guarding
                # constructor use; __module__ resolves to the same guard file.
                subject.GPUChecker.__module__ = "gpu_guard"
                result = subject.run(checkpoint, label_dir="missing", verify_only=True, audio_dir="missing")
            self.assertFalse(result["cuda_used"])
            self.assertEqual(result["model_forward_calls"], 0)

    def test_manifest_tamper_and_size_change_rejected(self):
        with temporary_directory() as temp:
            checkpoint = make_bundle(Path(temp))
            checkpoint.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "changed or is incomplete"):
                subject.verify_bundle(checkpoint)

    def test_lfs_pointer_has_download_instruction(self):
        with temporary_directory() as temp:
            checkpoint = Path(temp) / "model.pt"
            checkpoint.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:synthetic\nsize 1\n")
            with self.assertRaisesRegex(ValueError, "git lfs pull"):
                subject.verify_bundle(checkpoint)

    def test_path_traversal_rejected(self):
        with temporary_directory() as temp:
            for name in ("../x", "a/../x", "a\\x", "/x"):
                with self.assertRaises(ValueError):
                    subject.safe_file(Path(temp), name)

    def test_fixed_config_cannot_silently_round_thresholds(self):
        changed = config_fixture()
        changed["thresholds"] = [round(x, 3) for x in subject.FROZEN_THRESHOLDS]
        with self.assertRaisesRegex(ValueError, "exact frozen thresholds"):
            subject.validate_config(changed)

    def test_empty_and_korean_csv_with_bom_crlf(self):
        with temporary_directory() as temp:
            path = Path(temp) / "candidate.csv"
            rows = [{"filename": "a.json"}, {"filename": "b.json"}]
            subject.write_internal_csv(path, rows, [[0] * 9, [1] + [0] * 8])
            raw = path.read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
            self.assertEqual(raw.count(b"\r\n"), 3)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.reader(stream))
            self.assertEqual(records[0], ["labelfile", "symptom"])
            self.assertEqual(records[1], ["a.json", "[]"])
            self.assertEqual(json.loads(records[2][1]), ["고열"])
            with self.assertRaises(ValueError):
                subject.write_internal_csv(path, rows, [[0] * 9, [0] * 9])

    def test_checkpoint_must_be_neural_only_two_members(self):
        tensor = object()
        saved = {"kind": "roberta_multilabel", "labels": subject.LABELS,
                 "config": {"model_type": "roberta"}, "state_dict": {"synthetic.weight": tensor},
                 "max_length": 512, "stride": 128, "pooling": "max_logits"}
        package = {"kind": "mission3_neural_ensemble", "labels": subject.LABELS, "members": [saved, saved]}
        torch = SimpleNamespace(is_tensor=lambda value: value is tensor)
        self.assertIs(subject.validate_package(package, torch), package)
        leaked = {**saved, "training_texts": ["synthetic"]}
        with self.assertRaises(ValueError):
            subject.validate_package({**package, "members": [saved, leaked]}, torch)


class PoolingAndGuardTests(unittest.TestCase):
    def test_all_long_chunks_pool_across_batch_boundary(self):
        chunks = [{"row": [-10.0] * 9} for _ in range(19)]
        chunks[0]["row"][0] = 4.0
        chunks[17]["row"][1] = 9.0
        chunks[18]["row"][8] = 7.0
        calls = []
        def pad(batch, **kwargs):
            self.assertEqual(kwargs, {"padding": True, "pad_to_multiple_of": 8, "return_tensors": "pt"})
            calls.append(len(batch))
            return FakeBatch(rows=[chunk["row"] for chunk in batch])
        guard = SimpleNamespace(check=lambda: calls.append("guard"))
        model = lambda rows: SimpleNamespace(logits=FakeLogits(rows))
        torch = FakeTorch()
        result = subject.pooled_logits(chunks, SimpleNamespace(pad=pad), model, torch, guard)
        self.assertEqual(result.values, [4.0, 9.0] + [-10.0] * 6 + [7.0])
        self.assertEqual(calls, ["guard", 16, "guard", 3])
        self.assertEqual(torch.autocast_options, [("cuda", "bf16")])

    def test_two_idle_launch_and_one_spare_during_forward(self):
        gpu = f"0,{UUID0},4\n1,{UUID1},4\n2,{UUID2},4000\n"
        self.assertEqual(gpu_guard.occupancy(gpu, f"{UUID2},456\n", UUID0, 123, starting=True)["other_idle_gpu_n"], 1)
        allocated = f"0,{UUID0},6000\n1,{UUID1},4\n2,{UUID2},4000\n"
        self.assertEqual(gpu_guard.occupancy(allocated, f"{UUID0},123\n{UUID2},456\n", UUID0, 123)["other_idle_gpu_n"], 1)

    def test_zero_utilization_is_not_free_memory_or_process(self):
        gpu = f"0,{UUID0},4\n1,{UUID1},1000\n"
        with self.assertRaisesRegex(ValueError, "no idle spare"):
            gpu_guard.occupancy(gpu, "", UUID0, 123, starting=True)
        gpu = f"0,{UUID0},4\n1,{UUID1},4\n"
        with self.assertRaisesRegex(ValueError, "no idle spare"):
            gpu_guard.occupancy(gpu, f"{UUID1},456", UUID0, 123, starting=True)

    def test_other_user_on_allocated_gpu_and_bad_visibility_rejected(self):
        gpu = f"0,{UUID0},4\n1,{UUID1},4\n"
        for visible in ("0", "", UUID0 + "," + UUID1, "GPU-a-b"):
            with self.assertRaises(ValueError):
                gpu_guard.occupancy(gpu, "", visible, 123)
        with self.assertRaisesRegex(ValueError, "another process"):
            gpu_guard.occupancy(gpu, f"{UUID0},456", UUID0, 123)

    def test_guard_queries_only_gpu_inventory_and_compute_processes(self):
        commands = []
        def query(command, **kwargs):
            commands.append(command)
            return f"0,{UUID0},4\n1,{UUID1},4\n" if "--query-gpu=index,uuid,memory.used" in command else ""
        checker = gpu_guard.GPUChecker(query=query, environ={"CUDA_VISIBLE_DEVICES": UUID0}, own_pid=123)
        checker.check(starting=True)
        self.assertEqual(len(commands), 2)
        self.assertTrue(all(command[0] == "nvidia-smi" for command in commands))
        self.assertFalse(any("reset" in part or "kill" in part for command in commands for part in command))


if __name__ == "__main__":
    unittest.main()
