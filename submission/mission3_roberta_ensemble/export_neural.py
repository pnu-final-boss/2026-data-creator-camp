"""CPU-only export of the two selected neural members; no contest examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = ""

SOURCE_SHA = "b2f36866cd3bbefcdb41f18213bd09b3ff51d84c08599d5186210c4051becfe3"
LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
TOKENIZER_SHA = {
    "special_tokens_map.json": "a627d4e20da5e9482f6a08f667ead8809ff9a22dbf6284b91dacd57f217b4555",
    "tokenizer.json": "4a45131245e310d1650c760a22038de8180f7311ace3575ad96b24f85ef38e1b",
    "tokenizer_config.json": "e4ac45e7679c783ff5c5a5f9c1ed12c1c5fb86eda7f160b0c68a43d300ff6c91",
    "vocab.txt": "1ad4978b5dbe269dcc402a3c6eb71ceab21d712bca3b7d3b9994845e54cdcb39",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tensor_inventory(state, torch):
    result = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError("State must contain named CPU tensors only")
        blob = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
        result[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": hashlib.sha256(blob).hexdigest()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source, output = Path(args.source_dir).resolve(), Path(args.output_dir).resolve()
    if source == output or source in output.parents or output.exists():
        raise ValueError("Use a fresh separate output directory")
    if sha(source / "model.pt") != SOURCE_SHA:
        raise ValueError("Original checkpoint SHA differs")
    import torch
    package = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    if package.get("kind") != "mission3_probability_ensemble" or package.get("labels") != LABELS:
        raise ValueError("Unexpected original model package")
    original_members = [member["checkpoint"] for member in package["members"] if member["kind"] == "roberta"]
    if len(original_members) != 2:
        raise ValueError("Expected two neural members")
    members, inventories = [], []
    fields = ("kind", "labels", "config", "state_dict", "max_length", "stride", "pooling")
    for saved in original_members:
        if saved["kind"] != "roberta_multilabel" or saved["labels"] != LABELS or (saved["max_length"], saved["stride"], saved["pooling"]) != (512, 128, "max_logits"):
            raise ValueError("Unexpected frozen member")
        member = {key: saved[key] for key in fields}
        member["config"] = dict(saved["config"])
        # HF's local provenance path is not part of its architecture.
        member["config"].pop("_name_or_path", None)
        json.dumps(member["config"], allow_nan=False)
        members.append(member)
        inventories.append(tensor_inventory(member["state_dict"], torch))
    output.mkdir(parents=True)
    torch.save({"kind": "mission3_neural_ensemble", "labels": LABELS, "members": members}, output / "model.pt")
    exported = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    if set(exported) != {"kind", "labels", "members"} or any(set(member) != set(fields) for member in exported["members"]):
        raise ValueError("Unexpected public fields")
    for index, member in enumerate(exported["members"]):
        if tensor_inventory(member["state_dict"], torch) != inventories[index]:
            raise ValueError("Export changed a tensor")
        if member["config"] != members[index]["config"]:
            raise ValueError("Export changed an architecture")
    (output / "tokenizer").mkdir()
    for name, expected in TOKENIZER_SHA.items():
        if sha(source / "tokenizer" / name) != expected:
            raise ValueError("Original tokenizer SHA differs")
        shutil.copyfile(source / "tokenizer" / name, output / "tokenizer" / name)
        if sha(output / "tokenizer" / name) != expected:
            raise ValueError("Export changed a tokenizer")
    receipt = {
        "status": "passed", "source_checkpoint_sha256": SOURCE_SHA,
        "exported_checkpoint_sha256": sha(output / "model.pt"),
        "exported_checkpoint_bytes": (output / "model.pt").stat().st_size,
        "member_count": 2, "exported_member_fields": list(fields),
        "source_member_field_names": [sorted(member) for member in original_members],
        "member_config_fields": [sorted(member["config"]) for member in members],
        "member_state_dicts": inventories,
        "tensor_values_shapes_dtypes_and_names_exact": True,
        "tokenizer_sha256": TOKENIZER_SHA,
        "removed_components": ["non-neural baseline", "training/dev probabilities", "split IDs", "selection metadata", "HF local source path"],
        "forward_passes": 0, "cuda_initialized": torch.cuda.is_initialized(),
        "torch_version_used_for_export": torch.__version__,
        "note": "CPU tensor export proof only; no new Validation or published CLI replay is claimed.",
    }
    if receipt["cuda_initialized"]:
        raise ValueError("CPU export must not initialize CUDA")
    save_json(output / "export_receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "exported_checkpoint_sha256", "exported_checkpoint_bytes", "member_count", "forward_passes", "cuda_initialized")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
