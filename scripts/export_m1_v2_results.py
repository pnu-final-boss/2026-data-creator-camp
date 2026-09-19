"""Export the public, reproducible artifacts from an M1 v2 run directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FILES = (
    "config.json",
    "environment.json",
    "provenance.json",
    "dev_metrics.json",
    "dev_predictions.csv",
    "history.jsonl",
    "learning_curve.png",
    "result_summary.png",
)
DIRECTORIES = ("figures", "validation")


def export_results(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Run directory not found: {source}")

    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []

    for relative in FILES:
        item = source / relative
        if item.is_file():
            target = destination / relative
            shutil.copy2(item, target)
            copied.append(target)

    for relative in DIRECTORIES:
        item = source / relative
        if item.is_dir():
            target = destination / relative
            shutil.copytree(item, target, dirs_exist_ok=True)
            copied.append(target)

    required = destination / "validation" / "metrics.json"
    if not required.is_file():
        raise FileNotFoundError(
            "validation/metrics.json was not found; run the official validation "
            "evaluation before exporting."
        )

    print(f"Exported {len(copied)} result items to {destination}")
    print("Checkpoints and original data were intentionally excluded.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed runs/m1_v2/<run_id> directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/m1_v2"),
        help="Repository result directory (default: outputs/m1_v2)",
    )
    args = parser.parse_args()
    export_results(args.run_dir, args.output)


if __name__ == "__main__":
    main()
