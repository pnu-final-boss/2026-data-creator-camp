# 2026 Data Creator Camp

Shared code for the three university preliminary missions.

The final inference command must follow the organizer-provided interface:

```bash
python inference.py \
  --audio_dir {wav_folder} \
  --label_dir {json_folder} \
  --ckpt_path {checkpoint_file} \
  --output ./outputs/missionN.csv
```

## Local setup

Python 3.12 is recommended because the current PyTorch stack does not yet support
every newer Python release.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Preprocessing and EDA

The source WAV and JSON files are never modified. Raw/processed data and the
directly transformed Mel example stay local; aggregate EDA reports may be committed.

```bash
python -m src.preprocess --max_files_per_split 100 --seed 42
python -m src.eda
```

The preprocessing command samples 100 matched calls from each split by default.
Use `--max_files_per_split 0` only when a full-dataset run is needed.

Generated files:

- `data/raw/`: untouched source `Training` and `Validation` directories
- `data/processed/audio_manifest.csv`: audio metadata and pairing status
- `data/processed/mission1_segments.csv`: permitted segment fields + gender
- `data/processed/mission2_segments.csv`: segment boundaries + speaker
- `data/processed/mission3_calls.csv`: transcript text + filtered symptoms
- `data/processed/data_issues.csv`: missing or invalid pairs
- `docs/reports/eda/`: EDA report and locally generated plots
