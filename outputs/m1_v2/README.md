# Mission 1 WavLM v2 results

This directory contains the reproducible metrics, predictions, and figures from
the Mission 1 WavLM experiment executed on 2026-09-19.

## Result

| Evaluation | Accuracy | Macro F1 | AUC | Calls |
|---|---:|---:|---:|---:|
| Internal dev | 0.988356 | 0.988288 | 0.997495 | 5,840 |
| Official validation | 0.987363 | 0.987292 | 0.996465 | 3,640 |

The selected checkpoint is epoch 4. Training stopped after epoch 7 because the
internal-dev accuracy did not improve for three consecutive epochs. The official
validation confusion matrix is `[[1933, 26], [20, 1661]]` for labels `F=0` and
`M=1`.

## Included here

- `result_summary.png` and `learning_curve.png`
- dev and official-validation metrics and predictions
- confusion matrices, ROC/PR plots, duration reports, and error tables
- `config.json`, `environment.json`, and `provenance.json`
- the fully executed notebook at `../../notebooks/m1_v2/m1_v2_linux_r5.ipynb`
- exact package versions at `../../requirements-m1.txt`

## Checkpoints and full portable bundle

Files larger than GitHub's normal 100 MiB Git-object limit are published as
GitHub Release assets rather than committed to the repository:

- `best.pt` — selected epoch-4 model for inference
- `m1_v2_portable_20260919_165146.tar` — notebook, checkpoints, metrics,
  predictions, figures, environment metadata, and per-file checksums
- `m1_v2_portable_20260919_165146.tar.sha256` — archive checksum

Expected SHA-256 for the portable TAR:

```text
664d87d1cc44459261dc83798ba63b69758999dc413f532f8d6a9d630c14cef6
```

Extract and install the recorded environment with:

```bash
tar -xf m1_v2_portable_20260919_165146.tar
cd 20260919_165146_wavlm_experiment
python -m pip install -r requirements-lock.txt
```

`best.pt` contains the full trained state dict, WavLM backbone configuration,
label map, decision threshold, and training configuration. It can be restored
without downloading the original Hugging Face checkpoint. Recomputing the
official validation metrics still requires the original validation WAV/JSON data.
