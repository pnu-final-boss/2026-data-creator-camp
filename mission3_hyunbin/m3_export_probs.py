"""
Qwen LoRA 체크포인트 확률을 팀 공통 형식으로 저장 (재학습 없음)
  공통 val: 학습 시 저장된 val_probs.npz / Validation: 새로 추론
"""
import argparse, csv
from pathlib import Path
import numpy as np
import torch
from m3_train_qwen import SYMPTOMS
from m3_final_eval import load_docs, load_qwen, qwen_probs, VAL_LABEL_DIR


def write(path, names, probs):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["라벨파일명"] + SYMPTOMS)
        for n, p in zip(names, probs):
            w.writerow([n] + [f"{v:.6f}" for v in p])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    out = Path("outputs/m3_team"); out.mkdir(parents=True, exist_ok=True)
    z = np.load(f"{args.ckpt_dir}/val_probs.npz", allow_pickle=True)
    write(out / f"{args.name}_internal.csv", z["names"].tolist(), z["probs"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, ck = load_qwen(f"{args.ckpt_dir}/best.pt", device)
    vnames, vtexts, _ = load_docs(VAL_LABEL_DIR)
    write(out / f"{args.name}_validation.csv", vnames, qwen_probs(model, tok, ck, vtexts, 8, device))
    print(f"[done] {args.name}: internal {len(z['names']):,} / validation {len(vnames):,}")


if __name__ == "__main__":
    main()
