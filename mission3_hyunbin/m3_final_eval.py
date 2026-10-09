"""
Mission 3 — 최종 평가 (재학습 없음, 임계값 0.5 고정)
  1) Training 내부 val 에서 TF-IDF / Qwen / 평균 앙상블 비교 → 최고 구성 채택
  2) 채택 구성으로 Validation 1회 평가 (보고용) + 제출 형식 CSV 저장
입력은 전사 텍스트만 (m3_train_qwen.build_text 와 동일 전처리)
"""
import argparse, hashlib, json, re, time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

from m3_train_qwen import (SYMPTOMS, DEFAULTS as TD, build_text, build_labels,
                           f1_report, predict)

VAL_LABEL_DIR = ("/home/ldw2003/2026-data-creator-camp/data/raw/Validation/"
                 "2.라벨링데이터/VL_서울_구급")
DEFAULTS = dict(
    qwen_ckpt="checkpoints/m3_qwen15_pw_e5/best.pt",
    out_dir="outputs/m3_final",
    seed=42, val_ratio=0.1, eval_batch=8,
)


def load_docs(label_dir):
    names, texts, labels = [], [], []
    for jp in sorted(Path(label_dir).rglob("*.json")):
        try:
            d = json.load(open(jp, encoding="utf-8"))
        except Exception:
            continue
        t = build_text(d)
        if not t:
            continue
        names.append(jp.name); texts.append(t); labels.append(build_labels(d))
    return names, texts, np.array(labels, dtype=np.float32)


def split_like_training(texts, seed, val_ratio):
    """m3_train_qwen 과 동일한 '동일 전사문 그룹' 분할 재현."""
    keys = [hashlib.md5(re.sub(r"\s+", "", t).encode()).hexdigest() for t in texts]
    ug = sorted(set(keys))
    rng = np.random.RandomState(seed); rng.shuffle(ug)
    val_g = set(ug[:max(1, int(len(ug) * val_ratio))])
    va = [i for i, k in enumerate(keys) if k in val_g]
    tr = [i for i, k in enumerate(keys) if k not in val_g]
    return tr, va


def load_qwen(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    tok = AutoTokenizer.from_pretrained(ck["pretrained"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        ck["pretrained"], num_labels=len(SYMPTOMS),
        problem_type="multi_label_classification", torch_dtype=torch.bfloat16)
    base.config.pad_token_id = tok.pad_token_id
    lo = ck["lora"]
    model = get_peft_model(base, LoraConfig(
        task_type="SEQ_CLS", r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    res = model.load_state_dict(ck["trainable_state_dict"], strict=False)
    assert not res.unexpected_keys, f"예상치 못한 키: {res.unexpected_keys[:5]}"
    model.to(device).eval()
    print(f"[qwen] {ckpt_path} epoch={ck['epoch']} 학습시 val_macro_F1={ck['val_macro_f1']:.4f}",
          flush=True)
    return model, tok, ck


def qwen_probs(model, tok, ck, texts, bs, device):
    ids = tok(texts, add_special_tokens=True, truncation=True,
              max_length=ck["max_len"])["input_ids"]
    return predict(model, ids, bs, tok.pad_token_id, device)


def fmt(rep):
    return " | ".join(f"{s} {rep[s]['f1']:.3f}" for s in SYMPTOMS)


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ================= 1) 내부 val: 구성 결정 =================
    names, texts, Y = load_docs(TD["label_dir"])
    tr, va = split_like_training(texts, args.seed, args.val_ratio)
    print(f"[internal] train={len(tr):,} val={len(va):,}", flush=True)

    # TF-IDF + 클래스별 로지스틱 회귀 (내부 train 으로만 학습)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                          max_features=300000, sublinear_tf=True)
    Xtr = vec.fit_transform([texts[i] for i in tr])
    Xva = vec.transform([texts[i] for i in va])
    clfs = []
    for j in range(len(SYMPTOMS)):
        c = LogisticRegression(C=2.0, max_iter=3000, solver="liblinear")
        c.fit(Xtr, Y[tr, j]); clfs.append(c)
    p_tfidf = np.stack([c.predict_proba(Xva)[:, 1] for c in clfs], 1)
    print(f"[tfidf] 학습 완료 ({(time.time()-t0)/60:.1f}min)", flush=True)

    model, tok, ck = load_qwen(args.qwen_ckpt, device)
    p_qwen = qwen_probs(model, tok, ck, [texts[i] for i in va], args.eval_batch, device)
    p_ens = (p_qwen + p_tfidf) / 2

    cand = {"tfidf": p_tfidf, "qwen": p_qwen, "ensemble": p_ens}
    rep_int = {k: f1_report(Y[va], p) for k, p in cand.items()}
    chosen = max(rep_int, key=lambda k: rep_int[k]["macro_f1"])
    diff = abs(rep_int["qwen"]["macro_f1"] - ck["val_macro_f1"])

    L = []; W = L.append
    W("=" * 70)
    W("Mission 3 최종 평가 (재학습 없음, 임계값 0.5 고정, 입력=전사 텍스트만)")
    W("=" * 70)
    W(f"[1] 내부 val {len(va):,}건 — 구성 결정")
    for k in cand:
        W(f"  {k:<9} macro_F1 {rep_int[k]['macro_f1']:.4f} | {fmt(rep_int[k])}")
    W(f"  Qwen 로딩 검증: 학습시 {ck['val_macro_f1']:.4f} vs 재추론 "
      f"{rep_int['qwen']['macro_f1']:.4f} (차이 {diff:.4f}) "
      + ("OK" if diff < 0.005 else "⚠ 확인 필요"))
    W(f"  >>> 채택: {chosen}")

    # ================= 2) Validation: 1회 평가 =================
    vnames, vtexts, vY = load_docs(VAL_LABEL_DIR)
    vp_tfidf = np.stack([c.predict_proba(vec.transform(vtexts))[:, 1] for c in clfs], 1)
    vp_qwen = qwen_probs(model, tok, ck, vtexts, args.eval_batch, device)
    vcand = {"tfidf": vp_tfidf, "qwen": vp_qwen, "ensemble": (vp_qwen + vp_tfidf) / 2}
    rep_val = {k: f1_report(vY, p) for k, p in vcand.items()}
    fin = rep_val[chosen]

    W("")
    W(f"[2] Validation {len(vtexts):,}건 — 최종 수치 (채택 구성: {chosen})")
    W("-" * 70)
    W(f"  macro F1 : {fin['macro_f1']:.4f}")
    W(f"  {'증상':<8}{'F1':>8}{'Precision':>11}{'Recall':>9}")
    for s in SYMPTOMS:
        W(f"  {s:<8}{fin[s]['f1']:>8.3f}{fin[s]['precision']:>11.3f}{fin[s]['recall']:>9.3f}")
    W("")
    W("  (참고) 세 구성의 Validation macro F1 — 채택은 위 내부 val 결과로 이미 결정됨")
    for k in vcand:
        W(f"    {k:<9} {rep_val[k]['macro_f1']:.4f}")
    W("=" * 70)
    W(f"소요 {(time.time()-t0)/60:.1f}분 / 저장 {out}")

    text = "\n".join(L)
    print(text)
    (out / "report.txt").write_text(text, encoding="utf-8")
    json.dump({"chosen": chosen, "threshold": 0.5, "internal": rep_int, "validation": rep_val},
              open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 제출 형식: 라벨파일명, symptom (문자열)
    import csv
    pred = (vcand[chosen] >= 0.5)
    with open(out / "mission3.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["라벨파일명", "symptom"])
        for n, row in zip(vnames, pred):
            w.writerow([n, str([s for s, v in zip(SYMPTOMS, row) if v])])


if __name__ == "__main__":
    main()
