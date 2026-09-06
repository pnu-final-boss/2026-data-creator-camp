# -*- coding: utf-8 -*-
"""Mission 2 ablation.

검증할 두 가지 주장
  1. 통화별 평균차감(call normalization)이 실제로 기여하는가.
     — 대원은 고정 헤드셋/상황실, 신고자는 휴대폰/야외. 절대 음색이 아니라 "같은 통화 안에서의
       상대적 채널 차이"가 신호라면 raw 보다 norm/both 가 나아야 한다.
  2. 중첩 구간을 학습에서 빼면(라벨 노이즈 제거) 전체 성능이 오르는가.
"""
import json

from ..common.paths import CACHE
from .train import run

CONFIGS = [
    ("raw",  "all",   "원본 특징만"),
    ("norm", "all",   "통화별 평균차감만"),
    ("both", "all",   "원본 + 평균차감 (기본)"),
    ("both", "clean", "원본 + 평균차감, 비중첩 구간만 학습"),
]


def main():
    out = []
    for mode, train_on, desc in CONFIGS:
        r = run(mode=mode, train_on=train_on, model="hgb", save=False)
        r["desc"] = desc
        out.append(r)
        with open(CACHE / "m2_ablation.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[ablate] {desc:34s} acc={r['acc']:.4f} "
              f"(비중첩 {r['acc_nonoverlap']:.4f} / 중첩 {r['acc_overlap']:.4f})", flush=True)
    print("[ablate] done ->", CACHE / "m2_ablation.json")


if __name__ == "__main__":
    main()
