# -*- coding: utf-8 -*-
"""라벨 JSON 로더 + 캐시.

29,200 + 3,640 개의 JSON 을 매번 읽으면 느리므로 한 번 읽어 pickle 로 캐시한다.
반환 레코드는 미션별로 필요한 필드만 담는다 (규정상 허용 필드 구분은 각 미션 코드에서 한다).
"""
import json
import pickle
from concurrent.futures import ProcessPoolExecutor

from .paths import CACHE, TRAIN_LABEL, VAL_LABEL


def _read_one(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    utts = [
        {
            "startAt": u.get("startAt"),
            "endAt": u.get("endAt"),
            "speaker": u.get("speaker"),
            "text": u.get("text", ""),
        }
        for u in d.get("utterances", [])
    ]
    return {
        "stem": path.stem,                 # wav 파일명과 1:1 대응
        "gender": d.get("gender"),
        "symptom": d.get("symptom") or [],
        "startAt": d.get("startAt", 0),
        "endAt": d.get("endAt", 0),
        "utterances": utts,
    }


def load_split(split: str, workers: int = 8):
    """split: 'train' | 'val'. 캐시가 있으면 그대로 반환."""
    cache_file = CACHE / f"labels_{split}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    d = TRAIN_LABEL if split == "train" else VAL_LABEL
    files = sorted(d.glob("*.json"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(_read_one, files, chunksize=64))

    with open(cache_file, "wb") as f:
        pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return recs


if __name__ == "__main__":
    for s in ("train", "val"):
        r = load_split(s)
        print(f"{s}: {len(r)} records, e.g. {r[0]['stem']} gender={r[0]['gender']} "
              f"utts={len(r[0]['utterances'])}")
