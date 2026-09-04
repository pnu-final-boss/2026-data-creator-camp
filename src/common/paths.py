# -*- coding: utf-8 -*-
"""데이터 경로 및 상수. 다른 모듈은 전부 여기를 통해 경로를 얻는다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "대학부 데이터"

TRAIN_AUDIO = DATA / "Training" / "1.원천데이터" / "TS_서울_구급"
TRAIN_LABEL = DATA / "Training" / "2.라벨링데이터" / "TL_서울_구급"
VAL_AUDIO = DATA / "Validation" / "1.원천데이터" / "VS_서울_구급"
VAL_LABEL = DATA / "Validation" / "2.라벨링데이터" / "VL_서울_구급"

CACHE = ROOT / "cache"
FIGS = ROOT / "figs"
OUTPUTS = ROOT / "outputs"
CKPT = ROOT / "ckpt"
for _d in (CACHE, FIGS, OUTPUTS, CKPT):
    _d.mkdir(exist_ok=True)

# Mission 3: 평가 대상 9개 증상 클래스 (출제 PDF 9쪽)
SYMPTOM_9 = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]

# Mission 1 / 2 라벨
GENDER = ["F", "M"]              # 0=F, 1=M
SPEAKER = ["119대원", "신고자"]   # 0=대원, 1=신고자

SR = 8000  # 원천 데이터 샘플레이트
