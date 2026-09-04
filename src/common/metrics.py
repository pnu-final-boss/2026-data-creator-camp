# -*- coding: utf-8 -*-
"""평가 지표. 출제 PDF 10쪽의 macro-F1 정의를 그대로 구현한다.

PDF 정의: 통화별 정답/예측 → 레이블별 이진화 → 레이블별 TP/FP/FN → 클래스별 F1 → 단순 평균.
이는 sklearn 의 f1_score(average='macro', zero_division=0) 와 동일하다.
아래 macro_f1_pdf() 는 그 동치성을 직접 확인하기 위해 손으로 구현해 둔 것이다.
"""
import numpy as np
from sklearn.metrics import f1_score


def macro_f1_pdf(y_true: np.ndarray, y_pred: np.ndarray):
    """PDF 4단계 절차를 그대로 따른 macro-F1. (n_samples, n_classes) 이진 행렬 입력."""
    per_class = []
    for c in range(y_true.shape[1]):
        t, p = y_true[:, c], y_pred[:, c]
        tp = int(((t == 1) & (p == 1)).sum())
        fp = int(((t == 0) & (p == 1)).sum())
        fn = int(((t == 1) & (p == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class.append(dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1))
    macro = float(np.mean([d["f1"] for d in per_class]))
    return macro, per_class


def tune_thresholds(y_true: np.ndarray, y_score: np.ndarray, grid=None):
    """클래스별 임계값을 macro-F1 기준으로 각각 최적화한다.

    macro-F1 은 클래스별 F1 의 단순 평균이므로 각 클래스를 독립적으로 최적화하면
    전체 macro-F1 이 최적이 된다. 반드시 dev split 에서만 호출할 것.
    """
    if grid is None:
        grid = np.arange(0.05, 0.95, 0.01)
    n_classes = y_true.shape[1]
    best = np.full(n_classes, 0.5)
    for c in range(n_classes):
        t, s = y_true[:, c], y_score[:, c]
        scores = [f1_score(t, (s >= th).astype(int), zero_division=0) for th in grid]
        best[c] = float(grid[int(np.argmax(scores))])
    return best


def binary_confusion(y_true, y_pred):
    """2x2 혼동행렬을 [[TN, FP], [FN, TP]] 순서로 반환."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t in (0, 1):
        for p in (0, 1):
            cm[t, p] = int(((y_true == t) & (y_pred == p)).sum())
    return cm
