"""Base nonconformity scores and the corrected conformal quantile."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

SCORES = ("THR", "APS", "RAPS")
LAM, KREG = 0.1, 2  # RAPS regularisation (weight / penalty-free top-k)
EPS = 1e-12


def base_scores_all(p: np.ndarray) -> Dict[str, np.ndarray]:
    """THR / APS(U=0) / RAPS base score for every candidate, given softmax
    p."""
    order = np.argsort(-p)
    above = np.concatenate(
        [[0.0], np.cumsum(p[order])[:-1]]
    )  # mass strictly above
    ranks = np.arange(1, len(p) + 1)
    aps = np.empty_like(p)
    aps[order] = above
    raps = np.empty_like(p)
    raps[order] = above + LAM * np.maximum(0.0, ranks - KREG)
    return {"THR": 1.0 - p, "APS": aps, "RAPS": raps}


def softmax_valid(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Softmax over the valid (finite-logit) candidates: (p, valid_mask)."""
    valid = np.isfinite(logits)
    lv = logits[valid]
    e = np.exp(lv - lv.max())
    return np.clip(e / e.sum(), EPS, 1.0), valid


def teacher_fallback(score: str, n: int) -> float:
    """Worst-case base score when the teacher is not a valid candidate."""
    return 1.0 + LAM * max(0, n - KREG) if score == "RAPS" else 1.0


def conformal_quantile(scores, alpha: float) -> float:
    """ceil((n+1)(1-alpha))-th smallest score; +inf when n cannot certify."""
    s = np.sort(np.asarray(list(scores), dtype=float))
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    return float("inf") if k > n else float(s[k - 1])
