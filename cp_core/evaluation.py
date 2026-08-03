"""Calibration quantiles and coverage / efficiency evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cp_core.scores import conformal_quantile
from cp_core.split import Split


def epmax_quantile(
    split: Split,
    w: np.ndarray,
    score: str,
    alpha: float,
    ep_ptr: Optional[List[Tuple[int, int]]] = None,
) -> float:
    """Corrected quantile of per-episode maxima of s = base_teacher/(1+w)."""
    s = split.base_teacher[score] / (1.0 + w)
    ptr = ep_ptr if ep_ptr is not None else split.ep_ptr
    return conformal_quantile([s[a:b].max() for a, b in ptr if b > a], alpha)


def pooled_quantile(split: Split, score: str, alpha: float) -> float:
    """Step-pooled base-CP quantile (the collapsing baseline)."""
    return conformal_quantile(split.base_teacher[score], alpha)


def _set_sizes(split: Split, thr: np.ndarray, score: str) -> np.ndarray:
    cands = split.base_cands[score]
    return np.fromiter(
        (max(int(np.sum(cands[i] <= thr[i])), 1) for i in range(len(split))),
        dtype=int,
        count=len(split),
    )


def evaluate(
    split: Split, q: float, w: np.ndarray, score: str
) -> Dict[str, float]:
    """Coverage / efficiency with per-step effective threshold q * (1 + w).

    Beyond the headline cov/mean_set/singleton/saturation, reports the
    full set-size distribution (median, std, percentiles, min/max), a
    backbone-comparable normalised inefficiency (set_degree_ratio),
    standard errors of both coverage estimates, sample sizes, and the
    deployed per-step weight's own distribution -- everything needed to
    interpret a result cell without re-deriving it from the dump."""
    thr = q * (1.0 + w)
    covered = split.base_teacher[score] <= thr
    sizes = _set_sizes(split, thr, score)
    ep_rate = np.asarray([covered[a:b].mean() for a, b in split.ep_ptr])
    ep_simul = np.asarray(
        [bool(covered[a:b].all()) for a, b in split.ep_ptr], dtype=float
    )

    def se(x: np.ndarray) -> float:
        return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0

    return {
        "q": float(q),
        "cov_step": float(ep_rate.mean()),
        "cov_step_se": se(ep_rate),
        "cov_simul": float(ep_simul.mean()),
        "cov_simul_se": se(ep_simul),
        "mean_set": float(sizes.mean()),
        "median_set": float(np.median(sizes)),
        "set_std": float(sizes.std()),
        "set_min": int(sizes.min()),
        "set_max": int(sizes.max()),
        "set_p10": float(np.percentile(sizes, 10)),
        "set_p25": float(np.percentile(sizes, 25)),
        "set_p75": float(np.percentile(sizes, 75)),
        "set_p90": float(np.percentile(sizes, 90)),
        "set_p95": float(np.percentile(sizes, 95)),
        "set_degree_ratio": float(np.mean(sizes / split.degree)),
        "singleton": float(np.mean(sizes == 1)),
        "saturation": float(np.mean(sizes >= split.degree)),
        "n_episodes": int(len(ep_rate)),
        "n_steps": int(len(sizes)),
        "weight_mean": float(np.mean(w)),
        "weight_median": float(np.median(w)),
        "weight_std": float(np.std(w)),
    }


def conditional_diagnostics(
    split: Split, q: float, w: np.ndarray, score: str
) -> Dict[str, Any]:
    """Reviewer-facing conditionals: coverage by p_max quartile and degree
    class, plus the query-budget curve (recall of argmax errors when asking
    on the largest sets first vs the lowest-confidence steps first)."""
    thr = q * (1.0 + w)
    covered = split.base_teacher[score] <= thr
    sizes = _set_sizes(split, thr, score)

    quartile = np.searchsorted(
        np.quantile(split.p_max, [0.25, 0.5, 0.75]), split.p_max
    )
    by_quartile = [
        float(covered[quartile == b].mean()) if (quartile == b).any() else None
        for b in range(4)
    ]

    classes = {
        "small<=5": split.degree <= 5,
        "med6-10": (split.degree > 5) & (split.degree <= 10),
        "large>10": split.degree > 10,
    }
    by_degree = {
        k: {
            "cov": float(covered[m].mean()),
            "mean_set": float(sizes[m].mean()),
            "n": int(m.sum()),
        }
        for k, m in classes.items()
        if m.any()
    }

    err = split.argmax_err
    total = max(int(err.sum()), 1)
    ask_by_size = np.argsort(-sizes)
    ask_by_conf = np.argsort(split.p_max)
    budget = {
        f"{b:.2f}": {
            "set_size_trigger": float(
                err[ask_by_size[: max(1, int(b * len(err)))]].sum() / total
            ),
            "pmax_trigger": float(
                err[ask_by_conf[: max(1, int(b * len(err)))]].sum() / total
            ),
        }
        for b in (0.05, 0.10, 0.20, 0.30)
    }
    return {
        "cov_by_pmax_quartile": by_quartile,
        "cov_by_degree": by_degree,
        "query_budget": budget,
    }
