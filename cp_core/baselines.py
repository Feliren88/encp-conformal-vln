"""Review baselines: temperature, weighted CP, Mondrian, CIs, ties.

computed from the raw dumps:
  temperature_scaled_base_cp  R3-W2  fix overconfidence at the source?
  weighted_cp                 R3-W1  likelihood-ratio weights under shift
  mondrian_by_degree          R3-W1  per-degree-class calibration
  cluster_bootstrap_ci        R1-W3/W4  scan-cluster CIs on coverage
  tie_stats                   R1-Q3  fp16 tie rate in episode-max scores
  latency_us                  R2-W5  membership-test cost per step
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cp_core.scores import (
    SCORES,
    base_scores_all,
    softmax_valid,
    conformal_quantile,
)
from cp_core.split import Split
from cp_core.weights import WEIGHT_FAMILY
from cp_core.evaluation import epmax_quantile, evaluate
from cp_core.analyses import ALPHAS


# ---------------------------------------------------- temperature scaling
def _valid_logit_arrays(
    records_by_episode: Dict[str, Any],
) -> List[np.ndarray]:
    """Per step: (valid logits, teacher position among valid) pairs."""
    out = []
    for recs in records_by_episode.values():
        for r in recs:
            lg = r["logits"]
            lg = (
                lg.float().numpy()
                if isinstance(lg, torch.Tensor)
                else np.asarray(lg, float)
            )
            valid = np.isfinite(lg)
            ti = int(r["teacher_idx"])
            if not (0 <= ti < len(lg) and valid[ti]):
                continue
            out.append((lg[valid], int(valid[:ti].sum())))
    return out


def fit_temperature(
    cal_records: Dict[str, Any],
    grid=np.exp(np.linspace(np.log(0.25), np.log(8.0), 61)),
) -> float:
    """Grid-search the temperature minimising teacher NLL on calibration."""
    pairs = _valid_logit_arrays(cal_records)
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        nll = 0.0
        for lg, pos in pairs:
            z = lg / t
            z = z - z.max()
            nll -= z[pos] - np.log(np.exp(z).sum())
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def temperature_scaled_base_cp(
    cal_records: Dict[str, Any], test_records: Dict[str, Any], alphas=ALPHAS
) -> Dict[str, Any]:
    """The 'fix overconfidence at the source' baseline: fit T on val_seen,
    then run PLAIN step-pooled base CP on softmax(logits / T). The question
    is whether one global temperature restores the alpha-response that the
    concentrated policy destroys."""
    T = fit_temperature(cal_records)

    def scaled_split(records: Dict[str, Any]) -> Split:
        scaled = {
            eid: [
                {
                    **r,
                    "logits": (
                        r["logits"].float() / T
                        if isinstance(r["logits"], torch.Tensor)
                        else np.asarray(r["logits"], float) / T
                    ),
                }
                for r in recs
            ]
            for eid, recs in records.items()
        }
        return Split.from_records(scaled)

    cal, test = scaled_split(cal_records), scaled_split(test_records)
    out: Dict[str, Any] = {"temperature": T}
    for alpha in alphas:
        a = f"{alpha:.2f}"
        out[a] = {}
        for score in SCORES:
            q = conformal_quantile(cal.base_teacher[score], alpha)
            out[a][score] = evaluate(test, q, np.zeros(len(test)), score)
    return out


# ---------------------------------------------------- weighted CP (shift)
def _episode_features(split: Split) -> np.ndarray:
    """Per-episode covariates: mean of the 5 informative step features
    plus log-length (the alpha slot is excluded)."""
    rows = []
    for a, b in split.ep_ptr:
        rows.append(
            np.concatenate([split.feat[a:b, :5].mean(0), [np.log(b - a)]])
        )
    return np.asarray(rows, np.float32)


def _logistic_density_ratio(
    x_cal: np.ndarray, x_test: np.ndarray, epochs: int = 400, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Logistic regression cal-vs-test; returns unnormalised dQ/dP weights
    exp(logit) for calibration episodes and test episodes."""
    torch.manual_seed(seed)
    X = np.vstack([x_cal, x_test])
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xc = torch.from_numpy((x_cal - mu) / sd)
    Xt = torch.from_numpy((x_test - mu) / sd)
    Xall = torch.cat([Xc, Xt])
    y = torch.cat([torch.zeros(len(Xc)), torch.ones(len(Xt))])
    lin = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    with torch.enable_grad():
        for _ in range(epochs):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(
                lin(Xall).squeeze(-1), y
            ).backward()
            opt.step()
    with torch.no_grad():
        # odds ratio ~ dQ/dP up to the class prior, which cancels in the
        # normalised weighted quantile
        w_cal = torch.exp(lin(Xc).squeeze(-1)).numpy().astype(np.float64)
        w_test = torch.exp(lin(Xt).squeeze(-1)).numpy().astype(np.float64)
    clip = np.quantile(np.concatenate([w_cal, w_test]), 0.99)
    return np.minimum(w_cal, clip), np.minimum(w_test, clip)


def weighted_cp(
    cal: Split, test: Split, score: str = "THR", alphas=ALPHAS
) -> Dict[str, Any]:
    """Weighted split CP (Tibshirani et al. 2019) on the pf episode-max
    scores, with a logistic-regression density ratio over episode
    covariates. Targets the simultaneous coverage the seen->unseen shift
    pulls below 1-alpha."""
    w_cal_ep, w_test_ep = _logistic_density_ratio(
        _episode_features(cal), _episode_features(test)
    )
    s_cal = cal.base_teacher[score] / (2.0 - cal.p_max)
    s_test = test.base_teacher[score] / (2.0 - test.p_max)
    smax_cal = np.array([s_cal[a:b].max() for a, b in cal.ep_ptr])
    smax_test = np.array([s_test[a:b].max() for a, b in test.ep_ptr])
    order = np.argsort(smax_cal)
    s_sorted = smax_cal[order]
    w_sorted = w_cal_ep[order]
    cum = np.cumsum(w_sorted)
    W = cum[-1]
    out: Dict[str, Any] = {
        "ess_cal": float(w_cal_ep.sum() ** 2 / (w_cal_ep**2).sum())
    }
    for alpha in alphas:
        cut = (1.0 - alpha) * (W + w_test_ep)  # per test episode
        idx = np.searchsorted(cum, cut, side="left")
        q = np.where(
            idx < len(s_sorted),
            s_sorted[np.minimum(idx, len(s_sorted) - 1)],
            np.inf,
        )
        covered = smax_test <= q
        out[f"{alpha:.2f}"] = {
            "cov_simul": float(covered.mean()),
            "frac_q_inf": float(np.mean(~np.isfinite(q))),
            "median_q": (
                float(np.median(q[np.isfinite(q)]))
                if np.isfinite(q).any()
                else None
            ),
        }
    return out


# ---------------------------------------------------- Mondrian by degree
_DEGREE_CLASSES = {
    "small<=5": lambda d: d <= 5,
    "med6-10": lambda d: (d > 5) & (d <= 10),
    "large>10": lambda d: d > 10,
}


def mondrian_by_degree(
    cal: Split, test: Split, score: str = "THR", alpha: float = 0.10
) -> Dict[str, Any]:
    """Group-conditional (Mondrian) calibration by action-space size, on the
    pf-normalised STEP scores. Step-pooled (like base CP), so it carries no
    trajectory guarantee -- it answers the conditional-coverage/set-size ask,
    not the theorem's. Compared against the marginal episode-max method."""
    s_cal = cal.base_teacher[score] / (2.0 - cal.p_max)
    q_marginal = epmax_quantile(
        cal, WEIGHT_FAMILY["pf"](cal, alpha, {}), score, alpha
    )
    w_test = WEIGHT_FAMILY["pf"](test, alpha, {})
    thr_marginal = q_marginal * (1.0 + w_test)
    out: Dict[str, Any] = {"q_marginal": float(q_marginal)}
    for name, mask_fn in _DEGREE_CLASSES.items():
        m_cal, m_test = mask_fn(cal.degree), mask_fn(test.degree)
        if not m_cal.any() or not m_test.any():
            continue
        q_c = conformal_quantile(s_cal[m_cal], alpha)
        thr_c = q_c * (2.0 - test.p_max[m_test])
        bt = test.base_teacher[score][m_test]
        cands = [test.base_cands[score][i] for i in np.nonzero(m_test)[0]]
        sizes_c = np.array(
            [max(int(np.sum(c <= t)), 1) for c, t in zip(cands, thr_c)]
        )
        sizes_m = np.array(
            [
                max(int(np.sum(c <= t)), 1)
                for c, t in zip(cands, thr_marginal[m_test])
            ]
        )
        deg = test.degree[m_test]
        out[name] = {
            "n": int(m_test.sum()),
            "q_class": float(q_c),
            "mondrian": {
                "cov": float((bt <= thr_c).mean()),
                "mean_set": float(sizes_c.mean()),
                "saturation": float((sizes_c >= deg).mean()),
            },
            "marginal": {
                "cov": float((bt <= thr_marginal[m_test]).mean()),
                "mean_set": float(sizes_m.mean()),
                "saturation": float((sizes_m >= deg).mean()),
            },
        }
    return out


# ---------------------------------------------------- cluster bootstrap CI
def cluster_bootstrap_ci(
    test: Split,
    q: float,
    w: np.ndarray,
    score: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Percentile CI for cov_step / cov_simul, resampling SCANS with
    replacement (episodes cluster by building, so per-episode CIs are too
    narrow; R1-W3/W4)."""
    thr = q * (1.0 + w)
    covered = test.base_teacher[score] <= thr
    ep_scan, ep_rate, ep_sim = [], [], []
    for a, b in test.ep_ptr:
        ep_scan.append(test.scan[a])
        ep_rate.append(covered[a:b].mean())
        ep_sim.append(float(covered[a:b].all()))
    ep_scan = np.asarray(ep_scan)
    ep_rate = np.asarray(ep_rate)
    ep_sim = np.asarray(ep_sim)
    scans = np.unique(ep_scan)
    by_scan = {s: np.nonzero(ep_scan == s)[0] for s in scans}
    rng = np.random.RandomState(seed)
    stats = np.empty((n_boot, 2))
    for b in range(n_boot):
        pick = rng.choice(scans, size=len(scans), replace=True)
        idx = np.concatenate([by_scan[s] for s in pick])
        stats[b] = ep_rate[idx].mean(), ep_sim[idx].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    return {
        "n_scans": len(scans),
        "cov_step": float(ep_rate.mean()),
        "cov_step_ci95": [float(lo[0]), float(hi[0])],
        "cov_simul": float(ep_sim.mean()),
        "cov_simul_ci95": [float(lo[1]), float(hi[1])],
    }


# ---------------------------------------------------- ties and latency
def tie_stats(cal: Split, score: str = "THR") -> Dict[str, float]:
    """Fraction of calibration episodes whose pf episode-max score exactly
    equals another episode's (fp16 logits make ties non-negligible; R1-Q3)."""
    s = cal.base_teacher[score] / (2.0 - cal.p_max)
    smax = np.array([s[a:b].max() for a, b in cal.ep_ptr])
    _, counts = np.unique(smax, return_counts=True)
    return {
        "n_episodes": len(smax),
        "tie_rate": float((counts[counts > 1]).sum() / len(smax)),
    }


def latency_us(
    test: Split,
    q: float,
    score: str = "THR",
    n_steps: int = 20000,
    seed: int = 0,
) -> Dict[str, float]:
    """Wall-clock cost of the deployed membership test per step: base scores
    over the candidates + threshold comparison (R2-W5)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(test), size=min(n_steps, len(test)), replace=False)
    probs = [1.0 - test.base_cands["THR"][i] for i in idx]  # softmax vectors
    pmax = test.p_max[idx]
    t0 = time.perf_counter()
    for p, pm in zip(probs, pmax):
        scores = base_scores_all(p)[score]
        _ = scores <= q * (2.0 - pm)
    dt = time.perf_counter() - t0
    return {"us_per_step": dt / len(idx) * 1e6, "n_timed": int(len(idx))}


# ---------------------------------------------------- per-dump driver
def run_baselines(dump_path: str) -> Dict[str, Any]:
    d = torch.load(dump_path, weights_only=True)
    cal = Split.from_records(d["cal"])
    test = Split.from_records(d["test"])
    q_pf = epmax_quantile(cal, WEIGHT_FAMILY["pf"](cal, 0.10, {}), "THR", 0.10)
    w_pf = WEIGHT_FAMILY["pf"](test, 0.10, {})
    return {
        "condition": d["condition"],
        "temp_scaled_base": temperature_scaled_base_cp(d["cal"], d["test"]),
        "weighted_cp": weighted_cp(cal, test),
        "mondrian": mondrian_by_degree(cal, test),
        "ci": cluster_bootstrap_ci(test, q_pf, w_pf, "THR"),
        "ties": tie_stats(cal),
        "latency": latency_us(test, q_pf),
    }
