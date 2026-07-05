"""Condition-level analyses: family grid, shift, transfer, in-dist.

object head, cross-backbone threshold transfer, and the in-distribution
check. Everything here consumes Splits (or raw dump dicts) and returns
JSON-serialisable results.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from cp_core.scores import (
    SCORES,
    EPS,
    base_scores_all,
    conformal_quantile,
    softmax_valid,
)
from cp_core.split import Split
from cp_core.weights import (
    WEIGHT_FAMILY,
    WEIGHTS,
    PARAMETER_FREE,
    fit_weight_models,
)
from cp_core.evaluation import (
    epmax_quantile,
    pooled_quantile,
    evaluate,
    conditional_diagnostics,
)

ALPHAS = (0.10, 0.20, 0.30)


def evaluate_condition(
    cal: Split, test: Split, alphas=ALPHAS, seed: int = 0
) -> Dict[str, Any]:
    """The full weight-family grid for one condition.

    Per alpha, per score:
      base           step-pooled base CP (the collapsing baseline)
      family[w]      every member, split-matched (quantile on cal half 2)
      family_full[w] parameter-free members on the FULL calibration set
    Plus shift estimates and conditional diagnostics at the headline setting
    (pf, THR, alpha = 0.10, full-cal).
    """
    _, h2 = cal.halves()
    out: Dict[str, Any] = {
        "n_cal_ep": cal.n_episodes,
        "n_test_ep": test.n_episodes,
        "n_cal_half2_ep": len(h2),
        "shift": dtv_plugin(cal, test),
        "shift_sensitivity": dtv_sensitivity(cal, test),
    }
    for alpha in alphas:
        models = fit_weight_models(cal, alpha, seed=seed)
        w_cal = {v: WEIGHT_FAMILY[v](cal, alpha, models) for v in WEIGHTS}
        w_test = {v: WEIGHT_FAMILY[v](test, alpha, models) for v in WEIGHTS}
        block: Dict[str, Any] = {"base": {}, "family": {}, "family_full": {}}
        for score in SCORES:
            block["base"][score] = evaluate(
                test,
                pooled_quantile(cal, score, alpha),
                np.zeros(len(test)),
                score,
            )
            block["family"][score] = {
                v: evaluate(
                    test,
                    epmax_quantile(cal, w_cal[v], score, alpha, h2),
                    w_test[v],
                    score,
                )
                for v in WEIGHTS
            }
            block["family_full"][score] = {
                v: evaluate(
                    test,
                    epmax_quantile(cal, w_cal[v], score, alpha),
                    w_test[v],
                    score,
                )
                for v in PARAMETER_FREE
            }
        out[f"{alpha:.2f}"] = block

    w_pf = WEIGHT_FAMILY["pf"](test, 0.10, {})
    q_pf = epmax_quantile(cal, WEIGHT_FAMILY["pf"](cal, 0.10, {}), "THR", 0.10)
    out["diagnostics"] = conditional_diagnostics(test, q_pf, w_pf, "THR")
    return out


def dtv_plugin(cal: Split, test: Split, bins: int = 50) -> Dict[str, float]:
    """Plug-in total variation between calibration and test.

    Score piece: histogram of the THR teacher score (1 - p_teacher) on [0,1].
    Degree piece: histogram of |A_t| (attribution lower bound).
    """
    edges = np.linspace(0, 1, bins + 1)
    pc, _ = np.histogram(1.0 - cal.p_teacher, bins=edges)
    qt, _ = np.histogram(1.0 - test.p_teacher, bins=edges)
    pc = pc / max(pc.sum(), 1)
    qt = qt / max(qt.sum(), 1)
    m = int(max(cal.degree.max(), test.degree.max())) + 1
    hc = np.bincount(cal.degree, minlength=m) / max(len(cal), 1)
    ht = np.bincount(test.degree, minlength=m) / max(len(test), 1)
    return {
        "dTV_score": float(0.5 * np.abs(pc - qt).sum()),
        "dTV_degree": float(0.5 * np.abs(hc - ht).sum()),
    }


def dtv_sensitivity(
    cal: Split, test: Split, bin_grid=(20, 40, 50, 80, 100)
) -> Dict[str, float]:
    return {
        str(b): dtv_plugin(cal, test, bins=b)["dTV_score"] for b in bin_grid
    }


# ---- object-grounding head (REVERIE): one classification per episode --------
def _object_step(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ti = int(rec["teacher_idx"])
    lg = rec["obj_logits"]
    lg = (
        lg.float().numpy()
        if isinstance(lg, torch.Tensor)
        else np.asarray(lg, float)
    )
    if ti < 0 or lg.size == 0 or ti >= lg.size:
        return None
    e = np.exp(lg - lg.max())
    p = np.clip(e / e.sum(), EPS, 1.0)
    scores = base_scores_all(p)
    return {
        "scores": scores,
        "teacher": {k: float(scores[k][ti]) for k in SCORES},
        "p_max": float(p.max()),
    }


def evaluate_object_head(
    cal_obj: Dict[str, Any], test_obj: Dict[str, Any], alphas=ALPHAS
) -> Dict[str, Any]:
    """Plain split CP on the single-shot grounding classifier (episodes are
    exchangeable, no episode-max): base vs the parameter-free normalisation."""
    cal = [f for f in map(_object_step, cal_obj.values()) if f]
    test = [f for f in map(_object_step, test_obj.values()) if f]

    def metrics(q: float, score: str, normalised: bool) -> Dict[str, float]:
        cov, sizes = [], []
        for f in test:
            thr = q * (2.0 - f["p_max"]) if normalised else q
            sizes.append(max(int(np.sum(f["scores"][score] <= thr)), 1))
            cov.append(bool(f["teacher"][score] <= thr))
        if not test:
            return {"q": q, "cov": 0.0, "mean_set": 0.0, "singleton": 0.0}
        return {
            "q": float(q),
            "cov": float(np.mean(cov)),
            "mean_set": float(np.mean(sizes)),
            "singleton": float(np.mean(np.asarray(sizes) == 1)),
        }

    out: Dict[str, Any] = {
        "n_cal": len(cal_obj),
        "n_test": len(test_obj),
        "teacher_present_rate": (
            sum(int(r["teacher_idx"]) >= 0 for r in test_obj.values())
            / max(len(test_obj), 1)
        ),
    }
    for alpha in alphas:
        out[f"{alpha:.2f}"] = {
            score: {
                "base": metrics(
                    conformal_quantile(
                        [f["teacher"][score] for f in cal], alpha
                    ),
                    score,
                    False,
                ),
                "norm": metrics(
                    conformal_quantile(
                        [
                            f["teacher"][score] / (2.0 - f["p_max"])
                            for f in cal
                        ],
                        alpha,
                    ),
                    score,
                    True,
                ),
            }
            for score in SCORES
        }
    return out


# ---- cross-backbone threshold transfer
# ---------------------------------------
def threshold_transfer(
    dump_paths: List[str], score: str = "THR", alpha: float = 0.10
) -> Dict[str, Any]:
    """Calibrate q_hat (pf, full cal) on each condition, apply to every other
    condition's test split: the quantitative transfer experiment."""
    tests, q_hat = {}, {}
    for path in dump_paths:
        d = torch.load(path, weights_only=True)
        cal = Split.from_records(d["cal"])
        tests[d["condition"]] = Split.from_records(d["test"])
        q_hat[d["condition"]] = epmax_quantile(
            cal, WEIGHT_FAMILY["pf"](cal, alpha, {}), score, alpha
        )
    matrix: Dict[str, Dict[str, Dict[str, float]]] = {}
    for src, q in q_hat.items():
        matrix[src] = {}
        for tgt, split in tests.items():
            m = evaluate(
                split, q, WEIGHT_FAMILY["pf"](split, alpha, {}), score
            )
            matrix[src][tgt] = {
                "cov_step": m["cov_step"],
                "mean_set": m["mean_set"],
            }
    return {"score": score, "alpha": alpha, "q_hat": q_hat, "matrix": matrix}


# ---- in-distribution check
# ---------------------------------------------------
def evaluate_indist(
    dump_path: str, seed: int = 0, alphas=ALPHAS
) -> Dict[str, Any]:
    """Calibrate and test on exchangeable val_unseen halves. Isolates the
    seen->unseen shift as the only cause of undercoverage."""
    d = torch.load(dump_path, weights_only=True)

    def shuffled_halves(
        mapping: Dict[str, Any], salt: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ids = sorted(mapping.keys())
        np.random.RandomState(seed + salt).shuffle(ids)
        half = len(ids) // 2
        return (
            {k: mapping[k] for k in ids[:half]},
            {k: mapping[k] for k in ids[half:]},
        )

    cal_recs, test_recs = shuffled_halves(d["test"], 0)
    cal, tst = Split.from_records(cal_recs), Split.from_records(test_recs)
    out: Dict[str, Any] = {
        "condition": d["condition"],
        "nav": {},
        "object": None,
    }
    for alpha in alphas:
        out["nav"][f"{alpha:.2f}"] = {
            score: evaluate(
                tst,
                epmax_quantile(
                    cal, WEIGHT_FAMILY["pf"](cal, alpha, {}), score, alpha
                ),
                WEIGHT_FAMILY["pf"](tst, alpha, {}),
                score,
            )
            for score in SCORES
        }
    if d.get("test_obj"):
        cal_obj, test_obj = shuffled_halves(d["test_obj"], 1)
        out["object"] = evaluate_object_head(cal_obj, test_obj, alphas)
    return out


# ---- dense alpha sweep (the paper's "135 cells" claim)
# ------------------------
DENSE_ALPHAS = tuple(round(0.05 * k, 2) for k in range(1, 10))  # 0.05 .. 0.45


def dense_sweep(
    cal: Split, test: Split, alphas=DENSE_ALPHAS
) -> Dict[str, Any]:
    """Base CP vs the parameter-free score (full calibration set) over a dense
    alpha grid. Matches output_v9's cp_dense.json schema {base, zeroshot}."""
    out: Dict[str, Any] = {"base": {}, "zeroshot": {}}
    for alpha in alphas:
        a = f"{alpha:.2f}"
        out["base"][a], out["zeroshot"][a] = {}, {}
        w_cal = WEIGHT_FAMILY["pf"](cal, alpha, {})
        w_test = WEIGHT_FAMILY["pf"](test, alpha, {})
        for score in SCORES:
            out["base"][a][score] = evaluate(
                test,
                pooled_quantile(cal, score, alpha),
                np.zeros(len(test)),
                score,
            )
            out["zeroshot"][a][score] = evaluate(
                test, epmax_quantile(cal, w_cal, score, alpha), w_test, score
            )
    return out


# ---- qualitative worked episode (the paper's fig_qualitative source)
# ----------
def qualitative_episode(
    dump_path: str,
    instr_id: Optional[str] = None,
    score: str = "THR",
    alpha: float = 0.10,
) -> Dict[str, Any]:
    """Per-step trace of one test episode at the deployed pf threshold.

    Auto-selection (when instr_id is None) matches the paper's narrative: the
    episode starts with a confident singleton step and has the most
    confidently-wrong steps whose teacher the widened set still contains.
    """
    d = torch.load(dump_path, weights_only=True)
    cal = Split.from_records(d["cal"])
    q = epmax_quantile(cal, WEIGHT_FAMILY["pf"](cal, alpha, {}), score, alpha)

    def trace(recs) -> List[Dict[str, Any]]:
        steps = []
        for r in recs:
            lg = r["logits"]
            lg = (
                lg.float().numpy()
                if isinstance(lg, torch.Tensor)
                else np.asarray(lg, float)
            )
            p, valid = softmax_valid(lg)
            scores = base_scores_all(p)[score]
            p_max = float(p.max())
            thr = q * (2.0 - p_max)
            ti = int(r["teacher_idx"])
            in_cands = 0 <= ti < len(lg) and bool(valid[ti])
            t_pos = int(valid[:ti].sum()) if in_cands else -1
            steps.append(
                {
                    "step": int(r.get("step", len(steps))),
                    "n_valid": int(valid.sum()),
                    "p_max": p_max,
                    "set_size": max(int(np.sum(scores <= thr)), 1),
                    "teacher_in": bool(t_pos >= 0 and scores[t_pos] <= thr),
                    "argmax_err": bool(int(np.argmax(p)) != t_pos),
                    "scan": str(r.get("scan", "?")),
                }
            )
        return steps

    test = d["test"]
    if instr_id is None:

        def narrative_score(steps: List[Dict[str, Any]]) -> int:
            if not steps or steps[0]["set_size"] != 1:
                return -1
            return sum(1 for s in steps if s["argmax_err"] and s["teacher_in"])

        instr_id = max(test, key=lambda k: narrative_score(trace(test[k])))
    return {
        "instr_id": instr_id,
        "q_hat": float(q),
        "steps": trace(test[instr_id]),
    }


def run_condition(path: str, seed: int = 0) -> Dict[str, Any]:
    """Full analysis of one dump file."""
    d = torch.load(path, weights_only=True)
    out = {
        "condition": d["condition"],
        "sanity": d.get("sanity", {}),
        **evaluate_condition(
            Split.from_records(d["cal"]),
            Split.from_records(d["test"]),
            seed=seed,
        ),
    }
    if d.get("cal_obj") and d.get("test_obj"):
        out["object_cp"] = evaluate_object_head(d["cal_obj"], d["test_obj"])
    return out
