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
def _object_split(obj: Dict[str, Any]) -> Split:
    """One episode per object-grounding decision, reusing the `Split`
    abstraction: each episode has exactly one "step" (the terminal object
    classification). This makes the episode-max quantile, the weight
    family, and the half-1/half-2 learned-member bookkeeping identical to
    the navigation head -- no parallel implementation needed.

    Episodes without a valid teacher (`teacher_idx` negative, or out of
    range of the candidate logits) are dropped, matching the historical
    `_object_step` filter: `Split.from_records` otherwise scores a missing
    teacher via `teacher_fallback` (worst-case, 1.0 for THR) which is the
    right conservative choice for a per-step slot inside a multi-step nav
    episode, but wrong here -- REVERIE episodes commonly have no
    object-grounding teacher at all (`teacher_present_rate` < 1), and
    letting those saturate every score at 1.0 collapses the object-head
    thresholds to trivial full coverage."""
    records = {}
    for eid, rec in obj.items():
        ti = int(rec["teacher_idx"])
        lg = rec["obj_logits"]
        lg_arr = (
            lg.float().numpy()
            if isinstance(lg, torch.Tensor)
            else np.asarray(lg, float)
        )
        if ti < 0 or lg_arr.size == 0 or ti >= lg_arr.size:
            continue
        records[eid] = [
            {"logits": rec["obj_logits"], "teacher_idx": ti, "step": 0}
        ]
    return Split.from_records(records)


def _obj_metrics(e: Dict[str, float]) -> Dict[str, float]:
    """Pass through the full richer evaluate() dict (Task 1B: median/
    percentiles/set_degree_ratio/SE/weight stats) unchanged, adding one
    historical alias: `cov` = `cov_step` (with one step per episode,
    cov_step and cov_simul are identical; `cov` is the object head's
    long-standing key name and existing consumers read it)."""
    return {**e, "cov": e["cov_step"]}


def evaluate_object_head(
    cal_obj: Dict[str, Any],
    test_obj: Dict[str, Any],
    alphas=ALPHAS,
    seed: int = 0,
) -> Dict[str, Any]:
    """Split CP on the single-shot grounding classifier, reusing the same
    `Split`-based weight-family machinery as `evaluate_condition`: base
    (no normalisation), the full weight family split-matched on
    calibration half 2 (`family`), and the parameter-free members on the
    full calibration set (`family_full`). `base` and `norm` keep their
    historical full-cal values so existing consumers are unaffected;
    `norm` is numerically identical to `family_full["pf"]`."""
    cal = _object_split(cal_obj)
    test = _object_split(test_obj)
    out: Dict[str, Any] = {
        "n_cal": len(cal_obj),
        "n_test": len(test_obj),
        "teacher_present_rate": (
            sum(int(r["teacher_idx"]) >= 0 for r in test_obj.values())
            / max(len(test_obj), 1)
        ),
    }
    _, h2 = cal.halves()
    for alpha in alphas:
        models = fit_weight_models(cal, alpha, seed=seed)
        w_cal = {v: WEIGHT_FAMILY[v](cal, alpha, models) for v in WEIGHTS}
        w_test = {v: WEIGHT_FAMILY[v](test, alpha, models) for v in WEIGHTS}
        block: Dict[str, Any] = {}
        for score in SCORES:
            base = _obj_metrics(
                evaluate(
                    test,
                    pooled_quantile(cal, score, alpha),
                    np.zeros(len(test)),
                    score,
                )
            )
            family = {
                v: _obj_metrics(
                    evaluate(
                        test,
                        epmax_quantile(cal, w_cal[v], score, alpha, h2),
                        w_test[v],
                        score,
                    )
                )
                for v in WEIGHTS
            }
            family_full = {
                v: _obj_metrics(
                    evaluate(
                        test,
                        epmax_quantile(cal, w_cal[v], score, alpha),
                        w_test[v],
                        score,
                    )
                )
                for v in PARAMETER_FREE
            }
            block[score] = {
                "base": base,
                "norm": family_full["pf"],
                "family": family,
                "family_full": family_full,
            }
        out[f"{alpha:.2f}"] = block
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
DENSE_ALPHAS = tuple(round(0.05 * k, 2) for k in range(1, 11))  # 0.05 .. 0.50


def dense_sweep(
    cal: Split,
    test: Split,
    alphas=DENSE_ALPHAS,
    cal_obj: Optional[Dict[str, Any]] = None,
    test_obj: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Base CP vs the weight family over a dense alpha grid.

    `base`/`zeroshot` keep output_v9's original schema (full-cal pf) for
    backward compatibility. `family` adds the learned member (`mlp`,
    split-matched on calibration half 2) alongside `pf` so the score x
    weight-method comparison used by fig_reverie is available densely, for
    every dump (R2R and REVERIE alike -- no dataset-specific branching).
    When `cal_obj`/`test_obj` are given (REVERIE dumps), `object` adds the
    same base/{pf,mlp} comparison for the grounding head, reusing
    `_object_split` so the learned member is fit with the identical
    half-1/half-2 bookkeeping as the navigation head.
    """
    _, h2 = cal.halves()
    out: Dict[str, Any] = {"base": {}, "zeroshot": {}, "family": {}}
    for alpha in alphas:
        a = f"{alpha:.2f}"
        out["base"][a], out["zeroshot"][a], out["family"][a] = {}, {}, {}
        models = fit_weight_models(cal, alpha, seed=seed)
        w_cal_pf = WEIGHT_FAMILY["pf"](cal, alpha, {})
        w_test_pf = WEIGHT_FAMILY["pf"](test, alpha, {})
        w_cal_mlp = WEIGHT_FAMILY["mlp"](cal, alpha, models)
        w_test_mlp = WEIGHT_FAMILY["mlp"](test, alpha, models)
        for score in SCORES:
            out["base"][a][score] = evaluate(
                test,
                pooled_quantile(cal, score, alpha),
                np.zeros(len(test)),
                score,
            )
            out["zeroshot"][a][score] = evaluate(
                test,
                epmax_quantile(cal, w_cal_pf, score, alpha),
                w_test_pf,
                score,
            )
            out["family"][a][score] = {
                "pf": out["zeroshot"][a][score],
                "mlp": evaluate(
                    test,
                    epmax_quantile(cal, w_cal_mlp, score, alpha, h2),
                    w_test_mlp,
                    score,
                ),
            }

    if cal_obj and test_obj:
        obj_cal = _object_split(cal_obj)
        obj_test = _object_split(test_obj)
        _, obj_h2 = obj_cal.halves()
        out["object"] = {}
        for alpha in alphas:
            a = f"{alpha:.2f}"
            obj_models = fit_weight_models(obj_cal, alpha, seed=seed)
            ow_cal_pf = WEIGHT_FAMILY["pf"](obj_cal, alpha, {})
            ow_test_pf = WEIGHT_FAMILY["pf"](obj_test, alpha, {})
            ow_cal_mlp = WEIGHT_FAMILY["mlp"](obj_cal, alpha, obj_models)
            ow_test_mlp = WEIGHT_FAMILY["mlp"](obj_test, alpha, obj_models)
            out["object"][a] = {}
            for score in SCORES:
                base = _obj_metrics(
                    evaluate(
                        obj_test,
                        pooled_quantile(obj_cal, score, alpha),
                        np.zeros(len(obj_test)),
                        score,
                    )
                )
                pf = _obj_metrics(
                    evaluate(
                        obj_test,
                        epmax_quantile(obj_cal, ow_cal_pf, score, alpha),
                        ow_test_pf,
                        score,
                    )
                )
                mlp = _obj_metrics(
                    evaluate(
                        obj_test,
                        epmax_quantile(
                            obj_cal, ow_cal_mlp, score, alpha, obj_h2
                        ),
                        ow_test_mlp,
                        score,
                    )
                )
                out["object"][a][score] = {
                    "base": base,
                    "family": {"pf": pf, "mlp": mlp},
                }
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
