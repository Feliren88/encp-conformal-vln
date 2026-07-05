"""Unit tests for the CP core (pure CPU, synthetic data, seconds)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from cp_core.scores import (
    LAM,
    base_scores_all,
    softmax_valid,
    conformal_quantile,
)
from cp_core.split import Split
from cp_core.weights import (
    WEIGHT_FAMILY,
    WEIGHTS,
    PARAMETER_FREE,
    fit_weight_models,
)
from cp_core.evaluation import epmax_quantile, evaluate
from cp_core.analyses import evaluate_condition, evaluate_object_head


def run_tests() -> int:
    fails = 0

    def check(name: str, cond: bool) -> None:
        nonlocal fails
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        fails += 0 if cond else 1

    # n=9: k = ceil(10*(1-alpha)) exceeds n only when alpha < 1/(n+1) = 0.1
    check(
        "quantile: inf when k>n",
        conformal_quantile(range(9), 0.05) == float("inf"),
    )
    check(
        "quantile: k-th smallest",
        conformal_quantile([1, 2, 3, 4, 5, 6, 7, 8, 9], 0.5) == 5.0,
    )

    p = np.array([0.6, 0.3, 0.1])
    scores = base_scores_all(p)
    check("THR", np.allclose(scores["THR"], [0.4, 0.7, 0.9]))
    check("APS (U=0)", np.allclose(scores["APS"], [0.0, 0.6, 0.9]))
    check(
        "RAPS rank penalty", np.allclose(scores["RAPS"], [0.0, 0.6, 0.9 + LAM])
    )

    pv, valid = softmax_valid(np.array([1.0, -np.inf, 0.0]))
    check(
        "softmax_valid drops -inf",
        valid.tolist() == [True, False, True] and len(pv) == 2,
    )

    rng = np.random.RandomState(0)

    def synthetic_records(
        n_ep: int = 40, n_step: int = 5, n_cand: int = 6
    ) -> Dict[str, List[Dict[str, Any]]]:
        recs = {}
        for e in range(n_ep):
            steps = []
            for t in range(n_step):
                lg = rng.randn(n_cand) * 2
                lg[rng.randint(n_cand)] += 3.0
                steps.append(
                    {
                        "logits": lg,
                        "teacher_idx": int(np.argmax(lg)),
                        "step": t,
                        "scan": "s",
                    }
                )
            recs[f"ep{e}"] = steps
        return recs

    split = Split.from_records(synthetic_records())
    check("Split shape", split.n_episodes == 40 and len(split) == 200)

    w_pf = WEIGHT_FAMILY["pf"](split, 0.10, {})
    check("pf weight in [0,1]", bool((w_pf >= 0).all() and (w_pf <= 1).all()))
    q = epmax_quantile(split, w_pf, "THR", 0.10)
    check(
        "in-sample pf coverage >= 0.90",
        evaluate(split, q, w_pf, "THR")["cov_step"] >= 0.90,
    )

    models = fit_weight_models(split, 0.10, seed=0)
    for v in ("mlp", "hybrid", "random"):
        check(
            f"{v} weight >= 0",
            bool((WEIGHT_FAMILY[v](split, 0.10, models) >= 0).all()),
        )
    check(
        "hybrid >= pf floor",
        bool(
            (WEIGHT_FAMILY["hybrid"](split, 0.10, models) >= w_pf - 1e-9).all()
        ),
    )

    result = evaluate_condition(split, split, alphas=(0.10,), seed=0)
    check(
        "family complete", set(result["0.10"]["family"]["THR"]) == set(WEIGHTS)
    )
    check(
        "family_full is parameter-free only",
        set(result["0.10"]["family_full"]["THR"]) == set(PARAMETER_FREE),
    )
    check("diagnostics present", "query_budget" in result["diagnostics"])

    obj = {
        f"e{i}": {"obj_logits": rng.randn(8), "teacher_idx": i % 8}
        for i in range(60)
    }
    o = evaluate_object_head(obj, obj, alphas=(0.10,))
    check("object head runs", "0.10" in o and o["teacher_present_rate"] == 1.0)

    print(f"\n[test] {'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if run_tests() else 0)
