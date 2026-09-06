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

    from cp_core.analyses import DENSE_ALPHAS
    check(
        "dense alpha grid spans 0.05..0.50 in 10 steps",
        DENSE_ALPHAS == tuple(round(0.05 * k, 2) for k in range(1, 11)),
    )

    w_pf = WEIGHT_FAMILY["pf"](split, 0.10, {})
    check("pf weight in [0,1]", bool((w_pf >= 0).all() and (w_pf <= 1).all()))
    q = epmax_quantile(split, w_pf, "THR", 0.10)
    check(
        "in-sample pf coverage >= 0.90",
        evaluate(split, q, w_pf, "THR")["cov_step"] >= 0.90,
    )

    m = evaluate(split, q, w_pf, "THR")
    check(
        "evaluate(): richer stats present and consistent",
        m["median_set"] > 0
        and m["set_max"] >= m["median_set"] >= m["set_min"] > 0
        and m["n_steps"] == len(split)
        and m["n_episodes"] == split.n_episodes
        and 0.0 <= m["set_degree_ratio"] <= 1.0 + 1e-9
        and m["cov_step_se"] >= 0.0
        and m["weight_mean"] == float(np.mean(w_pf)),
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
    check(
        "object head: base/norm carry the historical keys plus the richer set",
        {"q", "cov", "mean_set", "median_set", "singleton", "set_degree_ratio"}
        <= set(o["0.10"]["THR"]["base"])
        and {"q", "cov", "mean_set", "median_set", "singleton"}
        <= set(o["0.10"]["THR"]["norm"]),
    )
    check(
        "object head: family has every weight member",
        set(o["0.10"]["THR"]["family"]) == set(WEIGHTS),
    )
    check(
        "object head: family_full is parameter-free only",
        set(o["0.10"]["THR"]["family_full"]) == set(PARAMETER_FREE),
    )
    check(
        "object head: norm equals family_full pf (both full-cal pf)",
        abs(
            o["0.10"]["THR"]["norm"]["cov"]
            - o["0.10"]["THR"]["family_full"]["pf"]["cov"]
        )
        < 1e-9,
    )

    from cp_core.analyses import dense_sweep, DENSE_ALPHAS

    dense = dense_sweep(split, split, alphas=(0.10, 0.50))
    check(
        "dense_sweep: base/zeroshot unchanged keys present",
        "base" in dense and "zeroshot" in dense,
    )
    check(
        "dense_sweep: new family block has pf and mlp",
        set(dense["family"]["0.10"]["THR"]) == {"pf", "mlp"},
    )
    check(
        "dense_sweep: zeroshot equals family pf",
        abs(
            dense["zeroshot"]["0.10"]["THR"]["cov_step"]
            - dense["family"]["0.10"]["THR"]["pf"]["cov_step"]
        )
        < 1e-9,
    )

    dense_obj = dense_sweep(
        split, split, alphas=(0.10,), cal_obj=obj, test_obj=obj
    )
    check(
        "dense_sweep: object block present when cal_obj/test_obj given",
        "object" in dense_obj
        and set(dense_obj["object"]["0.10"]["THR"]["family"]) == {"pf", "mlp"},
    )
    check(
        "dense_sweep: no object block when cal_obj/test_obj omitted",
        "object" not in dense,
    )

    # ---- Monte-Carlo validation of the finite-sample coverage guarantee ----
    # Theorem 1 predicts that, on exchangeable episodes, the marginal
    # whole-trajectory coverage equals k/(n+1) >= 1-alpha. We confirm this by
    # simulation, which is the empirical counterpart of the proof (and of the
    # in-distribution `indist` check on real data).

    # (a) Pure quantile lemma: n_cal+1 i.i.d. (hence exchangeable) episode
    # scores; q_hat is the k-th smallest of the first n_cal; the held-out score
    # is covered iff it falls at or below q_hat. Vectorised over many trials.
    rmc = np.random.RandomState(7)
    n_cal, n_trials, alpha = 200, 4000, 0.10
    k = int(np.ceil((n_cal + 1) * (1 - alpha)))
    pred = k / (n_cal + 1)
    Z = rmc.rand(n_trials, n_cal + 1)
    qhat = np.sort(Z[:, :n_cal], axis=1)[:, k - 1]
    emp = float((Z[:, n_cal] <= qhat).mean())
    check(
        f"MC quantile coverage {emp:.3f} >= 1-alpha ({1 - alpha:.2f})",
        emp >= (1 - alpha) - 0.02,
    )
    check(
        f"MC quantile coverage {emp:.3f} ~ k/(n+1) ({pred:.3f})",
        abs(emp - pred) <= 0.02,
    )

    # (b) End-to-end pipeline: a pool of exchangeable synthetic episodes;
    # repeatedly split into calibration/test and measure the SIMULTANEOUS
    # coverage of the parameter-free episode-max method. Exercises the reduced
    # score, the episode-maximum reduction, the corrected quantile, and the
    # coverage test together -- the same computations as the real pipeline.
    pool = Split.from_records(synthetic_records(n_ep=400, n_step=5, n_cand=6))
    s_pool = pool.base_teacher["THR"] / (2.0 - pool.p_max)
    ptr = pool.ep_ptr
    smax = np.array([s_pool[a:b].max() for a, b in ptr])
    nep, ncal = len(ptr), 200
    kk = int(np.ceil((ncal + 1) * (1 - alpha)))
    rp = np.random.RandomState(3)
    cov_sim = []
    for _ in range(80):
        perm = rp.permutation(nep)
        cal_e, test_e = perm[:ncal], perm[ncal:]
        q = float(np.sort(smax[cal_e])[kk - 1]) if kk <= ncal else np.inf
        for e in test_e:
            a, b = ptr[e]
            cov_sim.append(bool((s_pool[a:b] <= q).all()))
    emp2 = float(np.mean(cov_sim))
    check(
        f"MC pipeline simul coverage {emp2:.3f} >= 1-alpha ({1 - alpha:.2f})",
        emp2 >= (1 - alpha) - 0.03,
    )

    print(f"\n[test] {'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if run_tests() else 0)
