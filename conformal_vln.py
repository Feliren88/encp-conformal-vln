"""Entry point for the Normalised-CP weight family (all subcommands).

output_v10. Standalone (no imports from output_v8/_ablations/output_v9).
Orchestration only; the work lives in cp_core/ (pure CP domain, CPU) and
vln_backends/ (GPU/simulator side, imported lazily by `run` only).

GPU (one condition end-to-end: build -> argmax rollout on both splits ->
SR sanity gate -> dumps/<cond>.pt -> weight-family CP -> results/):

  PY=/home/vfeliren1/pr65_scratch2/vfvic1/conda/envs/vln_duet_conformal/bin/python
  $PY conformal_vln.py run --backend duet --action_space full          # R2R
  $PY conformal_vln.py run --backend duet --action_space local
  $PY conformal_vln.py run --backend hamt
  $PY conformal_vln.py run --backend recbert --recbert_variant prevalent
  $PY conformal_vln.py run --backend recbert --recbert_variant oscar
  $PY conformal_vln.py run --backend duet --dataset reverie            #
  REVERIE
  $PY conformal_vln.py run --backend hamt --dataset reverie

Offline (CPU, from dumps/*.pt):

  $PY conformal_vln.py analyze [--condition duet_full]  # ->
  results/cp_results.json
  $PY conformal_vln.py dense          # -> results/cp_dense.json
  $PY conformal_vln.py qualitative    # -> results/qualitative_episode.json
  $PY conformal_vln.py transfer | indist | figures | verify | test

RecBERT x REVERIE is an explicit placeholder: the public Recurrent-VLN-BERT
release is R2R-only (no REVERIE code or checkpoint was ever published).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

import cp_core as cp  # noqa: E402
from cp_core.reporting import make_figures, verify_results  # noqa: E402
from cp_core.tests import run_tests  # noqa: E402
from vln_backends.config import DUMP_DIR, RES_DIR, FIG_DIR  # noqa: E402


# ==============================================================================
# GPU: run one condition end-to-end
# ==============================================================================
def condition_name(a: argparse.Namespace) -> str:
    base = (
        f"duet_{a.action_space}"
        if a.backend == "duet"
        else (
            f"recbert_{a.recbert_variant}"
            if a.backend == "recbert"
            else "hamt"
        )
    )
    return base if a.dataset == "r2r" else f"{base}_{a.dataset}"


def cmd_run(a: argparse.Namespace) -> None:
    # vln_backends.bootstrap chdirs and imports MatterSim + DUET -- GPU only.
    import torch
    import vln_backends.bootstrap  # noqa: F401
    from vln_backends.builders import build_backend
    from vln_backends.rollouts import ROLLOUTS
    from vln_backends.metrics import sanity_gate

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(
        f"Python: {sys.version.split()[0]} | "
        f"PyTorch: {torch.__version__} | "
        f"CUDA: {gpu}"
    )
    cond = condition_name(a)
    os.makedirs(RES_DIR, exist_ok=True)
    t0 = time.time()

    print(f"[v10] building {cond} ...", flush=True)
    agent, cal_env, test_env, args, kind = build_backend(
        a.backend, a.action_space, a.recbert_variant, a.base_dir, a.dataset
    )
    rollout = ROLLOUTS[kind]
    if a.max_episodes > 0:
        cal_env.data = cal_env.data[: a.max_episodes]
        test_env.data = test_env.data[: a.max_episodes]
        print(
            f"[v10] DEV: capped to {a.max_episodes} episodes/split", flush=True
        )

    print("[v10] rolling out calibration split ...", flush=True)
    cal, _, cal_obj = rollout(agent, cal_env, args)
    print(
        f"[v10] calibration done: {len(cal)} episodes "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    print("[v10] rolling out test split ...", flush=True)
    test, test_preds, test_obj = rollout(agent, test_env, args)
    print(
        f"[v10] test done: {len(test)} episodes ({time.time() - t0:.0f}s)",
        flush=True,
    )

    sanity = sanity_gate(test_preds, test_env, a.dataset, cond)

    if not a.no_dump:
        torch.save(
            {
                "condition": cond,
                "cal": cal,
                "test": test,
                "cal_obj": cal_obj,
                "test_obj": test_obj,
                "sanity": sanity,
            },
            os.path.join(DUMP_DIR, f"{cond}.pt"),
        )
        print(f"[v10] dump -> dumps/{cond}.pt", flush=True)

    print("[v10] computing CP weight family (CPU) ...", flush=True)
    result = {
        "condition": cond,
        "sanity": sanity,
        **cp.evaluate_condition(
            cp.Split.from_records(cal),
            cp.Split.from_records(test),
            seed=a.seed,
        ),
    }
    if cal_obj and test_obj:
        result["object_cp"] = cp.evaluate_object_head(cal_obj, test_obj)
        o = result["object_cp"]["0.10"]["THR"]
        print(
            f"[v10] {cond} OBJECT-HEAD CP (THR, a=0.10): "
            f"base cov={o['base']['cov']:.3f} | "
            f"norm cov={o['norm']['cov']:.3f} "
            f"|C|={o['norm']['mean_set']:.2f}",
            flush=True,
        )

    with open(os.path.join(RES_DIR, f"{cond}.json"), "w") as f:
        json.dump(result, f, indent=2)
    pf_full = result["0.10"]["family_full"]["THR"]["pf"]
    family = result["0.10"]["family"]["THR"]
    print(
        f"[v10] {cond} (THR, a=0.10)  pf-full: "
        f"cov_step={pf_full['cov_step']:.3f} "
        f"cov_simul={pf_full['cov_simul']:.3f} |C|={pf_full['mean_set']:.2f}",
        flush=True,
    )
    print(
        "[v10] split-matched family cov_step: "
        + "  ".join(f"{v}={family[v]['cov_step']:.3f}" for v in cp.WEIGHTS),
        flush=True,
    )
    print(
        f"[v10] done in {(time.time() - t0) / 60:.1f} min "
        f"-> results/{cond}.json",
        flush=True,
    )


# ==============================================================================
# Offline: analyses from dumps
# ==============================================================================
def _dump_paths(dump_dir: str, condition: Optional[str]) -> List[str]:
    paths = (
        [os.path.join(dump_dir, f"{condition}.pt")]
        if condition
        else sorted(glob.glob(os.path.join(dump_dir, "*.pt")))
    )
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit(f"No dumps in {dump_dir}")
    os.makedirs(RES_DIR, exist_ok=True)
    return paths


def cmd_analyze(a: argparse.Namespace) -> None:
    results = []
    for p in _dump_paths(a.dump_dir, a.condition):
        r = cp.run_condition(p, seed=a.seed)
        results.append(r)
        pf = r["0.10"]["family_full"]["THR"]["pf"]
        print(
            f"{r['condition']:22s} pf THR a=0.10 "
            f"cov_step={pf['cov_step']:.3f} "
            f"cov_simul={pf['cov_simul']:.3f} |C|={pf['mean_set']:.2f} "
            f"sat={pf['saturation']:.2f} dTV={r['shift']['dTV_score']:.3f}",
            flush=True,
        )
    with open(os.path.join(RES_DIR, "cp_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {len(results)} conditions -> results/cp_results.json")


def cmd_dense(a: argparse.Namespace) -> None:
    import torch

    results = []
    for p in _dump_paths(a.dump_dir, a.condition):
        d = torch.load(p, weights_only=True)
        r = {
            "condition": d["condition"],
            **cp.dense_sweep(
                cp.Split.from_records(d["cal"]),
                cp.Split.from_records(d["test"]),
            ),
        }
        results.append(r)
        n_pass = sum(
            1
            for al in r["zeroshot"]
            for s in cp.SCORES
            if r["zeroshot"][al][s]["cov_step"] >= 1 - float(al)
        )
        print(
            f"{r['condition']:22s} zeroshot clears target in "
            f"{n_pass}/{len(r['zeroshot']) * len(cp.SCORES)} dense cells",
            flush=True,
        )
    with open(os.path.join(RES_DIR, "cp_dense.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {len(results)} conditions -> results/cp_dense.json")


def cmd_qualitative(a: argparse.Namespace) -> None:
    path = _dump_paths(a.dump_dir, a.condition)[0]
    r = cp.qualitative_episode(path, instr_id=a.instr_id)
    with open(os.path.join(RES_DIR, "qualitative_episode.json"), "w") as f:
        json.dump(r, f, indent=2)
    print(
        f"[qualitative] {r['instr_id']} q_hat={r['q_hat']:.4f} "
        f"{len(r['steps'])} steps -> results/qualitative_episode.json"
    )


def cmd_baselines(a: argparse.Namespace) -> None:
    """REVIEW.md #5-8, #11, #13: temperature scaling, weighted CP, Mondrian,
    scan-cluster CIs, tie rate, latency -- all offline from dumps."""
    from cp_core.baselines import run_baselines

    rows = []
    for p in _dump_paths(a.dump_dir, a.condition):
        r = run_baselines(p)
        rows.append(r)
        ts = r["temp_scaled_base"]
        print(
            f"[{r['condition']}] T={ts['temperature']:.2f} "
            f"tempAPS@0.30: q={ts['0.30']['APS']['q']:.3f} "
            f"sgl={ts['0.30']['APS']['singleton']:.2f} | "
            f"weightedCP cov_simul@0.10="
            f"{r['weighted_cp']['0.10']['cov_simul']:.3f} "
            f"(ess={r['weighted_cp']['ess_cal']:.0f}) | "
            f"CI cov_step={r['ci']['cov_step']:.3f} "
            f"[{r['ci']['cov_step_ci95'][0]:.3f},"
            f"{r['ci']['cov_step_ci95'][1]:.3f}] | "
            f"ties={r['ties']['tie_rate']:.3f} "
            f"lat={r['latency']['us_per_step']:.0f}us",
            flush=True,
        )
    with open(os.path.join(RES_DIR, "baselines.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[done] {len(rows)} conditions -> results/baselines.json")


def cmd_closedloop(a: argparse.Namespace) -> None:
    """REVIEW.md #15: closed-loop help-seeking on DUET-full R2R. A simulated
    operator supplies the teacher action when the trigger fires; sweep both
    triggers and report SR / SPL vs ask-rate."""
    import torch
    import vln_backends.bootstrap  # noqa: F401
    from vln_backends.builders import build_backend
    from vln_backends.rollouts import closedloop_duet_split
    from vln_backends.metrics import compute_nav_metrics

    d = torch.load(os.path.join(DUMP_DIR, "duet_full.pt"), weights_only=True)
    cal = cp.Split.from_records(d["cal"])
    q_hat = cp.epmax_quantile(
        cal, cp.WEIGHT_FAMILY["pf"](cal, a.alpha, {}), "THR", a.alpha
    )
    print(
        f"[closedloop] q_hat(pf, THR, a={a.alpha:.2f}) = {q_hat:.4f}",
        flush=True,
    )
    agent, _, test_env, args, _ = build_backend(
        "duet", "full", "prevalent", a.base_dir, "r2r"
    )
    if a.max_episodes > 0:
        test_env.data = test_env.data[: a.max_episodes]
    policies = (
        [("none", 0.0)]
        + [("set", tau) for tau in a.tau_grid]
        + [("pmax", c) for c in a.pmax_grid]
    )
    rows = []
    for trigger, param in policies:
        t0 = time.time()
        preds, n_asks, n_steps = closedloop_duet_split(
            agent, test_env, args, q_hat, trigger, param
        )
        m = compute_nav_metrics(preds, test_env)
        row = {
            "trigger": trigger,
            "param": param,
            "ask_rate": n_asks / max(n_steps, 1),
            "n_asks": n_asks,
            "n_steps": n_steps,
            "sr": m["sr"],
            "spl": m["spl"],
        }
        rows.append(row)
        print(
            f"[closedloop] {trigger:5s} param={param:<4} "
            f"ask={row['ask_rate']:.3f} SR={m['sr']:.2f} SPL={m['spl']:.2f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    out = {
        "alpha": a.alpha,
        "q_hat": q_hat,
        "condition": "duet_full",
        "policies": rows,
    }
    with open(os.path.join(RES_DIR, "closedloop.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] -> results/closedloop.json")


def cmd_transfer(a: argparse.Namespace) -> None:
    paths = [
        p
        for p in _dump_paths(a.dump_dir, None)
        if "reverie" not in os.path.basename(p)
    ]
    res = cp.threshold_transfer(paths)
    with open(os.path.join(RES_DIR, "transfer.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res["q_hat"], indent=2))
    print("[transfer] -> results/transfer.json")


def cmd_indist(a: argparse.Namespace) -> None:
    targets = {"0.10": 0.90, "0.20": 0.80, "0.30": 0.70}
    rows = []
    for p in _dump_paths(a.dump_dir, a.condition):
        nav = {k: [] for k in targets}
        obj = {k: [] for k in targets}
        cond = None
        for sd in range(a.indist_seeds):
            r = cp.evaluate_indist(p, seed=sd)
            cond = r["condition"]
            for k in targets:
                nav[k].append(r["nav"][k]["THR"]["cov_step"])
                if r["object"]:
                    obj[k].append(r["object"][k]["THR"]["norm"]["cov"])
        row = {
            "condition": cond,
            "nav": {k: float(np.mean(nav[k])) for k in targets},
            "object": {k: float(np.mean(obj[k])) for k in targets if obj[k]},
        }
        rows.append(row)
        print(
            f"[{cond}] "
            + "  ".join(
                f"a={k}: nav={row['nav'][k]:.3f}"
                + (
                    f" obj={row['object'][k]:.3f}"
                    if row["object"].get(k)
                    else ""
                )
                for k in targets
            )
        )
    with open(os.path.join(RES_DIR, "indist.json"), "w") as f:
        json.dump(rows, f, indent=2)


# ==============================================================================
# CLI
# ==============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Normalised-CP weight family for VLN (output_v10)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="GPU: one condition end-to-end")
    run.add_argument(
        "--backend", required=True, choices=["duet", "hamt", "recbert"]
    )
    run.add_argument(
        "--action_space", default="full", choices=["full", "local"]
    )
    run.add_argument(
        "--recbert_variant",
        default="prevalent",
        choices=["prevalent", "oscar"],
    )
    run.add_argument("--dataset", default="r2r", choices=["r2r", "reverie"])
    run.add_argument("--base_dir", default="/fs04/scratch2/pr65/vfvic1/thesis")
    run.add_argument("--no_dump", action="store_true")
    run.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="dev: cap episodes per split (0 = all)",
    )
    run.add_argument("--seed", type=int, default=0)
    run.set_defaults(fn=cmd_run)

    for name, fn, help_ in (
        ("analyze", cmd_analyze, "weight-family grid -> cp_results.json"),
        ("dense", cmd_dense, "dense alpha sweep -> cp_dense.json"),
        (
            "qualitative",
            cmd_qualitative,
            "worked episode -> qualitative_episode.json",
        ),
        ("transfer", cmd_transfer, "cross-backbone q-hat -> transfer.json"),
        ("indist", cmd_indist, "in-distribution check -> indist.json"),
        (
            "baselines",
            cmd_baselines,
            "review baselines (temp/weighted/Mondrian/CI) -> baselines.json",
        ),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--dump_dir", default=DUMP_DIR)
        p.add_argument("--condition", default=None)
        p.add_argument("--seed", type=int, default=0)
        if name == "qualitative":
            p.add_argument("--instr_id", default=None)
        if name == "indist":
            p.add_argument("--indist_seeds", type=int, default=20)
        p.set_defaults(fn=fn)

    cl = sub.add_parser(
        "closedloop",
        help="GPU: help-seeking rollout (DUET-full R2R) " "-> closedloop.json",
    )
    cl.add_argument("--alpha", type=float, default=0.10)
    cl.add_argument(
        "--tau_grid", type=float, nargs="*", default=[1, 2, 3, 4, 6, 8]
    )
    cl.add_argument(
        "--pmax_grid", type=float, nargs="*", default=[0.3, 0.5, 0.7, 0.8, 0.9]
    )
    cl.add_argument("--base_dir", default="/fs04/scratch2/pr65/vfvic1/thesis")
    cl.add_argument("--max_episodes", type=int, default=0)
    cl.set_defaults(fn=cmd_closedloop)

    fig = sub.add_parser("figures", help="aggregate HTML figures")
    fig.set_defaults(
        fn=lambda a: make_figures(
            os.path.join(RES_DIR, "cp_results.json"),
            os.path.join(FIG_DIR, "figures.html"),
        )
    )
    pf = sub.add_parser(
        "paperfigs", help="print PNGs for paper/figures (matplotlib)"
    )
    pf.set_defaults(
        fn=lambda a: __import__("paper_figures").make_all(
            RES_DIR,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "paper", "figures"
            ),
        )
    )
    ver = sub.add_parser(
        "verify", help="assert paper claims vs cp_results.json"
    )
    ver.set_defaults(
        fn=lambda a: sys.exit(
            1
            if verify_results(os.path.join(RES_DIR, "cp_results.json"))
            else 0
        )
    )
    tst = sub.add_parser("test", help="unit tests (CPU, seconds)")
    tst.set_defaults(fn=lambda a: sys.exit(1 if run_tests() else 0))

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
