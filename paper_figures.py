"""Print figures for the paper (matplotlib, IEEE column, 300 dpi PNG).

Presentation layer only: consumes results/*.json, writes paper/figures/.

Palette: reference categorical slots 1/2/6 (validated: CVD worst-pair
dE 21.2, all checks pass; the aqua contrast WARN is relieved by direct
labels + marker/linestyle secondary encoding on every series).
One axis per panel -- the qualitative trace uses two stacked panels
instead of a twin axis.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, AQUA, RED = "#2a78d6", "#1baf7a", "#e34948"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
ALPHAS = [0.10, 0.20, 0.30]
KEYS = ["0.10", "0.20", "0.30"]

plt.rcParams.update(
    {
        "font.size": 7.0,
        "axes.titlesize": 7.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.6,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


def _load(res_dir: str, name: str) -> Any:
    with open(os.path.join(res_dir, name)) as f:
        return json.load(f)


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_closedloop(res_dir: str, out: str) -> None:
    """SR vs operator ask-rate: prediction-set trigger vs confidence trigger.
    The safety narrative figure: at a matched ask budget, which signal buys
    more success?"""
    d = _load(res_dir, "closedloop.json")
    rows = d["policies"]
    base = next(r for r in rows if r["trigger"] == "none")
    sets = sorted(
        (r for r in rows if r["trigger"] == "set"), key=lambda r: r["ask_rate"]
    )
    pmax = sorted(
        (r for r in rows if r["trigger"] == "pmax"),
        key=lambda r: r["ask_rate"],
    )
    fig, ax = plt.subplots(figsize=(2.45, 1.75))
    ax.axhline(base["sr"], color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.annotate(
        f"no help ({base['sr']:.1f})",
        xy=(0.985, base["sr"]),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=MUTED,
    )
    ax.plot(
        [r["ask_rate"] for r in sets],
        [r["sr"] for r in sets],
        color=BLUE,
        marker="o",
        ms=3.4,
        lw=1.4,
        label=r"set trigger $|C_\alpha(x_t)|>\tau$",
    )
    ax.plot(
        [r["ask_rate"] for r in pmax],
        [r["sr"] for r in pmax],
        color=AQUA,
        marker="s",
        ms=3.2,
        lw=1.4,
        ls="--",
        label=r"confidence trigger $p_{\max}<c$",
    )
    for r in sets:
        if int(r["param"]) in (1, 4, 8, 12, 15):
            ax.annotate(
                rf"$\tau{{=}}{int(r['param'])}$",
                (r["ask_rate"], r["sr"]),
                textcoords="offset points",
                xytext=(3, -9),
                fontsize=5.8,
                color=BLUE,
            )
    ax.set_xlabel("operator ask rate (fraction of steps)")
    ax.set_ylabel("success rate (%)")
    ax.set_xlim(left=-0.01)
    _despine(ax)
    fig.legend(
        loc="lower left",
        bbox_to_anchor=(0.13, 0.86),
        ncol=1,
        handlelength=1.6,
        borderaxespad=0.0,
        labelspacing=0.2,
        frameon=False,
    )
    fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.85])
    fig.savefig(out)
    plt.close(fig)


def fig_qualitative(res_dir: str, out: str) -> None:
    """Per-step trace of one val-unseen episode at the deployed threshold.
    Two stacked panels (set size vs action space; policy confidence) --
    counts and probabilities never share an axis."""
    d = _load(res_dir, "qualitative_episode.json")
    steps = d["steps"]
    t = [s["step"] for s in steps]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(2.45, 1.95), sharex=True, height_ratios=[3, 2]
    )
    ax1.bar(
        t,
        [s["set_size"] for s in steps],
        width=0.62,
        color=BLUE,
        label=r"$|C_\alpha(x_t)|$",
        zorder=3,
    )
    ax1.step(
        [x - 0.5 for x in t] + [t[-1] + 0.5],
        [steps[0]["n_valid"]] + [s["n_valid"] for s in steps],
        color=MUTED,
        lw=1.0,
        label=r"$|\mathcal{A}_t|$",
        zorder=4,
    )
    saved = [s for s in steps if s["argmax_err"] and s["teacher_in"]]
    ax1.plot(
        [s["step"] for s in saved],
        [s["set_size"] + 0.55 for s in saved],
        ls="none",
        marker="v",
        ms=3.6,
        color=RED,
        label="argmax wrong, teacher in set",
        zorder=5,
    )
    ax1.set_ylabel("actions")
    ax1.set_ylim(0, max(s["n_valid"] for s in steps) + 2)
    _despine(ax1)
    fig.legend(
        loc="lower left",
        bbox_to_anchor=(0.12, 0.80),
        ncol=2,
        handlelength=1.2,
        borderaxespad=0.0,
        labelspacing=0.2,
        columnspacing=0.8,
        frameon=False,
    )
    ax2.plot(
        t, [s["p_max"] for s in steps], color=AQUA, marker="o", ms=2.8, lw=1.3
    )
    ax2.annotate(
        r"$p_{\max}$",
        (t[-1], steps[-1]["p_max"]),
        textcoords="offset points",
        xytext=(-2, 8),
        color=AQUA,
        fontsize=7,
    )
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("confidence")
    ax2.set_xlabel(r"step $t$")
    ax2.set_xticks(t)
    _despine(ax2)
    fig.tight_layout(pad=0.4, h_pad=0.6, rect=[0, 0, 1, 0.80])
    fig.savefig(out)
    plt.close(fig)


def fig_reverie(res_dir: str, out: str) -> None:
    """REVERIE over the dense alpha grid: the navigation head keeps the
    guarantee under shift (left); the grounding head loses it to the shift
    and in-distribution recalibration restores it (right).

    Navigation curves come from cp_dense.json; the object-head curves are
    recomputed densely from the dumps (CPU, seconds)."""
    import numpy as np
    import cp_core as cp
    import torch

    dense = {r["condition"]: r for r in _load(res_dir, "cp_dense.json")}
    alphas = sorted(float(a) for a in dense["duet_full_reverie"]["zeroshot"])
    keys = [f"{a:.2f}" for a in alphas]
    conds = ("duet_full_reverie", "hamt_reverie")

    obj_shift, obj_indist = {}, {}
    dump_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dumps"
    )
    for cond in conds:
        d = torch.load(os.path.join(dump_dir, f"{cond}.pt"), weights_only=True)
        o = cp.evaluate_object_head(d["cal_obj"], d["test_obj"], alphas=alphas)
        obj_shift[cond] = [o[a]["THR"]["norm"]["cov"] for a in keys]
        runs = []
        for seed in range(5):
            r = cp.evaluate_indist(
                os.path.join(dump_dir, f"{cond}.pt"), seed=seed, alphas=alphas
            )
            runs.append([r["object"][a]["THR"]["norm"]["cov"] for a in keys])
        obj_indist[cond] = np.mean(runs, axis=0)

    def nav(cond: str) -> List[float]:
        return [dense[cond]["zeroshot"][a]["THR"]["cov_step"] for a in keys]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(2.45, 1.45), sharey=True)
    for ax in (axA, axB):
        ax.plot(
            alphas,
            [1 - a for a in alphas],
            color=MUTED,
            lw=0.9,
            ls=(0, (4, 3)),
            zorder=2,
        )
        ax.set_xticks([0.1, 0.2, 0.3, 0.4])
        ax.set_xlabel(r"$\alpha$")
        _despine(ax)
    axA.annotate(
        r"target $1{-}\alpha$",
        xy=(0.30, 0.66),
        fontsize=6.0,
        color=MUTED,
        rotation=-33,
        ha="center",
        va="top",
    )
    axA.plot(
        alphas,
        nav("duet_full_reverie"),
        color=BLUE,
        marker="o",
        ms=2.6,
        lw=1.2,
    )
    axA.plot(
        alphas, nav("hamt_reverie"), color=AQUA, marker="s", ms=2.4, lw=1.2
    )
    axA.annotate(
        "DUET",
        xy=(alphas[-1], nav("duet_full_reverie")[-1]),
        textcoords="offset points",
        xytext=(-20, -10),
        color=BLUE,
        fontsize=6.6,
    )
    axA.annotate(
        "HAMT",
        xy=(alphas[-1], nav("hamt_reverie")[-1]),
        textcoords="offset points",
        xytext=(-20, 5),
        color=AQUA,
        fontsize=6.6,
    )
    axA.set_title("navigation head", fontsize=7.2)
    axA.set_ylabel("coverage")
    axB.plot(
        alphas,
        obj_shift["duet_full_reverie"],
        color=BLUE,
        marker="o",
        ms=2.6,
        lw=1.2,
    )
    axB.plot(
        alphas,
        obj_shift["hamt_reverie"],
        color=AQUA,
        marker="s",
        ms=2.4,
        lw=1.2,
    )
    axB.plot(
        alphas,
        obj_indist["duet_full_reverie"],
        color=BLUE,
        marker="o",
        ms=2.6,
        lw=1.0,
        ls=":",
    )
    axB.plot(
        alphas,
        obj_indist["hamt_reverie"],
        color=AQUA,
        marker="s",
        ms=2.4,
        lw=1.0,
        ls=":",
    )
    axB.set_title("grounding head", fontsize=7.2)
    # linestyle carries the condition, hue carries the agent -- legend uses
    # neutral proxies so it does not read as DUET-only
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([], [], color=MUTED, lw=1.3, label="seen$\\to$unseen"),
            Line2D(
                [],
                [],
                color=MUTED,
                lw=1.1,
                ls=":",
                label="recalibrated in-dist.",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.55, 0.865),
        ncol=2,
        handlelength=1.5,
        borderaxespad=0.0,
        columnspacing=1.0,
        frameon=False,
    )
    axA.set_ylim(0.3, 1.02)
    fig.tight_layout(pad=0.4, w_pad=0.8, rect=[0, 0, 1, 0.84])
    fig.savefig(out)
    plt.close(fig)


def make_all(res_dir: str, fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    jobs = [
        ("fig_qualitative.png", fig_qualitative),
        ("fig_reverie.png", fig_reverie),
        ("fig_closedloop.png", fig_closedloop),
    ]
    for name, fn in jobs:
        try:
            fn(res_dir, os.path.join(fig_dir, name))
            print(f"[paperfigs] {name}")
        except FileNotFoundError as e:
            print(f"[paperfigs] SKIP {name}: missing input ({e.filename})")
