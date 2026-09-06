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
import numpy as np

matplotlib.use(os.environ.get("PAPERFIG_BACKEND", "Agg"))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Under the PGF backend labels are typeset by LaTeX, where a bare % opens a
# comment; the raster backends take the character literally.
_PCT = r"\%" if matplotlib.get_backend().lower() == "pgf" else "%"

BLUE, AQUA, RED = "#2a78d6", "#1baf7a", "#e34948"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
ALPHAS = [0.10, 0.20, 0.30]
KEYS = ["0.10", "0.20", "0.30"]

# Per-backbone colour / marker / linestyle for the appendix figures, whose
# legends are placed OUTSIDE the axes (bbox_inches="tight" on save) so nothing
# in the plotting area is ever covered.
ORDER = [
    "duet_full", "hamt", "recbert_prevalent",
    "recbert_oscar", "duet_full_reverie", "hamt_reverie",
]
R2R = ORDER[:4]
FLABEL = {
    "duet_full": "DUET", "hamt": "HAMT",
    "recbert_prevalent": "RecBERT-P", "recbert_oscar": "RecBERT-O",
    "duet_full_reverie": "DUET-REV", "hamt_reverie": "HAMT-REV",
}
STYLE = {
    "duet_full": ("#2a78d6", "o", "-"),
    "hamt": ("#e34948", "^", "-"),
    "recbert_prevalent": ("#7b52d0", "D", "--"),
    "recbert_oscar": ("#e08a1e", "v", "-"),
    "duet_full_reverie": ("#17a2b8", "P", ":"),
    "hamt_reverie": ("#c13aa0", "X", ":"),
}

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
    """Simulated success versus ENCP set-size-trigger ask rate."""
    d = _load(res_dir, "closedloop.json")
    rows = d["policies"]
    base = next(r for r in rows if r["trigger"] == "none")
    sets = sorted(
        (r for r in rows if r["trigger"] == "set"), key=lambda r: r["ask_rate"]
    )
    fig, ax = plt.subplots(figsize=(2.55, 2.20), facecolor="white")
    ax.set_facecolor("white")
    ax.text(
        0.02,
        0.96,
        "SIMULATED ORACLE",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        fontweight="bold",
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        zorder=10,
    )
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
        label=r"ENCP set-size trigger $|\bar C_\alpha(x_t)|>\tau$",
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
    ax.set_xlabel("Operator ask rate (fraction of steps)")
    ax.set_ylabel(f"Success rate ({_PCT})")
    ax.set_xlim(left=-0.01)
    _despine(ax)
    fig.legend(
        loc="lower left",
        bbox_to_anchor=(0.13, 0.84),
        ncol=1,
        handlelength=1.6,
        borderaxespad=0.0,
        labelspacing=0.2,
        frameon=False,
    )
    # Extra top and bottom room prevents the legend and axis label from being
    # clipped when the PGF fragment is included in a one-column float.
    fig.tight_layout(pad=0.65, rect=[0, 0.04, 1, 0.80])
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.08)
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


_REV_SCORE_COLOR = {"THR": BLUE, "APS": AQUA, "RAPS": RED}
_REV_WEIGHT_STYLE = {"base": ":", "pf": "-", "mlp": "--"}
_REV_WEIGHT_MARKER = {"base": None, "pf": "o", "mlp": None}


def _reverie_alphas(cond: Dict[str, Any]) -> tuple:
    alphas = sorted(float(a) for a in cond["family"])
    return alphas, [f"{a:.2f}" for a in alphas]


def _reverie_head_values(
    cond: Dict[str, Any], keys: List[str], head: str
) -> Dict[str, Dict[str, List[float]]]:
    """head is 'nav' or 'object'. Returns {score: {base, pf, mlp}}."""
    out: Dict[str, Dict[str, List[float]]] = {}
    for score in ("THR", "APS", "RAPS"):
        if head == "nav":
            out[score] = {
                "base": [cond["base"][a][score]["cov_step"] for a in keys],
                "pf": [
                    cond["family"][a][score]["pf"]["cov_step"]
                    for a in keys
                ],
                "mlp": [
                    cond["family"][a][score]["mlp"]["cov_step"]
                    for a in keys
                ],
            }
        else:
            out[score] = {
                "base": [
                    cond["object"][a][score]["base"]["cov"] for a in keys
                ],
                "pf": [
                    cond["object"][a][score]["family"]["pf"]["cov"]
                    for a in keys
                ],
                "mlp": [
                    cond["object"][a][score]["family"]["mlp"]["cov"]
                    for a in keys
                ],
            }
    return out


def _reverie_target_line(ax, alphas: List[float]) -> None:
    ax.plot(
        alphas,
        [1 - a for a in alphas],
        color=MUTED,
        lw=0.9,
        ls=(0, (4, 3)),
        zorder=2,
    )
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel(r"$\alpha$")
    _despine(ax)


def _reverie_plot_head(ax, alphas: List[float], values: Dict) -> None:
    for score in ("THR", "APS", "RAPS"):
        for weight in ("base", "pf", "mlp"):
            ax.plot(
                alphas,
                values[score][weight],
                color=_REV_SCORE_COLOR[score],
                ls=_REV_WEIGHT_STYLE[weight],
                lw=1.1,
                marker=_REV_WEIGHT_MARKER[weight],
                ms=2.0,
            )


def _reverie_legend_handles() -> List:
    from matplotlib.lines import Line2D

    return [
        Line2D([], [], color=_REV_SCORE_COLOR[s], lw=1.3, label=s)
        for s in ("THR", "APS", "RAPS")
    ] + [
        Line2D([], [], color=MUTED, lw=1.1, ls=":", label="base"),
        Line2D([], [], color=MUTED, lw=1.1, ls="-", label="formula-based"),
        Line2D([], [], color=MUTED, lw=1.1, ls="--", label="learned"),
    ]


def fig_reverie(res_dir: str, out: str) -> None:
    """REVERIE, DUET only, dense alpha grid: coverage for every base score
    (THR/APS/RAPS) with no intervention (base, dotted), the formula-based
    weight (pf, solid), and the learned weight (mlp, dashed), navigation
    head (left) and grounding head (right). The base lines make the ENCP
    before/after contrast explicit in the same panel."""
    dense = _load(res_dir, "cp_dense.json")
    cond = next(r for r in dense if r["condition"] == "duet_full_reverie")
    alphas, keys = _reverie_alphas(cond)
    nav_vals = _reverie_head_values(cond, keys, "nav")
    obj_vals = _reverie_head_values(cond, keys, "object")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(3.6, 1.7), sharey=True)
    for ax in (axA, axB):
        _reverie_target_line(ax, alphas)
    axA.annotate(
        r"Target $1{-}\alpha$",
        xy=(0.35, 0.60),
        fontsize=6.0,
        color=MUTED,
        rotation=-30,
        ha="center",
        va="top",
    )
    _reverie_plot_head(axA, alphas, nav_vals)
    _reverie_plot_head(axB, alphas, obj_vals)
    axA.set_title("Navigation head", fontsize=7.2)
    axA.set_ylabel("Coverage")
    axB.set_title("Grounding head", fontsize=7.2)

    fig.legend(
        handles=_reverie_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.55, 0.865),
        ncol=6,
        handlelength=1.5,
        borderaxespad=0.0,
        columnspacing=0.8,
        frameon=False,
    )
    axA.set_ylim(0.15, 1.02)
    fig.tight_layout(pad=0.4, w_pad=0.8, rect=[0, 0, 1, 0.84])
    fig.savefig(out)
    plt.close(fig)


def _fig_reverie_single_head(res_dir: str, out: str, head: str) -> None:
    dense = _load(res_dir, "cp_dense.json")
    cond = next(r for r in dense if r["condition"] == "duet_full_reverie")
    alphas, keys = _reverie_alphas(cond)
    values = _reverie_head_values(cond, keys, head)

    fig, ax = plt.subplots(figsize=(2.6, 2.1))
    _reverie_target_line(ax, alphas)
    ax.annotate(
        r"Target $1{-}\alpha$",
        xy=(0.35, 0.60),
        fontsize=6.0,
        color=MUTED,
        rotation=-30,
        ha="center",
        va="top",
    )
    _reverie_plot_head(ax, alphas, values)
    ax.set_title(
        "Navigation head" if head == "nav" else "Grounding head",
        fontsize=8.2,
    )
    ax.set_ylabel("Coverage")
    ax.legend(
        handles=_reverie_legend_handles(),
        fontsize=5.4,
        frameon=False,
        loc="lower left",
        ncol=2,
        handlelength=1.4,
        columnspacing=0.7,
    )
    ax.set_ylim(0.15, 1.02)
    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    plt.close(fig)


def fig_reverie_nav(res_dir: str, out: str) -> None:
    """Navigation head alone -- same data/lines as fig_reverie's left
    panel, standalone."""
    _fig_reverie_single_head(res_dir, out, "nav")


def fig_reverie_grounding(res_dir: str, out: str) -> None:
    """Grounding head alone -- same data/lines as fig_reverie's right
    panel, standalone."""
    _fig_reverie_single_head(res_dir, out, "object")


_ALL_CONDITIONS = (
    "duet_full", "hamt", "recbert_prevalent", "recbert_oscar",
    "duet_full_reverie", "hamt_reverie",
)
_COND_LABEL = {
    "duet_full": "DUET (R2R)",
    "hamt": "HAMT (R2R)",
    "recbert_prevalent": "RecBERT-PREV (R2R)",
    "recbert_oscar": "RecBERT-OSCAR (R2R)",
    "duet_full_reverie": "DUET (REVERIE)",
    "hamt_reverie": "HAMT (REVERIE)",
}
_SCORE_COLOR = {"THR": BLUE, "APS": AQUA, "RAPS": RED}


def _fig_cov_single(res_dir: str, out: str, cond: str, score: str) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dense = _load(res_dir, "cp_dense.json")
    row = next(r for r in dense if r["condition"] == cond)
    alphas = sorted(float(a) for a in row["family"])
    keys = [f"{a:.2f}" for a in alphas]
    fig, ax = plt.subplots(figsize=(2.7, 2.1))
    ax.plot(
        alphas, [1 - a for a in alphas], color=MUTED, lw=0.9,
        ls=(0, (4, 3)),
    )
    for weight, ls, mk, lbl in (
        ("pf", "-", "o", "Formula-based"),
        ("mlp", "--", None, "Learned"),
    ):
        ax.plot(
            alphas,
            [row["family"][a][score][weight]["cov_step"] for a in keys],
            color=_SCORE_COLOR[score], ls=ls, lw=1.3, marker=mk, ms=2.4,
            label=lbl,
        )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Coverage")
    ax.set_title(f"{_COND_LABEL[cond]} -- {score}", fontsize=8.2)
    _despine(ax)
    ax.legend(fontsize=6.0, frameon=False, loc="lower left")
    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    plt.close(fig)


def _fig_singleton_single(
    res_dir: str, out: str, cond: str, score: str
) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dense = _load(res_dir, "cp_dense.json")
    row = next(r for r in dense if r["condition"] == cond)
    show = [a for a in ("0.10", "0.20", "0.30", "0.40", "0.50")
            if a in row["family"]]
    methods = ("base", "pf", "mlp")
    method_color = {"base": MUTED, "pf": BLUE, "mlp": AQUA}
    x = np.arange(len(show))
    width = 0.25
    fig, ax = plt.subplots(figsize=(2.9, 2.1))
    for i, m in enumerate(methods):
        vals = [
            row["base"][a][score]["singleton"]
            if m == "base"
            else row["family"][a][score][m]["singleton"]
            for a in show
        ]
        ax.bar(
            x + (i - 1) * width, vals, width, label=m,
            color=method_color[m],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(show, fontsize=6.5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Singleton rate")
    ax.set_title(f"{_COND_LABEL[cond]} -- {score}", fontsize=8.2)
    _despine(ax)
    ax.legend(
        fontsize=6.0, frameon=False, loc="upper left",
        bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _fig_cov_base_only(res_dir: str, out: str, cond: str, score: str) -> None:
    """Coverage vs dense alpha for the raw base score alone -- no ENCP
    (formula-based or learned) intervention at all."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dense = _load(res_dir, "cp_dense.json")
    row = next(r for r in dense if r["condition"] == cond)
    alphas = sorted(float(a) for a in row["base"])
    keys = [f"{a:.2f}" for a in alphas]
    fig, ax = plt.subplots(figsize=(2.7, 2.1))
    ax.plot(
        alphas, [1 - a for a in alphas], color=MUTED, lw=0.9,
        ls=(0, (4, 3)),
    )
    ax.plot(
        alphas,
        [row["base"][a][score]["cov_step"] for a in keys],
        color=_SCORE_COLOR[score], ls="-", lw=1.3, marker="o", ms=2.4,
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Coverage")
    ax.set_title(f"{_COND_LABEL[cond]} -- {score} (base)", fontsize=8.2)
    _despine(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    plt.close(fig)


def _fig_singleton_base_only(
    res_dir: str, out: str, cond: str, score: str
) -> None:
    """Singleton rate vs dense alpha for the raw base score alone -- no
    ENCP intervention at all. A line, not grouped bars, since there is
    only one series to show (unlike the pf/mlp/base comparison chart)."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dense = _load(res_dir, "cp_dense.json")
    row = next(r for r in dense if r["condition"] == cond)
    alphas = sorted(float(a) for a in row["base"])
    keys = [f"{a:.2f}" for a in alphas]
    fig, ax = plt.subplots(figsize=(2.7, 2.1))
    ax.plot(
        alphas,
        [row["base"][a][score]["singleton"] for a in keys],
        color=_SCORE_COLOR[score], ls="-", lw=1.3, marker="o", ms=2.4,
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Singleton rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{_COND_LABEL[cond]} -- {score} (base)", fontsize=8.2)
    _despine(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    plt.close(fig)


# ==========================================================================
# Appendix figures. Each is standalone, larger than the column figures above,
# and places its legend OUTSIDE the axes; every figure is saved with
# bbox_inches="tight" so the external legend is never clipped and nothing
# inside the plotting area is covered.
# ==========================================================================
def _legend_right(ax, ncol: int = 1, **kw) -> None:
    ax.legend(
        bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0,
        handlelength=1.8, labelspacing=0.35, fontsize=6.6, ncol=ncol, **kw
    )


def _legend_top(ax, ncol: int, **kw) -> None:
    ax.legend(
        bbox_to_anchor=(0.5, 1.02), loc="lower center", borderaxespad=0.0,
        handlelength=1.6, columnspacing=1.2, fontsize=6.8, ncol=ncol,
        frameon=False, **kw
    )


def _save(fig, out: str) -> None:
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_app_dense_cov(res_dir: str, out: str) -> None:
    """Parameter-free step coverage across the nine-level alpha grid: every
    backbone clears the target 1-alpha at every level."""
    dense = {r["condition"]: r for r in _load(res_dir, "cp_dense.json")}
    al = sorted(float(a) for a in dense["duet_full"]["zeroshot"])
    keys = [f"{a:.2f}" for a in al]
    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    ax.plot(al, [1 - a for a in al], color=MUTED, lw=1.1, ls=(0, (4, 3)),
            label=r"Target $1{-}\alpha$", zorder=2)
    for c in ORDER:
        col, mk, ls = STYLE[c]
        ax.plot(al, [dense[c]["zeroshot"][a]["THR"]["cov_step"] for a in keys],
                color=col, marker=mk, ms=3.0, lw=1.2, ls=ls, label=FLABEL[c])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Step coverage")
    ax.set_ylim(0.72, 1.005)
    _despine(ax)
    _legend_right(ax)
    _save(fig, out)


def fig_app_dense_size(res_dir: str, out: str) -> None:
    """Parameter-free mean set size shrinks smoothly with alpha -- the
    threshold is responsive, not collapsed."""
    dense = {r["condition"]: r for r in _load(res_dir, "cp_dense.json")}
    al = sorted(float(a) for a in dense["duet_full"]["zeroshot"])
    keys = [f"{a:.2f}" for a in al]
    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    for c in ORDER:
        col, mk, ls = STYLE[c]
        ax.plot(al, [dense[c]["zeroshot"][a]["THR"]["mean_set"] for a in keys],
                color=col, marker=mk, ms=3.0, lw=1.2, ls=ls, label=FLABEL[c])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"Mean set size $\overline{|C|}$")
    _despine(ax)
    _legend_right(ax)
    _save(fig, out)


def fig_app_collapse(res_dir: str, out: str) -> None:
    """Base-CP singleton rate versus alpha: on every backbone the base score
    collapses to all-singleton sets as alpha grows (APS shown; THR similar)."""
    dense = {r["condition"]: r for r in _load(res_dir, "cp_dense.json")}
    al = sorted(float(a) for a in dense["duet_full"]["base"])
    keys = [f"{a:.2f}" for a in al]
    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    for c in ORDER:
        col, mk, ls = STYLE[c]
        ax.plot(al, [dense[c]["base"][a]["APS"]["singleton"] for a in keys],
                color=col, marker=mk, ms=3.0, lw=1.2, ls=ls, label=FLABEL[c])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Base-CP singleton rate (APS)")
    ax.set_ylim(-0.02, 1.03)
    _despine(ax)
    _legend_right(ax)
    _save(fig, out)


def fig_app_indist_simul(res_dir: str, out: str) -> None:
    """The theorem, empirically: whole-trajectory (simultaneous) coverage
    sits far below target under the seen->unseen shift but returns to
    k/(n+1)~1-alpha on exchangeable in-distribution halves."""
    cp = {r["condition"]: r for r in _load(res_dir, "cp_results.json")}
    ind = {r["condition"]: r for r in _load(res_dir, "indist.json")}
    shifted = [cp[c]["0.10"]["family_full"]["THR"]["pf"]["cov_simul"]
               for c in ORDER]
    indist = [ind[c]["nav_simul"]["0.10"] for c in ORDER]
    x = np.arange(len(ORDER))
    w = 0.38
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.bar(x - w / 2, shifted, w, color=RED, label="Shifted (seen$\\to$unseen)")
    ax.bar(x + w / 2, indist, w, color=BLUE, label="In-distribution")
    ax.axhline(0.90, color=INK, lw=1.1, ls="--", label=r"Target $1{-}\alpha$")
    ax.set_xticks(x)
    ax.set_xticklabels([FLABEL[c] for c in ORDER], rotation=40, ha="right")
    ax.set_ylabel(r"Simultaneous coverage ($\alpha{=}0.10$)")
    ax.set_ylim(0.6, 1.0)
    _despine(ax)
    _legend_top(ax, ncol=3)
    _save(fig, out)


def fig_app_gap(res_dir: str, out: str) -> None:
    """Step-averaged versus simultaneous coverage under shift: the step
    average clears target while the whole-trajectory number does not."""
    cp = {r["condition"]: r for r in _load(res_dir, "cp_results.json")}
    step = [cp[c]["0.10"]["family_full"]["THR"]["pf"]["cov_step"]
            for c in ORDER]
    simul = [cp[c]["0.10"]["family_full"]["THR"]["pf"]["cov_simul"]
             for c in ORDER]
    x = np.arange(len(ORDER))
    w = 0.38
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.bar(x - w / 2, step, w, color=BLUE, label="Step-averaged")
    ax.bar(x + w / 2, simul, w, color=AQUA, label="Simultaneous")
    ax.axhline(0.90, color=INK, lw=1.1, ls="--", label=r"Target $1{-}\alpha$")
    ax.set_xticks(x)
    ax.set_xticklabels([FLABEL[c] for c in ORDER], rotation=40, ha="right")
    ax.set_ylabel(r"Coverage ($\alpha{=}0.10$)")
    ax.set_ylim(0.6, 1.0)
    _despine(ax)
    _legend_top(ax, ncol=3)
    _save(fig, out)


def fig_app_transfer(res_dir: str, out: str) -> None:
    """Cross-backbone threshold transfer: coverage when the row backbone's
    threshold is applied to the column backbone's test split."""
    tr = _load(res_dir, "transfer.json")
    order = ["duet_full", "hamt",
             "recbert_oscar", "recbert_prevalent"]
    lab = [FLABEL[c] for c in order]
    M = np.array([[tr["matrix"][s][t]["cov_step"] for t in order]
                  for s in order])
    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    im = ax.imshow(M, cmap="viridis", vmin=0.88, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(lab, rotation=40, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(lab)
    ax.set_xlabel("Applied to (test split)")
    ax.set_ylabel(r"Calibrated on ($\hat q$ source)")
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=5.8,
                    color="white" if M[i, j] < 0.96 else INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("step coverage", fontsize=6.6)
    cb.ax.tick_params(labelsize=6.0)
    _save(fig, out)


def fig_app_family(res_dir: str, out: str) -> None:
    """The weight family at alpha=0.10: every member clears the target, so
    the alpha-response is restored by any positive scale, not a tuned one."""
    cp = {r["condition"]: r for r in _load(res_dir, "cp_results.json")}
    members = ["w0", "pf", "mlp", "hybrid", "random"]
    mlab = {"w0": r"$w{=}0$", "pf": "pf", "mlp": "mlp",
            "hybrid": "hybrid", "random": "random"}
    mcol = {"w0": "#8c8c8c", "pf": BLUE, "mlp": "#e08a1e",
            "hybrid": "#1baf7a", "random": "#c13aa0"}
    x = np.arange(len(ORDER))
    nmemb = len(members)
    w = 0.15
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    for i, m in enumerate(members):
        ys = [cp[c]["0.10"]["family"]["THR"][m]["cov_step"] for c in ORDER]
        ax.bar(x + (i - (nmemb - 1) / 2) * w, ys, w,
               color=mcol[m], label=mlab[m])
    ax.axhline(0.90, color=INK, lw=1.0, ls="--", label=r"Target")
    ax.set_xticks(x)
    ax.set_xticklabels([FLABEL[c] for c in ORDER], rotation=40, ha="right")
    ax.set_ylabel(r"Step coverage ($\alpha{=}0.10$)")
    ax.set_ylim(0.9, 1.0)
    _despine(ax)
    _legend_top(ax, ncol=6)
    _save(fig, out)


def fig_app_dtv(res_dir: str, out: str) -> None:
    """Reduced-score shift versus the simultaneous-coverage shortfall: the
    larger the distribution shift, the further whole-trajectory coverage
    falls below target."""
    cp = {r["condition"]: r for r in _load(res_dir, "cp_results.json")}
    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    for c in ORDER:
        col, mk, _ = STYLE[c]
        d = cp[c]["shift"]["dTV_reduced"]
        short = 0.90 - cp[c]["0.10"]["family_full"]["THR"]["pf"]["cov_simul"]
        ax.scatter(d, short, s=42, color=col, marker=mk,
                   edgecolor="white", linewidth=0.4, label=FLABEL[c])
    ax.axhline(0.0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel(r"Reduced-score shift $\widehat{d}_{\mathrm{TV}}$")
    ax.set_ylabel(r"Simul. shortfall $(1{-}\alpha){-}\mathrm{cov}$")
    _despine(ax)
    _legend_right(ax)
    _save(fig, out)


def fig_app_conditional(res_dir: str, out: str) -> None:
    """Coverage by policy-confidence quartile: the only shortfall is on the
    most-confident quartile, where the set is a singleton and a confidently
    wrong step is missed."""
    cp = {r["condition"]: r for r in _load(res_dir, "cp_results.json")}
    q = [1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    ax.axhline(0.90, color=INK, lw=1.0, ls="--", label=r"Target $1{-}\alpha$")
    for c in ORDER:
        col, mk, ls = STYLE[c]
        ys = cp[c]["diagnostics"]["cov_by_pmax_quartile"]
        ax.plot(q, ys, color=col, marker=mk, ms=3.2, lw=1.2, ls=ls,
                label=FLABEL[c])
    ax.set_xticks(q)
    ax.set_xticklabels([r"Q1", r"Q2", r"Q3", r"Q4"])
    ax.set_xlabel(r"$p_{\max}$ quartile (least $\to$ most confident)")
    ax.set_ylabel(r"Step coverage ($\alpha{=}0.10$)")
    _despine(ax)
    _legend_right(ax)
    _save(fig, out)


def fig_app_budget(res_dir: str, out: str) -> None:
    """Query-budget recall of the policy's argmax errors, averaged over
    backbones: choosing the steps to ask by set size is close to choosing by
    lowest confidence."""
    cp = {r["condition"]: r for r in _load(res_dir, "cp_results.json")}
    budgets = ["0.05", "0.10", "0.20", "0.30"]
    xs = [float(b) for b in budgets]
    setr = np.mean([[cp[c]["diagnostics"]["query_budget"][b]
                     ["set_size_trigger"] for b in budgets]
                    for c in ORDER], axis=0)
    pmr = np.mean([[cp[c]["diagnostics"]["query_budget"][b]["pmax_trigger"]
                    for b in budgets] for c in ORDER], axis=0)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ax.plot(xs, setr, color=BLUE, marker="o", ms=3.4, lw=1.4,
            label="Set-size trigger")
    ax.plot(xs, pmr, color=AQUA, marker="s", ms=3.2, lw=1.4, ls="--",
            label=r"Confidence trigger")
    ax.plot([0, 0.3], [0, 0.3], color=MUTED, lw=0.8, ls=":", label="Random")
    ax.set_xlabel("Fraction of steps queried")
    ax.set_ylabel("Recall of argmax errors")
    _despine(ax)
    _legend_right(ax)
    _save(fig, out)


def fig_app_object(res_dir: str, out: str) -> None:
    """REVERIE object-grounding head: base vs parameter-free normalised split
    CP undercover out of distribution; both are shown against target."""
    cp = {r["condition"]: r for r in _load(res_dir, "cp_results.json")}
    keys = ["0.10", "0.20", "0.30"]
    al = [0.10, 0.20, 0.30]
    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    ax.plot(al, [1 - a for a in al], color=MUTED, lw=1.1, ls=(0, (4, 3)),
            label=r"Target $1{-}\alpha$")
    for c, col, mk in (("duet_full_reverie", BLUE, "o"),
                       ("hamt_reverie", RED, "^")):
        o = cp[c].get("object_cp")
        if not o:
            continue
        ax.plot(al, [o[a]["THR"]["base"]["cov"] for a in keys], color=col,
                marker=mk, ms=3.2, lw=1.3, label=f"{FLABEL[c]} base")
        ax.plot(al, [o[a]["THR"]["norm"]["cov"] for a in keys], color=col,
                marker=mk, ms=3.2, lw=1.1, ls=":",
                label=f"{FLABEL[c]} norm.")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Object-head coverage")
    ax.set_xticks(al)
    _despine(ax)
    _legend_right(ax)
    _save(fig, out)


def fig_app_mc(res_dir: str, out: str) -> None:
    """Monte-Carlo validation of the finite-sample bound on synthetic
    exchangeable episodes: per-trial whole-trajectory coverage concentrates
    at k/(n+1) ~ 1-alpha."""
    rng = np.random.RandomState(0)
    n_cal, n_test, alpha, trials = 200, 300, 0.10, 3000
    k = int(np.ceil((n_cal + 1) * (1 - alpha)))
    covs = np.empty(trials)
    for i in range(trials):
        z = rng.rand(n_cal + n_test)
        q = np.sort(z[:n_cal])[k - 1]
        covs[i] = (z[n_cal:] <= q).mean()
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ax.hist(covs, bins=28, color=BLUE, alpha=0.8, edgecolor="white", lw=0.3,
            label="Per-trial coverage")
    ax.axvline(1 - alpha, color=RED, lw=1.4, ls="--",
               label=r"Target $1{-}\alpha$")
    ax.axvline(float(covs.mean()), color=INK, lw=1.4,
               label=f"mean {covs.mean():.3f}")
    ax.set_xlabel("Whole-trajectory coverage per trial")
    ax.set_ylabel("Trials")
    _despine(ax)
    _legend_right(ax)
    _save(fig, out)


def fig_app_collapse2(res_dir: str, out: str) -> None:
    """Overconfidence collapse: on a concentrated backbone (DUET) the base APS
    teacher score is a point mass at zero, so the calibrated threshold has
    nowhere to sit; the parameter-free normalisation spreads the same scores
    across the unit interval and restores a responsive threshold."""
    import torch
    from cp_core.split import Split
    dump_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dumps"
    )
    d = torch.load(os.path.join(dump_dir, "duet_full.pt"), weights_only=True)
    sp = Split.from_records(d["test"])
    aps = sp.base_teacher["APS"]
    snorm = sp.base_teacher["THR"] / (2.0 - sp.p_max)
    bins = np.linspace(0, 1, 26)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.4, 2.0))
    axA.hist(aps, bins=bins, color=RED, alpha=0.85, edgecolor="white", lw=0.3)
    axA.set_title(r"Base APS score", fontsize=8.5)
    axA.set_xlabel("Teacher nonconformity score")
    axA.set_ylabel("Steps")
    axA.annotate(f"{(aps < 1e-9).mean() * 100:.0f}% at $0$",
                 xy=(0.05, 0.86), xycoords="axes fraction", fontsize=8,
                 color=RED)
    axB.hist(snorm, bins=bins, color=BLUE, alpha=0.85, edgecolor="white",
             lw=0.3)
    axB.set_title(r"Normalised score $s_{\mathrm{norm}}$", fontsize=8.5)
    axB.set_xlabel("Teacher nonconformity score")
    for ax in (axA, axB):
        _despine(ax)
    fig.tight_layout(w_pad=1.4)
    _save(fig, out)


def fig_app_shift_hist(res_dir: str, out: str) -> None:
    """Seen-to-unseen distribution shift: the density of the reduced score
    s~ under the calibration (val-seen) and test (val-unseen) laws for DUET;
    the visible gap is the total-variation distance d_TV=0.225 that pulls
    simultaneous coverage below target."""
    import torch
    from cp_core.split import Split
    dump_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dumps"
    )
    d = torch.load(os.path.join(dump_dir, "duet_full.pt"), weights_only=True)

    def epmax(recs):
        sp = Split.from_records(recs)
        s = sp.base_teacher["THR"] / (2.0 - sp.p_max)
        return np.clip([s[a:b].max() for a, b in sp.ep_ptr if b > a], 0, 1)

    rc, rt = epmax(d["cal"]), epmax(d["test"])
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(3.5, 2.15))
    ax.hist(rc, bins=bins, density=True, color=BLUE, alpha=0.55,
            label="Val-seen (calibration)")
    ax.hist(rt, bins=bins, density=True, color=RED, alpha=0.55,
            label="Val-unseen (test)")
    ax.set_xlabel(r"Reduced score $\tilde s=\varphi(E)$")
    ax.set_ylabel("Density")
    ax.annotate(r"$\widehat{d}_{\mathrm{TV}}=0.225$", xy=(0.03, 0.86),
                xycoords="axes fraction", fontsize=8)
    _despine(ax)
    _legend_top(ax, ncol=2)
    _save(fig, out)


def make_all(res_dir: str, fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    jobs = [
        ("fig_qualitative.png", fig_qualitative),
        ("fig_reverie.png", fig_reverie),
        ("fig_reverie_nav.png", fig_reverie_nav),
        ("fig_reverie_grounding.png", fig_reverie_grounding),
        ("fig_closedloop.png", fig_closedloop),
        ("fig_app_collapse2.png", fig_app_collapse2),
        ("fig_app_shift_hist.png", fig_app_shift_hist),
        ("fig_app_dense_cov.png", fig_app_dense_cov),
        ("fig_app_dense_size.png", fig_app_dense_size),
        ("fig_app_collapse.png", fig_app_collapse),
        ("fig_app_indist_simul.png", fig_app_indist_simul),
        ("fig_app_gap.png", fig_app_gap),
        ("fig_app_transfer.png", fig_app_transfer),
        ("fig_app_family.png", fig_app_family),
        ("fig_app_dtv.png", fig_app_dtv),
        ("fig_app_conditional.png", fig_app_conditional),
        ("fig_app_budget.png", fig_app_budget),
        ("fig_app_object.png", fig_app_object),
        ("fig_app_mc.png", fig_app_mc),
    ]
    for cond in _ALL_CONDITIONS:
        for score in ("THR", "APS", "RAPS"):
            jobs.append((
                os.path.join("coverage_grid", f"cov_{cond}_{score}.png"),
                lambda r, o, c=cond, s=score: _fig_cov_single(r, o, c, s),
            ))
            jobs.append((
                os.path.join(
                    "coverage_grid", f"singleton_{cond}_{score}.png"
                ),
                lambda r, o, c=cond, s=score: _fig_singleton_single(
                    r, o, c, s
                ),
            ))
            jobs.append((
                os.path.join(
                    "coverage_grid_base", f"cov_{cond}_{score}.png"
                ),
                lambda r, o, c=cond, s=score: _fig_cov_base_only(
                    r, o, c, s
                ),
            ))
            jobs.append((
                os.path.join(
                    "coverage_grid_base", f"singleton_{cond}_{score}.png"
                ),
                lambda r, o, c=cond, s=score: _fig_singleton_base_only(
                    r, o, c, s
                ),
            ))
    for name, fn in jobs:
        try:
            fn(res_dir, os.path.join(fig_dir, name))
            print(f"[paperfigs] {name}")
        except FileNotFoundError as e:
            print(f"[paperfigs] SKIP {name}: missing input ({e.filename})")
