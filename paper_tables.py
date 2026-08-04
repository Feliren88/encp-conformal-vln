"""Print LaTeX tables for the paper from results/*.json.

Presentation layer only: consumes results/cp_results.json, writes
paper/tables/. Mirrors paper_figures.py's module layout and _load
convention so both generators are regenerated the same way.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

ROWS: List[Dict[str, str]] = [
    {"cond": "duet_full", "label": "DUET", "dataset": "R2R", "third": "sgl"},
    {"cond": "hamt", "label": "HAMT", "dataset": "R2R", "third": "sgl"},
    {
        "cond": "recbert_prevalent",
        "label": "R-prev",
        "dataset": "R2R",
        "third": "sgl",
    },
    {
        "cond": "recbert_oscar",
        "label": "R-osc",
        "dataset": "R2R",
        "third": "sgl",
    },
    {
        "cond": "duet_full_reverie",
        "label": "DUET",
        "dataset": "REVERIE",
        "third": "sat",
    },
    {
        "cond": "hamt_reverie",
        "label": "HAMT",
        "dataset": "REVERIE",
        "third": "sat",
    },
]
SCORES = ("THR", "RAPS")
ALPHA_COLS = ("0.10", "0.30")


def _load(res_dir: str, name: str) -> Any:
    with open(os.path.join(res_dir, name)) as f:
        return json.load(f)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


_THIRD_FIELD = {"sgl": "singleton", "sat": "saturation"}


def _row_cells(
    by: Dict[str, Any], cond: str, score: str, third: str
) -> List[str]:
    field = _THIRD_FIELD[third]
    cells = []
    for a in ALPHA_COLS:
        m = by[cond][a]["family"][score]["pf"]
        cells += [_fmt(m["cov_step"]), f"{m['mean_set']:.1f}", _fmt(m[field])]
    return cells


def _delta_cov(by: Dict[str, Any], cond: str) -> str:
    fam = by[cond]["0.10"]["family"]["THR"]
    delta = fam["mlp"]["cov_step"] - fam["pf"]["cov_step"]
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}"


def make_tables(res_dir: str, tables_dir: str) -> None:
    os.makedirs(tables_dir, exist_ok=True)
    results = _load(res_dir, "cp_results.json")
    by = {r["condition"]: r for r in results}

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{ENCP (Episode-Normalized Conformal Prediction) on R2R "
        r"and REVERIE val-unseen: coverage, mean set size, and "
        r"singleton/saturation rate (sgl for R2R, sat for REVERIE) at "
        r"$\alpha=0.10$ and $\alpha=0.30$ ($\alpha=0.20$ is transitional "
        r"and omitted; APS matches THR to within the tolerance stated in "
        r"Section~\ref{ssec:main} and is omitted). $\Delta$cov is the "
        r"learned-minus-formula-based coverage gap (THR, $\alpha=0.10$).}"
    )
    lines.append(r"\label{tab:encp}")
    lines.append(r"\setlength{\tabcolsep}{3pt}\scriptsize")
    lines.append(r"\renewcommand{\arraystretch}{0.92}")
    lines.append(r"\begin{tabular}{ll ccc c ccc c c}")
    lines.append(r"\toprule")
    lines.append(
        r"& & \multicolumn{3}{c}{$\alpha{=}0.10$} & & "
        r"\multicolumn{3}{c}{$\alpha{=}0.30$} & & \\"
    )
    lines.append(r"\cmidrule{3-5}\cmidrule{7-9}")
    lines.append(
        r"\textbf{Backb.} & \textbf{Sc.} & cov & $\overline{|C|}$ & "
        r"sgl/sat & & cov & $\overline{|C|}$ & sgl/sat & & "
        r"$\Delta$cov \\"
    )
    lines.append(r"\midrule")

    prev_dataset = None
    for i, row in enumerate(ROWS):
        cond = row["cond"]
        if cond not in by:
            continue
        if (
            prev_dataset is not None
            and row["dataset"] != prev_dataset
            and lines[-1] != r"\midrule"
        ):
            lines.append(r"\midrule")
        prev_dataset = row["dataset"]
        for j, score in enumerate(SCORES):
            cells = _row_cells(by, cond, score, row["third"])
            prefix = (
                f"\\multirow{{{len(SCORES)}}}{{*}}{{{row['label']}}}"
                if j == 0
                else ""
            )
            delta = _delta_cov(by, cond) if j == 0 else ""
            lines.append(
                f" {prefix} & {score} & {cells[0]} & {cells[1]} & "
                f"{cells[2]} & & {cells[3]} & {cells[4]} & {cells[5]} & "
                f"& {delta} \\\\"
            )
        if i < len(ROWS) - 1:
            lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_path = os.path.join(tables_dir, "tab_encp.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tables] tab_encp.tex ({len(ROWS)} backbones) -> {out_path}")


def make_all(res_dir: str, tables_dir: str) -> None:
    make_tables(res_dir, tables_dir)
