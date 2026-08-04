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
    {"cond": "duet_full", "label": "DUET"},
    {"cond": "hamt", "label": "HAMT"},
    {"cond": "recbert_prevalent", "label": "R-prev"},
    {"cond": "recbert_oscar", "label": "R-osc"},
]
SCORES = ("THR", "APS", "RAPS")
ALPHA_COLS = ("0.10", "0.30")


def _load(res_dir: str, name: str) -> Any:
    with open(os.path.join(res_dir, name)) as f:
        return json.load(f)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _row_cells(by: Dict[str, Any], cond: str, score: str) -> List[str]:
    cells = []
    for a in ALPHA_COLS:
        base = by[cond][a]["base"][score]
        encp = by[cond][a]["family"][score]["pf"]
        cells += [
            _fmt(base["cov_step"]), f"{base['mean_set']:.1f}",
            _fmt(base["singleton"]),
            _fmt(encp["cov_step"]), f"{encp['mean_set']:.1f}",
            _fmt(encp["singleton"]),
        ]
    return cells


def make_tables(res_dir: str, tables_dir: str) -> None:
    """R2R, complete: every base score (THR/APS/RAPS) against its ENCP
    (formula-based weight) counterpart, all four R2R backbones."""
    os.makedirs(tables_dir, exist_ok=True)
    results = _load(res_dir, "cp_results.json")
    by = {r["condition"]: r for r in results}

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{R2R val-unseen, complete: every base score (THR/APS/"
        r"RAPS) against its ENCP (Episode-Normalized Conformal "
        r"Prediction, formula-based weight) counterpart, all four "
        r"backbones, at $\alpha=0.10$ and $\alpha=0.30$. cov is step-"
        r"averaged coverage, $\overline{|C|}$ the mean prediction-set "
        r"size, sgl the singleton rate.}"
    )
    lines.append(r"\label{tab:encp}")
    lines.append(r"\setlength{\tabcolsep}{3pt}\scriptsize")
    lines.append(r"\renewcommand{\arraystretch}{0.92}")
    lines.append(r"\begin{tabular}{ll ccc c ccc c ccc c ccc}")
    lines.append(r"\toprule")
    lines.append(
        r"& & \multicolumn{6}{c}{$\alpha{=}0.10$} & & "
        r"\multicolumn{6}{c}{$\alpha{=}0.30$} \\"
    )
    lines.append(r"\cmidrule{3-8}\cmidrule{10-15}")
    lines.append(
        r"& & \multicolumn{3}{c}{base} & \multicolumn{3}{c}{ENCP} & & "
        r"\multicolumn{3}{c}{base} & \multicolumn{3}{c}{ENCP} \\"
    )
    lines.append(r"\cmidrule{3-5}\cmidrule{6-8}\cmidrule{10-12}\cmidrule{13-15}")
    lines.append(
        r"\textbf{Backb.} & \textbf{Sc.} & cov & $\overline{|C|}$ & sgl & "
        r"cov & $\overline{|C|}$ & sgl & & "
        r"cov & $\overline{|C|}$ & sgl & cov & $\overline{|C|}$ & sgl \\"
    )
    lines.append(r"\midrule")

    for i, row in enumerate(ROWS):
        cond = row["cond"]
        if cond not in by:
            continue
        for j, score in enumerate(SCORES):
            cells = _row_cells(by, cond, score)
            prefix = (
                f"\\multirow{{{len(SCORES)}}}{{*}}{{{row['label']}}}"
                if j == 0
                else ""
            )
            lines.append(
                f" {prefix} & {score} & {cells[0]} & {cells[1]} & "
                f"{cells[2]} & {cells[3]} & {cells[4]} & {cells[5]} & & "
                f"{cells[6]} & {cells[7]} & {cells[8]} & {cells[9]} & "
                f"{cells[10]} & {cells[11]} \\\\"
            )
        if i < len(ROWS) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_path = os.path.join(tables_dir, "tab_encp.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tables] tab_encp.tex ({len(ROWS)} backbones) -> {out_path}")


REVERIE_ROWS: List[Dict[str, str]] = [
    {"cond": "duet_full_reverie", "label": "DUET"},
    {"cond": "hamt_reverie", "label": "HAMT"},
]
REVERIE_SCORES = ("THR", "APS", "RAPS")
REVERIE_ALPHA_COLS = ("0.10", "0.30")


def _reverie_full_cells(by: Dict[str, Any], cond: str, score: str) -> List[str]:
    cells = []
    for a in REVERIE_ALPHA_COLS:
        base = by[cond][a]["base"][score]
        encp = by[cond][a]["family"][score]["pf"]
        cells += [
            _fmt(base["cov_step"]), f"{base['mean_set']:.1f}",
            _fmt(base["saturation"]),
            _fmt(encp["cov_step"]), f"{encp['mean_set']:.1f}",
            _fmt(encp["saturation"]),
        ]
    return cells


def make_reverie_full_table(res_dir: str, tables_dir: str) -> None:
    """Standalone REVERIE-only table: every base score (THR/APS/RAPS)
    against its ENCP (formula-based) counterpart, both REVERIE backbones,
    no compression -- not wired into the paper, a results reference only."""
    os.makedirs(tables_dir, exist_ok=True)
    results = _load(res_dir, "cp_results.json")
    by = {r["condition"]: r for r in results}

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{REVERIE val-unseen, complete: every base score (THR/"
        r"APS/RAPS) against its ENCP (Episode-Normalized Conformal "
        r"Prediction, formula-based weight) counterpart, both backbones, "
        r"at $\alpha=0.10$ and $\alpha=0.30$. cov is step-averaged "
        r"coverage, $\overline{|C|}$ the mean prediction-set size, sat the "
        r"saturation rate (fraction of steps whose set spans the whole "
        r"action space).}"
    )
    lines.append(r"\label{tab:reverie-full}")
    lines.append(r"\setlength{\tabcolsep}{3pt}\scriptsize")
    lines.append(r"\renewcommand{\arraystretch}{0.92}")
    lines.append(r"\begin{tabular}{ll ccc c ccc c ccc c ccc}")
    lines.append(r"\toprule")
    lines.append(
        r"& & \multicolumn{6}{c}{$\alpha{=}0.10$} & & "
        r"\multicolumn{6}{c}{$\alpha{=}0.30$} \\"
    )
    lines.append(r"\cmidrule{3-8}\cmidrule{10-15}")
    lines.append(
        r"& & \multicolumn{3}{c}{base} & \multicolumn{3}{c}{ENCP} & & "
        r"\multicolumn{3}{c}{base} & \multicolumn{3}{c}{ENCP} \\"
    )
    lines.append(r"\cmidrule{3-5}\cmidrule{6-8}\cmidrule{10-12}\cmidrule{13-15}")
    lines.append(
        r"\textbf{Backb.} & \textbf{Sc.} & cov & $\overline{|C|}$ & sat & "
        r"cov & $\overline{|C|}$ & sat & & "
        r"cov & $\overline{|C|}$ & sat & cov & $\overline{|C|}$ & sat \\"
    )
    lines.append(r"\midrule")

    for i, row in enumerate(REVERIE_ROWS):
        cond = row["cond"]
        if cond not in by:
            continue
        for j, score in enumerate(REVERIE_SCORES):
            cells = _reverie_full_cells(by, cond, score)
            prefix = (
                f"\\multirow{{{len(REVERIE_SCORES)}}}{{*}}{{{row['label']}}}"
                if j == 0
                else ""
            )
            lines.append(
                f" {prefix} & {score} & {cells[0]} & {cells[1]} & "
                f"{cells[2]} & {cells[3]} & {cells[4]} & {cells[5]} & & "
                f"{cells[6]} & {cells[7]} & {cells[8]} & {cells[9]} & "
                f"{cells[10]} & {cells[11]} \\\\"
            )
        if i < len(REVERIE_ROWS) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_path = os.path.join(tables_dir, "tab_reverie_full.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(
        f"[tables] tab_reverie_full.tex ({len(REVERIE_ROWS)} backbones "
        f"x {len(REVERIE_SCORES)} scores) -> {out_path}"
    )


def make_all(res_dir: str, tables_dir: str) -> None:
    make_tables(res_dir, tables_dir)
    make_reverie_full_table(res_dir, tables_dir)
