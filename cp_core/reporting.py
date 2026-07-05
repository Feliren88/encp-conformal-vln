"""Claim verification (verify) and aggregate HTML figures (figures).

render the aggregate figures as one self-contained HTML (--figures).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from cp_core.scores import SCORES
from cp_core.weights import WEIGHTS
from cp_core.analyses import ALPHAS

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
_LABEL = {
    "duet_full": "DUET-full",
    "duet_local": "DUET-local",
    "hamt": "HAMT",
    "recbert_prevalent": "RecBERT-PREV",
    "recbert_oscar": "RecBERT-OSCAR",
    "duet_full_reverie": "DUET (REVERIE)",
    "hamt_reverie": "HAMT (REVERIE)",
}


def verify_results(results_path: str) -> int:
    """Assert the paper's load-bearing claims. Returns #failed checks."""
    with open(results_path) as f:
        by = {r["condition"]: r for r in json.load(f)}
    r2r = [
        c
        for c in (
            "duet_full",
            "duet_local",
            "hamt",
            "recbert_prevalent",
            "recbert_oscar",
        )
        if c in by
    ]
    reverie = [c for c in by if c.endswith("_reverie")]
    targets = {"0.10": 0.90, "0.20": 0.80, "0.30": 0.70}
    checks: List[Tuple[bool, str, str]] = []

    def ok(name: str, cond: bool, detail: str = "") -> None:
        checks.append((cond, name, detail))

    hits = [
        (c, a, s)
        for c in r2r
        for a, t in targets.items()
        for s in SCORES
        if by[c][a]["family_full"][s]["pf"]["cov_step"] >= t
    ]
    ok(
        f"pf clears target in all {len(r2r) * 9} R2R cells",
        len(hits) == len(r2r) * 9,
        f"{len(hits)}/{len(r2r) * 9}",
    )

    for c in ("duet_full", "hamt"):
        if c not in by:
            continue
        for s in ("APS", "RAPS"):
            b = by[c]["0.30"]["base"][s]
            ok(
                f"base collapse {c}/{s}@0.30",
                b["q"] < 1e-6 and b["singleton"] > 0.999,
                f"q={b['q']:.3f} sgl={b['singleton']:.3f}",
            )

    for c in r2r:
        fam = by[c]["0.10"]["family"]["THR"]
        gap = abs(fam["mlp"]["cov_step"] - fam["pf"]["cov_step"])
        ok(f"learned~pf gap<=0.02 [{c}]", gap <= 0.02, f"gap={gap:.4f}")
        ok(
            f"hybrid covers 0.90 [{c}]",
            fam["hybrid"]["cov_step"] >= 0.90,
            f"cov={fam['hybrid']['cov_step']:.3f}",
        )

    for c in reverie:
        for a, t in targets.items():
            cov = by[c][a]["family_full"]["THR"]["pf"]["cov_step"]
            ok(
                f"REVERIE nav clears target [{c}@{a}]",
                cov >= t,
                f"cov={cov:.3f}",
            )

    width = max(len(n) for _, n, _ in checks)
    fails = [n for passed, n, _ in checks if not passed]
    for passed, name, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<{width}}  {detail}")
    print(
        f"\n[verify] {len(checks) - len(fails)}/{len(checks)} checks passed."
    )
    return len(fails)


def make_figures(results_path: str, out_path: str) -> str:
    """All aggregate figures in one self-contained HTML (plotly via CDN)."""
    with open(results_path) as f:
        by = {r["condition"]: r for r in json.load(f)}
    conds = [c for c in _LABEL if c in by]
    labels = [_LABEL[c] for c in conds]
    alphas = list(ALPHAS)
    keys = [f"{a:.2f}" for a in alphas]

    def line(name: str, ys: List[Any]) -> Dict[str, Any]:
        return {"x": alphas, "y": ys, "mode": "lines+markers", "name": name}

    target_line = {
        "x": alphas,
        "y": [0.9, 0.8, 0.7],
        "mode": "lines",
        "line": {"dash": "dash"},
        "name": "target",
    }
    figs = []
    for key, title in (("cov_step", "step coverage"), ("mean_set", "|C|")):
        figs.append(
            (
                f"pf (full-cal) {title} vs alpha (THR)",
                [
                    line(
                        _LABEL[c],
                        [
                            by[c][a]["family_full"]["THR"]["pf"][key]
                            for a in keys
                        ],
                    )
                    for c in conds
                ]
                + ([target_line] if key == "cov_step" else []),
                {"xaxis": {"title": "alpha"}, "yaxis": {"title": title}},
            )
        )
    for key, title in (("cov_step", "coverage"), ("mean_set", "|C|")):
        figs.append(
            (
                f"Weight family @ alpha=0.10 (THR, split-matched): {title}",
                [
                    {
                        "x": labels,
                        "type": "bar",
                        "name": v,
                        "y": [
                            by[c]["0.10"]["family"]["THR"][v][key]
                            for c in conds
                        ],
                    }
                    for v in WEIGHTS
                ],
                {"barmode": "group", "yaxis": {"title": title}},
            )
        )
    figs.append(
        (
            "Base-CP singleton rate vs alpha (APS) -- the collapse",
            [
                line(
                    _LABEL[c],
                    [by[c][a]["base"]["APS"]["singleton"] for a in keys],
                )
                for c in conds
            ],
            {
                "xaxis": {"title": "alpha"},
                "yaxis": {"title": "singleton rate"},
            },
        )
    )
    for c in conds:
        o = by[c].get("object_cp")
        if o:
            figs.append(
                (
                    f"Object-head CP ({_LABEL[c]}): base vs normalised (THR)",
                    [
                        line(
                            "base", [o[a]["THR"]["base"]["cov"] for a in keys]
                        ),
                        line(
                            "normalised",
                            [o[a]["THR"]["norm"]["cov"] for a in keys],
                        ),
                        target_line,
                    ],
                    {
                        "xaxis": {"title": "alpha"},
                        "yaxis": {"title": "coverage"},
                    },
                )
            )

    sections, scripts = [], []
    for i, (heading, traces, layout) in enumerate(figs):
        sections.append(
            f"<section><h2>{heading}</h2><div id='g{i}' class='plot'>"
            "</div></section>"
        )
        layout = {**layout, "margin": {"t": 30}}
        scripts.append(
            f"Plotly.newPlot('g{i}',{json.dumps(traces)},"
            f"{json.dumps(layout)},{{responsive:true}});"
        )
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Normalised-CP weight family</title>"
        f"<script src='{_PLOTLY_CDN}'></script><style>"
        "body{margin:0 auto;max-width:1000px;padding:24px;"
        "font-family:sans-serif}"
        ".plot{width:100%;height:420px}section{margin-bottom:18px}"
        "</style></head><body><h1>Normalised-CP weight family</h1>"
        + "".join(sections)
        + "<script>"
        + "".join(scripts)
        + "</script></body></html>"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[figures] {len(figs)} figures -> {out_path}")
    return out_path
