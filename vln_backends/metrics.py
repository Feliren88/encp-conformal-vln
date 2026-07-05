"""Navigation metrics and the SR sanity gate."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from vln_backends.bootstrap import cal_dtw, cal_cls


def flatten_trajectory(raw_path: List) -> List[str]:
    flat: List[str] = []
    for sub in raw_path:
        flat.extend(sub) if isinstance(sub, list) else flat.append(sub)
    deduped = flat[:1]
    for v in flat[1:]:
        if v != deduped[-1]:
            deduped.append(v)
    return deduped


def compute_nav_metrics(preds: List[Dict], env: Any) -> Dict[str, float]:
    """SR / SPL / NE / nDTW / CLS from predicted viewpoint trajectories."""
    gt_by_id = {d["instr_id"]: d for d in env.data}
    rows = []
    for p in preds:
        d = gt_by_id.get(p["instr_id"])
        if d is None:
            continue
        gt = d["path"]
        pred = flatten_trajectory(p["trajectory"]) or [gt[0]]
        dist = env.shortest_distances.get(d["scan"], {})
        ne = dist.get(pred[-1], {}).get(gt[-1], 999.0)
        pred_len = sum(
            dist.get(a, {}).get(b, 0.0) for a, b in zip(pred, pred[1:])
        )
        gt_len = sum(dist.get(a, {}).get(b, 0.0) for a, b in zip(gt, gt[1:]))
        sr = float(ne < 3.0)
        try:
            ndtw = cal_dtw(dist, pred, gt, threshold=3.0)["nDTW"]
        except Exception:
            ndtw = 0.0
        try:
            cls_ = cal_cls(dist, pred, gt, threshold=3.0)
        except Exception:
            cls_ = 0.0
        rows.append(
            {
                "sr": sr,
                "spl": sr * gt_len / max(pred_len, gt_len, 1e-6),
                "ne": ne,
                "ndtw": ndtw,
                "cls": cls_,
            }
        )
    if not rows:
        return {"sr": 0.0, "spl": 0.0, "ne": 999.0}

    def mean(k: str) -> float:
        return float(np.mean([r[k] for r in rows]))

    return {
        "sr": mean("sr") * 100,
        "spl": mean("spl") * 100,
        "ne": mean("ne"),
        "ndtw": mean("ndtw") * 100,
        "cls": mean("cls") * 100,
    }


def sanity_gate(
    test_preds, test_env, dataset: str, cond: str
) -> Dict[str, float]:
    """The rollout must reproduce the published val_unseen SR before any CP
    number is trusted (coverage holds for any model, so it cannot validate
    the checkpoint)."""
    if not test_preds:
        return {}
    if dataset == "reverie" and hasattr(test_env, "eval_metrics"):
        agg, _ = test_env.eval_metrics(test_preds)
        sanity = {
            k: float(agg[k])
            for k in ("sr", "oracle_sr", "spl", "rgs", "rgspl")
            if k in agg
        }
    else:
        sanity = compute_nav_metrics(test_preds, test_env)
    line = " ".join(f"{k.upper()}={v:.2f}" for k, v in sanity.items())
    print(f"[v10] {cond} SANITY (val_unseen): {line}", flush=True)
    return sanity
