"""Domain model: one rollout split flattened to per-step arrays."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from cp_core.scores import (
    SCORES,
    EPS,
    base_scores_all,
    softmax_valid,
    teacher_fallback,
)

T_MAX = 15.0  # episode length cap (feature normalisation)


@dataclass
class Split:
    """One rollout split, flattened to per-step arrays.

    Steps are stored flat; `ep_ptr` holds [start, end) per episode. Base
    scores are precomputed for all three score functions over the valid
    candidates only.
    """

    p_max: np.ndarray  # top softmax prob per step
    p_teacher: np.ndarray  # softmax prob of the teacher action
    degree: np.ndarray  # |A_t| = number of valid candidates
    argmax_err: np.ndarray  # argmax != teacher (bool)
    scan: np.ndarray  # scene id per step (cluster unit)
    base_teacher: Dict[str, np.ndarray]  # score -> teacher base score
    base_cands: Dict[str, List[np.ndarray]]  # score -> ragged candidate scores
    feat: np.ndarray  # [H, p_max, margin, log|A|, t/T, alpha-slot]
    ep_ptr: List[Tuple[int, int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.p_max)

    @property
    def n_episodes(self) -> int:
        return len(self.ep_ptr)

    def halves(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """50/50 episode split by rollout order (train half / quantile
        half)."""
        half = max(self.n_episodes // 2, 1)
        return self.ep_ptr[:half], self.ep_ptr[half:]

    @classmethod
    def from_records(
        cls, records_by_episode: Dict[str, List[Dict[str, Any]]]
    ) -> "Split":
        """Build from raw rollout records ({episode_id: [step dicts]}).
        Each record needs `logits` and `teacher_idx`; `step` is optional."""
        p_max, p_teacher, degree, argmax_err, feat, scan = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        base_teacher = {k: [] for k in SCORES}
        base_cands = {k: [] for k in SCORES}
        ep_ptr, cur = [], 0
        for recs in records_by_episode.values():
            start = cur
            for r in recs:
                lg = r["logits"]
                lg = (
                    lg.float().numpy()
                    if isinstance(lg, torch.Tensor)
                    else np.asarray(lg, float)
                )
                p, valid = softmax_valid(lg)
                scores = base_scores_all(p)
                ti = int(r["teacher_idx"])
                in_cands = 0 <= ti < len(lg) and bool(valid[ti])
                t_pos = int(valid[:ti].sum()) if in_cands else -1
                for k in SCORES:
                    base_teacher[k].append(
                        float(scores[k][t_pos])
                        if t_pos >= 0
                        else teacher_fallback(k, len(p))
                    )
                    base_cands[k].append(scores[k])
                pm = float(p.max())
                top2 = np.sort(p)[::-1][:2]
                feat.append(
                    [
                        float(-np.sum(p * np.log(p))),
                        pm,
                        float(top2[0] - top2[1]) if len(top2) > 1 else 0.0,
                        math.log(max(len(p), 1)),
                        float(r.get("step", 0)) / T_MAX,
                        0.0,
                    ]
                )  # alpha slot, filled per fit
                p_max.append(pm)
                p_teacher.append(float(p[t_pos]) if t_pos >= 0 else EPS)
                degree.append(int(valid.sum()))
                argmax_err.append(int(np.argmax(p)) != t_pos)
                scan.append(str(r.get("scan", "?")))
                cur += 1
            if cur > start:
                ep_ptr.append((start, cur))
        return cls(
            p_max=np.asarray(p_max),
            p_teacher=np.asarray(p_teacher),
            degree=np.asarray(degree, int),
            argmax_err=np.asarray(argmax_err, bool),
            scan=np.asarray(scan),
            base_teacher={k: np.asarray(v) for k, v in base_teacher.items()},
            base_cands=base_cands,
            feat=np.asarray(feat, np.float32),
            ep_ptr=ep_ptr,
        )
