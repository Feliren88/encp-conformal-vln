"""The weight family s_w(x, a) = s_base(x, a) / (1 + w(x)), w >= 0.

Members (see WEIGHT_FAMILY): w0 (episode-max base), pf (parameter-free
1-p_max), mlp (learned), hybrid (parameter-free floor + learned residual --
the combination), random (frozen control). Learned members are fit on
calibration half 1 only; the quantile is taken on held-out half 2, which
preserves exchangeability and the coverage guarantee.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cp_core.split import Split


class WeightMLP(nn.Module):
    """Tiny MLP emitting a non-negative per-step weight (Softplus head)."""

    def __init__(self, in_dim: int = 6, hidden: int = 32, layers: int = 2):
        super().__init__()
        seq: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.ReLU()]
        seq += [nn.Linear(hidden, 1), nn.Softplus()]
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fit_mlp(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch: int = 512,
    seed: int = 0,
) -> WeightMLP:
    # Hyperparameters match output_v8's ALPSConfig (hidden 32, 2 layers,
    # 300 epochs, lr 1e-3, weight_decay 1e-3, batch 512).
    torch.manual_seed(seed)
    model = WeightMLP(in_dim=X.shape[1])
    opt = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    Xt = torch.from_numpy(np.asarray(X, np.float32))
    yt = torch.from_numpy(np.asarray(y, np.float32))
    n, bs = len(Xt), min(batch, len(Xt))
    with torch.enable_grad():
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                F.mse_loss(model(Xt[idx]), yt[idx]).backward()
                opt.step()
    model.eval()
    return model


def difficulty_target(p_teacher: np.ndarray, p_max: np.ndarray) -> np.ndarray:
    """SR-adaptive target (1-p_teacher)/(1-p_max), capped against outliers."""
    return np.clip(
        (1.0 - p_teacher) / np.clip(1.0 - p_max, 1e-8, None), 0.0, 10.0
    )


def fit_weight_models(
    cal: Split, alpha: float, seed: int = 0
) -> Dict[str, WeightMLP]:
    """Fit the learned members on calibration HALF 1 only.

    mlp        : target = difficulty ratio (paper's learned weight).
    mlp_noalpha: same target, alpha feature held at 0 -- functionally the
                 5-dim phi of the v8 A3 ablation (a constant input column is
                 absorbed by the first-layer bias).
    hybrid     : target = residual above the parameter-free floor; the
                 deployed weight is (1 - p_max) + r(phi), so hybrid >= pf.
    random     : untrained frozen MLP.
    """
    h1, _ = cal.halves()
    lo, hi = h1[0][0], h1[-1][1]
    X = cal.feat[lo:hi].copy()
    X[:, 5] = alpha
    X_noalpha = cal.feat[lo:hi].copy()
    X_noalpha[:, 5] = 0.0
    y = difficulty_target(cal.p_teacher[lo:hi], cal.p_max[lo:hi])
    y_resid = np.clip(y - (1.0 - cal.p_max[lo:hi]), 0.0, 10.0)
    torch.manual_seed(seed + 12345)
    frozen = WeightMLP(in_dim=X.shape[1]).eval()
    return {
        "mlp": _fit_mlp(X, y, seed=seed),
        "mlp_noalpha": _fit_mlp(X_noalpha, y, seed=seed + 2),
        "hybrid": _fit_mlp(X, y_resid, seed=seed + 1),
        "random": frozen,
    }


def _mlp_weights(model: WeightMLP, split: Split, alpha: float) -> np.ndarray:
    X = split.feat.copy()
    X[:, 5] = alpha
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy().astype(np.float64)


# Registry: variant name -> (split, alpha, models) -> per-step weights (>= 0).
# Adding a family member means adding one entry here.
WEIGHT_FAMILY: Dict[
    str, Callable[[Split, float, Dict[str, WeightMLP]], np.ndarray]
] = {
    "w0": lambda s, a, m: np.zeros(len(s)),
    "pf": lambda s, a, m: np.clip(1.0 - s.p_max, 0.0, None),
    "mlp": lambda s, a, m: _mlp_weights(m["mlp"], s, a),
    "mlp_noalpha": lambda s, a, m: _mlp_weights(m["mlp_noalpha"], s, 0.0),
    "hybrid": lambda s, a, m: np.clip(1.0 - s.p_max, 0.0, None)
    + _mlp_weights(m["hybrid"], s, a),
    "random": lambda s, a, m: _mlp_weights(m["random"], s, a),
}
WEIGHTS = tuple(WEIGHT_FAMILY)
PARAMETER_FREE = ("w0", "pf")  # entitled to the full calibration set
