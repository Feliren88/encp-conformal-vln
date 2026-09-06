"""Compose the paper's teaser figure from real data (CPU only).

Panels: instruction band (top); two skybox panorama strips (left,
steps 0 and 2 of episode 1070_1) with prediction-set markers at the
candidates' true world headings; the scan's actual connectivity graph
(right) with the instruction path, the executed path, and the step-2
prediction set.

Skybox convention (calibrated on this dataset in-session): faces 1..4
are horizontal, face k has centre yaw (270 - 90*(k-1)) mod 360, and
in-face x runs opposite to yaw (gnomonic, mirrored).
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Okabe-Ito (colour-vision-deficiency-safe); shapes carry identity too.
C_TEACHER = "#009E73"  # bluish green, filled circle
C_ARGMAX = "#D55E00"  # vermillion, X
C_SET = "#56B4E9"  # sky blue, open circle
C_AGENT = "#E69F00"  # orange, executed path
C_NODE = "#B0B0B0"

FACE_CENTER = {1: 270.0, 2: 180.0, 3: 90.0, 4: 0.0}


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def load_strip(
    skydir: str, vp: str, face_order: List[int], y0f: float, y1f: float
) -> np.ndarray:
    faces = []
    for k in face_order:
        img = Image.open(
            os.path.join(skydir, f"{vp}_skybox{k}_sami.jpg")
        ).convert("RGB")
        faces.append(np.asarray(img))
    strip = np.concatenate(faces, axis=1)
    h = strip.shape[0]
    return strip[int(y0f * h) : int(y1f * h)]


def project(
    yaw_deg: float,
    elev_rad: float,
    face_order: List[int],
    face_px: int,
    y0_px: float,
) -> Optional[Tuple[float, float]]:
    """(x, y) pixel of a world direction on the concatenated strip."""
    for slot, k in enumerate(face_order):
        off = _wrap(yaw_deg - FACE_CENTER[k])
        if abs(off) <= 45.0:
            u = -math.tan(math.radians(off))  # calibrated mirror
            x = face_px * (slot + (u + 1.0) / 2.0)
            y_full = 0.5 * face_px - (
                math.tan(elev_rad) / math.cos(math.radians(off))
            ) * (0.5 * face_px)
            return x, y_full - y0_px
    return None


def step_probs(step: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
    lg = np.array(step["logits"])
    valid = np.isfinite(lg)
    p = np.zeros_like(lg)
    p[valid] = np.exp(lg[valid] - lg[valid].max())
    p[valid] /= p[valid].sum()
    return p, valid, float(p[valid].max())


def in_set(step: Dict, q_hat: float) -> np.ndarray:
    p, valid, pmax = step_probs(step)
    snorm = np.where(valid, (1.0 - p) / (2.0 - pmax), np.inf)
    return snorm <= q_hat


def bearing(cur: np.ndarray, tgt: np.ndarray) -> float:
    return math.degrees(math.atan2(tgt[0] - cur[0], tgt[1] - cur[1])) % 360.0


def load_graph(conn_dir: str, scan: str):
    with open(os.path.join(conn_dir, f"{scan}_connectivity.json")) as f:
        raw = json.load(f)
    pos: Dict[str, np.ndarray] = {}
    idx: Dict[int, str] = {}
    for i, e in enumerate(raw):
        idx[i] = e["image_id"]
        if e["included"]:
            m = e["pose"]
            pos[e["image_id"]] = np.array([m[3], m[7], m[11]])
    edges = []
    for i, e in enumerate(raw):
        if not e["included"]:
            continue
        for j, ok in enumerate(e["unobstructed"]):
            if ok and j > i and raw[j]["included"]:
                edges.append((idx[i], idx[j]))
    return pos, edges


def draw_pano(
    ax,
    record: Dict,
    step: Dict,
    skydir: str,
    face_order: List[int],
    q_hat: float,
    title: str,
) -> None:
    Y0, Y1 = 0.32, 0.88
    strip = load_strip(skydir, step["viewpoint"], face_order, Y0, Y1)
    ax.imshow(strip)
    ax.set_xlim(0, strip.shape[1])
    ax.set_ylim(strip.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=7.2, pad=2.2, loc="left")

    p, valid, pmax = step_probs(step)
    member = in_set(step, q_hat)
    cur = np.array(step["position"])
    face_px = strip.shape[1] // 4
    adj = {a["vpid"]: a for a in step["adjacent"]}
    halo = dict(path_effects=None)

    seen_xy: List[Tuple[float, float]] = []
    for j, vpid in enumerate(step["vpids"]):
        if vpid == "STOP" or not valid[j] or not member[j]:
            continue
        if vpid in adj:
            yaw = math.degrees(adj[vpid]["heading"]) % 360.0
            elev = adj[vpid]["elevation"]
        else:
            continue  # non-adjacent frontier members: graph panel only
        xy = project(yaw, elev, face_order, face_px, Y0 * face_px)
        if xy is None:
            continue
        x, y = xy
        x = min(max(x, 16), strip.shape[1] - 16)
        y = min(max(y, 16), strip.shape[0] - 16)
        for px_, py_ in seen_xy:  # de-overlap coincident markers
            if abs(x - px_) < 60 and abs(y - py_) < 60:
                x += 95
        seen_xy.append((x, y))
        top_half = y < strip.shape[0] * 0.5
        dy = -13 if top_half else 13  # offset pts: down if top, up if bottom
        is_t = j == step["teacher_idx"]
        is_a = j == step["argmax_idx"]
        if is_a and not is_t:
            ax.plot(x, y, "x", ms=8.5, mew=2.8, color="white", zorder=4)
            ax.plot(x, y, "x", ms=7, mew=1.8, color=C_ARGMAX, zorder=5)
            ax.annotate(
                f"Policy choice (wrong)  p={p[j]:.2f}",
                (x, y),
                xytext=(5, int(dy * 2.2)),  # second label row
                textcoords="offset points",
                fontsize=6.0,
                color="white",
                va="center",
                bbox=dict(
                    facecolor=C_ARGMAX, edgecolor="none", pad=1.1, alpha=0.9
                ),
                zorder=6,
            )
        elif is_t:
            ax.plot(x, y, "o", ms=8.5, mfc=C_TEACHER, mec="white", mew=1.4,
                    zorder=5)
            lab = f"Correct action  p={p[j]:.2f}" if not is_a else (
                f"Correct = policy choice  p={p[j]:.2f}"
            )
            ax.annotate(
                lab,
                (x, y),
                xytext=(5, dy),
                textcoords="offset points",
                fontsize=6.0,
                color="white",
                va="center",
                bbox=dict(
                    facecolor=C_TEACHER, edgecolor="none", pad=1.1, alpha=0.9
                ),
                zorder=6,
            )
        else:
            ax.plot(x, y, "o", ms=6.5, mfc="none", mec="white", mew=2.4,
                    zorder=4)
            ax.plot(x, y, "o", ms=6.5, mfc="none", mec=C_SET, mew=1.3,
                    zorder=5)
    _ = halo


def draw_graph(ax, record: Dict, conn_dir: str, q_hat: float) -> None:
    scan = record["scan"]
    pos, edges = load_graph(conn_dir, scan)
    step = record["steps"][2]
    member = in_set(step, q_hat)
    set_vps = {
        v
        for j, v in enumerate(step["vpids"])
        if member[j] and v != "STOP"
    }
    gt = record["gt_path"]
    ag = record["agent_path"]

    keep = [p for v, p in pos.items()]
    ref = np.array([pos[v] for v in set(gt) | set(ag) if v in pos])
    lo = ref.min(0) - 4.0
    hi = ref.max(0) + 4.0
    _ = keep

    for a, b in edges:
        pa, pb = pos[a], pos[b]
        if (pa[:2] < lo[:2]).any() or (pa[:2] > hi[:2]).any():
            continue
        if (pb[:2] < lo[:2]).any() or (pb[:2] > hi[:2]).any():
            continue
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            color=C_NODE,
            lw=0.45,
            alpha=0.42,
            zorder=1,
        )
    for v, pxy in pos.items():
        if (pxy[:2] < lo[:2]).any() or (pxy[:2] > hi[:2]).any():
            continue
        ax.plot(pxy[0], pxy[1], "o", ms=1.8, mfc="#e4e4e4", mec="#b4b4b4",
                mew=0.35, zorder=2)

    gxy = np.array([pos[v][:2] for v in gt])
    ax.plot(gxy[:, 0], gxy[:, 1], "--", color=C_TEACHER, lw=1.8, zorder=3)
    axy = np.array([pos[v][:2] for v in ag])
    ax.plot(axy[:, 0], axy[:, 1], "-", color=C_AGENT, lw=1.6, alpha=0.95,
            zorder=4)
    for k in (1, 4):  # direction arrows on the executed path
        a0, a1 = axy[k], axy[k + 1]
        mid = (a0 + a1) / 2.0
        ax.annotate(
            "",
            xy=mid + (a1 - a0) * 0.18,
            xytext=mid - (a1 - a0) * 0.18,
            arrowprops=dict(arrowstyle="-|>", color=C_AGENT, lw=1.0,
                            mutation_scale=8),
            zorder=5,
        )
    ax.plot(*gxy[0], marker="s", ms=4.5, mfc="white", mec="black", mew=0.8,
            zorder=6)
    ax.plot(*gxy[-1], marker="*", ms=10, mfc="#F0E442", mec="black", mew=0.6,
            zorder=6)
    ax.plot(*axy[-1], marker="s", ms=4.0, mfc=C_AGENT, mec="black", mew=0.7,
            zorder=6)
    ax.annotate("Start", gxy[0], xytext=(4, -8), textcoords="offset points",
                fontsize=6.0, zorder=8)
    ax.annotate("Goal", gxy[-1], xytext=(3, 5), textcoords="offset points",
                fontsize=6.0, zorder=8)
    ax.annotate("Stop", axy[-1], xytext=(5, -3), textcoords="offset points",
                fontsize=6.0, color="#7a5200", zorder=8)

    for v in set_vps:
        if v not in pos:
            continue
        ax.plot(pos[v][0], pos[v][1], "o", ms=5.0, mfc="none", mec=C_SET,
                mew=1.4, zorder=5)
    tvp = step["vpids"][step["teacher_idx"]]
    avp = step["vpids"][step["argmax_idx"]]
    ax.plot(pos[tvp][0], pos[tvp][1], "o", ms=5.4, mfc=C_TEACHER,
            mec="white", mew=1.0, zorder=7)
    ax.plot(pos[avp][0], pos[avp][1], "x", ms=6.4, mew=2.0, color=C_ARGMAX,
            zorder=7)
    cvp = step["viewpoint"]
    ax.plot(pos[cvp][0], pos[cvp][1], "o", ms=9.5, mfc="none", mec="black",
            mew=1.0, zorder=7)
    ax.annotate(
        "Step 2",
        (pos[cvp][0], pos[cvp][1]),
        xytext=(-9, -12),
        textcoords="offset points",
        fontsize=6.2,
        ha="right",
    )

    # 2 m scale bar
    x0, y0 = lo[0] + 0.8, lo[1] + 0.8
    ax.plot([x0, x0 + 2.0], [y0, y0], "-", color="black", lw=1.2)
    ax.text(x0 + 1.0, y0 + 0.35, "2 m", ha="center", fontsize=6.0)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")
    ax.set_title(
        "Episode on the scan's navigation graph",
        fontsize=7.0,
        pad=2.2,
        loc="left",
    )


def render(
    record_path: str,
    scan_data_dir: str,
    conn_dir: str,
    out_path: str,
    pano_offset_deg: float = 0.0,
) -> None:
    _ = pano_offset_deg
    with open(record_path) as f:
        record = json.load(f)
    scan = record["scan"]
    skydir = os.path.join(
        scan_data_dir, scan, scan, "matterport_skybox_images"
    )
    q_hat = record["q_hat"]
    s0, s2 = record["steps"][0], record["steps"][2]
    p0, v0, pm0 = step_probs(s0)
    p2, v2, pm2 = step_probs(s2)
    n0, n2 = int(in_set(s0, q_hat).sum()), int(in_set(s2, q_hat).sum())

    fig = plt.figure(figsize=(7.16, 2.58), dpi=300)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.62, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.004,
        right=0.998,
        top=0.765,
        bottom=0.082,
        wspace=0.035,
        hspace=0.30,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    axg = fig.add_subplot(gs[:, 1])

    instr = record["instruction"].strip()
    words = instr.split()
    mid = len(instr) // 2
    acc, line1 = 0, []
    for w in words:
        if acc + len(w) > mid:
            break
        line1.append(w)
        acc += len(w) + 1
    l1, l2 = " ".join(line1), " ".join(words[len(line1):])
    fig.text(
        0.004,
        0.985,
        f"Instruction: “{l1}\n{l2}”",
        fontsize=6.9,
        va="top",
        ha="left",
        linespacing=1.35,
        family="DejaVu Sans",
    )

    draw_pano(
        ax0,
        record,
        s0,
        skydir,
        face_order=[4, 1, 2, 3],
        q_hat=q_hat,
        title=(
            f"Step 0 — policy confident ($p_{{\\max}}$={pm0:.2f}): "
            f"prediction set $C$ = 1 action"
        ),
    )
    draw_pano(
        ax2,
        record,
        s2,
        skydir,
        face_order=[3, 4, 1, 2],
        q_hat=q_hat,
        title=(
            f"Step 2 — first error ($p_{{\\max}}$={pm2:.2f}): "
            f"$C$ widens to all {n2} candidates, correct action kept"
        ),
    )
    draw_graph(axg, record, conn_dir, q_hat)

    handles = [
        Line2D([], [], marker="o", ls="none", mfc=C_TEACHER, mec="white",
               ms=6, label="Correct action (teacher)"),
        Line2D([], [], marker="x", ls="none", color=C_ARGMAX, mew=2.0,
               ms=6, label="Policy argmax (wrong)"),
        Line2D([], [], marker="o", ls="none", mfc="none", mec=C_SET,
               mew=1.3, ms=6, label="Prediction set $C_{0.1}$"),
        Line2D([], [], ls="--", color=C_TEACHER, lw=1.4,
               label="Instruction path"),
        Line2D([], [], ls="-", color=C_AGENT, lw=1.4,
               label="Executed path"),
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.002, -0.008),
        ncol=5,
        fontsize=6.0,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.45,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("wrote", out_path)
