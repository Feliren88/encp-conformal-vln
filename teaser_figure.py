"""Teaser figure for the paper (page 2).

Two subcommands:

  replay  -- re-run DUET-full on the single val-unseen episode 1070_1
             (the qualitative episode of the paper) and record, per step,
             the candidate viewpoint ids alongside the logits, so that the
             prediction set can be drawn on real imagery. Verifies the
             replay against results/qualitative_episode.json before
             writing results/teaser_episode.json.

  render  -- compose paper/figures/fig_teaser.png from the replay record,
             the Matterport3D skybox images, and the scan connectivity
             graph. CPU only.

Run replay with the vln_duet_conformal env on a GPU node, e.g.
  python teaser_figure.py replay
  python teaser_figure.py render
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

V10_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(V10_DIR, "results")
FIG_DIR = os.path.join(V10_DIR, "paper", "figures")
BASE_DIR = "/fs04/scratch2/pr65/vfvic1/thesis"
SCAN_DATA_DIR = os.path.join(BASE_DIR, "mp3d_mount", "v1", "scans")
CONN_DIR = os.path.join(BASE_DIR, "R2R", "connectivity")

INSTR_ID = "1070_1"
Q_HAT = 0.9886935967679575  # deployed threshold (THR, alpha=0.10, DUET-full)


# ------------------------------------------------------------------ replay
def cmd_replay(_: argparse.Namespace) -> None:
    import torch

    sys.path.insert(0, V10_DIR)
    from vln_backends.builders import build_backend
    from vln_backends.bootstrap import GraphMap

    agent, _cal_env, env, args, _ = build_backend(
        "duet", "full", "prevalent", BASE_DIR, "r2r"
    )
    env.data = [ep for ep in env.data if ep["instr_id"] == INSTR_ID]
    assert env.data, f"episode {INSTR_ID} not in val_unseen"
    env.batch_size = 1
    env.env.sims = env.env.sims[:1]  # getStates iterates every sim
    gt_path = list(env.data[0]["path"])

    agent.env = env
    env.reset_epoch(shuffle=False)
    obs = env.reset()
    agent._update_scanvp_cands(obs)
    i = next(k for k, ob in enumerate(obs) if ob["instr_id"] == INSTR_ID)
    instruction = obs[i]["instruction"]
    scan = obs[i]["scan"]

    bs = len(obs)
    gmaps = [GraphMap(ob["viewpoint"]) for ob in obs]
    for k, ob in enumerate(obs):
        gmaps[k].update_graph(ob)
    traj = [
        {"instr_id": ob["instr_id"], "path": [[ob["viewpoint"]]]}
        for ob in obs
    ]
    lang_in = agent._language_variable(obs)
    txt_emb = agent.vln_bert("language", lang_in)
    ended = np.array([False] * bs)
    steps = []

    with torch.no_grad():
        for t in range(args.max_action_len):
            for k, gm in enumerate(gmaps):
                if not ended[k]:
                    gm.node_step_ids[obs[k]["viewpoint"]] = t + 1
            pano_in = agent._panorama_feature_variable(obs)
            pano_emb, pano_masks = agent.vln_bert("panorama", pano_in)
            avg_pano = torch.sum(
                pano_emb * pano_masks.unsqueeze(2), 1
            ) / torch.sum(pano_masks, 1, keepdim=True)
            for k, gm in enumerate(gmaps):
                if not ended[k]:
                    gm.update_node_embed(
                        obs[k]["viewpoint"], avg_pano[k], rewrite=True
                    )
                    for j, cvp in enumerate(pano_in["cand_vpids"][k]):
                        if not gm.graph.visited(cvp):
                            gm.update_node_embed(cvp, pano_emb[k, j])
            nav_in = agent._nav_gmap_variable(obs, gmaps)
            nav_in.update(
                agent._nav_vp_variable(
                    obs,
                    gmaps,
                    pano_emb,
                    pano_in["cand_vpids"],
                    pano_in["view_lens"],
                    pano_in["nav_types"],
                )
            )
            nav_in.update(
                {"txt_embeds": txt_emb, "txt_masks": lang_in["txt_masks"]}
            )
            nav_out = agent.vln_bert("navigation", nav_in)
            logits, vpids, vmask = (
                nav_out["fused_logits"],
                nav_in["gmap_vpids"],
                nav_in["gmap_visited_masks"],
            )
            teacher = agent._teacher_action(
                obs, vpids, ended, visited_masks=vmask
            )
            _, a_t = logits.max(1)

            if not ended[i]:
                row = logits[i].detach().float().cpu().numpy()
                steps.append(
                    {
                        "step": t,
                        "viewpoint": obs[i]["viewpoint"],
                        "position": list(obs[i]["position"]),
                        "heading": float(obs[i]["heading"]),
                        "elevation": float(obs[i]["elevation"]),
                        "vpids": [
                            v if v is not None else "STOP"
                            for v in vpids[i]
                        ],
                        "logits": [float(x) for x in row],
                        "teacher_idx": int(teacher[i].item()),
                        "argmax_idx": int(a_t[i].item()),
                        # simulator's own headings for adjacent candidates
                        "adjacent": [
                            {
                                "vpid": c["viewpointId"],
                                "heading": float(
                                    c["heading"]
                                    + obs[i]["heading"]
                                ),
                                "elevation": float(
                                    c["elevation"]
                                    + obs[i]["elevation"]
                                ),
                                "position": [
                                    float(x) for x in c["position"]
                                ],
                            }
                            for c in obs[i]["candidate"]
                        ],
                    }
                )

            cpu_a = []
            for k in range(bs):
                stop = (
                    ended[k]
                    or a_t[k] == 0
                    or nav_in["no_vp_left"][k]
                    or t == args.max_action_len - 1
                )
                cpu_a.append(None if stop else vpids[k][a_t[k]])
            if not ended[i]:
                steps[-1]["action"] = (
                    cpu_a[i] if cpu_a[i] is not None else "STOP"
                )
            agent.make_equiv_action(cpu_a, gmaps, obs, traj)
            obs = env._get_obs()
            agent._update_scanvp_cands(obs)
            for k, ob in enumerate(obs):
                if not ended[k]:
                    gmaps[k].update_graph(ob)
            ended[:] = np.logical_or(ended, [a is None for a in cpu_a])
            if ended.all():
                break

    agent_path = [p[0] for p in traj[i]["path"]]
    record = {
        "instr_id": INSTR_ID,
        "scan": scan,
        "instruction": instruction,
        "q_hat": Q_HAT,
        "gt_path": gt_path,
        "agent_path": agent_path,
        "steps": steps,
    }

    # -------- verification gate against the paper's qualitative episode
    with open(os.path.join(RES_DIR, "qualitative_episode.json")) as f:
        ref = json.load(f)
    assert abs(ref["q_hat"] - Q_HAT) < 1e-9
    assert len(ref["steps"]) == len(steps), (
        f'step count {len(steps)} != {len(ref["steps"])}'
    )
    for r, s in zip(ref["steps"], steps):
        lg = np.array(s["logits"])
        valid = np.isfinite(lg)
        p = np.zeros_like(lg)
        p[valid] = np.exp(lg[valid] - lg[valid].max())
        p[valid] /= p[valid].sum()
        pmax = p[valid].max()
        snorm = (1.0 - p[valid]) / (2.0 - pmax)
        set_size = int((snorm <= Q_HAT).sum())
        assert int(valid.sum()) == r["n_valid"], (
            f'step {s["step"]}: n_valid {valid.sum()} != {r["n_valid"]}'
        )
        assert abs(pmax - r["p_max"]) < 2e-3, (
            f'step {s["step"]}: p_max {pmax:.4f} != {r["p_max"]:.4f}'
        )
        assert set_size == r["set_size"], (
            f'step {s["step"]}: |C| {set_size} != {r["set_size"]}'
        )
        ti = s["teacher_idx"]
        t_in = bool(valid[ti]) and (
            (1.0 - p[ti]) / (2.0 - pmax) <= Q_HAT
        )
        assert t_in == r["teacher_in"], f'step {s["step"]}: teacher_in'
        argmax_err = s["argmax_idx"] != ti
        assert argmax_err == r["argmax_err"], f'step {s["step"]}: err'
    print(f"gate PASSED: {len(steps)} steps match qualitative_episode.json")

    out = os.path.join(RES_DIR, "teaser_episode.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=1)
    print("wrote", out)


# ------------------------------------------------------------------ render
def cmd_render(a: argparse.Namespace) -> None:
    from render_teaser import render  # local module, CPU only

    render(
        os.path.join(RES_DIR, "teaser_episode.json"),
        SCAN_DATA_DIR,
        CONN_DIR,
        os.path.join(FIG_DIR, "fig_teaser.png"),
        pano_offset_deg=a.pano_offset,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("replay")
    r = sub.add_parser("render")
    r.add_argument("--pano_offset", type=float, default=0.0)
    a = ap.parse_args()
    if a.cmd == "replay":
        cmd_replay(a)
    else:
        cmd_render(a)


if __name__ == "__main__":
    main()
