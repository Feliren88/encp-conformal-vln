"""Record rollouts (argmax) and the closed-loop help-seeking rollout.

(recs_by_ep, preds, obj_recs_by_ep). Records are only collected while the
teacher is defined (on-path or STOP); the walk itself is always the plain
argmax walk, so the trajectory equals the baseline.

Three rollouts, deliberately not unified: the DUET graph-map loop, the
adapter loop (HAMT/RecBERT), and HAMT's navref REVERIE loop are genuinely
different algorithms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from vln_backends.bootstrap import GraphMap


def _record_step(
    recs_by_ep,
    ob,
    t: int,
    n_cands: int,
    logits_row,
    teacher_idx: int,
    probs_row,
) -> None:
    recs_by_ep.setdefault(ob["instr_id"], []).append(
        {
            "scan": ob["scan"],
            "step": t,
            "n_cands": n_cands,
            "logits": logits_row.detach().half().cpu(),
            "teacher_idx": teacher_idx,
            "p_max": float(probs_row.max().item()),
            "p_teacher": (
                float(probs_row[teacher_idx].item())
                if teacher_idx < probs_row.numel()
                else 0.0
            ),
        }
    )


@torch.no_grad()
def record_duet_split(
    agent: Any, env: Any, args: Any
) -> Tuple[
    Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], Dict[str, Any]
]:
    """DUET graph-map rollout; records nav logits (+ REVERIE object head)."""
    agent.env = env
    env.reset_epoch(shuffle=False)
    recs_by_ep: Dict[str, List[Dict[str, Any]]] = {}
    traj_by_ep: Dict[str, Dict[str, Any]] = {}
    obj_recs_by_ep: Dict[str, Dict[str, Any]] = {}
    use_local = getattr(args, "action_space", "full") == "local"
    n_done = 0

    while n_done < len(env.data):
        obs = env.reset()
        agent._update_scanvp_cands(obs)
        bs = len(obs)
        gmaps = [GraphMap(ob["viewpoint"]) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)
        traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [[ob["viewpoint"]]],
                "details": {},
            }
            for ob in obs
        ]
        lang_in = agent._language_variable(obs)
        txt_emb = agent.vln_bert("language", lang_in)
        ended = np.array([False] * bs)

        for t in range(args.max_action_len):
            for i, gm in enumerate(gmaps):
                if not ended[i]:
                    gm.node_step_ids[obs[i]["viewpoint"]] = t + 1
            pano_in = agent._panorama_feature_variable(obs)
            pano_emb, pano_masks = agent.vln_bert("panorama", pano_in)
            avg_pano = torch.sum(
                pano_emb * pano_masks.unsqueeze(2), 1
            ) / torch.sum(pano_masks, 1, keepdim=True)
            for i, gm in enumerate(gmaps):
                if not ended[i]:
                    gm.update_node_embed(
                        obs[i]["viewpoint"], avg_pano[i], rewrite=True
                    )
                    for j, cvp in enumerate(pano_in["cand_vpids"][i]):
                        if not gm.graph.visited(cvp):
                            gm.update_node_embed(cvp, pano_emb[i, j])
            nav_in = agent._nav_gmap_variable(obs, gmaps)
            vp_args = (
                obs,
                gmaps,
                pano_emb,
                pano_in["cand_vpids"],
                pano_in["view_lens"],
            )
            if "obj_lens" in pano_in:  # object-nav agent (REVERIE)
                vp_args += (pano_in["obj_lens"],)
            vp_args += (pano_in["nav_types"],)
            nav_in.update(agent._nav_vp_variable(*vp_args))
            nav_in.update(
                {"txt_embeds": txt_emb, "txt_masks": lang_in["txt_masks"]}
            )
            nav_out = agent.vln_bert("navigation", nav_in)

            if use_local:
                logits, vpids, vmask = (
                    nav_out["local_logits"],
                    nav_in["vp_cand_vpids"],
                    None,
                )
            else:
                logits, vpids, vmask = (
                    nav_out["fused_logits"],
                    nav_in["gmap_vpids"],
                    nav_in["gmap_visited_masks"],
                )
            probs = torch.softmax(logits, 1)
            for i, gm in enumerate(gmaps):
                if not ended[i]:
                    gm.node_stop_scores[obs[i]["viewpoint"]] = {
                        "stop": probs[i, 0].item()
                    }

            teacher = agent._teacher_action(
                obs, vpids, ended, visited_masks=vmask
            )
            for i in range(bs):
                if ended[i]:
                    continue
                ta = int(teacher[i].item())
                if ta != args.ignoreid:
                    _record_step(
                        recs_by_ep,
                        obs[i],
                        t,
                        len(vpids[i]),
                        logits[i],
                        ta,
                        probs[i],
                    )

            # Object head: overwrite every step so the surviving record is the
            # stop-viewpoint grounding distribution.
            if "obj_logits" in nav_out:
                for i in range(bs):
                    if ended[i]:
                        continue
                    objids = [str(x) for x in obs[i].get("obj_ids", [])]
                    if not objids:
                        continue
                    vl = int(pano_in["view_lens"][i])
                    obj_logits = (
                        nav_out["obj_logits"][i, vl + 1 : vl + 1 + len(objids)]
                        .detach()
                        .float()
                        .cpu()
                    )
                    gt = str(obs[i].get("gt_obj_id"))
                    teacher_obj = next(
                        (j for j, o in enumerate(objids) if o == gt), -1
                    )
                    obj_recs_by_ep[obs[i]["instr_id"]] = {
                        "obj_logits": obj_logits.half(),
                        "teacher_idx": int(teacher_obj),
                        "n_objs": len(objids),
                    }

            _, a_t = logits.max(1)
            cpu_a = []
            for i in range(bs):
                stop = (
                    ended[i]
                    or a_t[i] == 0
                    or nav_in["no_vp_left"][i]
                    or t == args.max_action_len - 1
                )
                cpu_a.append(None if stop else vpids[i][a_t[i]])
            agent.make_equiv_action(cpu_a, gmaps, obs, traj)
            obs = env._get_obs()
            agent._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)
            ended[:] = np.logical_or(ended, [a is None for a in cpu_a])
            if ended.all():
                break
        for ti in traj:
            traj_by_ep[ti["instr_id"]] = ti
        n_done += bs
    preds = [
        {"instr_id": k, "trajectory": v["path"]} for k, v in traj_by_ep.items()
    ]
    return recs_by_ep, preds, obj_recs_by_ep


@torch.no_grad()
def record_adapter_split(
    agent: Any, env: Any, args: Any
) -> Tuple[
    Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], Dict[str, Any]
]:
    """HAMT / RecBERT rollout through the shared adapter interface. Also
    records the walked trajectory so the SR sanity gate covers all backends."""
    agent.set_env(env)
    agent.feedback = "argmax"
    env.reset_epoch(shuffle=False)
    recs_by_ep: Dict[str, List[Dict[str, Any]]] = {}
    traj_by_ep: Dict[str, Dict[str, Any]] = {}
    ignoreid = getattr(args, "ignoreid", -100)
    n_done = 0

    while n_done < len(env.data):
        obs = env.reset()
        bs = len(obs)
        traj = [
            {"instr_id": ob["instr_id"], "path": [[ob["viewpoint"]]]}
            for ob in obs
        ]
        txt_embeds, txt_masks, state, state2 = agent.reset_episode(obs)
        ended = np.array([False] * bs)

        for t in range(args.max_action_len):
            logits, vpids = agent.step_logits(
                obs, state, state2, txt_embeds, txt_masks
            )
            teacher = agent.teacher_action(obs, ended)
            probs = torch.softmax(logits, 1)
            for i in range(bs):
                if ended[i]:
                    continue
                ta = int(teacher[i].item())
                if ta != ignoreid:
                    _record_step(
                        recs_by_ep,
                        obs[i],
                        t,
                        len(vpids[i]) + 1,
                        logits[i],
                        ta,
                        probs[i],
                    )
            _, a_t = logits.max(1)
            cpu_a = []
            for i in range(bs):
                ai = int(a_t[i].item())
                if ended[i] or ai == len(vpids[i]):  # STOP index
                    cpu_a.append(None)
                    ended[i] = True
                else:
                    cpu_a.append(ai)
            state, state2 = agent.update_history(
                obs, txt_embeds, txt_masks, cpu_a, state, state2, ended
            )
            agent.make_equiv_action(
                cpu_a, obs, [{"path": []} for _ in range(bs)]
            )
            obs = env._get_obs(t=t + 1)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]["path"].append([ob["viewpoint"]])
            if ended.all():
                break
        for ti in traj:
            traj_by_ep[ti["instr_id"]] = ti
        n_done += bs
    preds = [
        {"instr_id": k, "trajectory": v["path"]} for k, v in traj_by_ep.items()
    ]
    return recs_by_ep, preds, {}


@torch.no_grad()
def record_hamt_reverie_split(
    agent: Any, env: Any, args: Any
) -> Tuple[
    Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], Dict[str, Any]
]:
    """HAMT navref (REVERIE) rollout: nav logits + object head. Mirrors
    reverie/agent.py:rollout so the trajectory (and published SR) match."""
    import torch.nn.functional as F

    l2m = agent._v10_length2mask  # bound by build_hamt_reverie
    agent.env = env
    agent.feedback = "argmax"
    env.reset_epoch(shuffle=False)
    recs_by_ep: Dict[str, List[Dict[str, Any]]] = {}
    traj_by_ep: Dict[str, Dict[str, Any]] = {}
    obj_recs_by_ep: Dict[str, Dict[str, Any]] = {}
    n_done = 0

    while n_done < len(env.data):
        obs = env.reset()
        bs = len(obs)
        txt_ids, txt_masks, _ = agent._language_variable(obs)
        txt_embeds = agent.vln_bert(
            mode="language", txt_ids=txt_ids, txt_masks=txt_masks
        )
        traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [(ob["viewpoint"], ob["heading"], ob["elevation"])],
                "predObjId": str(None),
            }
            for ob in obs
        ]
        ended = np.array([False] * bs)
        hist_embeds = [agent.vln_bert("history").expand(bs, -1)]
        hist_lens = [1] * bs

        for t in range(args.max_action_len):
            (ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, _) = (
                agent._cand_pano_feature_variable(obs)
            )
            obj_feats, obj_angles, obj_poses, obj_lens = (
                agent._object_variable(obs)
            )
            ob_img_max_len = ob_img_feats.size(1)
            outputs = agent.vln_bert(
                mode="visual",
                txt_embeds=txt_embeds,
                txt_masks=txt_masks,
                hist_embeds=hist_embeds,
                hist_lens=hist_lens,
                ob_img_feats=ob_img_feats,
                ob_ang_feats=ob_ang_feats,
                ob_nav_types=ob_nav_types,
                ob_masks=l2m(ob_lens).logical_not(),
                obj_feats=obj_feats,
                obj_poses=obj_poses,
                obj_angles=obj_angles,
                obj_masks=l2m(obj_lens).logical_not(),
                return_states=False,
            )
            obj_logits = outputs["obj_logits"]
            # navref action space = candidates + "ground here" (best object).
            act_logits = torch.cat(
                [outputs["act_logits"], obj_logits.max(1).values.unsqueeze(1)],
                1,
            )
            target, ref_target = agent._teacher_action(
                obs, ended, ob_img_max_len
            )
            probs = F.softmax(act_logits, 1)
            for i in range(bs):
                if ended[i]:
                    continue
                ta = int(
                    target[i].item()
                    if hasattr(target[i], "item")
                    else target[i]
                )
                if ta == args.ignoreid:
                    continue
                _record_step(
                    recs_by_ep,
                    obs[i],
                    t,
                    ob_img_max_len + 1,
                    act_logits[i],
                    ta,
                    probs[i],
                )
                n_obj = int(
                    obj_lens[i].item()
                    if hasattr(obj_lens[i], "item")
                    else obj_lens[i]
                )
                if n_obj > 0:
                    rt = int(
                        ref_target[i].item()
                        if hasattr(ref_target[i], "item")
                        else ref_target[i]
                    )
                    obj_recs_by_ep[obs[i]["instr_id"]] = {
                        "obj_logits": obj_logits[i, :n_obj]
                        .detach()
                        .half()
                        .cpu(),
                        "teacher_idx": rt if 0 <= rt < n_obj else -1,
                        "n_objs": n_obj,
                    }

            cpu_a_t = act_logits.max(1).indices.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                stopping = (
                    (next_id >= ob_img_max_len)
                    or (t == args.max_action_len - 1)
                ) and not ended[i]
                if stopping:
                    if len(obs[i]["candidate_obj"][2]) == 0:
                        traj[i]["predObjId"] = str(None)
                    else:
                        ref = obj_logits[i, : obj_lens[i]].max(0).indices
                        traj[i]["predObjId"] = obs[i]["candidate_obj"][2][ref]
                if (
                    (next_id >= ob_img_max_len)
                    or (next_id == args.ignoreid)
                    or ended[i]
                ):
                    cpu_a_t[i] = -1

            (hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats) = (
                agent._history_variable(obs)
            )
            prev_act_angle = np.zeros((bs, args.angle_feat_size), np.float32)
            for i, next_id in enumerate(cpu_a_t):
                if next_id != -1:
                    prev_act_angle[i] = obs[i]["candidate"][next_id][
                        "feature"
                    ][-args.angle_feat_size :]
            hist_embeds.append(
                agent.vln_bert(
                    mode="history",
                    hist_img_feats=hist_img_feats,
                    hist_ang_feats=torch.from_numpy(prev_act_angle).cuda(),
                    hist_pano_img_feats=hist_pano_img_feats,
                    hist_pano_ang_feats=hist_pano_ang_feats,
                    ob_step=t,
                )
            )
            for i in range(bs):
                if not ended[i]:
                    hist_lens[i] += 1
            agent.make_equiv_action(cpu_a_t, obs, traj)
            obs = agent.env._get_obs()
            ended[:] = np.logical_or(ended, cpu_a_t == -1)
            if ended.all():
                break
        for tr in traj:
            traj_by_ep[tr["instr_id"]] = tr
        n_done += bs
    preds = [
        {"instr_id": k, "trajectory": v["path"], "predObjId": v["predObjId"]}
        for k, v in traj_by_ep.items()
    ]
    return recs_by_ep, preds, obj_recs_by_ep


@torch.no_grad()
def closedloop_duet_split(
    agent: Any,
    env: Any,
    args: Any,
    q_hat: float,
    trigger: str = "none",
    param: float = 0.0,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Closed-loop DUET rollout (REVIEW.md #15): when the trigger fires, a
    simulated operator supplies the teacher action; otherwise argmax.

    trigger 'set' : ask when |C_alpha(x_t)| > param (prediction-set trigger,
                    THR score, threshold q_hat * (2 - p_max));
    trigger 'pmax': ask when p_max < param (confidence baseline);
    trigger 'none': plain argmax (the no-help baseline).

    Returns (preds, n_asks, n_steps). Asks where the teacher is undefined
    (ignoreid) fall back to argmax and are not counted.
    """
    agent.env = env
    env.reset_epoch(shuffle=False)
    traj_by_ep: Dict[str, Dict[str, Any]] = {}
    n_asks = n_steps = 0
    n_done = 0

    while n_done < len(env.data):
        obs = env.reset()
        agent._update_scanvp_cands(obs)
        bs = len(obs)
        gmaps = [GraphMap(ob["viewpoint"]) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)
        traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [[ob["viewpoint"]]],
                "details": {},
            }
            for ob in obs
        ]
        lang_in = agent._language_variable(obs)
        txt_emb = agent.vln_bert("language", lang_in)
        ended = np.array([False] * bs)

        for t in range(args.max_action_len):
            for i, gm in enumerate(gmaps):
                if not ended[i]:
                    gm.node_step_ids[obs[i]["viewpoint"]] = t + 1
            pano_in = agent._panorama_feature_variable(obs)
            pano_emb, pano_masks = agent.vln_bert("panorama", pano_in)
            avg_pano = torch.sum(
                pano_emb * pano_masks.unsqueeze(2), 1
            ) / torch.sum(pano_masks, 1, keepdim=True)
            for i, gm in enumerate(gmaps):
                if not ended[i]:
                    gm.update_node_embed(
                        obs[i]["viewpoint"], avg_pano[i], rewrite=True
                    )
                    for j, cvp in enumerate(pano_in["cand_vpids"][i]):
                        if not gm.graph.visited(cvp):
                            gm.update_node_embed(cvp, pano_emb[i, j])
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
            logits = nav_out["fused_logits"]
            vpids = nav_in["gmap_vpids"]
            vmask = nav_in["gmap_visited_masks"]
            probs = torch.softmax(logits, 1)
            for i, gm in enumerate(gmaps):
                if not ended[i]:
                    gm.node_stop_scores[obs[i]["viewpoint"]] = {
                        "stop": probs[i, 0].item()
                    }

            teacher = agent._teacher_action(
                obs, vpids, ended, visited_masks=vmask
            )
            _, a_argmax = logits.max(1)
            chosen = a_argmax.cpu().numpy().copy()
            for i in range(bs):
                if ended[i]:
                    continue
                n_steps += 1
                p = probs[i].float().cpu().numpy()
                lg = logits[i].float().cpu().numpy()
                valid = np.isfinite(lg)
                pv = p[valid]
                pm = float(pv.max())
                if trigger == "set":
                    thr = q_hat * (2.0 - pm)
                    set_size = max(int(np.sum((1.0 - pv) <= thr)), 1)
                    ask = set_size > param
                elif trigger == "pmax":
                    ask = pm < param
                else:
                    ask = False
                ta = int(teacher[i].item())
                if ask and ta != args.ignoreid:
                    n_asks += 1
                    chosen[i] = ta
            cpu_a = []
            for i in range(bs):
                stop = (
                    ended[i]
                    or chosen[i] == 0
                    or nav_in["no_vp_left"][i]
                    or t == args.max_action_len - 1
                )
                cpu_a.append(None if stop else vpids[i][chosen[i]])
            agent.make_equiv_action(cpu_a, gmaps, obs, traj)
            obs = env._get_obs()
            agent._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)
            ended[:] = np.logical_or(ended, [a is None for a in cpu_a])
            if ended.all():
                break
        for ti in traj:
            traj_by_ep[ti["instr_id"]] = ti
        n_done += bs
    preds = [
        {"instr_id": k, "trajectory": v["path"]} for k, v in traj_by_ep.items()
    ]
    return preds, n_asks, n_steps


ROLLOUTS = {
    "duet": record_duet_split,
    "adapter": record_adapter_split,
    "hamt_reverie": record_hamt_reverie_split,
}
