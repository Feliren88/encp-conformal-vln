"""Backend adapters exposing one shared step interface.

reset_episode(obs)                         -> (txt_embeds, txt_masks, state,
state2)
  step_logits(obs, state, state2, txt, msk)  -> (logits, vpids)
  update_history(...)                        -> (state, state2)
  teacher_action(obs, ended)                 -> LongTensor (ignoreid off-path)
  make_equiv_action(cpu_a_t, obs, traj)         (None == STOP)

STOP convention for both adapters: index len(vpids[i]).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from vln_backends.isolation import ensure_hamt_imports


class VLNHAMTAgentAdapter:
    """Wrap HAMT's Seq2SeqCMTAgent behind the shared step interface."""

    def __init__(self, hamt_agent: Any) -> None:
        ensure_hamt_imports()
        self._agent = hamt_agent
        self.env = hamt_agent.env
        self.args = hamt_agent.args

    @property
    def feedback(self) -> str:
        return self._agent.feedback

    @feedback.setter
    def feedback(self, value: str) -> None:
        self._agent.feedback = value

    def set_env(self, env: Any) -> None:
        self._agent.env = env
        self.env = env

    @torch.no_grad()
    def reset_episode(
        self, obs: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[int]]:
        txt_ids, txt_masks, _ = self._agent._language_variable(obs)
        txt_embeds = self._agent.vln_bert(
            mode="language", txt_ids=txt_ids, txt_masks=txt_masks
        )
        hist_embeds = [self._agent.vln_bert("history").expand(len(obs), -1)]
        return txt_embeds, txt_masks, hist_embeds, [1] * len(obs)

    @torch.no_grad()
    def step_logits(
        self,
        obs: List[Dict[str, Any]],
        hist_embeds: List[Any],
        hist_lens: Any,
        txt_embeds: torch.Tensor,
        txt_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        (ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, _) = (
            self._agent._cand_pano_feature_variable(obs)
        )
        # NavCMT applies (1 - ob_masks) * -10000 internally: True = valid.
        # Inverting the polarity masks the real candidates and silently costs
        # ~20 SR points (the v6/v7 HAMT bug) -- do not wrap in `~`.
        ob_masks = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(n, dtype=torch.bool) for n in ob_lens],
            batch_first=True,
        ).cuda()
        outputs = self._agent.vln_bert(
            mode="visual",
            txt_embeds=txt_embeds,
            txt_masks=txt_masks,
            hist_embeds=hist_embeds,
            hist_lens=hist_lens,
            ob_img_feats=ob_img_feats,
            ob_ang_feats=ob_ang_feats,
            ob_nav_types=ob_nav_types,
            ob_masks=ob_masks,
            return_states=False,
        )
        vpids = [[c["viewpointId"] for c in ob["candidate"]] for ob in obs]
        return outputs[0], vpids

    @torch.no_grad()
    def update_history(
        self,
        obs: List[Dict[str, Any]],
        txt_embeds: torch.Tensor,
        txt_masks: torch.Tensor,
        cpu_a_t: List[Optional[int]],
        hist_embeds: List[Any],
        hist_lens: Any,
        ended: Any,
    ) -> Tuple[List[Any], Any]:
        (hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats) = (
            self._agent._history_variable(obs)
        )
        angle_size = self._agent.args.angle_feat_size
        prev_act_angle = np.zeros((len(obs), angle_size), np.float32)
        for i, act in enumerate(cpu_a_t):
            if act is not None and 0 <= act < len(obs[i]["candidate"]):
                prev_act_angle[i] = obs[i]["candidate"][act]["feature"][
                    -angle_size:
                ]
        step_embed = self._agent.vln_bert(
            mode="history",
            hist_img_feats=hist_img_feats,
            hist_ang_feats=torch.from_numpy(prev_act_angle).cuda(),
            hist_pano_img_feats=hist_pano_img_feats,
            hist_pano_ang_feats=hist_pano_ang_feats,
            ob_step=len(hist_embeds) - 1,
        )
        hist_embeds.append(step_embed)
        for i in range(len(obs)):
            if not ended[i]:
                hist_lens[i] += 1
        return hist_embeds, hist_lens

    def teacher_action(
        self, obs: List[Dict[str, Any]], ended: Any
    ) -> torch.Tensor:
        """Tolerant teacher: ignoreid when the argmax walk left the gt path
        (HAMT's strict teacher asserts on-path and would crash)."""
        ignoreid = self._agent.args.ignoreid
        a = np.full(len(obs), ignoreid, dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:
                continue
            for k, candidate in enumerate(ob["candidate"]):
                if candidate["viewpointId"] == ob["teacher"]:
                    a[i] = k
                    break
            else:
                if ob["teacher"] == ob["viewpoint"]:
                    a[i] = len(ob["candidate"])  # STOP
        return torch.from_numpy(a).cuda()

    def make_equiv_action(
        self,
        cpu_a_t: List[Optional[int]],
        obs: List[Dict[str, Any]],
        traj: Any,
    ) -> None:
        self._agent.make_equiv_action(
            [(-1 if a is None else a) for a in cpu_a_t], obs, traj
        )


class VLNRecBERTAgentAdapter:
    """Wrap RecBERT's Seq2SeqAgent behind the shared step interface.

    The recurrent state h_t rides in a 1-element list (None for OSCAR before
    the first visual pass) and advances INSIDE step_logits, so update_history
    is a no-op. The second state slot carries token_type_ids.
    """

    def __init__(
        self, recbert_agent, recbert_args, utils_mod, arch: str
    ) -> None:
        self._agent = recbert_agent
        self._utils = utils_mod
        self._arch = arch  # 'prevalent' | 'oscar'
        self.env = recbert_agent.env
        self.args = recbert_args

    @property
    def feedback(self) -> str:
        return getattr(self._agent, "feedback", "argmax")

    @feedback.setter
    def feedback(self, value: str) -> None:
        self._agent.feedback = value

    def set_env(self, env: Any) -> None:
        self._agent.env = getattr(env, "_env", env)
        self.env = env

    @torch.no_grad()
    def reset_episode(
        self, obs: List[Dict[str, Any]]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]], torch.Tensor
    ]:
        seq = (
            torch.from_numpy(np.array([ob["instr_encoding"] for ob in obs]))
            .long()
            .cuda()
        )
        mask = (seq != self._utils.padding_idx).long().cuda()
        token_type_ids = torch.zeros_like(mask).long().cuda()
        language_inputs = {
            "mode": "language",
            "sentence": seq,
            "attention_mask": mask,
            "lang_mask": mask,
            "token_type_ids": token_type_ids,
        }
        if self._arch == "oscar":
            language_features = self._agent.vln_bert(**language_inputs)
            h_t = None  # born at the first visual pass
        else:
            h_t, language_features = self._agent.vln_bert(**language_inputs)
        return language_features, mask, [h_t], token_type_ids

    @torch.no_grad()
    def step_logits(
        self,
        obs: List[Dict[str, Any]],
        hist_embeds: List[Any],
        hist_lens: Any,
        txt_embeds: torch.Tensor,
        txt_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        h_t = hist_embeds[0]
        input_a_t, candidate_feat, candidate_leng = self._agent.get_input_feat(
            obs
        )
        if h_t is not None:
            # Slot 0 carries the recurrent state; slots 1..L-1 are the frozen
            # language encoding (upstream agent.py keeps [:, 1:, :]).
            language_features = torch.cat(
                (h_t.unsqueeze(1), txt_embeds[:, 1:, :]), 1
            )
        else:
            language_features = txt_embeds
        visual_temp_mask = (
            self._utils.length2mask(candidate_leng) == 0
        ).long()
        self._agent.vln_bert.vln_bert.config.directions = max(candidate_leng)
        h_t_new, logits = self._agent.vln_bert(
            mode="visual",
            sentence=language_features,
            attention_mask=torch.cat((txt_masks, visual_temp_mask), dim=-1),
            lang_mask=txt_masks,
            vis_mask=visual_temp_mask,
            token_type_ids=hist_lens,
            action_feats=input_a_t,
            cand_feats=candidate_feat,
        )
        hist_embeds[0] = h_t_new
        logits = logits.masked_fill(
            self._utils.length2mask(candidate_leng), -float("inf")
        )
        vpids = [[c["viewpointId"] for c in ob["candidate"]] for ob in obs]
        return logits, vpids

    @torch.no_grad()
    def update_history(
        self,
        obs: List[Dict[str, Any]],
        txt_embeds: torch.Tensor,
        txt_masks: torch.Tensor,
        cpu_a_t: List[Optional[int]],
        hist_embeds: List[Any],
        hist_lens: Any,
        ended: Any,
    ) -> Tuple[List[Any], Any]:
        return hist_embeds, hist_lens  # state advanced in step_logits

    def teacher_action(
        self, obs: List[Dict[str, Any]], ended: Any
    ) -> torch.Tensor:
        # RecBERT's teacher is recomputed from the current state each step, so
        # it is always a visible candidate or STOP -- no off-path case.
        return self._agent._teacher_action(obs, ended)

    def make_equiv_action(
        self,
        cpu_a_t: List[Optional[int]],
        obs: List[Dict[str, Any]],
        traj: Any,
    ) -> None:
        self._agent.make_equiv_action(
            [(-1 if a is None else a) for a in cpu_a_t], obs, None, traj
        )


class RecBERTEnvWrapper:
    """Give RecBERT's R2RBatch the env surface the shared rollout expects."""

    def __init__(self, env: Any) -> None:
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_env"), name)

    def _get_obs(self, t: Optional[int] = None) -> Any:
        return self._env._get_obs()

    @property
    def shortest_distances(self) -> Dict[str, Any]:
        return self._env.distances
