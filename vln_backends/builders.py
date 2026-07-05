"""Backend builders returning (agent, cal_env, test_env, args, kind).

where rollout_kind selects the rollout in rollouts.ROLLOUTS.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from vln_backends import bootstrap as bt
from vln_backends.config import (
    DATASET_CONFIG,
    DUMP_DIR,
    HAMT_CKPT,
    HAMT_FINETUNE_SRC,
    HAMT_FT_FILE,
    RECBERT_CKPTS,
    RECBERT_FT_FILE,
    RECBERT_ROOT,
    RECBERT_SRC,
)
from vln_backends.isolation import (
    ensure_first_on_path,
    ensure_hamt_imports,
    hamt_context,
    recbert_context,
)
from vln_backends.adapters import (
    RecBERTEnvWrapper,
    VLNHAMTAgentAdapter,
    VLNRecBERTAgentAdapter,
)

logger = logging.getLogger("vln_backends.builders")


def _tmp_out(name: str) -> str:
    out = os.path.join(DUMP_DIR, name)
    for sub in ("ckpts", "logs", "preds"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return out


# ---------------------------------------------------------------------- DUET
def _duet_args(
    base_dir: str,
    ds: Dict[str, Any],
    dataset: str,
    out: str,
    action_space: str,
    conn: str,
    anno_dir: str,
) -> argparse.Namespace:
    """DUET Namespace matching the released checkpoints' training setup."""
    return argparse.Namespace(
        root_dir=base_dir,
        output_dir=out,
        seed=0,
        tokenizer="bert",
        dataset=dataset,
        fusion="dynamic",
        enc_full_graph=True,
        graph_sprels=True,
        num_l_layers=9,
        num_pano_layers=2,
        num_x_layers=4,
        image_feat_size=768,
        angle_feat_size=4,
        views=36,
        obj_feat_size=ds["obj_feat_size"],
        max_objects=ds["max_objects"],
        fix_lang_embedding=False,
        fix_pano_embedding=False,
        fix_local_branch=False,
        act_visited_nodes=False,
        expl_sample=False,
        expl_max_ratio=0.6,
        expert_policy="spl",
        max_instr_len=100 if dataset == "reverie" else 200,
        max_action_len=ds["max_action_len"],
        batch_size=8,
        ignoreid=-100,
        no_backtrack=False,
        detailed_output=True,
        optim="adamW",
        lr=1e-5,
        weight_decay=0.0,
        feedback="argmax",
        epsilon=0.1,
        gamma=0.9,
        normalize_loss="total",
        train_alg="dagger",
        ml_weight=0.2,
        entropy_loss_weight=0.01,
        dropout=0.5,
        feat_dropout=0.4,
        world_size=1,
        local_rank=-1,
        node_rank=0,
        resume_optimizer=False,
        bert_ckpt_file=None,
        aug=None,
        test=True,
        submit=False,
        img_ft_file=os.path.join(
            base_dir,
            "R2R",
            "features",
            "pth_vit_base_patch16_224_imagenet.hdf5",
        ),
        connectivity_dir=conn,
        scan_data_dir=os.path.join(base_dir, "Matterport3D", "v1", "scans"),
        ckpt_dir=os.path.join(out, "ckpts"),
        log_dir=os.path.join(out, "logs"),
        pred_dir=os.path.join(out, "preds"),
        resume_file=os.path.join(base_dir, ds["resume_file"]),
        anno_dir=anno_dir,
        backend="duet",
        action_space=action_space,
    )


def build_duet(
    base_dir: str, ds: Dict[str, Any], dataset: str, action_space: str
) -> Tuple[Any, Any, Any, argparse.Namespace, str]:
    anno_dir = os.path.join(base_dir, ds["anno_dir_key"], "annotations")
    conn = os.path.join(base_dir, "R2R", "connectivity")
    out = _tmp_out("_duet_tmp")
    feat_db = bt.ImageFeaturesDB(
        os.path.join(
            base_dir,
            "R2R",
            "features",
            "pth_vit_base_patch16_224_imagenet.hdf5",
        ),
        768,
    )
    args = _duet_args(base_dir, ds, dataset, out, action_space, conn, anno_dir)

    if dataset == "reverie":
        obj_db = bt.ReverieObjectFeatureDB(
            os.path.join(base_dir, ds["obj_ft_file"]), ds["obj_feat_size"]
        )
        obj2vps = bt.load_obj2vps(
            os.path.join(base_dir, "REVERIE", "annotations", "BBoxes.json")
        )

        def make_env(split) -> Tuple[Any, Any, Any, argparse.Namespace, str]:
            data = bt.reverie_construct_instrs(
                anno_dir,
                "reverie",
                [split],
                tokenizer="bert",
                max_instr_len=100,
            )
            return bt.ReverieObjectNavBatch(
                feat_db,
                obj_db,
                data,
                conn,
                obj2vps,
                batch_size=8,
                angle_feat_size=4,
                max_objects=ds["max_objects"],
                seed=0,
                name=split,
            )

    else:

        def make_env(split: str) -> Any:
            data = bt.r2r_construct_instrs(
                anno_dir, "r2r", [split], tokenizer="bert", max_instr_len=200
            )
            return bt.R2RNavBatch(
                feat_db,
                data,
                conn,
                batch_size=8,
                angle_feat_size=4,
                seed=0,
                name=split,
            )

    cal_env = make_env(ds["cal_split"])
    test_env = make_env(ds["test_split"])
    agent_cls = (
        bt.GMapObjectNavAgent if dataset == "reverie" else bt.GMapNavAgent
    )
    agent = agent_cls(args, cal_env, rank=0)
    start_iter = agent.load(args.resume_file)
    agent.vln_bert.eval()
    agent.critic.eval()
    logger.info(
        "DUET loaded: iter=%d agent=%s", start_iter, type(agent).__name__
    )
    return agent, cal_env, test_env, args, "duet"


# ---------------------------------------------------------------------- HAMT
def _resize_hamt_hist_pos_embeddings(
    vln_bert_wrapper: Any, num_positions: int
) -> None:
    import torch.nn as nn

    inner = vln_bert_wrapper.vln_bert
    old = inner.hist_embeddings.position_embeddings
    if old.num_embeddings != num_positions:
        inner.hist_embeddings.position_embeddings = nn.Embedding(
            num_positions, old.embedding_dim
        ).to(old.weight.device)


def build_hamt_r2r(
    base_dir: str, ds: Dict[str, Any]
) -> Tuple[Any, Any, Any, argparse.Namespace, str]:
    ensure_hamt_imports()
    anno_dir = os.path.join(base_dir, "R2R", "annotations")
    conn = os.path.join(
        HAMT_FINETUNE_SRC, "..", "datasets", "R2R", "connectivity"
    )
    out = _tmp_out("_hamt_tmp")
    with hamt_context():
        from r2r.env import R2RBatch as HamtR2RBatch
        from r2r.agent_cmt import Seq2SeqCMTAgent
    feat_db = bt.ImageFeaturesDB(HAMT_FT_FILE, 768)
    hamt_args = argparse.Namespace(
        root_dir=base_dir,
        dataset="r2r",
        tokenizer="bert",
        ob_type="pano",
        features="vitbase",
        image_feat_size=768,
        angle_feat_size=4,
        views=36,
        hist_enc_pano=True,
        num_h_layers=0,
        hist_pano_num_layers=2,
        num_l_layers=9,
        num_x_layers=4,
        no_lang_ca=False,
        act_pred_token="ob_txt",  # training value; 'ob_txt_hist' drifts
        fix_lang_embedding=False,
        fix_hist_embedding=False,
        fix_obs_embedding=False,
        max_instr_len=60,  # HAMT R2R finetuning truncation
        max_action_len=ds["max_action_len"],
        batch_size=8,
        ignoreid=-100,
        no_cand_backtrack=False,
        feedback="argmax",
        test=True,
        submit=False,
        resume_optimizer=False,
        optim="adamW",
        lr=1e-5,
        weight_decay=0.0,
        img_ft_file=HAMT_FT_FILE,
        connectivity_dir=conn,
        anno_dir=anno_dir,
        output_dir=out,
        ckpt_dir=os.path.join(out, "ckpts"),
        log_dir=os.path.join(out, "logs"),
        pred_dir=os.path.join(out, "preds"),
        world_size=1,
        local_rank=-1,
        seed=0,
        bert_ckpt_file=None,
        dropout=0.5,
        feat_dropout=0.4,
        hidden_size=768,
        num_attention_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
    )

    def instr_data(split: str) -> List[Dict[str, Any]]:
        return bt.r2r_construct_instrs(
            anno_dir, "r2r", [split], tokenizer="bert", max_instr_len=60
        )

    with hamt_context():
        cal_env = HamtR2RBatch(
            feat_db,
            instr_data("val_seen"),
            conn,
            batch_size=8,
            angle_feat_size=4,
            seed=0,
            name="val_seen",
        )
        test_env = HamtR2RBatch(
            feat_db,
            instr_data("val_unseen"),
            conn,
            batch_size=8,
            angle_feat_size=4,
            seed=0,
            name="val_unseen",
        )
        agent = Seq2SeqCMTAgent(hamt_args, cal_env, rank=0)
        # The e2e checkpoint pairs with the e2e features and 50 history
        # positions; mixing either silently costs 5-8 SR points.
        _resize_hamt_hist_pos_embeddings(agent.vln_bert, num_positions=50)
        start_iter = agent.load(HAMT_CKPT)
        agent.vln_bert.eval()
        agent.critic.eval()
    logger.info("HAMT loaded: iter=%d ckpt=%s", start_iter, HAMT_CKPT)
    args = argparse.Namespace(
        backend="hamt",
        max_action_len=ds["max_action_len"],
        ignoreid=-100,
        batch_size=8,
    )
    return VLNHAMTAgentAdapter(agent), cal_env, test_env, args, "adapter"


def build_hamt_reverie(
    base_dir: str, ds: Dict[str, Any]
) -> Tuple[Any, Any, Any, argparse.Namespace, str]:
    """HAMT navref (REVERIE) agent + envs via the REVERIE parser's defaults.

    The navref rollout needs HAMT's length2mask; it is bound inside the module
    context and attached to the agent as `_v10_length2mask`.
    """
    ensure_hamt_imports()
    out = _tmp_out("_hamt_tmp")
    root = os.path.join(base_dir, "VLN-HAMT", "datasets")
    ckpt = os.path.join(root, "REVERIE", "trained_models", "best_val_unseen")
    argv = [
        "prog",
        "--dataset",
        "reverie",
        "--ob_type",
        "pano",
        "--test",
        "--feedback",
        "argmax",
        "--root_dir",
        root,
        "--features",
        "vitbase_r2rfte2e",
        "--image_feat_size",
        "768",
        "--obj_feat_size",
        "768",
        "--max_objects",
        str(ds["max_objects"]),
        "--max_action_len",
        str(ds["max_action_len"]),
        "--max_instr_len",
        "80",
        "--batch_size",
        "8",
        "--hist_enc_pano",
        "--resume_file",
        ckpt,
        "--output_dir",
        out,
    ]
    with hamt_context():
        from reverie.parser import parse_args
        from reverie.agent import NavRefCMTAgent
        from reverie.data_utils import (
            ImageFeaturesDB as RevImgDB,
            construct_instrs,
            load_obj_database,
        )
        from reverie.env import ReverieNavRefBatch
        from utils.misc import length2mask

        old_argv, sys.argv = sys.argv, argv
        try:
            args = parse_args()
        finally:
            sys.argv = old_argv
        feat_db = RevImgDB(args.img_ft_file, args.image_feat_size)
        obj_db = load_obj_database(args.obj_ft_file, args.obj_feat_size)

        def make_env(split: str) -> Any:
            data = construct_instrs(
                args.anno_dir,
                args.dataset,
                [split],
                tokenizer=args.tokenizer,
                max_instr_len=args.max_instr_len,
            )
            return ReverieNavRefBatch(
                feat_db,
                obj_db,
                data,
                args.connectivity_dir,
                args.anno_dir,
                batch_size=args.batch_size,
                angle_feat_size=args.angle_feat_size,
                seed=args.seed,
                name=split,
                sel_data_idxs=None,
                multi_endpoints=False,
                multi_startpoints=False,
                max_objects=args.max_objects,
            )

        cal_env = make_env(ds["cal_split"])
        test_env = make_env(ds["test_split"])
        agent = NavRefCMTAgent(args, cal_env, rank=0)
        agent.load(args.resume_file)
        agent.vln_bert.eval()
        agent.critic.eval()
        agent._v10_length2mask = length2mask
    logger.info("HAMT REVERIE loaded: %s", type(agent).__name__)
    return agent, cal_env, test_env, args, "hamt_reverie"


# ------------------------------------------------------------------- RecBERT
def _patched_get_models(vlnbert_init: Any) -> Callable[..., Any]:
    """Fall back to a config-only VLNBert when the OSCAR/PREVALENT pretrained
    init files are absent. Safe for inference: Seq2SeqAgent.load() restores
    the FULL state dict from the trained snap checkpoint."""
    import importlib

    original = vlnbert_init.get_vlnbert_models

    def get_models_safe(args_rb: Any, config: Any = None) -> Any:
        try:
            return original(args_rb, config=config)
        except Exception as e:
            logger.warning(
                "RecBERT pretrained init unavailable (%s); "
                "config-only build.",
                e,
            )
            from pytorch_transformers import BertConfig

            cfg = BertConfig.from_pretrained("bert-base-uncased")
            cfg.img_feature_dim = 2176
            if args_rb.vlnbert == "oscar":
                mod = importlib.import_module("vlnbert.vlnbert_OSCAR")
                cfg.model_type = "visual"
                cfg.finetuning_task = "vln-r2r"
                cfg.hidden_dropout_prob = 0.3
                cfg.hidden_size = 768
                cfg.num_attention_heads = 12
                cfg.num_hidden_layers = 12
                cfg.num_labels = 2
            else:
                mod = importlib.import_module("vlnbert.vlnbert_PREVALENT")
                cfg.img_feature_type = ""
                cfg.vl_layers = 4
                cfg.la_layers = 9
            return mod.VLNBert(cfg)

    vlnbert_init.get_vlnbert_models = get_models_safe
    return get_models_safe


def _recbert_tokenizer(vlnbert_init: Any, args_rb: Any) -> Any:
    # pytorch_transformers returns None (not an exception) on a missing path;
    # a None tokenizer silently empties R2RBatch (tokenize errors swallowed).
    try:
        tok = vlnbert_init.get_tokenizer(args_rb)
    except Exception:
        tok = None
    if tok is None:
        from pytorch_transformers import BertTokenizer

        tok = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
    return tok


def build_recbert(
    variant: str, ds: Dict[str, Any]
) -> Tuple[Any, Any, Any, argparse.Namespace, str]:
    import importlib

    ckpt = RECBERT_CKPTS[variant]
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"RecBERT checkpoint not found: {ckpt}")
    # RecBERT resolves data paths against the cwd and MatterSim lazily reads
    # ./connectivity/ during rollouts -- chdir and STAY.
    os.chdir(RECBERT_ROOT)
    ensure_first_on_path(RECBERT_SRC)
    argv = [
        "conformal_vln_v10",
        "--vlnbert",
        variant,
        "--train",
        "validlistener",
        "--name",
        f"conformal_v10_{variant}",
        "--maxAction",
        str(ds["max_action_len"]),
        "--batchSize",
        "8",
        "--feedback",
        "argmax",
        "--angleFeatSize",
        "128",
        "--features",
        "places365",
        "--maxInput",
        "80",
        "--submit",
        "0",
        "--optim",
        "adamW",
    ]
    old_argv, sys.argv = sys.argv, argv
    try:
        with recbert_context():
            param_mod = importlib.import_module(
                "param"
            )  # parses argv at import
            utils_mod = importlib.import_module("utils")
            env_mod = importlib.import_module("env")
            vlnbert_init = importlib.import_module("vlnbert.vlnbert_init")
            get_models = _patched_get_models(vlnbert_init)
            # model_* bind get_vlnbert_models at import time; patch them too.
            for mod_name in ("model_OSCAR", "model_PREVALENT"):
                importlib.import_module(mod_name).get_vlnbert_models = (
                    get_models
                )
            agent_mod = importlib.import_module("agent")
            tok = _recbert_tokenizer(vlnbert_init, param_mod.args)
            logger.info(
                "Loading RecBERT features (3.9 GB TSV, several minutes)..."
            )
            feat_dict = utils_mod.read_img_features(
                RECBERT_FT_FILE, test_only=False
            )
            cal_env = env_mod.R2RBatch(
                feat_dict,
                batch_size=8,
                seed=0,
                splits=["val_seen"],
                tokenizer=tok,
            )
            test_env = env_mod.R2RBatch(
                feat_dict,
                batch_size=8,
                seed=0,
                splits=["val_unseen"],
                tokenizer=tok,
            )
            agent = agent_mod.Seq2SeqAgent(
                cal_env, "", tok, ds["max_action_len"]
            )
            start_iter = agent.load(ckpt)
            agent.vln_bert.eval()
            agent.critic.eval()
    finally:
        sys.argv = old_argv
    logger.info("RecBERT loaded: variant=%s iter=%d", variant, start_iter)
    adapter = VLNRecBERTAgentAdapter(agent, param_mod.args, utils_mod, variant)
    args = argparse.Namespace(
        backend="recbert",
        max_action_len=ds["max_action_len"],
        ignoreid=-100,
        batch_size=8,
    )
    return (
        adapter,
        RecBERTEnvWrapper(cal_env),
        RecBERTEnvWrapper(test_env),
        args,
        "adapter",
    )


# ---------------------------------------------------------------- dispatcher
def build_backend(
    backend: str,
    action_space: str,
    recbert_variant: str,
    base_dir: str,
    dataset: str,
) -> Tuple[Any, Any, Any, argparse.Namespace, str]:
    """Dispatch to one builder; returns (agent, cal_env, test_env, args,
    kind)."""
    ds = DATASET_CONFIG[dataset]
    bt.set_random_seed(0)
    os.makedirs(DUMP_DIR, exist_ok=True)
    if dataset == "reverie" and backend == "recbert":
        raise SystemExit(
            "REVERIE supports DUET and HAMT only: the public "
            "Recurrent-VLN-BERT "
            "release has no REVERIE code or checkpoint (placeholder cell)."
        )
    if backend == "duet":
        return build_duet(base_dir, ds, dataset, action_space)
    if backend == "hamt":
        return (
            build_hamt_reverie(base_dir, ds)
            if dataset == "reverie"
            else build_hamt_r2r(base_dir, ds)
        )
    return build_recbert(recbert_variant, ds)
