"""Machine-specific paths and dataset config (side-effect free).

from CPU-only code. Edit once per machine.
"""

from __future__ import annotations

import os

DUET_SRC = "/fs04/scratch2/pr65/vfvic1/thesis/VLN-DUET/map_nav_src"
MATTERSIM_BUILD = (
    "/fs04/scratch2/pr65/vfvic1/thesis/Matterport3DSimulator/build"
)
HAMT_FINETUNE_SRC = (
    "/home/vfeliren1/pr65_scratch2/vfvic1/thesis/VLN-HAMT/finetune_src"
)
HAMT_CKPT = (
    "/home/vfeliren1/pr65_scratch2/vfvic1/thesis/VLN-HAMT/datasets/"
    "R2R/trained_models/vitbase-finetune-e2e/best_val_unseen"
)
HAMT_FT_FILE = (
    "/home/vfeliren1/pr65_scratch2/vfvic1/thesis/VLN-HAMT/datasets/"
    "R2R/features/pth_vit_base_patch16_224_imagenet_r2r.e2e.ft.22k.hdf5"
)
RECBERT_ROOT = "/fs04/scratch2/pr65/vfvic1/thesis/Recurrent-VLN-BERT"
RECBERT_SRC = os.path.join(RECBERT_ROOT, "r2r_src")
RECBERT_FT_FILE = "img_features/ResNet-152-places365.tsv"  # cwd-relative
RECBERT_CKPTS = {
    "prevalent": os.path.join(
        RECBERT_ROOT,
        "snap",
        "VLNBERT-PREVALENT-final",
        "state_dict",
        "best_val_unseen",
    ),
    "oscar": os.path.join(
        RECBERT_ROOT,
        "snap",
        "VLNBERT-OSCAR-final",
        "state_dict",
        "best_val_unseen",
    ),
}

# output_v10 workspace (this package's parent directory)
V10_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP_DIR = os.path.join(V10_DIR, "dumps")
RES_DIR = os.path.join(V10_DIR, "results")
FIG_DIR = os.path.join(V10_DIR, "figures_html")

DATASET_CONFIG = {
    "r2r": {
        "anno_dir_key": "R2R",
        "resume_file": "R2R/trained_models/best_val_unseen",
        "obj_feat_size": 0,
        "max_objects": None,
        "obj_ft_file": None,
        "max_action_len": 15,
        "cal_split": "val_seen",
        "test_split": "val_unseen",
    },
    "reverie": {
        "anno_dir_key": "REVERIE",
        "resume_file": "REVERIE/trained_models/best_val_unseen",
        "obj_feat_size": 768,
        "max_objects": 20,
        "obj_ft_file": (
            "REVERIE/features/"
            "obj.avg.top3.min80_vit_base_patch16_224_imagenet.hdf5"
        ),
        "max_action_len": 15,
        "cal_split": "val_seen",
        "test_split": "val_unseen",
    },
}
