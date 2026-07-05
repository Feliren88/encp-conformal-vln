"""Order-sensitive bootstrap: chdir, MatterSim, and DUET imports.

1. chdirs into the DUET source tree (DUET resolves data paths relatively,
     and MatterSim lazily reads ./connectivity/ from the process cwd),
  2. puts DUET and the MatterSim build first on sys.path,
  3. imports MatterSim and the DUET modules that own the default top-level
     r2r/models/utils packages.
Every DUET symbol the rest of the package needs is re-exported from here,
so this is the ONLY module with import-time side effects.
"""

from __future__ import annotations

import os
import sys
import warnings

from vln_backends.config import DUET_SRC, MATTERSIM_BUILD

warnings.filterwarnings("ignore", category=UserWarning)

os.chdir(DUET_SRC)
sys.path.insert(0, DUET_SRC)
sys.path.insert(0, MATTERSIM_BUILD)

import MatterSim  # noqa: F401,E402  binding check

from utils.misc import set_random_seed  # noqa: E402
from utils.data import ImageFeaturesDB  # noqa: E402
from models.graph_utils import GraphMap  # noqa: E402
from r2r.data_utils import (
    construct_instrs as r2r_construct_instrs,
)  # noqa: E402
from r2r.env import R2RNavBatch  # noqa: E402
from r2r.agent import GMapNavAgent  # noqa: E402
from r2r.eval_utils import cal_dtw, cal_cls  # noqa: E402
from reverie.data_utils import (  # noqa: E402
    construct_instrs as reverie_construct_instrs,
    ObjectFeatureDB as ReverieObjectFeatureDB,
    load_obj2vps,
)
from reverie.env import ReverieObjectNavBatch  # noqa: E402
from reverie.agent_obj import GMapObjectNavAgent  # noqa: E402

__all__ = [
    "set_random_seed",
    "ImageFeaturesDB",
    "GraphMap",
    "r2r_construct_instrs",
    "R2RNavBatch",
    "GMapNavAgent",
    "cal_dtw",
    "cal_cls",
    "reverie_construct_instrs",
    "ReverieObjectFeatureDB",
    "load_obj2vps",
    "ReverieObjectNavBatch",
    "GMapObjectNavAgent",
]
