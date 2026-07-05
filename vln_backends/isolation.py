"""sys.modules swap contexts for the colliding upstream packages.

modules (r2r, utils, models, agent, ...). Whichever imports first wins in
sys.modules, so entering another repo's code requires swapping the cache:
evict the conflicting modules, restore that repo's own (stashed under a
private prefix), and undo on exit.
"""

from __future__ import annotations

import contextlib
import sys
from typing import ContextManager, Iterator, Tuple

import numpy as np

from vln_backends.config import HAMT_FINETUNE_SRC

_HAMT_PREFIXES = ("r2r", "reverie", "models", "utils")
_RECBERT_PREFIXES = (
    "param",
    "utils",
    "env",
    "agent",
    "eval",
    "model_OSCAR",
    "model_PREVALENT",
    "vlnbert",
)


@contextlib.contextmanager
def _module_swap(prefixes: Tuple[str, ...], stash: str) -> Iterator[None]:
    def owned(name: str) -> bool:
        return any(name == p or name.startswith(p + ".") for p in prefixes)

    saved = {n: sys.modules.pop(n) for n in list(sys.modules) if owned(n)}
    for n in list(sys.modules):
        if n.startswith(stash):
            sys.modules[n[len(stash) :]] = sys.modules[n]
    try:
        yield
    finally:
        for n in list(sys.modules):
            if owned(n):
                sys.modules[stash + n] = sys.modules.pop(n)
        sys.modules.update(saved)


def hamt_context() -> ContextManager[None]:
    return _module_swap(_HAMT_PREFIXES, "_hamt_")


def recbert_context() -> ContextManager[None]:
    return _module_swap(_RECBERT_PREFIXES, "_recbert_")


def ensure_first_on_path(src: str) -> None:
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)


def ensure_hamt_imports() -> None:
    if not hasattr(np, "bool"):  # numpy>=1.24 removed the alias
        np.bool = bool  # type: ignore[attr-defined]
    ensure_first_on_path(HAMT_FINETUNE_SRC)
