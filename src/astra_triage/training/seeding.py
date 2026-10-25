"""One seeding entry point, used by every entry point and restored on resume."""

from __future__ import annotations

import random

import numpy as np
import torch

DEFAULT_SEED: int = 20260825


def set_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError("a seed is non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng(seed: int) -> torch.Generator:
    handle = torch.Generator()
    handle.manual_seed(seed)
    return handle


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)
