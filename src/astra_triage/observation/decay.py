"""Recency decay on an observation's age.

Ref: Sec. 2.3.0.3, "A decay function is used to determine the importance of an
observation based on its age."

The decay is exponential in hours since the index time. The time constant is a
fixed property of the observation process, not a free parameter of the encoder:
letting the network choose it would let the model re-derive the sampling policy
from the record, which is the behaviour the admissible set forbids.
"""

from __future__ import annotations

import math

import torch

DEFAULT_DECAY_HOURS: float = 48.0


def decay_weight(age_hours: torch.Tensor, tau_hours: float = DEFAULT_DECAY_HOURS) -> torch.Tensor:
    """``exp(-age / tau)``, in (0, 1], with age measured back from the index time."""
    if tau_hours <= 0:
        raise ValueError("the decay constant must be positive")
    age = torch.clamp(age_hours, min=0.0)
    return torch.exp(-age / tau_hours)


def log_decay(age_hours: torch.Tensor, tau_hours: float = DEFAULT_DECAY_HOURS) -> torch.Tensor:
    """The same quantity in log space, which is what the pooling score adds."""
    return -torch.clamp(age_hours, min=0.0) / tau_hours


def half_life_hours(tau_hours: float = DEFAULT_DECAY_HOURS) -> float:
    return tau_hours * math.log(2.0)
