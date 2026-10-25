"""Arm registry: the eight rows of Table 2 plus the ablation variants.

Each name is the one used in the experiment configs. Learned arms are constructed
from the same architecture config so that width changes propagate to every row;
rule arms are constructed bare and receive their link at fit time.
"""

from __future__ import annotations

from torch import nn

from astra_triage.comparators.base import Comparator
from astra_triage.comparators.crp_rule import CReactiveProteinRule
from astra_triage.comparators.grud import GRUD
from astra_triage.comparators.last_value import LastValueLogistic
from astra_triage.comparators.late_fusion import LateFusionTransformer
from astra_triage.comparators.pas import PaediatricAppendicitisScore
from astra_triage.comparators.pews import PaediatricEarlyWarning
from astra_triage.comparators.tft import TFT
from astra_triage.model.astra import build_astra
from astra_triage.model.config import AstraConfig

Arm = nn.Module | Comparator

LEARNED_ARMS: tuple[str, ...] = (
    "astra",
    "late_fusion",
    "tft",
    "grud",
    "logistic_last_value",
)
RULE_ARMS: tuple[str, ...] = ("pews", "pas", "crp")

ALIASES: dict[str, str] = {
    "late_fusion_transformer": "late_fusion",
    "temporal_fusion_transformer": "tft",
    "logistic_last_value_features": "logistic_last_value",
    "paediatric_early_warning_score": "pews",
    "paediatric_appendicitis_score": "pas",
    "crp_threshold_rule": "crp",
}


def canonical(name: str) -> str:
    key = "_".join(name.strip().lower().replace("-", " ").split())
    return ALIASES.get(key, key)


def build(name: str, config: AstraConfig | None = None) -> Arm:
    key = canonical(name)
    settings = config if config is not None else AstraConfig()
    if key == "astra":
        return build_astra(settings)
    if key == "late_fusion":
        return LateFusionTransformer(settings)
    if key == "tft":
        return TFT(settings)
    if key == "grud":
        return GRUD(settings)
    if key == "logistic_last_value":
        return LastValueLogistic()
    if key == "pews":
        return PaediatricEarlyWarning()
    if key == "pas":
        return PaediatricAppendicitisScore()
    if key == "crp":
        return CReactiveProteinRule()
    raise KeyError(f"unknown arm: {name}")


def is_learned(name: str) -> bool:
    return canonical(name) in LEARNED_ARMS
