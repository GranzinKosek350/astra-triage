"""Decision-curve comparison at the pre-specified threshold.

Ref: abstract, "net benefit at the 0.20 threshold was 0.060 per episode against 0.033
for a C-reactive protein rule"; Sec. 2.4, which names the decision-curve script in
the code statement.

The table lists each arm's net benefit at 0.20 and the advantage over the
single-marker rule, with the treat-all reference so the reader can see whether any
arm clears the do-nothing line.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from astra_triage.endpoints.definition import TRIAGE_THRESHOLD
from astra_triage.metrics.net_benefit import curve, net_benefit, net_benefit_from_scores


@dataclass(frozen=True)
class DecisionRow:
    arm: str
    threshold: float
    net_benefit: float
    advantage_over_rule: float | None


def table(
    arms: dict[str, torch.Tensor],
    labels: torch.Tensor,
    rule: str = "crp_threshold_rule",
    threshold: float = TRIAGE_THRESHOLD,
) -> list[DecisionRow]:
    rule_value = net_benefit_from_scores(arms[rule], labels, threshold) if rule in arms else None
    rows: list[DecisionRow] = []
    for name, scores in arms.items():
        value = net_benefit_from_scores(scores, labels, threshold)
        rows.append(
            DecisionRow(
                arm=name,
                threshold=threshold,
                net_benefit=value,
                advantage_over_rule=(None if rule_value is None or name == rule else value - rule_value),
            )
        )
    return rows


def references(labels: torch.Tensor, threshold: float = TRIAGE_THRESHOLD) -> dict[str, float]:
    prevalence = float(labels.detach().mean())
    return {
        "treat_all": net_benefit(1.0, 0.0, prevalence, threshold),
        "treat_none": 0.0,
        "prevalence": prevalence,
    }


def curves(arms: dict[str, torch.Tensor], labels: torch.Tensor) -> dict[str, list[tuple[float, float, float]]]:
    return {name: curve(scores, labels) for name, scores in arms.items()}
