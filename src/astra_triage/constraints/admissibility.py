"""The admissible set of Sec. 2.3.0.1, assembled from its two measurable parts.

Ref: Sec. 2.3.0.1, "The estimator must belong to the admissible set defined in
Appendix A, where the two constraints are set out. The estimator must also be
count-blind ... The expression must also be roughly unchanged by the position".

Appendix A is not part of the manuscript file, so the set is implemented from the
two constraints the body states and from the three properties it attributes to the
appendix, which are implemented separately under ``properties``. A model is
admissible here when the replication and permutation probes are at numerical zero and
the site probe does not beat its own majority baseline by more than the threshold
below. That threshold is a design choice, stated once, not a number from the article.
"""

from __future__ import annotations

from dataclasses import dataclass

from astra_triage.constraints.count_blind import CountBlindResult, audit
from astra_triage.constraints.site_probe import ProbeResult, probe
from astra_triage.model.astra import Astra
from astra_triage.streams.case import Episode

SITE_PROBE_LIFT_LIMIT: float = 0.10


@dataclass(frozen=True)
class AdmissibilityReport:
    count_blind: CountBlindResult
    site_probe: ProbeResult
    lift_limit: float

    @property
    def site_invariant(self) -> bool:
        return self.site_probe.lift <= self.lift_limit

    @property
    def admissible(self) -> bool:
        return self.count_blind.passes and self.site_invariant

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "replication_deviation": self.count_blind.replication,
            "permutation_deviation": self.count_blind.permutation,
            "count_blind": self.count_blind.passes,
            "site_probe_accuracy": self.site_probe.accuracy,
            "site_majority_accuracy": self.site_probe.majority_accuracy,
            "site_lift": self.site_probe.lift,
            "site_lift_limit": self.lift_limit,
            "site_invariant": self.site_invariant,
            "admissible": self.admissible,
        }


def assess(
    arm: Astra,
    episodes: list[Episode],
    fit_slice: tuple[int, int] | None = None,
    tolerance: float = 1e-5,
) -> AdmissibilityReport:
    count_blind = audit(arm, episodes, tolerance=tolerance)
    split = fit_slice if fit_slice is not None else (0, max(1, len(episodes) // 2))
    fit_episodes = episodes[split[0] : split[1]]
    score_episodes = episodes[split[1] :] or episodes[:1]
    site = probe(arm, fit_episodes, score_episodes)
    return AdmissibilityReport(count_blind=count_blind, site_probe=site, lift_limit=SITE_PROBE_LIFT_LIMIT)
