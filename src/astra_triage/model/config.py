"""Configuration of the fused representation.

Ref: Sec. 2.3.0.1 to Sec. 2.3.0.4, Eq. (2)-(3).

The article does not print widths, head counts, dropout or the decay constant, so
every architectural number here is a stated engineering default. What the article
does fix is reproduced exactly: two attention directions, a recency-and-reliability
gate, a missingness embedding that fires beyond 24 hours, a site adapter that takes
only the site of origin, and a two-layer head. The ablation switches remove those
five objects one at a time and nothing else.

Widths default to a model of about half a million parameters, which is the scale a
five-site cohort of 3,412 episodes supports without over-parameterising the
between-site contrasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from astra_triage.observation.decay import DEFAULT_DECAY_HOURS
from astra_triage.streams.spec import SITES, marker_width, static_width, ultrasound_width

DEFAULT_PERIODS: tuple[float, ...] = (6.0, 12.0, 24.0, 48.0, 96.0, 192.0)


@dataclass(frozen=True)
class AstraConfig:
    d_model: int = 128
    heads: int = 4
    encoder_hidden: int = 128
    head_hidden: int = 64
    gate_hidden: int = 64
    dropout: float = 0.10
    periods: tuple[float, ...] = DEFAULT_PERIODS
    decay_hours: float = DEFAULT_DECAY_HOURS
    adapter_rank: int = 8
    site_count: int = len(SITES)
    capacity: int = 32
    ultrasound_width: int = field(default_factory=ultrasound_width)
    marker_width: int = field(default_factory=marker_width)
    static_width: int = field(default_factory=static_width)
    use_time_encoding: bool = True
    use_cross_attention: bool = True
    use_missingness_token: bool = True
    use_site_adapter: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.heads:
            raise ValueError("d_model must divide by the head count")
        if self.d_model <= 0 or self.adapter_rank <= 0:
            raise ValueError("widths must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout is a probability")
        if not self.periods:
            raise ValueError("at least one period is needed for the continuous encoding")

    @property
    def time_width(self) -> int:
        if not self.use_time_encoding:
            return 0
        return 4 * len(self.periods) + 2

    def variant(self, **changes: object) -> AstraConfig:
        """A copy with named fields replaced; used by the ablation table."""
        values = {
            "d_model": self.d_model,
            "heads": self.heads,
            "encoder_hidden": self.encoder_hidden,
            "head_hidden": self.head_hidden,
            "gate_hidden": self.gate_hidden,
            "dropout": self.dropout,
            "periods": self.periods,
            "decay_hours": self.decay_hours,
            "adapter_rank": self.adapter_rank,
            "site_count": self.site_count,
            "capacity": self.capacity,
            "use_time_encoding": self.use_time_encoding,
            "use_cross_attention": self.use_cross_attention,
            "use_missingness_token": self.use_missingness_token,
            "use_site_adapter": self.use_site_adapter,
        }
        values.update(changes)
        return AstraConfig(**values)  # type: ignore[arg-type]
