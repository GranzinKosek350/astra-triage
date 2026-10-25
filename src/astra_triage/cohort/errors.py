"""The one error type the cohort layer raises for a bad record."""

from __future__ import annotations


class RecordError(ValueError):
    """A row that cannot be turned into an episode, or a partition that cannot hold."""
