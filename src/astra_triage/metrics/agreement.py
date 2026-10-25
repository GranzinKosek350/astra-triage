"""Inter-reader agreement.

Ref: Table 7 footnote, "Inter-reader agreement between clinicians in the unaided arm
was kappa = 0.58."

Cohen's kappa on the two raw reads of the unsupported arm, before adjudication. A
weighted form is not used: the instrument is binary.
"""

from __future__ import annotations

import numpy as np


def cohen_kappa(first: list[int], second: list[int], levels: int = 2) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("kappa needs two aligned, non-empty read vectors")
    table = np.zeros((levels, levels), dtype=np.float64)
    for a, b in zip(first, second, strict=True):
        if not 0 <= a < levels or not 0 <= b < levels:
            raise ValueError("a read falls outside the instrument's levels")
        table[a, b] += 1.0
    total = table.sum()
    observed = float(np.trace(table) / total)
    row = table.sum(axis=1) / total
    column = table.sum(axis=0) / total
    expected = float(np.sum(row * column))
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def prevalence_adjusted_bias(first: list[int], second: list[int]) -> float:
    """Difference in the positive rate between the two readers."""
    if len(first) != len(second) or not first:
        raise ValueError("two aligned read vectors are needed")
    return float(np.mean(first) - np.mean(second))
