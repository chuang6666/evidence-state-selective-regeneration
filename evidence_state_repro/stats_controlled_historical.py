"""Unchanged historical pure function bodies; no original runners imported."""
from __future__ import annotations
import numpy as np
from typing import Sequence, Any
EXPECTED_N = 1096
BOOTSTRAP_REPLICATES = 10000
def fail(message):
    raise ValueError(message)

def paired_bootstrap_delta(
    left: Sequence[float], right: Sequence[float], seed: int
) -> dict[str, Any]:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) != EXPECTED_N:
        fail("invalid paired bootstrap vectors")
    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for i in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, len(a), size=len(a))
        deltas[i] = float((a[idx] - b[idx]).mean())
    low, high = np.quantile(deltas, [0.025, 0.975])
    observed = float((a - b).mean())
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "observed_delta": observed,
        "observed_delta_percentage_points": 100.0 * observed,
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci95": [float(low), float(high)],
        "ci95_percentage_points": [100.0 * float(low), 100.0 * float(high)],
        "ci_method": "paired percentile bootstrap",
    }
