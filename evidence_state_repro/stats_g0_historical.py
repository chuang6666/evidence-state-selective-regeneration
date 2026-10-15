"""Unchanged historical pure function bodies; no original runners imported."""
from __future__ import annotations
import math, random, statistics
from typing import Sequence, Any
BOOTSTRAP_REPS = 10000

def paired_bootstrap(a: Sequence[float], b: Sequence[float], seed: int) -> dict[str, Any]:
    delta = [float(x) - float(y) for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(delta)
    values = sorted(sum(delta[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_REPS))

    def percentile(p: float) -> float:
        pos = p * (len(values) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return values[lo]
        return values[lo] * (hi - pos) + values[hi] * (pos - lo)

    return {
        "delta": statistics.mean(delta), "delta_pp": 100 * statistics.mean(delta),
        "ci95_low": percentile(0.025), "ci95_high": percentile(0.975),
        "ci95_low_pp": 100 * percentile(0.025), "ci95_high_pp": 100 * percentile(0.975),
        "repetitions": BOOTSTRAP_REPS, "seed": seed,
    }
