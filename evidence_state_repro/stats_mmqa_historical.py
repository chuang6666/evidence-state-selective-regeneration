"""Unchanged historical pure function bodies; no original runners imported."""
from __future__ import annotations
import math, random, statistics
from typing import Sequence, Dict, Any, List
BOOTSTRAP_SAMPLES = 10000
class EvalError(ValueError):
    pass

def exact_mcnemar_p(a_better: int, b_better: int) -> float:
    n = a_better + b_better
    if n == 0:
        return 1.0
    k = min(a_better, b_better)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)

def paired_bootstrap_delta(
    values_a: Sequence[float], values_b: Sequence[float], seed: int
) -> Dict[str, Any]:
    if len(values_a) != len(values_b) or not values_a:
        raise EvalError("Invalid paired bootstrap vectors")
    deltas = [float(a) - float(b) for a, b in zip(values_a, values_b)]
    observed = statistics.mean(deltas)
    rng = random.Random(seed)
    n = len(deltas)
    boot: List[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        boot.append(sum(deltas[rng.randrange(n)] for _j in range(n)) / n)
    boot.sort()

    def quantile(p: float) -> float:
        pos = p * (len(boot) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return boot[lo]
        w = pos - lo
        return boot[lo] * (1.0 - w) + boot[hi] * w

    return {
        "delta": observed,
        "delta_pp": 100.0 * observed,
        "ci95_low": quantile(0.025),
        "ci95_high": quantile(0.975),
        "ci95_low_pp": 100.0 * quantile(0.025),
        "ci95_high_pp": 100.0 * quantile(0.975),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": seed,
    }
