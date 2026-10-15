"""Unchanged historical pure function bodies; no original runners imported."""
from __future__ import annotations
import numpy as np
from typing import Dict, Any
from .m3_frozen import ProtocolError
EXPECTED_TOTAL = 13125
BOOTSTRAP_SEED = 20260823
BOOTSTRAP_REPS = 20000

def paired_bootstrap_ci(deltas: np.ndarray) -> Dict[str, Any]:
    if deltas.ndim != 1 or len(deltas) != EXPECTED_TOTAL:
        raise ProtocolError("paired bootstrap received the wrong population")
    values, counts = np.unique(deltas, return_counts=True)
    probabilities = counts.astype(np.float64) / len(deltas)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.multinomial(len(deltas), probabilities, size=BOOTSTRAP_REPS)
    means = (draws @ values.astype(np.float64)) / len(deltas)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "method": "paired empirical percentile bootstrap",
        "repetitions": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "delta_accuracy_pp_95ci_lower": float(lower * 100.0),
        "delta_accuracy_pp_95ci_upper": float(upper * 100.0),
        "bootstrap_probability_delta_le_zero": float(np.mean(means <= 0.0)),
        "unique_delta_values": int(len(values)),
    }
