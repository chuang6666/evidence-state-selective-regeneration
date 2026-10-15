"""Unchanged historical pure function bodies; no original runners imported."""
from __future__ import annotations
import numpy as np

def fixed_p(ranks,labels,scores):
    # Exact P@220 projection of the project's fast_bootstrap_metrics.
    return float(labels[np.lexsort((ranks,-scores))[:220]].sum()/220)
