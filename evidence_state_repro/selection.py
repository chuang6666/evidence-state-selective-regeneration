"""Transparent replay of the documented budget rule, not a historical runner.

This helper never fits a model or creates new risk scores.  Membership is
checked against saved selections by the replay command.
"""

from decimal import Decimal, ROUND_CEILING
import math


def fixed_budget(qids, risk_scores, forced_qids=(), fraction="0.20"):
    qids = list(qids)
    if len(qids) != len(set(qids)):
        raise ValueError("Duplicate QIDs")
    if not all(isinstance(q, str) and q for q in qids):
        raise ValueError("QIDs must be nonempty strings")
    forced = set(forced_qids)
    if not forced.issubset(qids):
        raise ValueError("Forced QID outside the population")
    rho = Decimal(str(fraction))
    if not 0 <= rho <= 1:
        raise ValueError("Selection fraction outside [0, 1]")
    budget = int((rho * len(qids)).to_integral_value(rounding=ROUND_CEILING))
    if len(forced) > budget:
        raise ValueError("Forced set exceeds selected-case budget")
    candidates = []
    for qid in qids:
        if qid in forced:
            continue
        if qid not in risk_scores or not math.isfinite(float(risk_scores[qid])):
            raise ValueError("Missing or non-finite non-forced risk score")
        candidates.append(qid)
    ranked = sorted(candidates, key=lambda q: (-float(risk_scores[q]), q))
    slots = budget - len(forced)
    if len(ranked) < slots:
        raise ValueError("Insufficient scored non-forced candidates")
    return sorted(forced) + ranked[:slots]
