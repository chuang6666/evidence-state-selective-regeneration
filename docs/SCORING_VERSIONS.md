# Scoring versions and interpretation

## MMQA historical development-label implementation

`evidence_state_repro/mmqa_dev_historical.py` contains mechanically extracted
functions from `run_mmqa_v6_1_topology_routed_raw_visual.py`. Its normalization
uses NFKC, casefold, hyphen-to-space replacement and the original character
filter before article/whitespace handling. The bodies are preserved, rather
than replaced by a newly unified normalizer.

## MMQA frozen endpoint implementation

`evidence_state_repro/mmqa_endpoint_historical.py` contains the functions from
`audit_mmqa_v3_locked1341_frozen_boundary_v1_1.py` used by the audited endpoint
evaluation entry points. Article removal precedes ASCII punctuation removal.
EM and token F1 take the best match over answer members under that contract.
This is not an assertion of equivalence to every upstream list/set metric.

On the actual frozen Dev1096 Initial answers, author-private verification checks
the previously documented 20 F1 differences, zero EM differences, and agreement
of the saved wrong-answer labels. This finite-set result does not establish
equivalence for arbitrary future answers or prove that all training/ranking
stages would be unchanged under arbitrary metric substitutions.

## M3 frozen local answer-list implementation

`evidence_state_repro/m3_frozen.py` preserves the normalization constants,
answer-list conversion, reference-group validation and `official_accuracy`
function from `evaluate_m3vqa_frozen13125_one_time_official_v1.py`.

The historical identifier and docstrings containing `official` are retained
for traceability. They must not be read as a new claim that this implementation
matches the currently published upstream `run_eval.py` in every detail. The
manuscript and SI explicitly distinguish the study's frozen contract.

Let p be the number of predicted items, g the number of reference groups,
k the number of predicted items matching at least one group, and u the number
of distinct groups covered. The frozen function uses m = min(k, u) and
m / (p + g - m). It is not replaced with a different matching algorithm.
Strict full-match transitions use a score of 1 and are separate from partial-
credit answer-list Accuracy. In particular, M3's Accuracy ranking and strict
Net ranking need not agree.

## Scope of the new portable code

Function source segments are checked by SHA256 and AST provenance. Only module
headers/imports are added for standalone use. The new `selection.py` and
`replay.py` helpers are transparent replays, not renamed historical executables.
Historical script names and B3 identifiers are retained.

No original scoring source or frozen prediction file is edited. Differences
detected during preparation must be reported separately, never corrected by
silently changing the experiment's official stored results.
