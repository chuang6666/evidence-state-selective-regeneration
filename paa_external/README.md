# PAA external comparisons

This directory replays the frozen numerical records of Suff-only,
G12+Suff-LR, MiniCheck-only, and G12+MiniCheck-LR. Apply this update to
the pinned repository version documented in the
[PAA manuscript mapping](../docs/PAA_MANUSCRIPT_MAPPING.md).
Use Python 3.10 or newer. The replay uses only the standard library
and the repository's existing pure evaluation functions.

From the repository root:

```bash
python paa_external/replay.py
python paa_external/replay.py --statistics
```

The commands print JSON to standard output. The optional statistics
pass may take several tens of seconds. The script checks the release
manifest, joins Dev1096 to the base release's frozen binary
initial-error labels, reconstructs Top-220 membership and AP/AUC, and
verifies locked1341 rankings, 269-case memberships, EM, token F1,
selected errors, W2R, R2W, Net, and repair yield. The statistics pass
also replays the saved paired EM/F1 intervals and exact McNemar tests
against B3-Refit using the original ordering and seeds. A mismatch
ends the command with an error.

## Files and scope

- `dev1096.jsonl` contains 1,096 QIDs, risk, Top-220 membership,
  direct signal values, and held-out folds for the LR combinations.
- `locked1341.jsonl` contains 1,341 QIDs, selector risk and status,
  forced flags, rank, selection, and frozen per-QID EM/F1. The
  initial and B3-Refit projections agree with the base snapshot.
- `expected.json` holds frozen aggregate targets and paired-test
  targets. `provenance.json` records the checked source hashes.

Suff is a frozen binary judgment over the question and complete,
ordered Exact Top-6, with Beijing-region qwen3.7-plus. MiniCheck uses
a frozen v1 claim transformation, ordered text evidence, and the
official MiniCheck-Flan-T5-Large model, revision
`96eafd01cee2d16cf81aaa2fb226b14f422a37b3`, with official code revision
`b58b9fa69acbd1015ec970fa65dd752413a053d2`. Its adaptation takes
maximum support across chunks and averages across claims. The G12
combinations add the respective external signal to the general
features and use the original training-only folds and LR settings.
On locked1341, original forced cases precede the score order. Suff
also prioritizes unavailable judgments. MiniCheck's forced cases
have inapplicable scores.

These files support evaluation replay, including membership and
paired statistics for the four external configurations. Rebuilding
the upstream sufficiency judgments, MiniCheck claims or model
scores, G12 feature matrices or LR fits, and generation requests
requires additional input/code assets and access arrangements.
See manuscript Sections 4.1.2–4.1.4 and Online Resource 1, Section S6.

## Independent answer rescoring

Obtain the original MultiModalQA references from the provider under
its terms. See [reference access](../docs/DATA_ACCESS.md). Prepare a
local JSONL with 1,341 unique frozen QIDs and a nonempty reference
`answers` list per QID. An approved local selected-answer projection
has `qid` and a `methods` dictionary with the effective resolved
answer of each method selecting that question.

```bash
python paa_external/replay.py --references ../local-references/locked1341.jsonl --answers ../authorized/selected_answers.jsonl
```

This optional route checks each answer score against the frozen
projection, using the base release's Initial values for unselected
questions. Provider references and the selected-answer projection
are outside this update. An access arrangement for the latter
requires author approval. Neither route trains a model, calls an
API, regenerates answers, or writes into the release.
