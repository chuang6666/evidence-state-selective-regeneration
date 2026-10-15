# Evidence-State risk estimation and selective regeneration

Code and frozen-output reproduction materials for *Evidence-State Risk
Estimation for Selective Regeneration in Multimodal Multi-Hop Question
Answering*. The scientific binding is the JIIS v6 text / v7 Figure 1 preview,
with B3's **65 allowed raw fields and 75 vectorized dimensions**, not the later
P2 experimental line.

This version adds author-approved frozen predictions and numeric projections.
**Original benchmark reference answers, images, source evidence, complete API
responses, credentials and model weights are not redistributed here.** No
project-wide license has been selected on the authors' behalf.

## Reproduction scope

- Reaggregate the six formal OOF variants and nine explanatory variants:
  Error Precision/Recall at Top-220, AP and ROC-AUC.
- Reproduce main Tables 2–7 from frozen scores, including all reported
  comparators and five fixed Random-20 assignments, sample SDs, selected
  errors, repair yields and strict W2R/R2W/Net.
- Independently rescore frozen MMQA and M3-VQA answers after obtaining the
  original reference data from their providers.
- Reconstruct the **specified main-text CIs/tests** using original pure
  statistical functions and documented RNG/order. This is not a claim that
  every statistic in all 17 supplementary tables is covered.
- Replay B3 membership at MMQA budget 269 (6 forced) and M3 budget 2,625
  (515 forced). This is not a recovered original M3 selection runner.

No API credentials, training or new answer generation are needed for these
commands. Complete upstream model retraining and deterministic commercial-model
output regeneration are not claimed.

## Quickstart without reference data

Use Python 3.10+. These commands require only the standard library:

```bash
python -m unittest discover -s tests -v
python -m evidence_state_repro
python -m evidence_state_repro tables --output-dir ../reproduced-saved-tables
```

SHA256s are checked before replay. Output directories must be new/empty and
**outside the repository**, to avoid overwriting frozen inputs or accidentally
publishing references.

## From frozen answers to newly computed scores

Obtain the original data under the providers' terms:

- MultiModalQA: https://github.com/allenai/multimodalqa
- M3-VQA: https://github.com/CASIA-IVA-Lab/M3VQA

See [data access and schemas](docs/DATA_ACCESS.md). With a matching MMQA
`MMQA_dev.jsonl.gz` and M3 `quesions.jsonl` (historical spelling), run:

```bash
python -m evidence_state_repro prepare-references --mmqa-data /path/to/MMQA_dev.jsonl.gz --m3-data /path/to/quesions.jsonl --output-dir ../local-references
python -m evidence_state_repro rescore --reference-dir ../local-references --output-dir ../rescored
python -m evidence_state_repro tables --scores-dir ../rescored --output-dir ../reproduced-rescored-tables
```

Every method/QID is checked against frozen score projections. Mismatches stop
the command; original outputs are not adjusted. References remain local.

## Specified confidence intervals and tests

The optional statistics command needs NumPy; the actual validation runtime is
recorded in `provenance/candidate_validation.json`.

```bash
python -m pip install -r requirements-statistics.txt
python -m evidence_state_repro statistics --scores-dir ../rescored --output-dir ../reproduced-statistics
```

Omit `--scores-dir` to use bundled scores; output metadata distinguishes the
two modes. The command may take several minutes. It preserves original
directions, RNGs, seeds, QID multiplicity and continuous component-pair order.
Random-20 is averaged within QID before paired resampling. Covered checks are
listed in `provenance/statistics_expectations.json`; unrelated SI statistics
are not implied validated.

## Files and provenance

- `frozen_predictions/`: QIDs and frozen method answer values only.
- `frozen_scores/`: numeric projections, folds/labels and memberships.
- `contracts/`: unchanged contracts and allowed-key dictionaries.
- `evidence_state_repro/`: provider-free scoring, table generation and explicit
  statistics. Historical pure bodies have source/function hashes.
- `historical/`: byte-preserved archive, **not a quickstart**. Some scripts
  retain machine paths and training/provider entry points; do not run them as
  the offline workflow.
- `provenance/`: manuscript binding, hashes, expected values and validation.

See [scoring versions](docs/SCORING_VERSIONS.md). Dev-label and endpoint
normalization are not asserted identical: Dev1096 Initial had 20 F1 differences
and no EM-label differences. MMQA remains reference-member matching; M3 uses
the study's frozen answer-list contract, not a claim of complete equivalence
to a current upstream scorer.

## Limits and release status

Verifier/checkpoint loading, risk-model refitting and state extraction are not
validated by this workflow. Original M3 B3 runner provenance is incomplete;
old B2/P2 code is not substituted. Case score transitions are linked by QID,
but complete source evidence and all supplementary statistics are not bundled.

Read `RELEASE_STATUS.json` and `provenance/candidate_validation.json` for actual
state. No visibility change, license grant, DOI or release tag is implied.
