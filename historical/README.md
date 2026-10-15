# Read-only historical code archive

These are byte-identical copies of named experiment sources, plus identified
direct local Python import dependencies. Source hashes and original relative
locations are in `provenance/historical_sources.json`. Dynamic imports, files
loaded through strings, checkpoints and source datasets are not declared
complete simply because direct imports have been copied.

The portable quickstart is `python -m evidence_state_repro`, not a historical
script in this directory. Historical sources may have absolute local paths,
derive roots from their own path, fit models, write outputs, or contain provider
execution paths. No such entry point was executed in preparing the candidate.

Do not run them against the original project directories or assume they form a
single clean install. Consult stage-specific historical dependency receipts.
No provider credentials, raw API logs or model objects are intentionally added.

An archived name/comment calling a scorer `official` records the historical
name only. `docs/SCORING_VERSIONS.md` governs the current interpretation of the
two MMQA normalizers and the frozen local M3 answer-list scorer.

These copies are not automatically licensed for public redistribution.
Review the release checklist before moving the candidate to any remote.
