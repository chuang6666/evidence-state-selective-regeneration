# PAA final-closure manuscript mapping

This mapping matches the 23-page PAA English final-closure draft
and the 29-page Online Resource 1. The base public snapshot is pinned to commit
`4c435de79c782ee17df476e8b79a2f95bcd09d2d`. See the
[manuscript binding](../provenance/paa_manuscript_binding.json)
for the exact PDFs audited against this mapping.

- PAA Table 1 is a method specification.
- PAA Table 2 maps to historical Table 2. G0 and B0–B4 retain
  their original frozen OOF projections.
- PAA Table 3 uses G0/B3 from the base snapshot and four external
  rows from the [external replay](../paa_external/README.md).
- PAA Table 4 maps to historical Table 3 for Initial, Random-20,
  G0-20, B3-Refit, Instability-20 and VC-20. Suff-20 and
  MiniCheck-20 come from the external replay.
- PAA Table 5 maps to historical Table 4 for the original
  comparators. Its four external rows and the G12 combination
  endpoints in Section 4.3.2 come from the external replay.
- PAA Table 6 maps to historical Table 5. Answer-list Accuracy
  averages the frozen per-question scores. Strict full-match
  W2R, R2W and Net count binary changes from Initial.
- PAA Tables 7 and 8 map to historical Tables 6 and 7,
  respectively, for ablation and the controlled endpoints.
- PAA Table 9 maps to the historical
  [case mapping](../provenance/qualitative_case_qid_map.json).
  It contains case identifiers, transitions and localization.
- The extension also covers the external paired endpoint
  statistics in Online Resource 1, Section S6. Other supplementary
  implementation and upstream execution records have their
  own coverage boundaries.

## Online Resource 1 (ESM_1)

- **S1:** Transition requirements and verification contract. See
  [frozen contracts](../contracts/) for the archived definitions.
  The release does not refit the transition models.
- **S2:** Selector-side computation and call accounting. These are
  historical execution records described in the supplement.
- **S3:** Direction of paired M3-VQA answer-list score changes.
  The two per-question series are in
  [M3-VQA frozen scores](../frozen_scores/m3vqa13125.jsonl).
  This table uses answer-list scores, whereas Table 6 also reports
  binary full-match transitions.
- **S4-S6:** Transition-supervision sizes, deployed verifier
  configurations, thresholds and weights. The supplement reports
  the historical settings. Model checkpoints and training
  replay are outside the offline workflow.
- **S7:** Component-level OOF dependencies. The replay uses the
  saved folds and scores. It does not repeat upstream fits.
- **S8-S14:** Categorical dimensions, variant counts, and G12,
  E21, M6, localization and N6 feature dictionaries. The
  [feature contracts](../contracts/) identify the released
  field definitions and allowed-key sets.
- **S15-S16:** Cache-matching conditions and M3-VQA execution
  history. Endpoint scores are replayed from the frozen outputs.
  Historical request and cache logs are outside this release.
- **S17:** Full-coverage BM25/BGE endpoint references under a
  separate budget contract. The offline package does not
  reconstruct these rows.
- **Section S5:** The original public
  [statistics expectations](../provenance/statistics_expectations.json)
  enumerate the covered intervals and tests.
- **Section S6.2:** The external extension replays the four
  adaptations' paired EM/F1 and McNemar comparisons with B3-Refit.
  Its input and historical generation boundaries are in the
  [external replay guide](../paa_external/README.md).

The original public replay, numeric assets, and documentation of
the KAIS numbering retain their historical identities. The earlier
JIIS binding and local release-status record remain provenance.
See the [historical KAIS section](../README.md#historical-kais-manuscript-linkage-23-october-2026)
and the [checked statistics](../provenance/statistics_expectations.json).
