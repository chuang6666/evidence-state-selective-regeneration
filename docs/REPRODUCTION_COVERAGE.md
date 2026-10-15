# Exact reproduction boundary

## Implemented offline paths

`rescore` uses supplied original reference data and bundled frozen answer
values. It compares every computed per-QID/method score with frozen numerical
projections, stopping on a mismatch. Merely reaggregating saved scores is a
different mode and is labelled as such.

`tables` produces main Tables 2–7's numeric columns, including six formal and
nine explanatory risk configurations, Random five-assignment means and sample
SD, selected-error/repair-yield counts, MMQA EM/token F1, controlled outcomes,
and M3 answer-list Accuracy separate from strict full-match transitions.

`statistics` checks 24 specified metric/CI entries: B3 versus Initial, VC and
within-QID Random mean; G0 versus B3; Instability versus B3; three controlled
pairwise comparisons; M3 versus Initial and Random; OOF B3-G0 P@20%; and five
component P@20% comparisons. Specified EM McNemar p-values are also checked.
Exact directions and unrounded stored expectations are in
`provenance/statistics_expectations.json`. This is not all historical or SI
statistics, selection-adjusted uncertainty or regeneration variability.

The original statistical function bodies are exported without executing
historical runners. The risk P@20% bootstrap uses the archived audit's exact
`fixed_p` mathematical projection; it preserves multiplicity, reranking,
Top-220 and original continuous five-pair RNG order, without refitting.

Table 8's QID/transition mapping is supplied, not its full evidence text.
Table 1 is a method specification, not an endpoint computed by these commands.

## Deliberately not asserted

- Loading verifier/risk checkpoints or reconstructing all upstream features.
- Refitting OOF models or recovering the original M3 B3 selection runner.
- Repeating commercial-model generation with identical outputs.
- Reproducing every supplementary table, training record or cost statement.
- Granting a new license to benchmark data, pretrained weights or third-party
  material; a project-wide license remains an author decision.

Original sources/results/manuscript/figures are not edited. Higher-endpoint
comparators and non-leading results remain in the reproduction scope.
