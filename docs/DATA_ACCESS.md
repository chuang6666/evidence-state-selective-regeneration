# Reference data are obtained separately

The repository supplies author-approved frozen predictions and numeric
derivatives, not benchmark Gold, questions, evidence, images or weights.
Follow the original providers' terms and access procedures:

- MMQA: https://github.com/allenai/multimodalqa
- M3-VQA: https://github.com/CASIA-IVA-Lab/M3VQA

The local preparation command accepts MMQA JSONL (plain or gzip) with `qid` or
`question_id` and original `answers`, plus M3 JSONL with `data_id` and
`answer_evals`. M3 `data_id` is the frozen QID mapping. For scoring, the original
images and large knowledge base are not needed.

```bash
python -m evidence_state_repro prepare-references --mmqa-data /your/local/MMQA_dev.jsonl.gz --m3-data /your/local/quesions.jsonl --output-dir ../local-references
```

It creates the following **local-only** canonical files:

- `locked1341.jsonl`: 1,341 unique QIDs, fields `qid`, `answers`.
- `dev1096_controlled.jsonl`: 1,096 unique QIDs, fields `qid`, `answers`.
- `m3vqa13125.jsonl`: 13,125 unique QIDs, fields `qid`, `answer_evals`.

MMQA `answers` is a nonempty list of original reference-member strings. M3
`answer_evals` is a nonempty list of nonempty alias/reference groups copied
unchanged. Do not concatenate members, rewrite Gold or swap in another scorer
to force agreement. Missing QIDs or malformed groups stop the command.
The subsequent `rescore` command verifies per-QID numerical agreement.

Synthetic format examples only (not real benchmark records):

```json
{"qid":"synthetic-mmqa","answers":["synthetic reference"]}
{"qid":"synthetic-m3","answer_evals":[["synthetic reference","synthetic alias"]]}
```

Keep all references and generated results outside Git. An access-on-request
arrangement must not be described as available before actually established.
