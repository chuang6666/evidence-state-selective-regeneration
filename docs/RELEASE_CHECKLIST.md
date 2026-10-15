# Checklist before any GitHub upload or public release

This file is a checklist, not evidence that permission has already been granted.

V2 scope update: the user confirmed author/group approval to upload own code,
frozen predictions and numeric derivatives on 2026-10-15. This does not select
a project-wide license or authorize third-party Gold/images/weights. Items
below concerning full upstream artifacts remain outside this limited release.

- [ ] All contributors and the appropriate institutional rights holders approve
  the release scope and choose the license for author-owned code.
- [ ] The complete author order, name spellings and citation metadata are confirmed.
- [ ] Derived QIDs/labels/scores and their relationship to third-party data are
  reviewed for redistribution. Data/code licenses are not conflated.
- [ ] The security scan and a manual review cover all files and Git history
  that will actually be uploaded, including any later-added files.
- [ ] No `.env`, provider token, private key, credential store, request body,
  personal information, raw Gold, source image, or unapproved weight is included.
- [ ] Original M3 B3 selection-runner provenance is resolved or its absence is
  explicitly documented. No old B2 runner is substituted.
- [ ] Historical dynamic dependencies, local path assumptions, verifier models,
  feature generation and environment/checkpoint compatibility are documented.
- [ ] A suitable environment for full training/inference is separately tested
  before claiming full upstream reproducibility.
- [ ] Confidence-interval/statistical reruns are verified before claiming all
  reported uncertainty estimates are reproducible from this bundle.
- [ ] All formal comparators and non-leading results retain their records.
- [ ] `provenance/file_manifest.json`, commands and validation records are current.
- [ ] Author-private materials outside this directory are not uploaded.
- [ ] Repository ownership, visibility, version tag and final release contents
  are approved. Actual upload requires a separate user instruction.
- [ ] Only after successful access checks are the manuscript's Data/Code
  Availability statements updated to describe the actual released materials.

Do not create a release or change repository visibility from any bundled script.
This candidate has no release/deployment GitHub Actions workflow and performs
no authentication, `git init`, `git push`, or remote repository mutation.
