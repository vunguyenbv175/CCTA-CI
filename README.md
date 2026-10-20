# CCTA-CI — public CI source/test closure

This repository is an owner-authorized public export of the CCTA Studio
source/test closure.  It is **not** the private CCTA repository, has **no
shared git history**, and is **not** a substitute for the private governed
candidate or its frozen gates.

## Provenance

- Private source: `vunguyenbv175/CCTA` (private)
- Private candidate: `6a43807215b14a828fad23716871ef548a5f5128` (T2)
- Public snapshot: an independent, clean git repository.  Public SHAs are
  different commit identities; no descendant lineage is claimed or forged.
- Exact per-file private blob hashes, public hashes and public commit/tree
  identity are recorded in `PUBLIC_EXPORT_MANIFEST.json` and
  `RUN_EVIDENCE.md`.

## What is included

Product source (`apps/`, `core/`, `models/`, `validation/`, `visualization/`,
`plugins/`, `webapp/src/`) and the self-contained Python product test closure
under `tests/apps`, `tests/core`, `tests/models`, `tests/validation`,
`tests/plugins`, with synthetic fixtures only.

## What is deliberately not included

- Internal governance, `docs/`, `reports/`, `orchestration/`, runtime evidence,
  agent/session/attachment data.
- Browser/CDP test suites that require Chrome plus the webapp build graph.
- The served viewer bundle `apps/viewer/static/app.js`: it embeds third-party
  Cornerstone code and no file-level redistribution notice was proven for a
  public export, so the bundle and the two tests that require its exact bytes
  (`tests/apps/test_viewer_server.py`, `tests/apps/test_viewer_seed_wait_words.py`)
  stay private.
- Tests that read private internal reports (`test_qc_001_gate_census.py`,
  `test_qc_001_human_census.py`) and the seed-provenance panel test that reads
  a private browser-digest test file.
- Two host-calibrated frozen-digest assertions
  (`test_phantom48_evidence_identity_is_frozen`,
  `test_fixture_response_digest_is_frozen`) require the owner's exact WSL2
  CPU/Python/glibc fingerprint.  The public-only root `conftest.py` deselects
  exactly those two nodes with a printed reason; every other test in those
  modules runs.  The public run does not evaluate or claim those host-parity
  digests.
- Vendored wheels/npm tarballs (dependencies are installed from PyPI; no
  third-party package source is redistributed here).

## CI

`.github/workflows/tests.yml` runs **only** on manual `workflow_dispatch`
(the single owner-authorized bounded trial).  It uses standard public
GitHub-hosted `ubuntu-latest` runners, `permissions: contents: read`, no
private access, no pull_request/pull_request_target trigger, and a fixed
disk-remediation allowlist plus a free-space preflight guard.  The pytest
command is the frozen serial argv:

```
python -m pytest -q -p no:cacheprovider -m "not gpu and not network" --durations=20
```

This public run is cross-repository corroboration of the included closure
only.  It does **not** create an admissible private `pytest_full` gate record,
does not change the private gate status, and must not be reported as a full
private PASS.  The private gates remain BLOCKED until an authorized
independent acceptance chain supports cross-repo evidence.

## Data safety

No real patient data, DICOM studies, PHI, credentials, tokens, private URLs,
or internal reports are included.  DICOM fixtures are synthetic and generated
at test time by `tests/apps/fixture_factory.py`.  See `NOTICE`.
