"""Regression runner golden-digest tests (WS02 §B9)."""

from __future__ import annotations

import json
from pathlib import Path

from apps.workstation.round_trip import RoundTripVerificationEngine
from tests.apps.pipeline import rewrite_manifest, run_full_pipeline
from validation.regression.runner import RegressionRunner


def test_golden_write_then_fresh_run_matches(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    run_full_pipeline(workspace)
    runner = RegressionRunner(RoundTripVerificationEngine(workspace))
    golden_path = tmp_path / "golden" / "report.golden.json"

    fresh = runner.run_fresh(workspace)
    assert fresh.verdict == "PASS"
    runner.write_golden(fresh, golden_path)

    second = runner.run_fresh(workspace)
    result = runner.compare(second, runner.load_golden(golden_path))
    assert result.matched, [d.to_dict() for d in result.differences]
    assert result.verdict == "PASS"


def test_identity_keys_are_pruned_from_comparison() -> None:
    from validation.regression.runner import IDENTITY_IGNORE_KEYS, _prune

    pruned, removals = _prune({"study_id": "NEW", "verdict": "PASS"}, "")
    assert pruned == {"verdict": "PASS"}
    assert any("identity" in r for r in removals)
    assert "session_digest" in IDENTITY_IGNORE_KEYS


def test_content_drift_produces_fail_with_pointers(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    run_full_pipeline(workspace)
    runner = RegressionRunner(RoundTripVerificationEngine(workspace))
    golden_path = tmp_path / "golden.json"
    fresh = runner.run_fresh(workspace)
    runner.write_golden(fresh, golden_path)

    # drift: mutate the committed measurement payload and reseal the manifest
    artifacts_path = workspace / "session" / "artifacts.json"
    document = json.loads(artifacts_path.read_text(encoding="utf-8"))
    for _artifact_id, doc in document["payload_documents"].items():
        if doc.get("algorithm") == "central-box-hu-stats":
            doc["voxel_count"] = int(doc["voxel_count"]) + 1
            break
    artifacts_path.write_text(
        json.dumps(document, sort_keys=True, indent=2), encoding="utf-8"
    )
    rewrite_manifest(workspace)

    drifted = runner.run_fresh(workspace)
    result = runner.compare(drifted, runner.load_golden(golden_path))
    assert not result.matched
    assert result.verdict == "FAIL"
    pointers = " | ".join(d.pointer for d in result.differences)
    assert "voxel_count" in pointers or "measurement_recompute_equality" in pointers
