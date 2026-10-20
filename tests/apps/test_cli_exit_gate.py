"""CLI exit-gate end-to-end tests: import→derive→save→reopen→verify (§B5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from apps.workstation.cli import main
from tests.apps.fixture_factory import build_synthetic_study


def run_cli(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, dict[str, object]]:
    code = main(list(args))
    out = capsys.readouterr().out
    return code, cast(dict[str, object], json.loads(out))


def _checks(document: dict[str, object]) -> dict[str, dict[str, object]]:
    checks = document["checks"]
    assert isinstance(checks, dict)
    return cast(dict[str, dict[str, object]], checks)


def test_full_exit_gate_sequence_passes(tmp_path: Path, capsys) -> None:
    source = build_synthetic_study(tmp_path / "dicom", slices_per_phase=3)
    ws = tmp_path / "ws"

    steps = [
        ("import", "--workspace", str(ws), "--dicom-dir", str(source), "--id-seed", "42"),
        ("derive", "--workspace", str(ws), "--id-seed", "42"),
        ("save", "--workspace", str(ws)),
        ("reopen", "--workspace", str(ws)),
        ("verify", "--workspace", str(ws)),
    ]
    digests: list[tuple[str, str]] = []
    for step in steps:
        code, document = run_cli(capsys, *step)
        assert code == 0, (step, document)
        digest = document.get("session_digest")
        if isinstance(digest, str):
            digests.append((step[0], digest))
    assert document["verdict"] == "PASS"
    for check in _checks(document).values():
        assert check["passed"] is True
    # import committed the pre-derive bundle; derive advanced it; then
    # save/reopen/verify all saw exactly that same committed bundle
    by_command = dict(digests)
    assert by_command["import"] != by_command["derive"]
    assert by_command["derive"] == by_command["save"] == by_command["reopen"]


def test_verify_fails_closed_on_missing_workspace(tmp_path: Path, capsys) -> None:
    code, document = run_cli(
        capsys, "verify", "--workspace", str(tmp_path / "never-created")
    )
    assert code == 1
    assert document["error"] == "SessionMissingError"


def test_import_abstains_on_garbage_directory(tmp_path: Path, capsys) -> None:
    garbage = tmp_path / "garbage"
    garbage.mkdir()
    (garbage / "a.bin").write_bytes(b"\x00" * 64)
    code, document = run_cli(
        capsys,
        "import",
        "--workspace",
        str(tmp_path / "ws"),
        "--dicom-dir",
        str(garbage),
    )
    assert code == 1
    assert document["error"] == "NotAnalyzableError"


def test_inspect_and_export_audit_are_stable_and_audited(
    tmp_path: Path,
    capsys,
) -> None:
    source = build_synthetic_study(tmp_path / "dicom")
    ws = tmp_path / "ws"
    base = ("--workspace", str(ws))
    assert main(["import", *base, "--dicom-dir", str(source), "--id-seed", "7"]) == 0
    assert main(["derive", *base, "--id-seed", "7"]) == 0
    capsys.readouterr()

    code_one, inspect_one = run_cli(capsys, "inspect", *base)
    assert code_one == 0

    # identical seeded pipeline in a second workspace => identical inventory facts
    other = build_synthetic_study(tmp_path / "dicom2")
    ws2 = tmp_path / "ws2"
    base2 = ("--workspace", str(ws2))
    assert main(["import", *base2, "--dicom-dir", str(other), "--id-seed", "7"]) == 0
    assert main(["derive", *base2, "--id-seed", "7"]) == 0
    capsys.readouterr()
    code_two, inspect_two = run_cli(capsys, "inspect", *base2)
    assert code_two == 0
    # inventories carry kinds/states/composite digests only (no minted ids)
    assert inspect_one["series"] == inspect_two["series"]
    assert inspect_one["artifacts"] == inspect_two["artifacts"]

    # determinism: repeating a command against an UNCHANGED workspace emits
    # byte-identical stdout (no wall-clock values leak into output)
    _, doc_a = run_cli(capsys, "verify", *base)
    assert doc_a["verdict"] == "PASS"
    _, doc_b = run_cli(capsys, "verify", *base)
    assert json.dumps(doc_a, sort_keys=True) == json.dumps(doc_b, sort_keys=True)

    code_export, export_document = run_cli(capsys, "export-audit", *base)
    assert code_export == 0
    assert export_document["status"] == "EXPORTED"
    exported = (ws / "exports" / "audit_export.json").read_text(encoding="utf-8")
    assert '"schema": "ccta.audit.export.v1"' in exported


def test_backup_restore_round_trip_via_cli(tmp_path: Path, capsys) -> None:
    source = build_synthetic_study(tmp_path / "dicom")
    ws = tmp_path / "ws"
    base = ("--workspace", str(ws))
    assert main(["import", *base, "--dicom-dir", str(source), "--id-seed", "9"]) == 0
    assert main(["derive", *base, "--id-seed", "9"]) == 0
    capsys.readouterr()

    code, document = run_cli(
        capsys, "backup", *base, "--dest", str(tmp_path / "backups"), "--include-source"
    )
    assert code == 0
    archive_name = document["archive"]
    assert isinstance(archive_name, str)
    archive = tmp_path / "backups" / archive_name

    target = tmp_path / "restored"
    code, restore_document = run_cli(
        capsys, "restore", "--archive", str(archive), "--workspace", str(target)
    )
    assert code == 0
    assert int(cast(int, restore_document["restored_files"])) > 0
    assert int(cast(int, restore_document["derived_state_files"])) > 0
    assert int(cast(int, restore_document["dicom_source_files"])) > 0

    restored_ws = target / "derived_state"
    code, verify_document = run_cli(capsys, "verify", "--workspace", str(restored_ws))
    assert code == 0
    assert verify_document["verdict"] == "PASS"
