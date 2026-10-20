"""Session persistence tests: total-or-fail reopen (WS02 §B3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.workstation.session_store import (
    MANIFEST_NAME,
    DerivedStateStoreManager,
    SessionIntegrityError,
    SessionMissingError,
    SessionSnapshot,
    WorkstationPaths,
)
from apps.workstation.workspace import StudyWorkspaceManager
from core.events import EventLog
from tests.apps.fixture_factory import build_synthetic_study, counter_id_factory
from tests.apps.pipeline import run_full_pipeline


def test_save_reopen_reproduces_identical_records(tmp_path: Path) -> None:
    session = run_full_pipeline(tmp_path / "ws")
    assert len(session.series) == 1
    assert len(session.volumes) == 1
    kinds = sorted(artifact.kind.value for artifact in session.artifacts)
    assert kinds == ["COORDINATE_TRANSFORM", "MEASUREMENT_SET"]
    # journal replayed from genesis reproduces the captured projections
    for entity_id, fields in session.projections.items():
        assert session.log.project_state(entity_id).to_dict() == fields


def test_double_save_is_byte_stable(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    source = build_synthetic_study(workspace.parent / "dicom")
    factory = counter_id_factory(11)
    manager = StudyWorkspaceManager(workspace, id_factory=factory)
    imported = manager.import_study(source)
    store = DerivedStateStoreManager(imported.paths)

    def snapshot(log: EventLog) -> SessionSnapshot:
        return SessionSnapshot(
            study=imported.study,
            series=tuple(item.series for item in imported.series),
            volumes=(),
            artifacts=(),
            payload_documents={},
            provenance=imported.provenance,
            log=log,
            skipped_instances=imported.skipped,
        )

    first = store.save(snapshot(EventLog()))
    second = store.save(snapshot(EventLog()))
    assert first == second


def test_tampered_file_aborts_reopen_totally(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    run_full_pipeline(workspace)
    blob_files = list((workspace / "blobs").rglob("*.bin"))
    data = bytearray(blob_files[0].read_bytes())
    data[0] ^= 0xFF
    blob_files[0].write_bytes(bytes(data))

    with pytest.raises(SessionIntegrityError) as excinfo:
        DerivedStateStoreManager(
            WorkstationPaths.create(workspace)
        ).reopen()
    problems = " | ".join(excinfo.value.violations)
    assert "sha256 mismatch" in problems


def test_undeclared_extra_blob_aborts_reopen(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    run_full_pipeline(workspace)
    (workspace / "blobs" / "smuggled.bin").write_bytes(b"x")

    with pytest.raises(SessionIntegrityError) as excinfo:
        DerivedStateStoreManager(WorkstationPaths.create(workspace)).reopen()
    assert any("undeclared extra file" in p for p in excinfo.value.violations)


def test_missing_manifest_raises_session_missing(tmp_path: Path) -> None:
    paths = WorkstationPaths.create(tmp_path / "empty-ws")
    with pytest.raises(SessionMissingError):
        DerivedStateStoreManager(paths).reopen()


def test_manifest_digest_covers_all_session_and_blob_files(tmp_path: Path) -> None:
    from typing import cast

    workspace = tmp_path / "ws"
    session = run_full_pipeline(workspace)
    manifest = json_load(workspace / "session" / MANIFEST_NAME)
    files_map = cast(dict[str, object], manifest["files"])
    listed = set(files_map)
    on_disk = {
        p.relative_to(workspace).as_posix()
        for base in ("blobs", "session")
        for p in (workspace / base).rglob("*")
        if p.is_file() and p.suffix != ".tmp" and p.name != MANIFEST_NAME
    }
    assert listed == on_disk
    recorded = manifest["session_digest"]
    assert isinstance(recorded, str) and len(recorded) == 64
    assert session.session_digest == recorded


def json_load(path: Path) -> dict[str, object]:
    import json
    from typing import cast

    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return cast(dict[str, object], loaded)
