"""WorkspaceBackupManager tests: backup→restore→verify property (WS02 §B10)."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from apps.workstation.backup import (
    CLASS_DICOM_SOURCE,
    BackupIntegrityError,
    WorkspaceBackupManager,
)
from apps.workstation.round_trip import RoundTripVerificationEngine
from tests.apps.pipeline import run_full_pipeline

MANIFEST = "manifest.json"


def _make_archive(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "ws"
    run_full_pipeline(workspace)
    dest = tmp_path / "backups"
    archive = WorkspaceBackupManager().backup(
        workspace, dest, source_dirs=(tmp_path / "ws-dicom",)
    )
    return archive, workspace


def test_backup_manifest_separates_source_from_derived(tmp_path: Path) -> None:
    archive, _ = _make_archive(tmp_path)
    with tarfile.open(archive, "r:gz") as tar:
        manifest = json.loads(
            tar.extractfile("MANIFEST.sha256.json").read().decode("utf-8")  # type: ignore[union-attr]
        )
    classes = {member["class"] for member in manifest["members"]}
    assert classes == {CLASS_DICOM_SOURCE, "derived_state"}
    derived = [m for m in manifest["members"] if m["class"] == "derived_state"]
    assert any(m["path"].startswith("derived_state/session/") for m in derived)
    assert any(m["path"].startswith("derived_state/blobs/") for m in derived)


def test_backup_restore_verify_property_holds(tmp_path: Path) -> None:
    archive, _ = _make_archive(tmp_path)
    target = tmp_path / "restored"
    summary = WorkspaceBackupManager().restore(archive, target)

    restored_root = target / "derived_state"
    assert (restored_root / "session" / MANIFEST).exists()
    report = RoundTripVerificationEngine(restored_root).verify(
        _reopen(restored_root)
    )
    assert summary.restored_files > 0
    assert summary.dicom_source_files > 0
    assert report.verdict == "PASS"
    # original DICOM bytes came back under dicom_source/
    sources = list((target / "dicom_source").rglob("*.dcm"))
    assert sources


def test_tampered_member_is_rejected_before_extraction(tmp_path: Path) -> None:
    archive, _ = _make_archive(tmp_path)

    # rewrite the archive with one corrupted member payload
    with tarfile.open(archive, "r:gz") as tar:
        members = []
        for info in tar.getmembers():
            handle = tar.extractfile(info)
            data = handle.read() if handle is not None else b""
            if info.name != "MANIFEST.sha256.json":
                mutable = bytearray(data)
                if mutable and info.name.endswith(".bin"):
                    mutable[0] ^= 0xFF
                data = bytes(mutable)
            members.append((info, data))
    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(tampered, "w:gz") as tar:
        for info, data in members:
            if info.name == "MANIFEST.sha256.json":
                tar.addfile(info, io.BytesIO(data))
            else:
                info_copy = tarfile.TarInfo(name=info.name)
                info_copy.size = len(data)
                tar.addfile(info_copy, io.BytesIO(data))

    target = tmp_path / "must-not-exist"
    with pytest.raises(BackupIntegrityError):
        WorkspaceBackupManager().restore(tampered, target)
    assert not target.exists()


def test_restore_refuses_unsafe_member_names(tmp_path: Path) -> None:
    evil = tmp_path / "evil.tar.gz"
    payload = b"owned"
    with tarfile.open(evil, "w:gz") as tar:
        manifest = {
            "schema": "ccta.backup.manifest.v1",
            "members": [
                {"path": "../escape.txt", "sha256": "0" * 64, "bytes": 5, "class": "x"}
            ],
        }
        blob = json.dumps(manifest).encode()
        info = tarfile.TarInfo(name="MANIFEST.sha256.json")
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
        member = tarfile.TarInfo(name="../escape.txt")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))

    target = tmp_path / "target"
    with pytest.raises(BackupIntegrityError):
        WorkspaceBackupManager().restore(evil, target)


def test_restore_refuses_non_empty_target(tmp_path: Path) -> None:
    archive, _ = _make_archive(tmp_path)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("do not clobber", encoding="utf-8")
    with pytest.raises(Exception):  # noqa: B017 - CanonicalValidationError expected
        WorkspaceBackupManager().restore(archive, occupied)
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "do not clobber"


def _reopen(workspace_root: Path):
    from apps.workstation.session_store import (
        DerivedStateStoreManager,
        WorkstationPaths,
    )

    manager = DerivedStateStoreManager(WorkstationPaths.create(workspace_root))
    return manager.reopen()
