"""WorkspaceBackupManager — tarfile + SHA-256 manifest baseline (§B10).

``backup`` produces one ``.tar.gz`` whose FIRST member is a manifest sealing
every other member with its SHA-256 and a class label separating raw DICOM
source snapshots from derived state (Rule A separation, pre-research §2.1).

``restore`` is fail-closed in the strict order:

1. read the manifest member;
2. stream-verify EVERY member digest against it BEFORE extracting anything;
3. only then extract into a staging directory and atomically move it into
   place. A tampered archive never touches the target filesystem.

Encryption at rest is explicitly deferred to the threat-model task (Part E3);
integrity, not confidentiality, is the Phase-0 guarantee.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from core.errors import CanonicalValidationError

BACKUP_MANIFEST_SCHEMA: Final[str] = "ccta.backup.manifest.v1"
MANIFEST_MEMBER: Final[str] = "MANIFEST.sha256.json"
CLASS_DICOM_SOURCE: Final[str] = "dicom_source"
CLASS_DERIVED_STATE: Final[str] = "derived_state"
_PREFIX_CLASSES: Final[dict[str, str]] = {
    "dicom_source/": CLASS_DICOM_SOURCE,
    "derived_state/": CLASS_DERIVED_STATE,
}


class BackupIntegrityError(CanonicalValidationError):
    """Archive failed pre-extraction digest verification."""


@dataclass(frozen=True)
class ManifestEntry:
    """One sealed archive member."""

    path: str
    sha256: str
    byte_size: int
    member_class: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.byte_size,
            "class": self.member_class,
        }


@dataclass(frozen=True)
class RestoreSummary:
    """Counts of what restore placed on disk."""

    restored_files: int
    dicom_source_files: int
    derived_state_files: int


def _classify(member_name: str) -> str | None:
    for prefix, klass in _PREFIX_CLASSES.items():
        if member_name.startswith(prefix):
            return klass
    return None


def _is_safe_member_name(name: str) -> bool:
    if not name or name.startswith(("/", "\\")):
        return False
    if ":" in name.split("/")[0]:
        return False
    parts = name.replace("\\", "/").split("/")
    return ".." not in parts


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class WorkspaceBackupManager:
    """Backup/restore of one workstation workspace (§B10)."""

    def __init__(self, *, chunk_size: int = 1 << 20) -> None:
        self._chunk = chunk_size

    # -- backup ------------------------------------------------------------------

    def backup(
        self,
        workspace_root: Path,
        dest_dir: Path,
        *,
        source_dirs: tuple[Path, ...] = (),
        archive_stem: str | None = None,
    ) -> Path:
        """Archive *workspace_root* (+ optional DICOM source snapshots)."""
        root = Path(workspace_root)
        if not root.is_dir():
            raise CanonicalValidationError(f"workspace {root} does not exist")
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        members: list[tuple[str, Path]] = []
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(root).as_posix()
            members.append((f"{CLASS_DERIVED_STATE}/{relative}", file_path))
        for source_dir in source_dirs:
            resolved = Path(source_dir).resolve()
            if not resolved.is_dir():
                raise CanonicalValidationError(
                    f"source dir {resolved.name} does not exist"
                )
            for file_path in sorted(resolved.rglob("*")):
                if not file_path.is_file():
                    continue
                relative = file_path.relative_to(resolved).as_posix()
                members.append(
                    (
                        f"{CLASS_DICOM_SOURCE}/{resolved.name}/{relative}",
                        file_path,
                    )
                )

        unique: dict[str, Path] = {}
        for name, path in members:
            if name in unique:
                raise CanonicalValidationError(f"duplicate archive member {name}")
            unique[name] = path
        ordered = sorted(unique.items())

        entries: list[ManifestEntry] = []
        for name, path in ordered:
            entries.append(
                ManifestEntry(
                    path=name,
                    sha256=_hash_bytes(path.read_bytes()),
                    byte_size=path.stat().st_size,
                    member_class=_classify(name) or CLASS_DERIVED_STATE,
                )
            )
        manifest_document = {
            "schema": BACKUP_MANIFEST_SCHEMA,
            "members": [entry.to_dict() for entry in entries],
        }
        manifest_bytes = json.dumps(
            manifest_document, sort_keys=True, indent=2
        ).encode("utf-8")

        stem = archive_stem or f"{root.name}-backup"
        archive_path = dest / f"{stem}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            info = tarfile.TarInfo(name=MANIFEST_MEMBER)
            info.size = len(manifest_bytes)
            archive.addfile(info, io.BytesIO(manifest_bytes))
            for name, path in ordered:
                archive.add(str(path), arcname=name)
        return archive_path

    # -- restore ----------------------------------------------------------------------

    def restore(self, archive_path: Path, target: Path) -> RestoreSummary:
        """Verify EVERY digest first; extract only after the whole archive passes."""
        archive = Path(archive_path)
        target_dir = Path(target)
        if target_dir.exists() and any(target_dir.iterdir()):
            raise CanonicalValidationError("restore target must be empty or nonexistent")

        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
            if MANIFEST_MEMBER not in names:
                raise BackupIntegrityError(["archive has no backup manifest"])
            manifest_handle = tar.extractfile(MANIFEST_MEMBER)
            if manifest_handle is None:
                raise BackupIntegrityError(["manifest member is not a regular file"])
            try:
                document = json.loads(manifest_handle.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BackupIntegrityError([f"manifest unreadable: {exc}"]) from None
            rows = document.get("members")
            if not isinstance(rows, list):
                raise BackupIntegrityError(["manifest has no members array"])

            declared: dict[str, ManifestEntry] = {}
            problems: list[str] = []
            for row in rows:
                if not isinstance(row, dict):
                    problems.append("malformed manifest entry")
                    continue
                name = str(row.get("path", ""))
                sha = str(row.get("sha256", ""))
                size = int(row.get("bytes", -1))
                klass = str(row.get("class", CLASS_DERIVED_STATE))
                if not _is_safe_member_name(name):
                    problems.append(f"{name}: unsafe member path")
                    continue
                if name == MANIFEST_MEMBER:
                    problems.append("manifest lists itself")
                    continue
                declared[name] = ManifestEntry(
                    path=name, sha256=sha, byte_size=size, member_class=klass
                )
            if problems:
                raise BackupIntegrityError(problems)

            actual_members = {
                name for name in names if name != MANIFEST_MEMBER and not name.endswith("/")
            }
            undeclared = sorted(actual_members - set(declared))
            if undeclared:
                raise BackupIntegrityError(
                    [f"{name}: member not covered by manifest" for name in undeclared]
                )

            # PASS 1: verify every member digest BEFORE any extraction.
            payloads: dict[str, bytes] = {}
            for name in sorted(actual_members):
                handle = tar.extractfile(name)
                if handle is None:
                    raise BackupIntegrityError([f"{name}: not a regular file"])
                expected = declared[name]
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = handle.read(self._chunk)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                data = b"".join(chunks)
                if total != expected.byte_size or _hash_bytes(data) != expected.sha256:
                    raise BackupIntegrityError(
                        [
                            f"{name}: digest verification failed "
                            f"(size {total}, expected {expected.byte_size})"
                        ]
                    )
                payloads[name] = data

        # PASS 2: extraction into staging, then atomic swap into place.
        staging = target_dir.with_name(target_dir.name + ".staging")
        staging.mkdir(parents=True, exist_ok=True)
        source_count = 0
        derived_count = 0
        try:
            for name, data in sorted(payloads.items()):
                out_path = staging / name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(data)
                if declared[name].member_class == CLASS_DICOM_SOURCE:
                    source_count += 1
                else:
                    derived_count += 1
            if target_dir.exists():
                os.rmdir(target_dir)
            os.replace(staging, target_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        return RestoreSummary(
            restored_files=len(payloads),
            dicom_source_files=source_count,
            derived_state_files=derived_count,
        )


__all__ = [
    "BACKUP_MANIFEST_SCHEMA",
    "BackupIntegrityError",
    "CLASS_DICOM_SOURCE",
    "CLASS_DERIVED_STATE",
    "MANIFEST_MEMBER",
    "ManifestEntry",
    "RestoreSummary",
    "WorkspaceBackupManager",
]
