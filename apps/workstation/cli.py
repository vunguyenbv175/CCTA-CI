"""WorkstationCliHarness — automation surface for the exit gate (§B5).

``python -m apps.workstation.cli {import,inspect,derive,save,reopen,verify,
export-audit}``. Deterministic JSON on stdout (sorted keys; no wall-clock
values — identifiers live in the workspace, not in the output), and the
process exit code IS the verdict: 0 = PASS/OK, 1 = any fail-closed refusal.

The no-network guard (§B7) is installed by default for every command;
``--no-network-guard`` is an explicit operator override and is logged as an
operational audit event when used.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from apps.workstation.backup import WorkspaceBackupManager
from apps.workstation.round_trip import RoundTripVerificationEngine
from apps.workstation.session_store import (
    DerivedStateStoreManager,
    RestoredSession,
    SessionSnapshot,
    WorkstationPaths,
)
from apps.workstation.volume_loader import MEASUREMENT_ALGORITHM
from apps.workstation.workspace import (
    SOFTWARE_VERSION,
    NotAnalyzableError,
    StudyWorkspaceManager,
)
from core.audit_log import OPERATIONAL_STREAM, PhiSafeAuditSink, iter_stream_records
from core.enums import PrivacyClass
from core.errors import CanonicalValidationError
from core.events import EventLog
from core.network_guard import (
    NetworkEgressBlockedError,
    deactivate_network_guard,
    install_import_guard,
    install_network_guard,
    remove_import_guard,
)

AUDIT_EXPORT_SCHEMA: Final[str] = "ccta.audit.export.v1"
PSEUDONYM_KEY_FILE: Final[str] = ".pseudonym_key"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apps.workstation.cli",
        description="CCTA Phase-0 workstation shell (headless exit-gate harness).",
    )
    parser.add_argument("--version", action="version", version=SOFTWARE_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workspace", required=True, help="workspace root directory")
    common.add_argument(
        "--id-seed",
        type=int,
        default=None,
        help="deterministic id sequence seed (reproducible runs / golden tests)",
    )
    common.add_argument(
        "--no-network-guard",
        action="store_true",
        help="operator override: skip installing the no-network egress guard",
    )

    p_import = subparsers.add_parser("import", parents=[common], help="import a DICOM directory")
    p_import.add_argument("--dicom-dir", required=True)
    p_import.add_argument(
        "--privacy-class",
        default="SYNTHETIC",
        choices=["SYNTHETIC", "DE_IDENTIFIED", "IDENTIFIABLE", "PUBLIC_DATASET"],
    )

    for name, help_text in (
        ("inspect", "print a stable session inventory"),
        ("derive", "materialize volume + deterministic artifacts"),
        ("save", "re-persist the current session bundle"),
        ("reopen", "total-or-fail restore of the saved session"),
        ("verify", "run the five-check round-trip exit gate"),
        ("export-audit", "export the PHI-safe operational audit log"),
    ):
        subparsers.add_parser(name, parents=[common], help=help_text)

    p_backup = subparsers.add_parser(
        "backup", parents=[common], help="archive the workspace (tar.gz + sha256 manifest)"
    )
    p_backup.add_argument("--dest", required=True)
    p_backup.add_argument(
        "--include-source",
        action="store_true",
        help="snapshot the original DICOM tree into dicom_source/ members",
    )
    p_restore = subparsers.add_parser(
        "restore", parents=[common], help="restore a verified backup archive"
    )
    p_restore.add_argument("--archive", required=True)

    return parser


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _emit(document: dict[str, object]) -> None:
    print(json.dumps(document, sort_keys=True, indent=2))


def _ensure_pseudonym_key(paths: WorkstationPaths) -> bytes:
    key_file = paths.logs / PSEUDONYM_KEY_FILE
    if key_file.exists():
        return bytes.fromhex(key_file.read_text(encoding="utf-8").strip())
    key = secrets.token_bytes(32)
    key_file.write_text(key.hex(), encoding="utf-8", newline="\n")
    return key


def _open_sink(paths: WorkstationPaths) -> PhiSafeAuditSink:
    return PhiSafeAuditSink(paths.logs, pseudonym_key=_ensure_pseudonym_key(paths))


def _arm_guard(paths: WorkstationPaths, *, disabled: bool) -> PhiSafeAuditSink | None:
    if disabled:
        sink = _open_sink(paths)
        sink.record(
            entity="workstation_cli",
            entity_id="network-guard",
            event_type="GUARD_OVERRIDE",
            to_value="DISABLED_BY_OPERATOR",
        )
        return sink
    sink = _open_sink(paths)
    install_network_guard(sink=sink)
    install_import_guard()
    return sink


def _disarm() -> None:
    deactivate_network_guard()
    remove_import_guard()


def _load_session(paths: WorkstationPaths) -> RestoredSession:
    return DerivedStateStoreManager(paths).reopen()


def _privacy_class(name: str) -> PrivacyClass:
    return PrivacyClass[name]


def _id_factory(seed: int | None):
    """Deterministic id sequence when --id-seed is given (golden runs)."""
    if seed is None:
        return None
    from itertools import count

    numbers = count()

    def factory() -> str:
        return f"ID{seed:03d}{next(numbers):05d}"

    return factory


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def _cmd_import(args: argparse.Namespace) -> int:
    paths = WorkstationPaths.create(Path(args.workspace))
    _arm_guard(paths, disabled=args.no_network_guard)
    try:
        manager = StudyWorkspaceManager(
            Path(args.workspace), id_factory=_id_factory(args.id_seed)
        )
        imported = manager.import_study(
            Path(args.dicom_dir),
            privacy_class=_privacy_class(args.privacy_class),
        )
        snapshot = SessionSnapshot(
            study=imported.study,
            series=tuple(item.series for item in imported.series),
            volumes=(),
            artifacts=(),
            payload_documents={},
            provenance=imported.provenance,
            log=EventLog(),
            skipped_instances=imported.skipped,
        )
        digest = DerivedStateStoreManager(imported.paths).save(snapshot)
        _emit(
            {
                "command": "import",
                "dicom_study_instance_uid": imported.study.dicom_study_instance_uid,
                "exam_type": imported.study.exam_classification.exam_type.value,
                "instance_count": sum(
                    s.series.manifest.instance_count for s in imported.series
                ),
                "multiphase": imported.study.exam_classification.multiphase,
                "session_digest": digest,
                "series_composite_sha256": [
                    item.series.manifest.composite_series_sha256
                    for item in imported.series
                ],
                "series_count": len(imported.series),
                "skipped_count": len(imported.skipped),
                "status": "IMPORTED",
            }
        )
        return 0
    finally:
        _disarm()


def _cmd_inspect(args: argparse.Namespace) -> int:
    paths = WorkstationPaths.create(Path(args.workspace))
    _arm_guard(paths, disabled=args.no_network_guard)
    try:
        session = _load_session(paths)
        _emit(
            {
                "command": "inspect",
                "artifact_count": len(session.artifacts),
                "artifacts": [
                    {
                        "kind": artifact.kind.value,
                        "review_state": artifact.review.review_state.value,
                    }
                    for artifact in session.artifacts
                ],
                "dicom_study_instance_uid": session.study.dicom_study_instance_uid,
                "measurement_algorithm": MEASUREMENT_ALGORITHM,
                "provenance_records": len(session.provenance),
                "series": [
                    {
                        "acquisition_phase": (
                            series.manifest.acquisition_phase.label
                            if series.manifest.acquisition_phase
                            else None
                        ),
                        "composite_series_sha256": series.manifest.composite_series_sha256,
                        "instance_count": series.manifest.instance_count,
                        "modality": series.modality,
                        "series_instance_uid": series.dicom_series_instance_uid,
                    }
                    for series in session.series
                ],
                "session_digest": session.session_digest,
                "skipped_count": len(session.skipped_instances),
                "volume_count": len(session.volumes),
            }
        )
        return 0
    finally:
        _disarm()


def _cmd_derive(args: argparse.Namespace) -> int:
    paths = WorkstationPaths.create(Path(args.workspace))
    _arm_guard(paths, disabled=args.no_network_guard)
    try:
        session = _load_session(paths)
        manager = StudyWorkspaceManager(
            Path(args.workspace), id_factory=_id_factory(args.id_seed)
        )
        manager.paths = paths
        outcome = manager.derive(session)
        documents = dict(session.payload_documents)
        documents.update(outcome.payload_documents)
        snapshot = SessionSnapshot(
            study=session.study,
            series=session.series,
            volumes=session.volumes + outcome.volumes,
            artifacts=session.artifacts + outcome.artifacts,
            payload_documents=documents,
            provenance=session.provenance + outcome.provenance,
            log=session.log,
            skipped_instances=session.skipped_instances,
        )
        digest = DerivedStateStoreManager(paths).save(snapshot)
        _emit(
            {
                "command": "derive",
                "already_derived_series": list(outcome.already_derived_series),
                "artifact_payload_sha256": [
                    artifact.payload.blob_sha256 for artifact in outcome.artifacts
                ],
                "derived_artifact_kinds": sorted(
                    {artifact.kind.value for artifact in outcome.artifacts}
                ),
                "new_artifacts": len(outcome.artifacts),
                "new_volumes": len(outcome.volumes),
                "session_digest": digest,
                "status": "DERIVED",
            }
        )
        return 0
    finally:
        _disarm()


def _cmd_save(args: argparse.Namespace) -> int:
    paths = WorkstationPaths.create(Path(args.workspace))
    _arm_guard(paths, disabled=args.no_network_guard)
    try:
        session = _load_session(paths)
        snapshot = SessionSnapshot(
            study=session.study,
            series=session.series,
            volumes=session.volumes,
            artifacts=session.artifacts,
            payload_documents=dict(session.payload_documents),
            provenance=session.provenance,
            log=session.log,
            skipped_instances=session.skipped_instances,
        )
        digest = DerivedStateStoreManager(paths).save(snapshot)
        _emit({"command": "save", "session_digest": digest, "status": "SAVED"})
        return 0
    finally:
        _disarm()


def _cmd_reopen(args: argparse.Namespace) -> int:
    paths = WorkstationPaths.create(Path(args.workspace))
    _arm_guard(paths, disabled=args.no_network_guard)
    try:
        session = _load_session(paths)
        _emit(
            {
                "command": "reopen",
                "artifact_count": len(session.artifacts),
                "integrity": "TOTAL_OR_FAIL_OK",
                "journal_events": len(session.log.events),
                "provenance_records": len(session.provenance),
                "session_digest": session.session_digest,
                "status": "REOPENED",
                "volume_count": len(session.volumes),
            }
        )
        return 0
    finally:
        _disarm()


def _cmd_verify(args: argparse.Namespace) -> int:
    paths = WorkstationPaths.create(Path(args.workspace))
    _arm_guard(paths, disabled=args.no_network_guard)
    try:
        session = _load_session(paths)
        report = RoundTripVerificationEngine(paths.root).verify(session)
        document = report.to_dict()
        document["command"] = "verify"
        _emit(document)
        return 0 if report.verdict == "PASS" else 1
    finally:
        _disarm()


def _cmd_export_audit(args: argparse.Namespace) -> int:
    paths = WorkstationPaths.create(Path(args.workspace))
    sink = _arm_guard(paths, disabled=args.no_network_guard)
    try:
        records = list(iter_stream_records(paths.logs, OPERATIONAL_STREAM))
        export_path = paths.exports / "audit_export.json"
        export_path.write_text(
            json.dumps(
                {"schema": AUDIT_EXPORT_SCHEMA, "records": records},
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
            newline="\n",
        )
        if sink is not None:
            sink.record(
                entity="workstation_cli",
                entity_id="export-audit",
                event_type="EXPORTED",
                to_value="exports/audit_export.json",
                payload={"record_count": len(records)},
            )
        _emit(
            {
                "command": "export-audit",
                "path": "exports/audit_export.json",
                "record_count": len(records),
                "status": "EXPORTED",
            }
        )
        return 0
    finally:
        _disarm()


def _cmd_backup(args: argparse.Namespace) -> int:
    paths = WorkstationPaths.create(Path(args.workspace))
    _arm_guard(paths, disabled=args.no_network_guard)
    try:
        source_dirs: tuple[Path, ...] = ()
        if args.include_source:
            session = _load_session(paths)
            roots = {
                Path(series.original_dicom_ref.root_path) for series in session.series
            }
            source_dirs = tuple(sorted(roots))
        archive = WorkspaceBackupManager().backup(
            Path(args.workspace), Path(args.dest), source_dirs=source_dirs
        )
        _emit(
            {
                "command": "backup",
                "archive": archive.name,
                "included_source_dirs": [p.name for p in source_dirs],
                "status": "BACKED_UP",
            }
        )
        return 0
    finally:
        _disarm()


def _cmd_restore(args: argparse.Namespace) -> int:
    paths_root = Path(args.workspace)
    summary = WorkspaceBackupManager().restore(Path(args.archive), paths_root)
    _emit(
        {
            "command": "restore",
            "dicom_source_files": summary.dicom_source_files,
            "derived_state_files": summary.derived_state_files,
            "restored_files": summary.restored_files,
            "status": "RESTORED",
        }
    )
    return 0


_COMMANDS = {
    "import": _cmd_import,
    "inspect": _cmd_inspect,
    "derive": _cmd_derive,
    "save": _cmd_save,
    "reopen": _cmd_reopen,
    "verify": _cmd_verify,
    "export-audit": _cmd_export_audit,
    "backup": _cmd_backup,
    "restore": _cmd_restore,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS[args.command]
    try:
        return int(handler(args))
    except (CanonicalValidationError, NotAnalyzableError, NetworkEgressBlockedError) as exc:
        _emit({"command": args.command, "error": exc.__class__.__name__, "reasons": [
            *getattr(exc, "violations", ())
        ] or [str(exc)]})
        return 1
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        workspace = getattr(args, "workspace", None)
        if workspace is not None:
            try:
                _open_sink(WorkstationPaths.create(Path(workspace))).record_exception(exc)
            except Exception:  # noqa: BLE001 - diagnostics must never mask the verdict
                pass
        _emit(
            {
                "command": args.command,
                "error": exc.__class__.__name__,
                "note": "details recorded to the diagnostic audit stream",
            }
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
