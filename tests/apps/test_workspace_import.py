"""Import-path tests: StudyWorkspaceManager (WS02 §B1, Rule A, H-009)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import pytest

from apps.workstation.workspace import (
    NotAnalyzableError,
    StudyWorkspaceManager,
    WorkstationPaths,
)
from core.enums import ContrastClass
from tests.apps.fixture_factory import (
    SERIES_UID_TEMPLATE,
    build_synthetic_study,
    counter_id_factory,
)


def _tree_digest(root: Path) -> str:
    sha = hashlib.sha256()
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file():
            sha.update(str(file_path.relative_to(root)).encode())
            sha.update(file_path.read_bytes())
    return sha.hexdigest()


def test_import_single_phase_creates_frozen_records(tmp_path: Path) -> None:
    source = build_synthetic_study(tmp_path / "dicom", slices_per_phase=3)
    manager = StudyWorkspaceManager(tmp_path / "ws", id_factory=counter_id_factory(1))
    imported = manager.import_study(source)

    assert len(imported.series) == 1
    series = imported.series[0].series
    assert not imported.study.exam_classification.multiphase
    assert series.manifest.instance_count == 3
    assert len(series.manifest.ordered_sop_uids) == 3
    assert imported.study.privacy_class.value == "SYNTHETIC"
    # provenance chain: CLASSIFIED + IMPORTED
    event_types = sorted(p.event_type.value for p in imported.provenance)
    assert event_types == ["CLASSIFIED", "IMPORTED"]
    assert imported.study.import_info.import_job_ref == imported.provenance[1].provenance_id


def test_import_multiphase_requires_and_binds_acquisition_phase(tmp_path: Path) -> None:
    source = build_synthetic_study(
        tmp_path / "dicom", phases=("75% R-R", "40% R-R"), slices_per_phase=2
    )
    manager = StudyWorkspaceManager(tmp_path / "ws", id_factory=counter_id_factory(2))
    imported = manager.import_study(source)

    assert imported.study.exam_classification.multiphase
    phases = {
        item.series.dicom_series_instance_uid: item.series.manifest.acquisition_phase
        for item in imported.series
    }
    labels = {phase.label for phase in phases.values() if phase is not None}
    assert labels == {"75% R-R", "40% R-R"}
    contrast_classes = {
        item.series.classification.contrast_class for item in imported.series
    }
    assert ContrastClass.ARTERIAL in contrast_classes


def test_import_never_writes_into_the_dicom_source_tree(tmp_path: Path) -> None:
    source = build_synthetic_study(tmp_path / "dicom", slices_per_phase=2)
    before = _tree_digest(source)
    mtimes_before = {
        p: p.stat().st_mtime for p in sorted(source.rglob("*")) if p.is_file()
    }

    manager = StudyWorkspaceManager(tmp_path / "ws", id_factory=counter_id_factory(3))
    manager.import_study(source)

    assert _tree_digest(source) == before
    mtimes_after = {
        p: p.stat().st_mtime for p in sorted(source.rglob("*")) if p.is_file()
    }
    assert mtimes_after == mtimes_before


def test_unparseable_files_are_skipped_with_reasons_not_silently(tmp_path: Path) -> None:
    source = build_synthetic_study(
        tmp_path / "dicom",
        slices_per_phase=2,
        include_corrupt_file=True,
    )
    manager = StudyWorkspaceManager(tmp_path / "ws", id_factory=counter_id_factory(4))
    imported = manager.import_study(source)

    assert [item.series.manifest.instance_count for item in imported.series] == [2]
    assert len(imported.skipped) == 1
    assert "broken.dcm" in imported.skipped[0].path
    assert "magic" in imported.skipped[0].reason


def test_directory_with_no_analyzable_instance_abstains(tmp_path: Path) -> None:
    source = build_synthetic_study(
        tmp_path / "dicom", include_corrupt_file=True
    )
    # leave only the corrupt file
    for valid in source.glob("phase1_slice*.dcm"):
        valid.unlink()
    manager = StudyWorkspaceManager(tmp_path / "ws", id_factory=counter_id_factory(5))
    with pytest.raises(NotAnalyzableError):
        manager.import_study(source)


def test_mixed_study_uids_refuse_to_guess(tmp_path: Path) -> None:
    source = build_synthetic_study(
        tmp_path / "dicom", include_foreign_study_file=True
    )
    manager = StudyWorkspaceManager(tmp_path / "ws", id_factory=counter_id_factory(6))
    with pytest.raises(NotAnalyzableError):
        manager.import_study(source)


def test_workspace_layout_is_explicit(tmp_path: Path) -> None:
    paths = WorkstationPaths.create(tmp_path / "ws")
    assert paths.session.is_dir()
    assert paths.blobs.is_dir()
    assert paths.logs.is_dir()
    assert paths.exports.is_dir()


def test_series_uid_template_is_stable() -> None:
    assert SERIES_UID_TEMPLATE.format(phase=1).count(".") >= 6


# ---------------------------------------------------------------------------
# R1 fix round F2: the derive path routes through the stack adapter
# ---------------------------------------------------------------------------


def test_derive_persists_canonical_v2_volume_with_source_bit_depth(
    tmp_path: Path,
) -> None:
    """Import -> derive -> save -> reopen emits adapter-owned provenance.

    The volume envelope records the four source_bits_* fields and the
    canonical codec decision of the stack adapter; a conformant 12-bit import
    derives an MPR ccta.mpr.view.v2; a uniform 16-bit import derives v2 as
    well; and the persisted session round-trips (the exit gate stays PASS).
    """
    from apps.workstation.round_trip import RoundTripVerificationEngine
    from apps.workstation.session_store import (
        DerivedStateStoreManager,
        SessionSnapshot,
    )
    from apps.workstation.volume_loader import CanonicalVolumeLoader
    from core.events import EventLog

    source = build_synthetic_study(
        tmp_path / "dicom",
        slices_per_phase=3,
        rows=2,
        columns=2,
    )
    manager = StudyWorkspaceManager(tmp_path / "ws", id_factory=counter_id_factory(21))
    imported = manager.import_study(source)
    store = DerivedStateStoreManager(imported.paths)
    store.save(
        SessionSnapshot(
            study=imported.study,
            series=tuple(item.series for item in imported.series),
            volumes=(),
            artifacts=(),
            payload_documents={},
            provenance=imported.provenance,
            log=EventLog(),
            skipped_instances=imported.skipped,
        )
    )
    session = store.reopen()
    outcome = manager.derive(session)
    assert len(outcome.volumes) == 1
    volume = outcome.volumes[0]

    loader = CanonicalVolumeLoader(imported.paths.blobs)
    envelope = loader.read_envelope(volume.blob_ref.sha256)
    # The adapter is the single authority: canonical codec + full record.
    # The seeded 16-bit noise exceeds the signed int16 range, so the frozen
    # canonical decision is float32-le (the legacy path stored raw uint16-le).
    assert envelope["codec"] == "float32-le"
    assert envelope["source_bits_allocated"] == 16
    assert envelope["source_bits_stored"] == 16
    assert envelope["source_high_bit"] == 15
    assert envelope["source_pixel_representation"] == 0

    store.save(
        SessionSnapshot(
            study=session.study,
            series=session.series,
            volumes=session.volumes + outcome.volumes,
            artifacts=session.artifacts + outcome.artifacts,
            payload_documents={**session.payload_documents, **outcome.payload_documents},
            provenance=session.provenance + outcome.provenance,
            log=session.log,
            skipped_instances=session.skipped_instances,
        )
    )
    reopened = store.reopen()
    assert RoundTripVerificationEngine(imported.paths.root).verify(reopened).verdict == "PASS"


def test_derive_of_conformant_12_bit_series_emits_mpr_v2(tmp_path: Path) -> None:
    """A 12-bit import derives through the adapter and emits ccta.mpr.view.v2."""
    from apps.workstation.session_store import (
        DerivedStateStoreManager,
        SessionSnapshot,
    )
    from apps.workstation.volume_loader import CanonicalVolumeLoader
    from core.coordinates import transform_for_frame
    from core.events import EventLog
    from tests.core.dicom.test_imaging_core_002_admission import TWELVE_BIT, _build
    from visualization.mpr import build_mpr

    source_dir = tmp_path / "dicom"
    source_dir.mkdir()
    for index in range(3):
        _build(
            source_dir / f"slice{index}.dcm",
            index=index + 1,
            rows=2,
            columns=2,
            position=(0.0, 0.0, float(index)),
            **cast(dict[str, Any], TWELVE_BIT),
        )
    # The default id factory mints ULIDs: build_mpr's source_volume_ref
    # contract is volume_id:<ULID>, so this test cannot use a counter factory.
    manager = StudyWorkspaceManager(tmp_path / "ws")
    imported = manager.import_study(source_dir)
    store = DerivedStateStoreManager(imported.paths)
    store.save(
        SessionSnapshot(
            study=imported.study,
            series=tuple(item.series for item in imported.series),
            volumes=(),
            artifacts=(),
            payload_documents={},
            provenance=imported.provenance,
            log=EventLog(),
            skipped_instances=imported.skipped,
        )
    )
    session = store.reopen()
    outcome = manager.derive(session)
    assert len(outcome.volumes) == 1
    volume = outcome.volumes[0]
    loader = CanonicalVolumeLoader(imported.paths.blobs)
    envelope = loader.read_envelope(volume.blob_ref.sha256)
    assert envelope["source_bits_stored"] == 12
    assert envelope["source_high_bit"] == 11
    assert envelope["codec"] == "int16-le"

    # The derive product feeds a v2 MPR frame with the recorded provenance.
    from core.ids import new_ulid

    transform = transform_for_frame(
        volume.coordinate_frame,
        series_instance_uid="1.2.3.4",
    )
    result = build_mpr(
        volume,
        transform,
        plane="AXIAL",
        origin=(0.0, 0.0, 0.0),
        in_plane_spacing_mm=(0.5, 0.5),
        matrix_size=(1, 1),
        method="NEAREST",
        out_of_bounds_fill_value=0.0,
        loader=loader,
        transform_artifact_ref=f"coordinate_transform:{new_ulid()}",
        provenance_ref=f"provenance:{new_ulid()}",
        study_instance_uid=imported.study.dicom_study_instance_uid,
        series_instance_uid="1.2.3.4",
    )
    assert result.payload["schema_version"] == "ccta.mpr.view.v2"
    assert result.payload["bits_allocated"] == 16
    assert result.payload["bits_stored"] == 12
    assert result.payload["high_bit"] == 11
    assert result.payload["pixel_representation"] == 0
