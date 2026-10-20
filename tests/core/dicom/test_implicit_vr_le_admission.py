"""DICOM_IMPLICIT_VR_LE_ADMISSION_001 — Implicit VR LE admission tests.

Implicit VR Little Endian (``1.2.840.10008.1.2``) is admitted alongside
Explicit VR Little Endian (``1.2.840.10008.1.2.1``) through pydicom's native
parser: no transcoding, no metadata rewrite, and the original
TransferSyntaxUID is carried on the canonical :class:`DicomInstance`.  This
module proves the admission end-to-end at the canonical layer:

  (a) implicit positive   — ORIGINAL/PRIMARY series parses, stack admits,
                            classification is importable (Syntax=VALID);
  (b) explicit regression — the same shape on explicit VR stays admitted;
  (c) semantic equivalence— explicit and implicit encodings of the same
                            series differ ONLY in ``transfer_syntax_uid`` and
                            classify/reduce identically (no digest forcing);
  (d) unsupported negative— a genuinely unsupported syntax (Deflated Explicit
                            VR LE) still refuses with UNSUPPORTED_TRANSFER_SYNTAX;
                            the JPEG Baseline UID is outside the admitted set;
  (e) DERIVED/MPR guard   — ImageType (DERIVED, SECONDARY, MPR) stays refused
                            with DERIVED_IMAGE_UNSUPPORTED even under the newly
                            admitted syntax.

Fixtures are written by ``tests.apps.fixture_factory.build_instance`` with a
matching ``transfer_syntax_uid`` (pydicom writes the declaration-correct
encoding); no byte is hand-crafted.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from core.dicom import (
    ADMITTED_TRANSFER_SYNTAXES,
    TRANSFER_SYNTAX_EXPLICIT_VR_LE,
    TRANSFER_SYNTAX_IMPLICIT_VR_LE,
    ClassificationOutcome,
    DicomAbstentionError,
    DicomInstance,
    classify_series,
    read_dicom_instance,
    validate_stack,
)
from tests.apps.fixture_factory import build_instance

STUDY_UID: str = "1.2.826.0.1.3680043.8.498.77.101"
SERIES_UID: str = "1.2.826.0.1.3680043.8.498.77.102"
FRAME_UID: str = "1.2.826.0.1.3680043.8.498.77.109"
SOP_UID_ROOT: str = "1.2.826.0.1.3680043.8.498.77.103"
SLICE_COUNT: int = 3
SLICE_STEP_MM: float = 1.0
PIXEL_SEED: int = 2026

#: Deflated Explicit VR LE — a syntactically valid UID outside the admitted
#: set (pydicom round-trips it, so it exercises the reader's syntax refusal,
#: unlike a compressed syntax whose raw PixelData pydicom will not write).
UNSUPPORTED_TRANSFER_SYNTAX_UID: str = "1.2.840.10008.1.2.1.99"
#: JPEG Baseline (Process 1) — the compressed UID named by the task order;
#: it is outside the admitted set (fixture-construction is covered by the
#: Deflated arm above, which pydicom can write without an encoder).
JPEG_BASELINE_UID: str = "1.2.840.10008.1.2.4.50"


def _build_series(root: Path, transfer_syntax_uid: str) -> tuple[Path, ...]:
    """Write one deterministic three-slice ORIGINAL/PRIMARY CT series."""
    return tuple(
        build_instance(
            root / f"slice-{index}.dcm",
            sop_instance_uid=f"{SOP_UID_ROOT}.{index + 1}",
            series_instance_uid=SERIES_UID,
            study_instance_uid=STUDY_UID,
            frame_of_reference_uid=FRAME_UID,
            position=(0.0, 0.0, index * SLICE_STEP_MM),
            pixel_seed=PIXEL_SEED + index,
            transfer_syntax_uid=transfer_syntax_uid,
        )
        for index in range(SLICE_COUNT)
    )


def _read_series(root: Path, transfer_syntax_uid: str) -> tuple[DicomInstance, ...]:
    paths = _build_series(root, transfer_syntax_uid)
    return tuple(read_dicom_instance(path) for path in paths)


def test_implicit_vr_le_original_primary_series_is_admitted(tmp_path: Path) -> None:
    """(a) Implicit VR LE parses natively and the stack/classification admit it."""
    instances = _read_series(tmp_path / "implicit", TRANSFER_SYNTAX_IMPLICIT_VR_LE)

    # The ORIGINAL TransferSyntaxUID is carried on every canonical instance.
    assert all(
        instance.transfer_syntax_uid == TRANSFER_SYNTAX_IMPLICIT_VR_LE
        for instance in instances
    )

    stack = validate_stack(instances)
    assert stack.admitted
    assert stack.slice_spacing_mm == pytest.approx(SLICE_STEP_MM)

    result = classify_series(instances, stack=stack)
    assert result.importable
    assert result.outcome == ClassificationOutcome.IMPORTABLE_UNKNOWN
    assert result.detail_codes == ()


def test_explicit_vr_le_regression_is_admitted(tmp_path: Path) -> None:
    """(b) The explicit VR arm keeps the same positive behavior."""
    instances = _read_series(tmp_path / "explicit", TRANSFER_SYNTAX_EXPLICIT_VR_LE)

    assert all(
        instance.transfer_syntax_uid == TRANSFER_SYNTAX_EXPLICIT_VR_LE
        for instance in instances
    )

    stack = validate_stack(instances)
    assert stack.admitted
    assert stack.slice_spacing_mm == pytest.approx(SLICE_STEP_MM)

    result = classify_series(instances, stack=stack)
    assert result.importable
    assert result.outcome == ClassificationOutcome.IMPORTABLE_UNKNOWN
    assert result.detail_codes == ()


def test_explicit_and_implicit_series_are_semantically_equivalent(tmp_path: Path) -> None:
    """(c) Encoding changes only the syntax UID — canonical semantics match."""
    explicit_paths = _build_series(tmp_path / "explicit", TRANSFER_SYNTAX_EXPLICIT_VR_LE)
    implicit_paths = _build_series(tmp_path / "implicit", TRANSFER_SYNTAX_IMPLICIT_VR_LE)

    # The two encodings really differ byte-wise; equivalence is asserted at
    # the canonical layer, never by forcing source bytes/digests to match.
    assert explicit_paths[0].read_bytes() != implicit_paths[0].read_bytes()

    explicit = tuple(read_dicom_instance(path) for path in explicit_paths)
    implicit = tuple(read_dicom_instance(path) for path in implicit_paths)
    assert len(explicit) == len(implicit) == SLICE_COUNT

    for explicit_instance, implicit_instance in zip(explicit, implicit):
        differences = {
            field.name
            for field in dataclasses.fields(DicomInstance)
            if getattr(explicit_instance, field.name)
            != getattr(implicit_instance, field.name)
        }
        assert differences == {"transfer_syntax_uid"}

    explicit_stack = validate_stack(explicit)
    implicit_stack = validate_stack(implicit)
    assert explicit_stack.admitted and implicit_stack.admitted
    assert explicit_stack.slice_spacing_mm == implicit_stack.slice_spacing_mm
    assert explicit_stack.normal == implicit_stack.normal

    explicit_result = classify_series(explicit, stack=explicit_stack)
    implicit_result = classify_series(implicit, stack=implicit_stack)
    assert explicit_result.protocol == implicit_result.protocol
    assert explicit_result.contrast == implicit_result.contrast
    assert explicit_result.phase == implicit_result.phase
    assert explicit_result.outcome == implicit_result.outcome
    assert explicit_result.visible_reason == implicit_result.visible_reason
    assert explicit_result.detail_codes == implicit_result.detail_codes
    assert explicit_result.provenance == implicit_result.provenance


def test_genuinely_unsupported_transfer_syntax_still_refuses(tmp_path: Path) -> None:
    """(d) A non-admitted syntax parses and still refuses at the syntax gate."""
    assert JPEG_BASELINE_UID not in ADMITTED_TRANSFER_SYNTAXES
    assert UNSUPPORTED_TRANSFER_SYNTAX_UID not in ADMITTED_TRANSFER_SYNTAXES

    path = build_instance(
        tmp_path / "unsupported.dcm",
        sop_instance_uid=f"{SOP_UID_ROOT}.9",
        series_instance_uid=SERIES_UID,
        study_instance_uid=STUDY_UID,
        frame_of_reference_uid=FRAME_UID,
        position=(0.0, 0.0, 0.0),
        transfer_syntax_uid=UNSUPPORTED_TRANSFER_SYNTAX_UID,
    )
    with pytest.raises(DicomAbstentionError) as error:
        read_dicom_instance(path)
    assert error.value.reason_code == "DICOM_SYNTAX_INVALID"
    assert error.value.detail_code == "UNSUPPORTED_TRANSFER_SYNTAX"


def test_derived_secondary_mpr_stays_refused_under_implicit_vr_le(tmp_path: Path) -> None:
    """(e) The admitted syntax does not widen the ImageType boundary."""
    path = build_instance(
        tmp_path / "derived-implicit.dcm",
        sop_instance_uid=f"{SOP_UID_ROOT}.8",
        series_instance_uid=SERIES_UID,
        study_instance_uid=STUDY_UID,
        frame_of_reference_uid=FRAME_UID,
        position=(0.0, 0.0, 0.0),
        image_type=("DERIVED", "SECONDARY", "MPR"),
        transfer_syntax_uid=TRANSFER_SYNTAX_IMPLICIT_VR_LE,
    )
    with pytest.raises(DicomAbstentionError) as error:
        read_dicom_instance(path)
    assert error.value.reason_code == "DERIVED_IMAGE_UNSUPPORTED"
