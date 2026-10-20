"""IMAGING_CORE_002 F18: the viewer's HU-semantic frame-construction gate.

Astra finding F18 (REPRODUCED, P1 acceptance blocker): a pre-F15 chain whose
canonical source volume records the **STORED** scalar domain reopens, its
already-derived series is skipped, and ``apps/viewer/frames.py`` accepted
every ACCEPTED MPR_VIEW with NO scalar-domain predicate — so the HU-semantic
frozen presets (VIEWER_001_SPEC Part 4) were applied directly to decoded
STORED samples.  The probe record ``reports/IMAGING_CORE_002_F18_PROBE.md``
captures the harm end to end: stored 1324 renders byte 255 (white-clipped)
under ``CT_VESSEL_CONTRAST`` where the HU-equivalent 300 renders 128, and the
persisted measurement carries ``mean_hu=1324.0``.

This battery builds that exact persisted session (the probe's fixture
skeleton: loader-direct STORED materialize + accepted MPR + save through the
real session store) and asserts the FIXED behavior:

* reopen -> frame-list construction raises the TYPED refusal
  :class:`~apps.viewer.frames.NonHUScalarDomainError` — never a silent
  render, never a skipped frame, never ``*_hu`` labeling of STORED pixels;
* the fresh HU chain stays completely green (render 128, ``mean_hu=300``);
* a record-free / absent domain is its own distinct refusal state (the gate
  never guesses and never defaults a domain).

Every expected number is the hand-computed oracle of the admission battery
(``1324 * 1 - 1024 = 300``; byte 128 vs 255 under center 300 / width 800) —
never a value read back from the implementation under test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from apps.viewer.engine import ViewerEngine, ViewerOpenError
from apps.viewer.frames import (
    FrameListConstructionError,
    NonHUScalarDomainError,
    build_geometry_lists,
    render_png_for_record,
)
from apps.viewer.grammar import RejectionCode
from apps.viewer.presets import preset_by_id
from apps.viewer.transfer import window_bounds, window_byte, window_bytes
from apps.workstation.volume_loader import CanonicalVolumeLoader
from apps.workstation.workspace import StudyWorkspaceManager
from core.enums import ArtifactKind
from tests.apps.viewer_pipeline import build_viewer_session
from tests.core.dicom.test_imaging_core_002_admission import (
    F15_HU,
    F15_STORED,
    F15_WL_BYTE_HU,
    F15_WL_BYTE_STORED,
    F15_WL_CENTER,
    F15_WL_WIDTH,
    _f15_series,
)
from visualization.mpr import decode_pixel_blob

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

# ---------------------------------------------------------------------------
# The hand oracle (independent of every implementation path under test)
# ---------------------------------------------------------------------------


def test_hand_oracle_is_the_probe_contrast() -> None:
    """1324 stored -> 300 HU -> byte 128; the same preset clips STORED to 255."""

    assert F15_STORED == 1324
    assert F15_HU == 300
    assert F15_WL_BYTE_HU == 128
    assert F15_WL_BYTE_STORED == 255
    lo, hi = window_bounds(F15_WL_CENTER, F15_WL_WIDTH)
    assert (lo, hi) == (-100.0, 700.0)
    assert window_byte(float(F15_HU), lo, hi) == F15_WL_BYTE_HU
    assert window_byte(float(F15_STORED), lo, hi) == F15_WL_BYTE_STORED


# ---------------------------------------------------------------------------
# Fixtures: the probe's two chains (STORED persisted, fresh HU)
# ---------------------------------------------------------------------------


def build_stored_domain_session(
    workspace: Path, *, dicom_dir: Path
) -> Any:
    """Build the persisted pre-F15 STORED accepted session, exactly as the probe.

    Only the OLDER scalar-domain selection is emulated, by intercepting the
    single ``materialize_dicom_stack`` call of the real derive path and
    passing ``scalar_domain="STORED"``.  Import, derive, transform,
    measurement, MPR construction, HUMAN-role ACCEPTED transitions,
    provenance, session save and reopen are the untouched real pipeline.
    """

    original = CanonicalVolumeLoader.materialize_dicom_stack
    calls: list[dict[str, Any]] = []

    def legacy_selection(self: CanonicalVolumeLoader, instances: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return original(self, instances, scalar_domain="STORED", intensity_unit="STORED")

    with patch.object(CanonicalVolumeLoader, "materialize_dicom_stack", legacy_selection):
        fixture = build_viewer_session(workspace, dicom_dir=dicom_dir)
    # The wrapper is removed before the code under test is exercised.
    assert CanonicalVolumeLoader.materialize_dicom_stack is original
    assert len(calls) == 1, "exactly the one derive materialization is emulated"
    return fixture


@pytest.fixture(scope="module")
def f15_dicom(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three 4x5 slices, every stored sample F15_STORED (slope 1, -1024)."""

    return _f15_series(
        tmp_path_factory.mktemp("f18-f15-dicom"), slices=3, rows=4, columns=5
    )


@pytest.fixture()
def stored_chain(tmp_path: Path, f15_dicom: Path) -> Any:
    """The pre-F15 STORED accepted session, reopened as the viewer would."""

    fixture = build_stored_domain_session(tmp_path / "stored", dicom_dir=f15_dicom)
    loader = CanonicalVolumeLoader(fixture.workspace / "blobs")
    session = fixture.store.reopen()
    assert loader.read_envelope(session.volumes[0].blob_ref.sha256)["scalar_domain"] == "STORED"
    return fixture


@pytest.fixture()
def hu_chain(tmp_path: Path, f15_dicom: Path) -> Any:
    """The current fresh HU chain (the probe's step-5 contrast case)."""

    fixture = build_viewer_session(tmp_path / "hu", dicom_dir=f15_dicom)
    loader = CanonicalVolumeLoader(fixture.workspace / "blobs")
    session = fixture.store.reopen()
    assert loader.read_envelope(session.volumes[0].blob_ref.sha256)["scalar_domain"] == "HU"
    return fixture


def _envelope_domains(fixture: Any) -> dict[str, str]:
    """The recorded scalar domain of every restored volume (never defaulted)."""

    session = fixture.session
    loader = CanonicalVolumeLoader(fixture.workspace / "blobs")
    domains: dict[str, str] = {}
    for volume in session.volumes:
        domain = loader.read_envelope(volume.blob_ref.sha256).get("scalar_domain")
        if isinstance(domain, str):
            domains[volume.volume_id] = domain
    return domains


def _reopen_after_derive_skip(fixture: Any) -> Any:
    """Reopen and run derive, as the probe does: the known series is skipped."""

    session = fixture.store.reopen()
    manager = StudyWorkspaceManager(fixture.workspace)
    manager.initialize()
    skipped = manager.derive(session)
    assert skipped.already_derived_series == (session.series[0].series_id,)
    assert not skipped.volumes and not skipped.artifacts
    return session


def _geometry_kwargs(fixture: Any, session: Any) -> dict[str, Any]:
    return {
        "session_id": session.study.study_id,
        "artifacts": session.artifacts,
        "payload_documents": session.payload_documents,
        "volume_ids": tuple(volume.volume_id for volume in session.volumes),
        "volume_scalar_domains": _envelope_domains(fixture),
    }


def _first_axial_pixels(fixture: Any, session: Any, geometry: Any) -> np.ndarray:
    frame = geometry["AXIAL"][0]
    payload = session.payload_documents[frame.mpr_artifact_id]
    blob = CanonicalVolumeLoader(fixture.workspace / "blobs").read_blob(
        cast(dict[str, Any], payload["pixel_blob_ref"])["sha256"]
    )
    return decode_pixel_blob(payload, blob)


# ---------------------------------------------------------------------------
# 1. The gate: a STORED accepted MPR is refused, typed, at frame construction
# ---------------------------------------------------------------------------


def test_stored_accepted_mpr_refuses_frame_construction_typed(stored_chain: Any) -> None:
    session = _reopen_after_derive_skip(stored_chain)
    accepted = [
        artifact
        for artifact in session.artifacts
        if artifact.kind == ArtifactKind.MPR_VIEW
    ]
    assert accepted, "the fixture must carry accepted MPR views"
    assert all(
        session.payload_documents[a.artifact_id]["scalar_domain"] == "STORED"
        for a in accepted
    )

    with pytest.raises(NonHUScalarDomainError) as error:
        build_geometry_lists(**_geometry_kwargs(stored_chain, session))
    message = str(error.value)
    assert "STORED" in message and "HU" in message
    # A typed refusal from the viewer's closed vocabulary — never a bare
    # string, never a silent skip.
    assert isinstance(error.value, FrameListConstructionError)


def test_refusal_is_not_a_silent_skip_or_a_byte_255_render(stored_chain: Any) -> None:
    """Fail closed: no frame list, so no HU preset can touch STORED pixels.

    The harm the gate prevents is asserted from the hand oracle: those very
    samples ARE 1324 and the frozen preset maps 1324 to the clipped byte 255.
    Only the gate stands between them and a white-clipped, HU-labeled frame.
    """

    session = _reopen_after_derive_skip(stored_chain)
    with pytest.raises(NonHUScalarDomainError):
        build_geometry_lists(**_geometry_kwargs(stored_chain, session))

    # Bypassing the gate is the defect: the same payload still decodes to
    # STORED samples that CT_VESSEL_CONTRAST would white-clip.
    payload = next(
        session.payload_documents[artifact.artifact_id]
        for artifact in session.artifacts
        if artifact.kind == ArtifactKind.MPR_VIEW
    )
    blob = CanonicalVolumeLoader(stored_chain.workspace / "blobs").read_blob(
        cast(dict[str, Any], payload["pixel_blob_ref"])["sha256"]
    )
    pixels = decode_pixel_blob(payload, blob)
    assert int(pixels[0, 0]) == F15_STORED
    preset = preset_by_id("CT_VESSEL_CONTRAST")
    assert window_byte(float(F15_STORED), *window_bounds(preset.center, preset.width)) == (
        F15_WL_BYTE_STORED
    )
    assert np.all(
        window_bytes(pixels, preset.center, preset.width) == F15_WL_BYTE_STORED
    )
    # Historical evidence is untouched: no envelope/artifact rewriting (the
    # gate refuses at open/use only).
    assert payload["scalar_domain"] == "STORED"
    measurement = next(
        document
        for document in session.payload_documents.values()
        if document.get("algorithm") == "central-box-hu-stats"
    )
    assert measurement["mean_hu"] == float(F15_STORED)


def test_gate_keys_on_the_recorded_domain_even_without_the_envelope_map(
    stored_chain: Any,
) -> None:
    """The payload-side gate alone still refuses: nothing is defaulted to HU."""

    session = _reopen_after_derive_skip(stored_chain)
    with pytest.raises(NonHUScalarDomainError):
        build_geometry_lists(
            session_id=session.study.study_id,
            artifacts=session.artifacts,
            payload_documents=session.payload_documents,
            volume_ids=tuple(volume.volume_id for volume in session.volumes),
        )


def test_record_free_domain_is_its_own_refusal_state(hu_chain: Any) -> None:
    """An absent payload or envelope domain is a distinct fail-closed state."""

    session = _reopen_after_derive_skip(hu_chain)
    kwargs = _geometry_kwargs(hu_chain, session)
    mpr_id = next(
        artifact.artifact_id
        for artifact in session.artifacts
        if artifact.kind == ArtifactKind.MPR_VIEW
    )

    # (a) payload key absent
    payloads = dict(session.payload_documents)
    stripped = {key: dict(value) for key, value in payloads.items()}
    stripped[mpr_id].pop("scalar_domain")
    with pytest.raises(NonHUScalarDomainError) as error:
        build_geometry_lists(**{**kwargs, "payload_documents": stripped})
    assert "record" in str(error.value).lower()

    # (b) envelope record absent for the source volume
    volume_id = str(stripped[mpr_id]["source_volume_ref"]).split(":", 1)[1]
    domains = dict(kwargs["volume_scalar_domains"])
    domains.pop(volume_id)
    with pytest.raises(NonHUScalarDomainError):
        build_geometry_lists(**{**kwargs, "volume_scalar_domains": domains})


def test_payload_and_envelope_domain_disagreement_refuses(hu_chain: Any) -> None:
    payloads = {
        key: dict(value) for key, value in hu_chain.session.payload_documents.items()
    }
    mpr_id = next(
        artifact.artifact_id
        for artifact in hu_chain.session.artifacts
        if artifact.kind == ArtifactKind.MPR_VIEW
    )
    payloads[mpr_id]["scalar_domain"] = "STORED"
    with pytest.raises(NonHUScalarDomainError) as error:
        build_geometry_lists(
            session_id=hu_chain.session.study.study_id,
            artifacts=hu_chain.session.artifacts,
            payload_documents=payloads,
            volume_ids=tuple(v.volume_id for v in hu_chain.session.volumes),
            volume_scalar_domains=_envelope_domains(hu_chain),
        )
    assert "disagree" in str(error.value)


def test_closed_vocabulary_is_extended_by_specialisation_only() -> None:
    """The new typed refusal is a subclass: no new top-level viewer type."""

    assert issubclass(NonHUScalarDomainError, FrameListConstructionError)
    # The 16 closed action rejection codes are unchanged (VIEWER_001 Part 3.1).
    assert len(RejectionCode) == 16


# ---------------------------------------------------------------------------
# 2. The fresh HU chain is completely unaffected
# ---------------------------------------------------------------------------


def test_fresh_hu_chain_stays_green_end_to_end(hu_chain: Any) -> None:
    session = _reopen_after_derive_skip(hu_chain)
    geometry = build_geometry_lists(**_geometry_kwargs(hu_chain, session))
    assert {plane: len(frames) for plane, frames in geometry.items()} == {
        "AXIAL": 2,
        "CORONAL": 2,
        "SAGITTAL": 2,
    }
    pixels = _first_axial_pixels(hu_chain, session, geometry)
    assert pixels.shape == (4, 4)
    assert np.all(pixels == F15_HU)

    preset = preset_by_id("CT_VESSEL_CONTRAST")
    lo, hi = window_bounds(preset.center, preset.width)
    assert window_byte(float(pixels[0, 0]), lo, hi) == F15_WL_BYTE_HU
    canvas = window_bytes(pixels, preset.center, preset.width)
    assert canvas[0, 0].tolist() == [F15_WL_BYTE_HU] * 3

    frame = geometry["AXIAL"][0]
    payload = session.payload_documents[frame.mpr_artifact_id]
    loader = CanonicalVolumeLoader(hu_chain.workspace / "blobs")
    blob = loader.read_blob(cast(dict[str, Any], payload["pixel_blob_ref"])["sha256"])
    png = render_png_for_record(payload, blob, preset.center, preset.width)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    measurement = next(
        document
        for document in session.payload_documents.values()
        if document.get("algorithm") == "central-box-hu-stats"
    )
    assert measurement["mean_hu"] == 300.0
    assert measurement["min_hu"] == 300
    assert measurement["max_hu"] == 300


# ---------------------------------------------------------------------------
# 3. The engine boundary: the typed refusal surfaces as SESSION_OPEN_FAILED
# ---------------------------------------------------------------------------


def _deterministic_ulids() -> Any:
    class _Factory:
        def __init__(self) -> None:
            self._n = 0

        def __call__(self, at_ms: int | None = None) -> str:
            del at_ms
            self._n += 1
            value = (0x01A9 << 80) | self._n
            return "".join(
                _ULID_ALPHABET[(value >> (shift * 5)) & 0x1F]
                for shift in range(25, -1, -1)
            )

    return _Factory()


@pytest.fixture(autouse=True)
def _run_inside_bound_checkout(
    clean_checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F11 pattern (as in ``tests/apps/test_viewer_e2e.py``).

    The governed build-identity resolver is anchored at the bound checkout of
    the exact bytes under test and every engine operation runs with that
    checkout as its CWD.  In the governed configuration (a clean checkout of
    the candidate) the executing code root IS this root.
    """

    monkeypatch.chdir(clean_checkout)


def test_viewer_open_refuses_the_stored_chain_with_session_open_failed(
    stored_chain: Any, clean_checkout: Path
) -> None:
    session = _reopen_after_derive_skip(stored_chain)
    engine = ViewerEngine(
        session,
        id_factory=_deterministic_ulids(),
        now=lambda: "2026-09-07T00:00:00Z",
        repo_root=clean_checkout,
    )
    with pytest.raises(ViewerOpenError) as error:
        engine.open()
    assert error.value.code is RejectionCode.SESSION_OPEN_FAILED
    assert "STORED" in "; ".join(error.value.problems)
    # Zero partial state: nothing rendered, no provenance minted.
    assert engine.new_provenance() == ()


def test_viewer_open_still_succeeds_for_the_hu_chain(
    hu_chain: Any, clean_checkout: Path
) -> None:
    session = _reopen_after_derive_skip(hu_chain)
    engine = ViewerEngine(
        session,
        id_factory=_deterministic_ulids(),
        now=lambda: "2026-09-07T00:00:00Z",
        repo_root=clean_checkout,
    )
    record, _state = engine.open()
    assert record.frame_lists_ref
    for plane in ("AXIAL", "CORONAL", "SAGITTAL"):
        frames = engine.frames_document(plane)["frames"]
        assert isinstance(frames, list)
        assert len(frames) == 2
