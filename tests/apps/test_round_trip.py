"""RoundTripVerificationEngine tests: the §61 exit gate (WS02 §B4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.workstation.round_trip import (
    CHECK_BLOBS,
    CHECK_COORDS,
    CHECK_MEASURE,
    CHECK_PROV,
    MANDATED_CHECKS,
    RoundTripVerificationEngine,
)
from apps.workstation.session_store import WorkstationPaths
from core.errors import CanonicalValidationError
from tests.apps.pipeline import rewrite_manifest, run_full_pipeline


class _RepoRootedFixture:
    """``ViewerSessionFixture`` + the bound clean checkout as repo_root."""

    def __init__(self, fixture: object, repo_root: Path) -> None:
        self._fixture = fixture
        self.repo_root = repo_root

    def __getattr__(self, name: str) -> object:
        return getattr(self._fixture, name)


@pytest.fixture(autouse=True)
def _run_inside_bound_checkout(clean_checkout: Path, monkeypatch: pytest.MonkeyPatch):
    """F11 pattern: run engine operations from inside the bound checkout."""

    monkeypatch.chdir(clean_checkout)


def test_all_five_checks_pass_on_synthetic_fixture(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    session = run_full_pipeline(workspace)
    report = RoundTripVerificationEngine(workspace).verify(session)
    assert report.verdict == "PASS"
    for name in MANDATED_CHECKS:
        check = report.checks[name]
        assert check.passed, f"{name}: {check.details}"
    document = report.to_dict()
    schema = document["schema"]
    checks_map = document["checks"]
    assert isinstance(schema, str) and schema.startswith("ccta.round_trip.report.")
    assert isinstance(checks_map, dict) and set(checks_map) >= set(MANDATED_CHECKS)
    # informational source check ran against the live fixture tree
    assert report.source_check is not None
    assert report.source_check.status in {"PASS", "SKIPPED"}


def test_blob_digest_tamper_is_caught_even_after_manifest_reseal(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    run_full_pipeline(workspace)
    blob = next((workspace / "blobs").rglob("*.bin"))
    data = bytearray(blob.read_bytes())
    data[-1] ^= 0x01
    blob.write_bytes(bytes(data))
    rewrite_manifest(workspace)  # simulate corruption invisible to the manifest

    session = WorkstationPaths.create(workspace)  # noqa: F841 - reopen sanity
    from apps.workstation.session_store import DerivedStateStoreManager

    restored = DerivedStateStoreManager(WorkstationPaths.create(workspace)).reopen()
    report = RoundTripVerificationEngine(workspace).verify(restored)
    assert not report.checks[CHECK_BLOBS].passed
    assert report.checks[CHECK_BLOBS].status == "FAIL"
    assert report.verdict == "FAIL"


def test_measurement_payload_tamper_breaks_recompute_check(tmp_path: Path) -> None:

    workspace = tmp_path / "ws"
    run_full_pipeline(workspace)

    artifacts_path = workspace / "session" / "artifacts.json"
    document = json.loads(artifacts_path.read_text(encoding="utf-8"))
    docs = document["payload_documents"]
    measurement_ids = [
        artifact["f"]["artifact_id"]
        for artifact in document["artifacts"]
        if artifact["f"]["kind"]["__e__"].endswith(":MEASUREMENT_SET")
    ]
    tampered = dict(docs[measurement_ids[0]])
    tampered["mean_hu"] = float(tampered["mean_hu"]) + 1000.0
    docs[measurement_ids[0]] = tampered
    artifacts_path.write_text(json.dumps(document, sort_keys=True, indent=2), encoding="utf-8")
    rewrite_manifest(workspace)

    from apps.workstation.session_store import DerivedStateStoreManager

    restored = DerivedStateStoreManager(WorkstationPaths.create(workspace)).reopen()
    report = RoundTripVerificationEngine(workspace).verify(restored)
    assert not report.checks[CHECK_MEASURE].passed


def test_provenance_subject_mismatch_fails_dag_check(tmp_path: Path) -> None:

    workspace = tmp_path / "ws"
    run_full_pipeline(workspace)

    prov_path = workspace / "session" / "provenance.json"
    document = json.loads(prov_path.read_text(encoding="utf-8"))
    derived = [
        record
        for record in document["records"]
        if record["f"]["event_type"]["__e__"].endswith(":DERIVED")
    ]
    derived[0]["f"]["subject_ref"] = "drifted-subject"
    prov_path.write_text(json.dumps(document, sort_keys=True, indent=2), encoding="utf-8")
    rewrite_manifest(workspace)

    from apps.workstation.session_store import DerivedStateStoreManager

    restored = DerivedStateStoreManager(WorkstationPaths.create(workspace)).reopen()
    report = RoundTripVerificationEngine(workspace).verify(restored)
    assert not report.checks[CHECK_PROV].passed


def test_journal_truncation_breaks_replay_equality(tmp_path: Path) -> None:

    workspace = tmp_path / "ws"
    run_full_pipeline(workspace)

    events_path = workspace / "session" / "journal_events.json"
    document = json.loads(events_path.read_text(encoding="utf-8"))
    if document["events"]:
        document["events"][0]["to_value"] = "MUTATED"
        events_path.write_text(
            json.dumps(document, sort_keys=True, indent=2), encoding="utf-8"
        )
        rewrite_manifest(workspace)

        from apps.workstation.session_store import DerivedStateStoreManager

        with pytest.raises(CanonicalValidationError):
            DerivedStateStoreManager(WorkstationPaths.create(workspace)).reopen()


def test_verify_engine_requires_derived_content(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    run_full_pipeline(workspace)
    # strip artifacts to prove the engine refuses vacuous sessions
    artifacts_path = workspace / "session" / "artifacts.json"
    volumes_path = workspace / "session" / "volumes.json"
    for path, key in ((artifacts_path, "artifacts"), (volumes_path, "volumes")):
        document = json.loads(path.read_text(encoding="utf-8"))
        document[key] = []
        if key == "artifacts":
            document["payload_documents"] = {}
        path.write_text(json.dumps(document, sort_keys=True, indent=2), encoding="utf-8")
    rewrite_manifest(workspace)

    from apps.workstation.session_store import DerivedStateStoreManager

    restored = DerivedStateStoreManager(WorkstationPaths.create(workspace)).reopen()
    report = RoundTripVerificationEngine(workspace).verify(restored)
    for name in (CHECK_BLOBS, CHECK_COORDS, CHECK_MEASURE):
        assert not report.checks[name].passed
    assert report.verdict == "FAIL"


# ---------------------------------------------------------------------------
# R2-03 (r2): the seventh check validates COMPLETE geometry/render bijection
# ---------------------------------------------------------------------------


def _viewer_session(tmp_path: Path, repo_root: Path):
    from tests.apps.viewer_pipeline import build_viewer_session

    fixture = build_viewer_session(tmp_path / "ws")
    from apps.viewer.engine import ViewerEngine
    from tests.apps.test_viewer_e2e import _deterministic_ulids

    engine = ViewerEngine(
        fixture.session,
        id_factory=_deterministic_ulids(),
        now=lambda: "2026-09-07T00:00:00Z",
        repo_root=repo_root,
    )
    engine.open()
    return _RepoRootedFixture(fixture, repo_root), engine


def _mutated_reopen(tmp_path: Path, repo_root: Path, mutate):
    """Persist a valid viewer session, mutate the state document, reseal."""

    from tests.apps.pipeline import rewrite_manifest

    fixture, engine = _viewer_session(tmp_path, repo_root)
    engine.persist(fixture.store)
    state_path = fixture.workspace / "session" / "artifacts.json"
    document = json.loads(state_path.read_text(encoding="utf-8"))
    for key, payload in document["payload_documents"].items():
        if key.startswith("viewer_state:"):
            mutate(payload)
    state_path.write_text(json.dumps(document, sort_keys=True, indent=2), encoding="utf-8")
    rewrite_manifest(fixture.workspace)
    from apps.workstation.session_store import DerivedStateStoreManager

    return fixture, DerivedStateStoreManager(
        WorkstationPaths.create(fixture.workspace)
    ).reopen()


def test_viewer_bijection_valid_round_trip_remains_pass(
    tmp_path: Path, clean_checkout: Path
) -> None:
    from apps.workstation.round_trip import CHECK_VIEWER

    fixture, engine = _viewer_session(tmp_path, clean_checkout)
    engine.persist(fixture.store)
    from apps.workstation.session_store import DerivedStateStoreManager

    reopened = DerivedStateStoreManager(
        WorkstationPaths.create(fixture.workspace)
    ).reopen()
    report = RoundTripVerificationEngine(
        fixture.workspace, viewer_repo_root=fixture.repo_root
    ).verify(reopened)
    assert report.checks[CHECK_VIEWER].passed
    assert report.verdict == "PASS"


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda payload: payload["renders"].update(
                {"AXIAL": {"wl_preset_id": None, "records": []}}
            ),
            id="delete-whole-plane-records",
        ),
        pytest.param(
            lambda payload: payload["renders"]["CORONAL"].update(
                {"records": payload["renders"]["CORONAL"]["records"][:1]}
            ),
            id="delete-one-record",
        ),
        pytest.param(
            lambda payload: payload["renders"]["AXIAL"].update(
                {
                    "records": [
                        payload["renders"]["AXIAL"]["records"][0],
                        payload["renders"]["AXIAL"]["records"][0],
                    ]
                }
            ),
            id="duplicate-record",
        ),
        pytest.param(
            lambda payload: payload["renders"]["SAGITTAL"].update(
                {"records": list(reversed(payload["renders"]["SAGITTAL"]["records"]))}
            ),
            id="reorder-records",
        ),
        pytest.param(
            lambda payload: payload["renders"]["AXIAL"]["records"].__setitem__(
                0, payload["renders"]["CORONAL"]["records"][0]
            ),
            id="wrong-plane-binding",
        ),
        pytest.param(
            lambda payload: payload["renders"]["AXIAL"]["records"][0].__setitem__(
                "frame_index", 9
            ),
            id="out-of-range-index",
        ),
        pytest.param(
            lambda payload: payload["renders"]["AXIAL"]["records"][0].__setitem__(
                "frame_index", -1
            ),
            id="negative-frame-index",
        ),
        pytest.param(
            lambda payload: payload["renders"]["AXIAL"]["records"][0].__setitem__(
                "frame_index", "0"
            ),
            id="string-frame-index",
        ),
        pytest.param(
            lambda payload: payload["renders"]["AXIAL"]["records"][0].__setitem__(
                "frame_index", True
            ),
            id="bool-frame-index",
        ),
        pytest.param(
            lambda payload: payload["renders"]["AXIAL"]["records"][0].__setitem__(
                "frame_index", None
            ),
            id="null-frame-index",
        ),
        pytest.param(
            lambda payload: payload["renders"]["AXIAL"].__setitem__(
                "records", None
            ),
            id="null-records-container",
        ),
        pytest.param(
            lambda payload: payload["renders"]["AXIAL"].__setitem__(
                "records", "malformed"
            ),
            id="string-records-container",
        ),
        pytest.param(
            lambda payload: payload["renders"]["AXIAL"].__setitem__(
                "records", {"not": "a-list"}
            ),
            id="dict-records-container",
        ),
        pytest.param(
            lambda payload: (
                payload["renders"]["AXIAL"]["records"][0].update(
                    {
                        "wl_preset_id": "CT_BONE",
                        "wl_applied": {"center": 300.0, "width": 1500.0},
                        "png_sha256": "f" * 64,
                    }
                )
            ),
            id="conflicting-same-id-record",
        ),
        pytest.param(
            lambda payload: (
                payload["render_archive"].pop(
                    payload["renders"]["AXIAL"]["records"][0]["provenance_ref"].split(":")[1],
                    None,
                )
            ),
            id="deleted-current-archive-membership",
        ),
    ],
)
def test_viewer_bijection_mutations_fail_the_seventh_check(
    tmp_path: Path, clean_checkout: Path, mutation
) -> None:
    from apps.viewer.engine import ViewerEngine, ViewerOpenError
    from apps.viewer.grammar import RejectionCode
    from apps.workstation.round_trip import CHECK_VIEWER

    _fixture, reopened = _mutated_reopen(tmp_path, clean_checkout, mutation)
    report = RoundTripVerificationEngine(
        _fixture.workspace, viewer_repo_root=_fixture.repo_root
    ).verify(reopened)
    # the SEVENTH check itself must fail (not only transport manifests)
    assert not report.checks[CHECK_VIEWER].passed, report.to_dict()
    # ... and the engine refuses the snapshot with the typed open failure
    engine = ViewerEngine(reopened, repo_root=_fixture.repo_root)
    with pytest.raises(ViewerOpenError) as excinfo:
        engine.open()
    assert excinfo.value.code is RejectionCode.SESSION_OPEN_FAILED


def test_viewer_bijection_genuinely_empty_geometry_plane_passes(
    tmp_path: Path, clean_checkout: Path
) -> None:
    """A legal empty plane (zero accepted MPRs) must not fail the check."""

    from apps.viewer.engine import ViewerEngine
    from apps.workstation.round_trip import CHECK_VIEWER
    from apps.workstation.session_store import (
        DerivedStateStoreManager,
        RestoredSession,
    )
    from tests.apps.test_viewer_e2e import _deterministic_ulids
    from tests.apps.viewer_pipeline import build_viewer_session

    # Build the governed fixture, then strip the CORONAL MPR artifacts
    # BEFORE the viewer opens: CORONAL geometry is then genuinely empty at
    # session open (the legal empty-plane case of Part 5.1).
    fixture = build_viewer_session(tmp_path / "ws")
    session = fixture.session
    keep_artifacts = tuple(
        artifact
        for artifact in session.artifacts
        if not (
            artifact.kind.value == "MPR_VIEW"
            and isinstance(
                session.payload_documents.get(artifact.artifact_id), dict
            )
            and session.payload_documents[artifact.artifact_id].get("plane")
            == "CORONAL"
        )
    )
    payload_documents = {
        artifact_id: payload
        for artifact_id, payload in session.payload_documents.items()
        if not (
            isinstance(payload, dict)
            and payload.get("plane") == "CORONAL"
            and "pixel_blob_ref" in payload
        )
    }
    stripped = RestoredSession(
        paths=session.paths,
        study=session.study,
        series=session.series,
        volumes=session.volumes,
        artifacts=keep_artifacts,
        payload_documents=payload_documents,
        provenance=session.provenance,
        log=session.log,
        skipped_instances=session.skipped_instances,
        projections=session.projections,
        session_digest=session.session_digest,
        pixel_blobs=dict(session.pixel_blobs),
    )
    engine = ViewerEngine(
        stripped,
        id_factory=_deterministic_ulids(),
        now=lambda: "2026-09-07T00:00:00Z",
        repo_root=clean_checkout,
    )
    engine.open()
    assert engine.frames_document("CORONAL")["frames"] == []
    assert engine.frames_document("AXIAL")["frames"]
    engine.persist(fixture.store)
    reopened = DerivedStateStoreManager(
        WorkstationPaths.create(fixture.workspace)
    ).reopen()
    report = RoundTripVerificationEngine(
        fixture.workspace, viewer_repo_root=clean_checkout
    ).verify(reopened)
    assert report.checks[CHECK_VIEWER].passed, report.to_dict()
    reopened_engine = ViewerEngine(reopened, repo_root=clean_checkout)
    reopened_engine.open()
    assert reopened_engine.frames_document("CORONAL")["frames"] == []
