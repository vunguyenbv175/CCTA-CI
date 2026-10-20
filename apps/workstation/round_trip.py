"""RoundTripVerificationEngine — the Phase-0 exit gate made executable (§B4).

Given a REOPENED session, mechanically verifies the Master Plan §61 sentence:

    import -> save derived state -> reopen -> reproduce identical
    geometry/measurement identity and provenance

as five independent, machine-readable checks:

(a) ``artifact_blob_digest_equality``      — every blob byte-digests to its
    recorded content address; every artifact payload document digests to its
    sealed ``payload.blob_sha256``;
(b) ``coordinate_round_trip_identity``     — persisted voxel->world transforms
    round-trip through their inverse via :func:`core.coordinates.assert_round_trip`
    and equal the canonical FrameSpec affine exactly;
(c) ``measurement_recompute_equality``     — deterministic recomputation over
    restored voxels reproduces the stored measurement document bit-for-bit;
(d) ``provenance_completeness_dag_integrity`` — §44 chain verifies, every
    provenance_ref resolves back to its subject, upstream DAG is acyclic;
(e) ``eventlog_replay_equality``           — journal replay reproduces the
    projections captured at save time AND every artifact's live triple.

Output is a JSON-ready :class:`VerificationReport`; verdict PASS only when all
five mandated checks pass. Source-immutability is reported as a sixth,
independent informational check (it concerns the ORIGINAL tree, Rule A, not
the derived-state reproduction this gate owns).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from apps.workstation.session_store import RestoredSession, decode_value
from apps.workstation.volume_loader import (
    CANONICAL_VOLUME_CODECS,
    CanonicalVolumeLoader,
    compute_measurements,
    measurement_parameters_digest,
)
from core.coordinates import CoordinateTransform, assert_round_trip
from core.digest import canonical_digest
from core.enums import ArtifactKind
from core.provenance import NotApplicable, ProvenanceRecord, verify_chain

REPORT_SCHEMA: Final[str] = "ccta.round_trip.report.v1"

CHECK_BLOBS: Final[str] = "artifact_blob_digest_equality"
CHECK_COORDS: Final[str] = "coordinate_round_trip_identity"
CHECK_MEASURE: Final[str] = "measurement_recompute_equality"
CHECK_PROV: Final[str] = "provenance_completeness_dag_integrity"
CHECK_REPLAY: Final[str] = "eventlog_replay_equality"
CHECK_MPR: Final[str] = "mpr_persistence_identity"
CHECK_VIEWER: Final[str] = "viewer_frame_identity"
CHECK_SOURCE: Final[str] = "source_immutability_informational"

MANDATED_CHECKS: Final[tuple[str, ...]] = (
    CHECK_BLOBS,
    CHECK_COORDS,
    CHECK_MEASURE,
    CHECK_PROV,
    CHECK_REPLAY,
    CHECK_MPR,
    CHECK_VIEWER,
)

_STATUS_PASS: Final[str] = "PASS"
_STATUS_FAIL: Final[str] = "FAIL"
_STATUS_SKIPPED: Final[str] = "SKIPPED"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one exit-gate check."""

    name: str
    status: str
    details: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == _STATUS_PASS

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "passed": self.passed,
            "details": list(self.details),
        }


@dataclass(frozen=True)
class VerificationReport:
    """Machine-readable exit-gate report consumed by acceptance evidence."""

    session_digest: str
    checks: dict[str, CheckResult]
    source_check: CheckResult | None

    @property
    def verdict(self) -> str:
        return _STATUS_PASS if all(
            self.checks[name].passed for name in MANDATED_CHECKS
        ) else _STATUS_FAIL

    def to_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
            "schema": REPORT_SCHEMA,
            "session_digest": self.session_digest,
            "verdict": self.verdict,
            "checks": {name: check.to_dict() for name, check in self.checks.items()},
            "mandated_checks": list(MANDATED_CHECKS),
        }
        if self.source_check is not None:
            document["source_immutability"] = self.source_check.to_dict()
        return document


def _result(name: str, problems: list[str]) -> CheckResult:
    return CheckResult(
        name=name,
        status=_STATUS_PASS if not problems else _STATUS_FAIL,
        details=tuple(problems),
    )


def _upstream_of(record: ProvenanceRecord) -> tuple[str, ...]:
    if isinstance(record.upstream_refs, NotApplicable):
        return ()
    return tuple(record.upstream_refs)


def _compute_canonical_measurements(
    voxels: object, shape: tuple[int, int, int]
) -> dict[str, object]:
    """Recompute the frozen measurement document over canonical scalar data.

    The legacy measurement helper intentionally accepts only ``uint16-le``
    bytes.  Canonical materialization may instead persist signed ``int16`` or
    ``float32`` samples; those codecs must be verified with the same central
    ROI rule without decoding them as unsigned legacy bytes.
    """
    array = cast(Any, voxels)
    actual_shape = getattr(array, "shape", None)
    if actual_shape is None or tuple(int(value) for value in actual_shape) != shape:
        raise ValueError("canonical voxel shape does not match the Volume FrameSpec")
    lo = tuple(dim // 3 for dim in shape)
    hi = tuple(dim - low for dim, low in zip(shape, lo))
    total: int | float = 0
    count = 0
    minimum: int | float | None = None
    maximum: int | float | None = None
    for k in range(lo[0], hi[0]):
        for j in range(lo[1], hi[1]):
            for i in range(lo[2], hi[2]):
                value = array[k, j, i]
                item = getattr(value, "item", None)
                if callable(item):
                    value = item()
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("canonical voxel contains a non-numeric sample")
                total += value
                count += 1
                if minimum is None or value < minimum:
                    minimum = value
                if maximum is None or value > maximum:
                    maximum = value
    if minimum is None or maximum is None or count == 0:
        raise ValueError("canonical measurement ROI is empty")
    return {
        "schema": "ccta.payload.measurement_set.v1",
        "algorithm": "central-box-hu-stats",
        "params_version": "v1",
        "params_sha256": measurement_parameters_digest(),
        "shape": list(shape),
        "roi_lo": list(lo),
        "roi_hi": list(hi),
        "mean_hu": total / count,
        "min_hu": minimum,
        "max_hu": maximum,
        "voxel_count": count,
    }


class RoundTripVerificationEngine:
    """Runs the five §B4 checks against one reopened session."""

    def __init__(
        self, workspace_root: Path, *, viewer_repo_root: Path | str | None = None
    ) -> None:
        self._loader = CanonicalVolumeLoader(Path(workspace_root) / "blobs")
        # The viewer engine resolves its governed build identity from the
        # code checkout; tests run from a dirty dev tree pass an explicit
        # clean bound-checkout root here (F11 practical-testing pattern).
        self._viewer_repo_root = viewer_repo_root

    # -- public API ------------------------------------------------------------

    def verify(self, session: RestoredSession) -> VerificationReport:
        checks: dict[str, CheckResult] = {
            CHECK_BLOBS: self._check_blobs(session),
            CHECK_COORDS: self._check_coordinates(session),
            CHECK_MEASURE: self._check_measurements(session),
            CHECK_PROV: self._check_provenance(session),
            CHECK_REPLAY: self._check_replay(session),
            CHECK_MPR: self._check_mpr(session),
            CHECK_VIEWER: self._check_viewer(session),
        }
        return VerificationReport(
            session_digest=session.session_digest,
            checks=checks,
            source_check=self._check_source(session),
        )

    # -- (a) digests ---------------------------------------------------------------

    def _check_blobs(self, session: RestoredSession) -> CheckResult:
        problems: list[str] = []
        if not session.volumes or len(session.artifacts) < 2:
            problems.append("session carries no derived content to verify")
        for volume in session.volumes:
            try:
                data = self._loader.read_blob(volume.blob_ref.sha256)
            except Exception as exc:  # noqa: BLE001 - any read failure is a check failure
                problems.append(f"volume {volume.volume_id}: blob unreadable ({exc})")
                continue
            if len(data) != volume.blob_ref.byte_size:
                problems.append(
                    f"volume {volume.volume_id}: blob size {len(data)} != "
                    f"{volume.blob_ref.byte_size}"
                )
        for artifact in session.artifacts:
            document = session.payload_documents.get(artifact.artifact_id)
            if document is None:
                problems.append(f"artifact {artifact.artifact_id}: payload document missing")
                continue
            if canonical_digest(document) != artifact.payload.blob_sha256:
                problems.append(
                    f"artifact {artifact.artifact_id}: payload digest mismatch"
                )
        return _result(CHECK_BLOBS, problems)

    # -- (b) coordinates ---------------------------------------------------------------

    def _check_coordinates(self, session: RestoredSession) -> CheckResult:
        problems: list[str] = []
        checked = 0
        volumes_by_id = {v.volume_id: v for v in session.volumes}
        for artifact in session.artifacts:
            if artifact.kind != ArtifactKind.COORDINATE_TRANSFORM:
                continue
            encoded = session.payload_documents.get(artifact.artifact_id)
            if encoded is None:
                problems.append(f"{artifact.artifact_id}: transform payload missing")
                continue
            try:
                decoded = decode_value(encoded)
            except Exception as exc:  # noqa: BLE001 - decode failure fails the check
                problems.append(f"{artifact.artifact_id}: undecodable transform ({exc})")
                continue
            if not isinstance(decoded, CoordinateTransform):
                problems.append(f"{artifact.artifact_id}: payload is not a CoordinateTransform")
                continue
            ref = next(iter(artifact.lineage.input_volume_refs), None)
            if ref is None:
                problems.append(f"{artifact.artifact_id}: no input volume bound")
                continue
            volume = volumes_by_id.get(ref)
            if volume is None:
                problems.append(f"{artifact.artifact_id}: input volume {ref!r} unresolved")
                continue
            expected = volume.coordinate_frame.voxel_to_world_matrix()
            if decoded.matrix != expected:
                problems.append(f"{artifact.artifact_id}: matrix diverges from FrameSpec affine")
            try:
                assert_round_trip(decoded, decoded.inverse)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{artifact.artifact_id}: round-trip failed ({exc})")
            checked += 1
        if checked == 0:
            problems.append("no COORDINATE_TRANSFORM artifact found to exercise")
        return _result(CHECK_COORDS, problems)

    # -- (c) measurements ------------------------------------------------------------------

    def _check_measurements(self, session: RestoredSession) -> CheckResult:
        problems: list[str] = []
        checked = 0
        volumes_by_id = {v.volume_id: v for v in session.volumes}
        for artifact in session.artifacts:
            if artifact.kind != ArtifactKind.MEASUREMENT_SET:
                continue
            stored = session.payload_documents.get(artifact.artifact_id)
            if stored is None:
                problems.append(f"{artifact.artifact_id}: measurement payload missing")
                continue
            ref = next(iter(artifact.lineage.input_volume_refs), None)
            if ref is None:
                problems.append(f"{artifact.artifact_id}: no input volume bound")
                continue
            volume = volumes_by_id.get(ref)
            if volume is None:
                problems.append(f"{artifact.artifact_id}: input volume {ref!r} unresolved")
                continue
            try:
                envelope = self._loader.read_envelope(volume.blob_ref.sha256)
                if envelope.get("codec") in CANONICAL_VOLUME_CODECS:
                    voxels = self._loader.read_array(
                        volume.blob_ref.sha256, envelope=envelope
                    )
                    recomputed = _compute_canonical_measurements(
                        voxels, volume.coordinate_frame.dimensions
                    )
                else:
                    voxels = self._loader.read_blob(volume.blob_ref.sha256)
                    recomputed = compute_measurements(
                        voxels, volume.coordinate_frame.dimensions
                    )
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{artifact.artifact_id}: recomputation failed ({exc})")
                continue
            if recomputed != stored:
                problems.append(
                    f"{artifact.artifact_id}: recomputed measurement differs from stored "
                    "(H-003 measurement drift)"
                )
            checked += 1
        if checked == 0:
            problems.append("no MEASUREMENT_SET artifact found to exercise")
        return _result(CHECK_MEASURE, problems)

    # -- (d) provenance ------------------------------------------------------------------------

    def _check_provenance(self, session: RestoredSession) -> CheckResult:
        problems: list[str] = []
        records = session.provenance
        by_id = {record.provenance_id: record for record in records}
        _, chain_problems = verify_chain(records)
        problems.extend(chain_problems)

        for artifact in session.artifacts:
            record = by_id.get(artifact.provenance_ref)
            if record is None:
                problems.append(f"artifact {artifact.artifact_id}: provenance_ref unresolvable")
                continue
            if record.subject_ref != artifact.artifact_id:
                problems.append(
                    f"artifact {artifact.artifact_id}: provenance subject mismatch "
                    f"({record.subject_ref})"
                )
        for volume in session.volumes:
            record = by_id.get(volume.provenance_ref)
            if record is None:
                problems.append(f"volume {volume.volume_id}: provenance_ref unresolvable")
            elif record.subject_ref != volume.volume_id:
                problems.append(
                    f"volume {volume.volume_id}: provenance subject mismatch "
                    f"({record.subject_ref})"
                )
        import_record = by_id.get(session.study.import_info.import_job_ref)
        if import_record is None:
            problems.append("study.import_job_ref unresolvable")

        # DAG integrity: upstream refs resolve + graph is acyclic (Kahn).
        indegree: dict[str, int] = defaultdict(int)
        children: dict[str, list[str]] = defaultdict(list)
        for record in records:
            rid = record.provenance_id
            indegree.setdefault(rid, 0)
            for up in _upstream_of(record):
                if up not in by_id:
                    problems.append(f"{rid}: unresolvable upstream {up!r}")
                    continue
                children[up].append(rid)
                indegree[rid] += 1
        ready = [rid for rid, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            current = ready.pop()
            visited += 1
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(indegree):
            cyclic = sorted(rid for rid, degree in indegree.items() if degree > 0)
            problems.append(f"provenance DAG contains a cycle involving {cyclic}")
        return _result(CHECK_PROV, problems)

    # -- (e) event-log replay ---------------------------------------------------------------

    def _check_replay(self, session: RestoredSession) -> CheckResult:
        problems: list[str] = []
        for entity_id, saved in session.projections.items():
            try:
                live = session.log.project_state(entity_id)
            except KeyError:
                problems.append(f"projection {entity_id}: unknown to the replayed journal")
                continue
            if live.to_dict() != saved:
                problems.append(f"projection {entity_id}: replay != saved snapshot")
        for artifact in session.artifacts:
            try:
                live = session.log.project_state(
                    artifact.artifact_id, entity="derived_artifact"
                )
            except KeyError:
                problems.append(f"artifact {artifact.artifact_id}: absent from journal")
                continue
            expected = {
                "review_state": artifact.review.review_state.value,
                "status": artifact.status.value,
                "release_lane": artifact.release_lane.value,
            }
            actual = {key: live.get(key) for key in expected}
            if actual != expected:
                problems.append(
                    f"artifact {artifact.artifact_id}: journal triple {actual} != {expected}"
                )
        return _result(CHECK_REPLAY, problems)

    # -- (f) MPR persistence ------------------------------------------------

    def _check_mpr(self, session: RestoredSession) -> CheckResult:
        """Verify MPR payload and pixel bytes survived reopen unchanged."""
        problems: list[str] = []
        mpr_artifacts = tuple(
            artifact for artifact in session.artifacts if artifact.kind == ArtifactKind.MPR_VIEW
        )
        if not mpr_artifacts:
            return CheckResult(
                name=CHECK_MPR,
                status=_STATUS_PASS,
                details=(
                    "mpr_payload_digest_match=1.0",
                    "mpr_pixel_blob_digest_match=1.0",
                    "no MPR_VIEW artifacts present",
                ),
            )
        from visualization.mpr import validate_mpr_payload

        payload_matches = 0
        pixel_matches = 0
        for artifact in mpr_artifacts:
            document = session.payload_documents.get(artifact.artifact_id)
            if document is None:
                problems.append(f"{artifact.artifact_id}: MPR payload missing")
                continue
            try:
                validated = validate_mpr_payload(document)
            except Exception as exc:  # noqa: BLE001 - verification must fail closed
                problems.append(f"{artifact.artifact_id}: MPR payload invalid ({exc})")
                continue
            payload_match = canonical_digest(validated) == artifact.payload.blob_sha256
            if not payload_match:
                problems.append(f"{artifact.artifact_id}: mpr_payload_digest_match=0.0")
            else:
                payload_matches += 1
            pixel_ref = validated.get("pixel_blob_ref")
            if not isinstance(pixel_ref, dict):
                problems.append(f"{artifact.artifact_id}: mpr_pixel_blob_digest_match=0.0")
                continue
            digest = pixel_ref.get("sha256")
            expected_size = pixel_ref.get("byte_size")
            pixel_match = False
            try:
                if isinstance(digest, str):
                    data = self._loader.read_blob(digest)
                    from visualization.mpr import decode_pixel_blob

                    decode_pixel_blob(validated, data)
                    pixel_match = isinstance(expected_size, int) and len(data) == expected_size
                    if pixel_match:
                        pixel_match = (
                            digest in session.pixel_blobs
                            and session.pixel_blobs[digest] == data
                        )
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{artifact.artifact_id}: MPR pixel blob unreadable ({exc})")
            if not pixel_match:
                problems.append(f"{artifact.artifact_id}: mpr_pixel_blob_digest_match=0.0")
            else:
                pixel_matches += 1
        expected = len(mpr_artifacts)
        details = [
            f"mpr_payload_digest_match={1.0 if payload_matches == expected else 0.0}",
            f"mpr_pixel_blob_digest_match={1.0 if pixel_matches == expected else 0.0}",
            *problems,
        ]
        return CheckResult(
            name=CHECK_MPR,
            status=_STATUS_PASS if not problems else _STATUS_FAIL,
            details=tuple(details),
        )

    # -- (g) VIEWER_001 seventh check: viewer frame identity -----------------

    def _check_viewer(self, session: RestoredSession) -> CheckResult:
        """Verify every viewer frame record is bit-identical after reopen.

        The seventh mandated check (VIEWER_001_SPEC Part 2): for every
        persisted ``ccta.viewer.session.v1`` document, rebuild the viewer
        session over the reopened WS-02 session and re-derive every current
        LAYER-R record — all frame identity tuples + ``png_sha256`` +
        ``png_byte_size`` must be bit-identical (frame ULIDs excluded per
        Part 5.1), and the LAYER-G ``frame_lists_ref`` must never have moved.
        Sessions without viewer documents pass trivially (nothing to check;
        the check never fabricates state).
        """

        session_documents = [
            (key, document)
            for key, document in session.payload_documents.items()
            if isinstance(document, dict)
            and document.get("schema_version") == "ccta.viewer.session.v1"
        ]
        if not session_documents:
            return CheckResult(
                name=CHECK_VIEWER,
                status=_STATUS_PASS,
                details=("no viewer session documents present",),
            )
        problems: list[str] = []
        from apps.viewer.build_identity import repo_root_from_file
        from apps.viewer.engine import ViewerEngine

        resolved_root = repo_root_from_file()
        if (
            self._viewer_repo_root is not None
            and Path(self._viewer_repo_root).resolve() != resolved_root
        ):
            problems.append(
                f"supplied viewer_repo_root {self._viewer_repo_root} does not match "
                f"executing code root {resolved_root}"
            )
            return _result(CHECK_VIEWER, problems)

        for key, document in session_documents:
            engine = ViewerEngine(session, repo_root=self._viewer_repo_root)
            try:
                engine.open()
            except Exception as exc:  # noqa: BLE001 - reopen failure fails the check
                problems.append(f"{key}: viewer session failed to reopen ({exc})")
                continue
            for problem in engine.verify_reopen_identity():
                problems.append(f"{key}: {problem}")
        return _result(CHECK_VIEWER, problems)

    # -- informational: original-source immutability (Rule A) ---------------------------------

    def _check_source(self, session: RestoredSession) -> CheckResult:
        source_root = Path(session.study.import_info.source_root_path)
        if not source_root.is_dir():
            return CheckResult(
                name=CHECK_SOURCE,
                status=_STATUS_SKIPPED,
                details=("original DICOM tree unavailable at this location",),
            )
        problems: list[str] = []
        for series in session.series:
            for entry in series.manifest.instances:
                full = source_root / entry.file_path
                if not full.exists():
                    problems.append(f"{series.series_id}/{entry.file_path}: source vanished")
                    continue
                digest = hashlib.sha256(full.read_bytes()).hexdigest()
                if digest != entry.sha256:
                    problems.append(
                        f"{series.series_id}/{entry.file_path}: original bytes modified"
                    )
        return _result(CHECK_SOURCE, problems)


def summarize_volumes(session: RestoredSession) -> list[dict[str, object]]:
    """Stable inventory helper reused by the CLI inspect command."""
    out: list[dict[str, object]] = []
    for volume in session.volumes:
        out.append(
            {
                "blob_sha256": volume.blob_ref.sha256,
                "byte_size": volume.blob_ref.byte_size,
                "dimensions": list(volume.coordinate_frame.dimensions),
                "kind": volume.kind.value,
                "volume_id": volume.volume_id,
            }
        )
    return out


__all__ = [    "CHECK_BLOBS",
    "CHECK_COORDS",
    "CHECK_MEASURE",
    "CHECK_PROV",
    "CHECK_REPLAY",
    "CHECK_MPR",
    "CHECK_VIEWER",
    "CHECK_SOURCE",
    "MANDATED_CHECKS",
    "REPORT_SCHEMA",
    "RoundTripVerificationEngine",
    "CheckResult",
    "VerificationReport",
    "summarize_volumes",
]
