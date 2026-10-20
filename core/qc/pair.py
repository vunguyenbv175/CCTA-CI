"""Registered-pair validation, construction, and H-010 geometry tracking."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from core.artifacts import (
    ClaimBinding,
    DerivedArtifact,
    DerivedArtifactStore,
    GeometryBase,
    Lineage,
    PayloadRef,
    Producer,
    create_derived_artifact,
)
from core.coordinates import FrameSpec, invert_matrix
from core.digest import canonical_digest, canonical_json, is_sha256_hex
from core.enums import ArtifactKind, ArtifactStatus, ProducerType, ReleaseLane, VolumeKind
from core.events import utc_now_iso
from core.ids import new_ulid
from core.provenance import (
    Actor,
    BlobRef,
    Environment,
    InputRef,
    NotApplicable,
    ProvenanceRecord,
    create_provenance,
    verify_chain,
)
from core.volume import Volume

from .constants import QC_PREPROCESSING, REGISTRATION_PAYLOAD_SCHEMA
from .types import QCValidationError, RegisteredPairInput

_ULID = re.compile(r"^[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}$")
_PHASE = re.compile(r"^phase:[A-Z0-9_]{1,32}$")
_REGISTRATION_KEYS = frozenset(
    {
        "schema_version",
        "registration_ref",
        "registration_geometry_id",
        "registration_version",
        "first_volume_ref",
        "second_volume_ref",
        "first_volume_blob_sha256",
        "second_volume_blob_sha256",
        "first_volume_provenance_ref",
        "second_volume_provenance_ref",
        "first_series_ref",
        "second_series_ref",
        "first_phase_ref",
        "second_phase_ref",
        "first_geometry_id",
        "second_geometry_id",
        "first_geometry_version",
        "second_geometry_version",
        "first_frame_spec_sha256",
        "second_frame_spec_sha256",
        "study_ref",
        "frame_of_reference_uid_digest",
        "transform_direction",
        "affine_4x4_row_major",
        "converged",
        "overlap_lower_lps_mm",
        "overlap_upper_lps_mm",
        "algorithm_id",
        "algorithm_version",
        "parameters_sha256",
    }
)
_ZERO_COMMIT = "0" * 40
_DEFAULT_SOFTWARE = "ccta-qc-registration-v1"
_DEFAULT_CLAIM = ClaimBinding(
    claim_id="CLM-QC-001",
    claim_version="v0-draft",
    population_ref="QC_001-D:qc-synthetic-v1",
    metric_refs=("QC_001-D",),
    snapshot_ref=None,
)


@dataclass(frozen=True, slots=True)
class RegistrationBuildResult:
    """Construction result retained alongside the closed pair DTO."""

    pair: RegisteredPairInput
    artifact: DerivedArtifact
    provenance: ProvenanceRecord
    document: Mapping[str, object]

    @property
    def registration_artifact(self) -> DerivedArtifact:
        return self.artifact

    @property
    def registration_document(self) -> Mapping[str, object]:
        return self.document


def _fail(detail: str) -> QCValidationError:
    return QCValidationError(detail)


def _as_status(value: object) -> str:
    return value.value if isinstance(value, ArtifactStatus) else str(value)


def _valid_ulid(value: object) -> bool:
    return isinstance(value, str) and _ULID.fullmatch(value) is not None


def _canonical_volume_ref(volume: Volume) -> str:
    if not _valid_ulid(volume.volume_id):
        raise _fail("Volume id must be a valid ULID for a registered pair")
    return f"volume_id:{volume.volume_id}"


def _canonical_study_ref(value: str) -> str:
    if value.startswith("study_id:"):
        suffix = value.split(":", 1)[1]
    else:
        suffix = value
    if not _valid_ulid(suffix):
        raise _fail("study_ref must be study_id:<ULID>")
    return f"study_id:{suffix}"


def _require_sha(value: object, label: str) -> str:
    if not is_sha256_hex(value):
        raise _fail(f"{label} must be lowercase 64-hex SHA-256")
    return cast(str, value)


def _frame_digest(frame: FrameSpec) -> str:
    return canonical_digest(frame)


def _array_digest(array: Any) -> str:
    if not isinstance(array, np.ndarray):
        raise _fail("registered pair arrays must be NumPy arrays")
    if array.ndim != 3 or not array.flags.c_contiguous or bool(array.flags.writeable):
        raise _fail("registered pair arrays must be read-only C-order 3-D arrays")
    if array.dtype.kind not in {"i", "u", "f"} or not bool(np.all(np.isfinite(array))):
        raise _fail("registered pair arrays must contain finite numeric values")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _geometry_id(volume: Volume) -> str:
    return f"volume_geometry:{volume.volume_id}"


def registration_geometry_id(
    *,
    algorithm_id: str,
    first_geometry_id: str,
    first_phase_ref: str,
    second_geometry_id: str,
    second_phase_ref: str,
) -> str:
    """Stable logical registration geometry identity from the exact preimage."""

    preimage = {
        "algorithm_id": algorithm_id,
        "first_geometry_id": first_geometry_id,
        "first_phase_ref": first_phase_ref,
        "second_geometry_id": second_geometry_id,
        "second_phase_ref": second_phase_ref,
    }
    digest = hashlib.sha256(canonical_json(preimage).encode("utf-8")).hexdigest()
    return f"qc_registration_geometry:{digest}"


def pair_geometry_id(pair: RegisteredPairInput) -> str:
    document = pair.registration_document
    preimage = {
        "first_geometry_id": pair.first_geometry_base.geometry_id,
        "registration_geometry_id": document["registration_geometry_id"],
        "second_geometry_id": pair.second_geometry_base.geometry_id,
    }
    return (
        f"qc_pair_geometry:{hashlib.sha256(canonical_json(preimage).encode('utf-8')).hexdigest()}"
    )


def pair_geometry_version(pair: RegisteredPairInput) -> int:
    registration_version = pair.registration_document["registration_version"]
    if isinstance(registration_version, bool) or not isinstance(registration_version, int):
        raise QCValidationError("registration_version must be an integer")
    return (
        pair.first_geometry_base.geometry_version
        + pair.second_geometry_base.geometry_version
        + registration_version
        - 2
    )


def _validate_float_array(value: object, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise _fail(f"{label} must contain exactly {length} values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise _fail(f"{label} must contain numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise _fail(f"{label} must contain finite values")
    return result


def _validate_registration_document(pair: RegisteredPairInput) -> None:
    document = pair.registration_document
    if not isinstance(document, Mapping):
        raise _fail("registration_document must be a mapping")
    if set(document) != set(_REGISTRATION_KEYS):
        raise _fail("registration_document has missing, duplicate, or extra keys")
    if document["schema_version"] != REGISTRATION_PAYLOAD_SCHEMA:
        raise _fail("registration payload schema is closed")
    registration_ref = document["registration_ref"]
    if (
        not isinstance(registration_ref, str)
        or re.fullmatch(
            r"registration:[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}",
            registration_ref,
        )
        is None
    ):
        raise _fail("registration_ref must be registration:<ULID>")
    artifact = pair.registration_artifact
    if not isinstance(artifact, DerivedArtifact):
        raise _fail("registration_artifact must be a DerivedArtifact")
    if registration_ref != f"registration:{artifact.artifact_id}":
        raise _fail("registration_ref does not bind the outer artifact id")
    if not _valid_ulid(artifact.artifact_id):
        raise _fail("registration artifact id must be a valid ULID")
    version = document["registration_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise _fail("registration_version must be an integer >= 1")
    first = pair.first_volume
    second = pair.second_volume
    if not isinstance(first, Volume) or not isinstance(second, Volume):
        raise _fail("registered pair requires two Volume objects")
    if (
        not isinstance(first.status, ArtifactStatus)
        or not isinstance(second.status, ArtifactStatus)
        or first.status != ArtifactStatus.ACTIVE
        or second.status != ArtifactStatus.ACTIVE
    ):
        raise _fail("registered pair volumes must be ACTIVE")
    if first.kind != VolumeKind.CANONICAL or second.kind != VolumeKind.CANONICAL:
        raise _fail("registered pair requires canonical source volumes")
    if first.volume_id == second.volume_id:
        raise _fail("registered pair volumes must be distinct")
    if first.study_ref != second.study_ref:
        raise _fail("registered pair volumes must share study identity")
    canonical_study = _canonical_study_ref(first.study_ref)
    if first.study_ref != canonical_study or second.study_ref != canonical_study:
        raise _fail("registered pair volumes must use canonical study_id refs")
    if len(first.source_series_refs) != 1 or len(second.source_series_refs) != 1:
        raise _fail("registered pair volumes must each have one source series")
    if first.source_series_refs[0] == second.source_series_refs[0]:
        raise _fail("registered pair source series must be distinct")
    if _canonical_study_ref(artifact.study_ref) != _canonical_study_ref(first.study_ref):
        raise _fail("registration artifact study identity does not bind the pair")
    if artifact.study_ref != canonical_study:
        raise _fail("registration artifact study_ref is not the canonical pair ref")
    if artifact.lineage.input_volume_refs != (first.volume_id, second.volume_id):
        raise _fail("registration artifact volume lineage is not ordered and exact")
    if artifact.lineage.input_series_refs != (
        first.source_series_refs[0],
        second.source_series_refs[0],
    ):
        raise _fail("registration artifact series lineage is not ordered and exact")
    if artifact.lineage.input_artifact_refs:
        raise _fail("registration artifact may not claim extra input artifacts")
    if artifact.lineage.registration_ref != registration_ref:
        raise _fail("registration artifact registration_ref lineage is not exact")
    first_ref = _canonical_volume_ref(first)
    second_ref = _canonical_volume_ref(second)
    if document["first_volume_ref"] != first_ref or document["second_volume_ref"] != second_ref:
        raise _fail("registration volume refs do not bind the supplied volumes")
    if first_ref == second_ref:
        raise _fail("registration volume refs must be distinct")
    if (
        document["first_volume_blob_sha256"] != first.blob_ref.sha256
        or document["second_volume_blob_sha256"] != second.blob_ref.sha256
    ):
        raise _fail("registration volume blob refs do not bind the supplied volumes")
    _require_sha(document["first_volume_blob_sha256"], "first_volume_blob_sha256")
    _require_sha(document["second_volume_blob_sha256"], "second_volume_blob_sha256")
    if first.blob_ref.byte_size <= 0 or second.blob_ref.byte_size <= 0:
        raise _fail("registered pair volume blob sizes must be positive")
    if (
        document["first_volume_provenance_ref"] != first.provenance_ref
        or document["second_volume_provenance_ref"] != second.provenance_ref
    ):
        raise _fail("registration volume provenance refs do not bind the volumes")
    if not isinstance(first.provenance_ref, str) or not first.provenance_ref:
        raise _fail("first volume provenance ref is required")
    if not isinstance(second.provenance_ref, str) or not second.provenance_ref:
        raise _fail("second volume provenance ref is required")
    if (
        document["first_series_ref"] != first.source_series_refs[0]
        or document["second_series_ref"] != second.source_series_refs[0]
    ):
        raise _fail("registration series refs do not bind the volumes")
    for series_ref in (document["first_series_ref"], document["second_series_ref"]):
        if (
            not isinstance(series_ref, str)
            or not series_ref.startswith("series_id:")
            or not _valid_ulid(series_ref.split(":", 1)[1])
        ):
            raise _fail("registration series refs must be series_id:<ULID>")
    if (
        document["first_phase_ref"] != pair.first_phase_ref
        or document["second_phase_ref"] != pair.second_phase_ref
    ):
        raise _fail("registered pair phase fields disagree")
    if not isinstance(pair.first_phase_ref, str) or _PHASE.fullmatch(pair.first_phase_ref) is None:
        raise _fail("first_phase_ref has invalid closed syntax")
    if (
        not isinstance(pair.second_phase_ref, str)
        or _PHASE.fullmatch(pair.second_phase_ref) is None
    ):
        raise _fail("second_phase_ref has invalid closed syntax")
    if pair.first_phase_ref == pair.second_phase_ref:
        raise _fail("registered pair phases must be distinct")
    first_geometry = document["first_geometry_id"]
    second_geometry = document["second_geometry_id"]
    if first_geometry != _geometry_id(first) or second_geometry != _geometry_id(second):
        raise _fail("registration geometry ids do not bind volume ids")
    if (
        first_geometry == second_geometry
        or not isinstance(first_geometry, str)
        or not isinstance(second_geometry, str)
    ):
        raise _fail("registration geometry ids must be distinct")
    if (
        not first_geometry.startswith("volume_geometry:")
        or not second_geometry.startswith("volume_geometry:")
        or not _valid_ulid(first_geometry.split(":", 1)[1])
        or not _valid_ulid(second_geometry.split(":", 1)[1])
    ):
        raise _fail("registration geometry ids must be volume_geometry:<ULID>")
    if not isinstance(pair.first_geometry_base, GeometryBase) or not isinstance(
        pair.second_geometry_base, GeometryBase
    ):
        raise _fail("registered pair requires two GeometryBase objects")
    if not isinstance(first.coordinate_frame, FrameSpec) or not isinstance(
        second.coordinate_frame, FrameSpec
    ):
        raise _fail("registered pair volumes require FrameSpec objects")
    first_version = pair.first_geometry_base.geometry_version
    second_version = pair.second_geometry_base.geometry_version
    if (
        pair.first_geometry_base.geometry_id != first_geometry
        or pair.second_geometry_base.geometry_id != second_geometry
    ):
        raise _fail("supplied GeometryBase ids disagree with registration document")
    if (
        document["first_geometry_version"] != first_version
        or document["second_geometry_version"] != second_version
    ):
        raise _fail("registration geometry versions do not bind supplied GeometryBase")
    if (
        isinstance(first_version, bool)
        or not isinstance(first_version, int)
        or first_version < 1
        or isinstance(second_version, bool)
        or not isinstance(second_version, int)
        or second_version < 1
    ):
        raise _fail("geometry versions must be integers >= 1")
    if document["first_frame_spec_sha256"] != _frame_digest(first.coordinate_frame) or document[
        "second_frame_spec_sha256"
    ] != _frame_digest(second.coordinate_frame):
        raise _fail("FrameSpec digest mismatch")
    _require_sha(document["first_frame_spec_sha256"], "first_frame_spec_sha256")
    _require_sha(document["second_frame_spec_sha256"], "second_frame_spec_sha256")
    study_ref = _canonical_study_ref(first.study_ref)
    if document["study_ref"] != study_ref or _canonical_study_ref(second.study_ref) != study_ref:
        raise _fail("registration study_ref is not shared and canonical")
    if (
        first.coordinate_frame.frame_of_reference_uid
        != second.coordinate_frame.frame_of_reference_uid
    ):
        raise _fail("FrameOfReferenceUID values must be byte-equal")
    uid_digest = hashlib.sha256(
        first.coordinate_frame.frame_of_reference_uid.encode("utf-8")
    ).hexdigest()
    if document["frame_of_reference_uid_digest"] != uid_digest:
        raise _fail("FrameOfReferenceUID digest mismatch")
    _require_sha(document["frame_of_reference_uid_digest"], "frame_of_reference_uid_digest")
    if document["transform_direction"] != "FIRST_WORLD_TO_SECOND_WORLD":
        raise _fail("registration transform direction is closed")
    affine = _validate_float_array(document["affine_4x4_row_major"], 16, "affine_4x4_row_major")
    if tuple(affine[12:]) != (0.0, 0.0, 0.0, 1.0):
        raise _fail("registration affine homogeneous row must be exact")
    matrix = tuple(tuple(affine[row * 4 + column] for column in range(4)) for row in range(4))
    try:
        invert_matrix(cast(Any, matrix))
    except ValueError as exc:
        raise _fail("registration affine upper 3x3 is singular") from exc
    if not isinstance(document["converged"], bool) or document["converged"] is not True:
        raise _fail("registration must be converged true")
    lower = _validate_float_array(document["overlap_lower_lps_mm"], 3, "overlap_lower_lps_mm")
    upper = _validate_float_array(document["overlap_upper_lps_mm"], 3, "overlap_upper_lps_mm")
    if any(a >= b for a, b in zip(lower, upper)):
        raise _fail("registration overlap lower must be below upper")
    algorithm_id = document["algorithm_id"]
    algorithm_version = document["algorithm_version"]
    if (
        not isinstance(algorithm_id, str)
        or not algorithm_id
        or not isinstance(algorithm_version, str)
        or not algorithm_version
    ):
        raise _fail("registration algorithm identity is required")
    _require_sha(document["parameters_sha256"], "parameters_sha256")
    if (
        not isinstance(artifact.kind, ArtifactKind)
        or artifact.kind != ArtifactKind.REGISTRATION
        or not isinstance(artifact.status, ArtifactStatus)
        or artifact.status != ArtifactStatus.ACTIVE
    ):
        raise _fail("registration artifact must be ACTIVE kind=REGISTRATION")
    if (
        artifact.payload.payload_schema != REGISTRATION_PAYLOAD_SCHEMA
        or artifact.payload.blob_sha256 != canonical_digest(document)
    ):
        raise _fail("registration payload schema/digest mismatch")
    if artifact.producer.producer_type != ProducerType.DETERMINISTIC_ENGINE:
        raise _fail("registration producer must be DETERMINISTIC_ENGINE")
    if (
        artifact.producer.algorithm != algorithm_id
        or artifact.producer.algorithm_version != algorithm_version
        or artifact.producer.parameters_blob_sha256 != document["parameters_sha256"]
    ):
        raise _fail("registration producer binding mismatch")
    if artifact.provenance_ref == "" or not isinstance(artifact.provenance_ref, str):
        raise _fail("registration artifact provenance is required")
    if (
        _array_digest(pair.first_array) != first.blob_ref.sha256
        or _array_digest(pair.second_array) != second.blob_ref.sha256
    ):
        raise _fail("registered pair array bytes do not match Volume blob refs")
    if (
        int(pair.first_array.nbytes) != int(first.blob_ref.byte_size)
        or int(pair.second_array.nbytes) != int(second.blob_ref.byte_size)
    ):
        raise _fail("registered pair array byte sizes do not match Volume blob refs")
    if tuple(pair.first_array.shape) != tuple(first.coordinate_frame.dimensions) or tuple(
        pair.second_array.shape
    ) != tuple(second.coordinate_frame.dimensions):
        raise _fail("registered pair array shapes do not match FrameSpecs")
    expected_registration_geometry = registration_geometry_id(
        algorithm_id=cast(str, algorithm_id),
        first_geometry_id=cast(str, first_geometry),
        first_phase_ref=pair.first_phase_ref,
        second_geometry_id=cast(str, second_geometry),
        second_phase_ref=pair.second_phase_ref,
    )
    if document["registration_geometry_id"] != expected_registration_geometry:
        raise _fail("stable registration_geometry_id preimage mismatch")
    if not isinstance(document["registration_geometry_id"], str) or not re.fullmatch(
        r"qc_registration_geometry:[0-9a-f]{64}", cast(str, document["registration_geometry_id"])
    ):
        raise _fail("registration_geometry_id is not a stable lowercase SHA-256 ref")
    _validate_pair_provenance(pair)


def _validate_pair_provenance(pair: RegisteredPairInput) -> None:
    records = pair.provenance_records
    if not records:
        raise _fail("registered pair requires nonempty provenance_records")
    if any(not isinstance(record, ProvenanceRecord) for record in records):
        raise _fail("registered pair provenance_records must contain ProvenanceRecord objects")
    by_id = {record.provenance_id: record for record in records}
    if len(by_id) != len(records):
        raise _fail("registered pair provenance records contain duplicate ids")
    ok, problems = verify_chain(records)
    if not ok:
        raise _fail("registered pair provenance chain failed: " + "; ".join(problems))
    artifact_record = by_id.get(pair.registration_artifact.provenance_ref)
    if artifact_record is None:
        raise _fail("registration artifact provenance ref is unresolved")
    document = pair.registration_document
    if artifact_record.provenance_id != pair.registration_artifact.provenance_ref:
        raise _fail("registration provenance id does not bind outer artifact")
    if artifact_record.subject_ref != pair.registration_artifact.artifact_id:
        raise _fail("registration provenance subject does not bind outer artifact")
    if (
        artifact_record.actor.actor_id != document["algorithm_id"]
        or artifact_record.actor.actor_type.value != "ENGINE"
    ):
        raise _fail("registration provenance actor does not bind algorithm")
    if artifact_record.environment.software_version != document["algorithm_version"]:
        raise _fail("registration provenance algorithm version/environment mismatch")
    parameters_ref = artifact_record.parameters_ref
    if (
        isinstance(parameters_ref, NotApplicable)
        or parameters_ref.sha256 != document["parameters_sha256"]
    ):
        raise _fail("registration provenance parameter binding mismatch")
    if artifact_record.preprocessing != QC_PREPROCESSING:
        raise _fail("registration preprocessing must be QC_INPUT_NORMALIZATION_V1")
    first = pair.first_volume
    second = pair.second_volume
    expected_upstreams = {first.provenance_ref, second.provenance_ref}
    actual_upstreams = (
        set(artifact_record.upstream_refs)
        if not isinstance(artifact_record.upstream_refs, NotApplicable)
        else set()
    )
    if not expected_upstreams.issubset(actual_upstreams):
        raise _fail("registration provenance does not bind both volume upstreams")
    expected_inputs = {
        (f"volume_id:{first.volume_id}", first.blob_ref.sha256),
        (f"volume_id:{second.volume_id}", second.blob_ref.sha256),
        (
            f"frame_spec:{_frame_digest(first.coordinate_frame)}",
            _frame_digest(first.coordinate_frame),
        ),
        (
            f"frame_spec:{_frame_digest(second.coordinate_frame)}",
            _frame_digest(second.coordinate_frame),
        ),
    }
    if isinstance(artifact_record.inputs, NotApplicable):
        raise _fail("registration provenance inputs must be concrete")
    actual_inputs = {(item.ref, item.sha256_at_use, item.kind) for item in artifact_record.inputs}
    actual_input_pairs = {(ref, digest) for ref, digest, _kind in actual_inputs}
    if not expected_inputs.issubset(actual_input_pairs):
        raise _fail("registration provenance inputs do not bind volumes and FrameSpecs")
    expected_kinds = {
        (f"volume_id:{first.volume_id}", first.blob_ref.sha256): "volume",
        (f"volume_id:{second.volume_id}", second.blob_ref.sha256): "volume",
        (
            f"frame_spec:{_frame_digest(first.coordinate_frame)}",
            _frame_digest(first.coordinate_frame),
        ): "artifact",
        (
            f"frame_spec:{_frame_digest(second.coordinate_frame)}",
            _frame_digest(second.coordinate_frame),
        ): "artifact",
    }
    if any(
        (ref, digest, expected_kinds[(ref, digest)]) not in actual_inputs
        for ref, digest in expected_kinds
    ):
        raise _fail("registration provenance input kinds do not bind components")
    for label, volume in (("first", first), ("second", second)):
        record = by_id.get(volume.provenance_ref)
        if record is None or record.provenance_id != volume.provenance_ref:
            raise _fail(f"{label} volume provenance ref does not bind raw record id")
        if isinstance(record.inputs, NotApplicable):
            input_binds_volume = False
        else:
            input_binds_volume = any(
                item.kind == "volume"
                and item.ref == f"volume_id:{volume.volume_id}"
                and item.sha256_at_use == volume.blob_ref.sha256
                for item in record.inputs
            )
        if record.subject_ref not in {
            volume.volume_id,
            f"volume_id:{volume.volume_id}",
        } and not input_binds_volume:
            raise _fail(f"{label} volume provenance record does not bind its Volume")


def validate_registered_pair_input(pair: RegisteredPairInput) -> bool:
    """Validate every closed pair binding; return ``True`` on success."""

    try:
        _validate_registration_document(pair)
    except QCValidationError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise QCValidationError(f"registered pair invalid: {exc}") from exc
    return True


def pair_affine_and_overlap(
    pair: RegisteredPairInput,
) -> tuple[Any, tuple[float, ...], tuple[float, ...]]:
    validate_registered_pair_input(pair)
    document = pair.registration_document
    flat = tuple(float(value) for value in cast(Sequence[float], document["affine_4x4_row_major"]))
    matrix = tuple(tuple(flat[row * 4 + column] for column in range(4)) for row in range(4))
    lower = tuple(float(value) for value in cast(Sequence[float], document["overlap_lower_lps_mm"]))
    upper = tuple(float(value) for value in cast(Sequence[float], document["overlap_upper_lps_mm"]))
    return matrix, lower, upper


def _world_corners(frame: FrameSpec) -> tuple[tuple[float, float, float], ...]:
    matrix = frame.voxel_to_world_matrix()
    values: list[tuple[float, float, float]] = []
    for k in (0, int(frame.dimensions[0]) - 1):
        for j in (0, int(frame.dimensions[1]) - 1):
            for i in (0, int(frame.dimensions[2]) - 1):
                values.append(
                    (
                        matrix[0][0] * i + matrix[0][1] * j + matrix[0][2] * k + matrix[0][3],
                        matrix[1][0] * i + matrix[1][1] * j + matrix[1][2] * k + matrix[1][3],
                        matrix[2][0] * i + matrix[2][1] * j + matrix[2][2] * k + matrix[2][3],
                    )
                )
    return tuple(values)


def _default_source_record(
    *,
    subject_ref: str,
    provenance_id: str,
    volume: Volume,
    at: str,
    created_by: str,
    code_commit_sha: str,
) -> ProvenanceRecord:
    return create_provenance(
        subject_ref=subject_ref,
        event_type=__import__(
            "core.enums", fromlist=["ProvenanceEventType"]
        ).ProvenanceEventType.DERIVED,
        actor=Actor(
            actor_type=__import__("core.enums", fromlist=["ActorType"]).ActorType.ENGINE,
            actor_id="qc-source-volume",
        ),
        inputs=(
            InputRef(
                ref=f"volume_id:{volume.volume_id}",
                kind="volume",
                sha256_at_use=volume.blob_ref.sha256,
            ),
        ),
        preprocessing="QC_SOURCE_VOLUME_BINDING_V1",
        parameters_ref=BlobRef(
            blob_ref=f"volume:{volume.volume_id}", sha256=volume.blob_ref.sha256
        ),
        human_edit=NotApplicable(value=None, reason="source volume record has no human edit"),
        upstream_refs=NotApplicable(value=None, reason="source volume is a supplied root"),
        environment=Environment(
            software_version=_DEFAULT_SOFTWARE, code_commit_sha=code_commit_sha
        ),
        model_record_ref=None,
        created_by=created_by,
        at=at,
        provenance_id=provenance_id,
    )


def _records_for_volumes(
    supplied: Sequence[ProvenanceRecord],
    first: Volume,
    second: Volume,
    *,
    at: str,
    created_by: str,
    code_commit_sha: str,
) -> tuple[ProvenanceRecord, ...]:
    by_id = {record.provenance_id: record for record in supplied}
    records = list(supplied)
    for volume in (first, second):
        if volume.provenance_ref not in by_id:
            record = _default_source_record(
                subject_ref=volume.volume_id,
                provenance_id=volume.provenance_ref,
                volume=volume,
                at=at,
                created_by=created_by,
                code_commit_sha=code_commit_sha,
            )
            records.append(record)
            by_id[record.provenance_id] = record
    return tuple(records)


def create_registered_pair_input(
    *,
    first_volume: Volume,
    second_volume: Volume,
    first_array: Any,
    second_array: Any,
    first_phase_ref: str,
    second_phase_ref: str,
    affine_4x4_row_major: Sequence[float] | None = None,
    overlap_lower_lps_mm: Sequence[float] | None = None,
    overlap_upper_lps_mm: Sequence[float] | None = None,
    registration_version: int = 1,
    algorithm_id: str = "qc-registration-v1",
    algorithm_version: str = "1",
    parameters_sha256: str | None = None,
    registration_artifact_id: str | None = None,
    registration_provenance_id: str | None = None,
    provenance_records: Sequence[ProvenanceRecord] = (),
    first_geometry_version: int = 1,
    second_geometry_version: int = 1,
    release_lane: ReleaseLane = ReleaseLane.PERSONAL_RESEARCH,
    claim_binding: ClaimBinding = _DEFAULT_CLAIM,
    created_by: str = "qc_001_det",
    created_at: str | None = None,
    code_commit_sha: str = _ZERO_COMMIT,
    software_version: str = "1",
    id_factory: Callable[[], str] | None = None,
    artifact_store: DerivedArtifactStore | None = None,
) -> RegistrationBuildResult:
    """Build and validate the immutable registration artifact/pair surface."""

    mint = id_factory or new_ulid
    if not isinstance(first_volume, Volume) or not isinstance(second_volume, Volume):
        raise QCValidationError("registered pair requires two Volume objects")
    if (
        first_volume.status != ArtifactStatus.ACTIVE
        or second_volume.status != ArtifactStatus.ACTIVE
    ):
        raise QCValidationError("registered pair volumes must be ACTIVE")
    if first_volume.kind != VolumeKind.CANONICAL or second_volume.kind != VolumeKind.CANONICAL:
        raise QCValidationError("registered pair requires canonical source volumes")
    if first_volume.study_ref != second_volume.study_ref:
        raise QCValidationError("registered pair volumes must share study identity")
    canonical_study = _canonical_study_ref(first_volume.study_ref)
    if first_volume.study_ref != canonical_study:
        raise QCValidationError("registered pair study_ref must be canonical")
    if first_volume.volume_id == second_volume.volume_id:
        raise QCValidationError("registered pair volumes must be distinct")
    if len(first_volume.source_series_refs) != 1 or len(second_volume.source_series_refs) != 1:
        raise QCValidationError("registered pair volumes must each have one source series")
    if first_volume.source_series_refs[0] == second_volume.source_series_refs[0]:
        raise QCValidationError("registered pair source series must be distinct")
    for series_ref in (first_volume.source_series_refs[0], second_volume.source_series_refs[0]):
        if (
            not isinstance(series_ref, str)
            or not series_ref.startswith("series_id:")
            or not _valid_ulid(series_ref.split(":", 1)[1])
        ):
            raise QCValidationError("registered pair source series must be series_id:<ULID>")
    if (
        first_volume.coordinate_frame.frame_of_reference_uid
        != second_volume.coordinate_frame.frame_of_reference_uid
    ):
        raise QCValidationError("registered pair FrameOfReferenceUID values must be byte-equal")
    if not isinstance(first_phase_ref, str) or _PHASE.fullmatch(first_phase_ref) is None:
        raise QCValidationError("first_phase_ref has invalid closed syntax")
    if not isinstance(second_phase_ref, str) or _PHASE.fullmatch(second_phase_ref) is None:
        raise QCValidationError("second_phase_ref has invalid closed syntax")
    if first_phase_ref == second_phase_ref:
        raise QCValidationError("registered pair phases must be distinct")
    artifact_id = registration_artifact_id or mint()
    provenance_id = registration_provenance_id or mint()
    if not all(_valid_ulid(item) for item in (artifact_id, provenance_id)):
        raise QCValidationError("registration artifact/provenance ids must be ULIDs")
    if (
        not isinstance(registration_version, int)
        or isinstance(registration_version, bool)
        or registration_version < 1
    ):
        raise QCValidationError("registration_version must be an integer >= 1")
    if (
        not isinstance(first_geometry_version, int)
        or isinstance(first_geometry_version, bool)
        or first_geometry_version < 1
    ):
        raise QCValidationError("first_geometry_version must be an integer >= 1")
    if (
        not isinstance(second_geometry_version, int)
        or isinstance(second_geometry_version, bool)
        or second_geometry_version < 1
    ):
        raise QCValidationError("second_geometry_version must be an integer >= 1")
    if affine_4x4_row_major is None:
        raise QCValidationError("registration affine must be supplied explicitly")
    if overlap_lower_lps_mm is None or overlap_upper_lps_mm is None:
        raise QCValidationError("registration overlap bounds must be supplied explicitly")
    affine = _validate_float_array(affine_4x4_row_major, 16, "affine_4x4_row_major")
    if tuple(affine[12:]) != (0.0, 0.0, 0.0, 1.0):
        raise QCValidationError("registration affine homogeneous row must be exact")
    try:
        invert_matrix(
            cast(
                Any,
                tuple(
                    tuple(affine[row * 4 + column] for column in range(4))
                    for row in range(4)
                ),
            )
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise QCValidationError("registration affine upper 3x3 is singular") from exc
    lower = _validate_float_array(overlap_lower_lps_mm, 3, "overlap_lower_lps_mm")
    upper = _validate_float_array(overlap_upper_lps_mm, 3, "overlap_upper_lps_mm")
    if any(a >= b for a, b in zip(lower, upper)):
        raise QCValidationError("registration overlap lower must be below upper")
    if not isinstance(algorithm_id, str) or not algorithm_id:
        raise QCValidationError("registration algorithm_id is required")
    if not isinstance(algorithm_version, str) or not algorithm_version:
        raise QCValidationError("registration algorithm_version is required")
    if parameters_sha256 is not None and not is_sha256_hex(parameters_sha256):
        raise QCValidationError("parameters_sha256 must be lowercase 64-hex SHA-256")
    _array_digest(first_array)
    _array_digest(second_array)
    if tuple(first_array.shape) != tuple(first_volume.coordinate_frame.dimensions):
        raise QCValidationError("first registration array shape does not match FrameSpec")
    if tuple(second_array.shape) != tuple(second_volume.coordinate_frame.dimensions):
        raise QCValidationError("second registration array shape does not match FrameSpec")
    if int(first_array.nbytes) != int(first_volume.blob_ref.byte_size):
        raise QCValidationError("first registration array byte size does not match Volume")
    if int(second_array.nbytes) != int(second_volume.blob_ref.byte_size):
        raise QCValidationError("second registration array byte size does not match Volume")
    first_geometry = GeometryBase(_geometry_id(first_volume), first_geometry_version)
    second_geometry = GeometryBase(_geometry_id(second_volume), second_geometry_version)
    first_frame_digest = _frame_digest(first_volume.coordinate_frame)
    second_frame_digest = _frame_digest(second_volume.coordinate_frame)
    parameter_digest = parameters_sha256 or canonical_digest(
        {
            "algorithm_id": algorithm_id,
            "algorithm_version": algorithm_version,
            "registration": "QC_REGISTRATION_V1",
        }
    )
    document: dict[str, object] = {
        "schema_version": REGISTRATION_PAYLOAD_SCHEMA,
        "registration_ref": f"registration:{artifact_id}",
        "registration_geometry_id": registration_geometry_id(
            algorithm_id=algorithm_id,
            first_geometry_id=first_geometry.geometry_id,
            first_phase_ref=first_phase_ref,
            second_geometry_id=second_geometry.geometry_id,
            second_phase_ref=second_phase_ref,
        ),
        "registration_version": registration_version,
        "first_volume_ref": _canonical_volume_ref(first_volume),
        "second_volume_ref": _canonical_volume_ref(second_volume),
        "first_volume_blob_sha256": first_volume.blob_ref.sha256,
        "second_volume_blob_sha256": second_volume.blob_ref.sha256,
        "first_volume_provenance_ref": first_volume.provenance_ref,
        "second_volume_provenance_ref": second_volume.provenance_ref,
        "first_series_ref": first_volume.source_series_refs[0],
        "second_series_ref": second_volume.source_series_refs[0],
        "first_phase_ref": first_phase_ref,
        "second_phase_ref": second_phase_ref,
        "first_geometry_id": first_geometry.geometry_id,
        "second_geometry_id": second_geometry.geometry_id,
        "first_geometry_version": first_geometry.geometry_version,
        "second_geometry_version": second_geometry.geometry_version,
        "first_frame_spec_sha256": first_frame_digest,
        "second_frame_spec_sha256": second_frame_digest,
        "study_ref": canonical_study,
        "frame_of_reference_uid_digest": hashlib.sha256(
            first_volume.coordinate_frame.frame_of_reference_uid.encode("utf-8")
        ).hexdigest(),
        "transform_direction": "FIRST_WORLD_TO_SECOND_WORLD",
        "affine_4x4_row_major": affine,
        "converged": True,
        "overlap_lower_lps_mm": lower,
        "overlap_upper_lps_mm": upper,
        "algorithm_id": algorithm_id,
        "algorithm_version": algorithm_version,
        "parameters_sha256": parameter_digest,
    }
    stamp = created_at or utc_now_iso()
    records = _records_for_volumes(
        tuple(provenance_records),
        first_volume,
        second_volume,
        at=stamp,
        created_by=created_by,
        code_commit_sha=code_commit_sha,
    )
    inputs = (
        InputRef(
            ref=f"volume_id:{first_volume.volume_id}",
            kind="volume",
            sha256_at_use=first_volume.blob_ref.sha256,
        ),
        InputRef(
            ref=f"volume_id:{second_volume.volume_id}",
            kind="volume",
            sha256_at_use=second_volume.blob_ref.sha256,
        ),
        InputRef(
            ref=f"frame_spec:{first_frame_digest}",
            kind="artifact",
            sha256_at_use=first_frame_digest,
        ),
        InputRef(
            ref=f"frame_spec:{second_frame_digest}",
            kind="artifact",
            sha256_at_use=second_frame_digest,
        ),
    )
    registration_provenance = create_provenance(
        subject_ref=artifact_id,
        event_type=__import__(
            "core.enums", fromlist=["ProvenanceEventType"]
        ).ProvenanceEventType.DERIVED,
        actor=Actor(
            actor_type=__import__("core.enums", fromlist=["ActorType"]).ActorType.ENGINE,
            actor_id=algorithm_id,
        ),
        inputs=inputs,
        preprocessing=QC_PREPROCESSING,
        parameters_ref=BlobRef(
            blob_ref=f"registration-parameters:{parameter_digest}", sha256=parameter_digest
        ),
        human_edit=NotApplicable(value=None, reason="deterministic registration has no human edit"),
        upstream_refs=(first_volume.provenance_ref, second_volume.provenance_ref),
        environment=Environment(software_version=software_version, code_commit_sha=code_commit_sha),
        model_record_ref=None,
        created_by=created_by,
        at=stamp,
        provenance_id=provenance_id,
    )
    records = records + (registration_provenance,)
    artifact = create_derived_artifact(
        artifact_id=artifact_id,
        study_ref=first_volume.study_ref,
        kind=ArtifactKind.REGISTRATION,
        payload=PayloadRef(
            payload_schema=REGISTRATION_PAYLOAD_SCHEMA,
            blob_sha256=canonical_digest(document),
            inline_summary=(
                ("registration_geometry_id", str(document["registration_geometry_id"])),
            ),
        ),
        lineage=Lineage(
            input_volume_refs=(first_volume.volume_id, second_volume.volume_id),
            input_series_refs=(
                first_volume.source_series_refs[0],
                second_volume.source_series_refs[0],
            ),
            registration_ref=str(document["registration_ref"]),
        ),
        claim_binding=claim_binding,
        producer=Producer(
            producer_type=ProducerType.DETERMINISTIC_ENGINE,
            algorithm=algorithm_id,
            algorithm_version=algorithm_version,
            parameters_blob_sha256=parameter_digest,
        ),
        release_lane=release_lane,
        provenance_ref=provenance_id,
        created_by=created_by,
        created_at=stamp,
    )
    pair = RegisteredPairInput(
        first_volume=first_volume,
        second_volume=second_volume,
        first_array=first_array,
        second_array=second_array,
        first_geometry_base=first_geometry,
        second_geometry_base=second_geometry,
        first_phase_ref=first_phase_ref,
        second_phase_ref=second_phase_ref,
        registration_artifact=artifact,
        registration_document=document,
        provenance_records=records,
    )
    if artifact_store is not None:
        artifact_store.add(artifact)
    return RegistrationBuildResult(
        pair=pair, artifact=artifact, provenance=registration_provenance, document=document
    )


build_registered_pair_input = create_registered_pair_input
make_registered_pair = create_registered_pair_input


class PairGeometryTracker:
    """Advance only the stable composite geometry key in the existing DAG."""

    def __init__(self, store: DerivedArtifactStore, pair: RegisteredPairInput) -> None:
        validate_registered_pair_input(pair)
        self.store = store
        self.pair = pair
        self.composite_geometry_id = pair_geometry_id(pair)
        self._first_version = pair.first_geometry_base.geometry_version
        self._second_version = pair.second_geometry_base.geometry_version
        registration_version = pair.registration_document["registration_version"]
        if isinstance(registration_version, bool) or not isinstance(registration_version, int):
            raise QCValidationError("registration_version must be an integer")
        self._registration_version = registration_version
        try:
            store.register_new_geometry_version(
                self.composite_geometry_id, pair_geometry_version(pair)
            )
        except ValueError:
            # ``DerivedArtifactStore.add`` normally registered it when the QC
            # artifact was added; an existing same-version binding is safe.
            pass

    @property
    def composite_version(self) -> int:
        return self._first_version + self._second_version + self._registration_version - 2

    def _bump(self, component: str, increment: int = 1) -> tuple[str, ...]:
        if increment < 1:
            raise ValueError("component geometry increment must be positive")
        if component == "first":
            self._first_version += increment
        elif component == "second":
            self._second_version += increment
        elif component == "registration":
            self._registration_version += increment
        else:
            raise ValueError("component must be first, second, or registration")
        return self.store.register_new_geometry_version(
            self.composite_geometry_id, self.composite_version
        )

    def bump_first(self, increment: int = 1) -> tuple[str, ...]:
        return self._bump("first", increment)

    def bump_second(self, increment: int = 1) -> tuple[str, ...]:
        return self._bump("second", increment)

    def bump_registration(self, increment: int = 1) -> tuple[str, ...]:
        return self._bump("registration", increment)

    bump_first_geometry = bump_first
    bump_second_geometry = bump_second
    bump_registration_geometry = bump_registration
    bump_component = _bump


QCPairGeometryTracker = PairGeometryTracker
RegisteredPair = RegisteredPairInput


__all__ = [
    "PairGeometryTracker",
    "QCPairGeometryTracker",
    "RegisteredPair",
    "RegistrationBuildResult",
    "build_registered_pair_input",
    "create_registered_pair_input",
    "make_registered_pair",
    "pair_affine_and_overlap",
    "pair_geometry_id",
    "pair_geometry_version",
    "registration_geometry_id",
    "validate_registered_pair_input",
]
