"""The straightened CPR view read surface of the viewer (display-only).

CORONARY_REVIEW_VIEW_V1, Decision B: the dedicated straightened-view panel
reads the accepted ``ccta.cpr.view.v1`` computation through the narrowly
admitted WS-02 seam (``apps/workstation/session_store`` mirrors the MPR
digest checks for CPR payloads; the MPR path is byte-identical).  This module
owns the closed ``ccta.viewer.cpr_view.v1`` VIEW document: sealed CPR payload
metadata + digest-bound pixel dimensions + the bound centerline generation
(``centerline_ref`` + ``centerline_payload_sha256`` +
``centerline_generation``).  Pixel bytes travel separately through the EXISTING
frozen PNG encoder/transport (``apps.viewer.cpr_render``); this document
carries NO pixels and NO geometry of its own (Rule B: the client invents no
geometry; Rule C: no measurement).

Exactly one ACTIVE CPR view (an ``MPR_VIEW`` whose outer ``payload_schema``
is ``ccta.cpr.view.v1``) may be present; zero / multiple / malformed /
stale-or-unbound views are typed, visible refusals - never a silent pick and
never a stale fallback:

* ``CPR_VIEW_UNAVAILABLE`` - no ACTIVE CPR view in the session;
* ``CPR_VIEW_INVALID`` - malformed payload, multiple ACTIVE candidates, or an
  unbound centerline reference;
* ``CPR_VIEW_STALE`` - the bound centerline (or the view itself) is no longer
  ACTIVE after an EDIT/REJECT successor transition.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from core.digest import is_sha256_hex
from core.enums import ArtifactKind, ArtifactStatus
from core.ids import validate_ulid
from core.qc.gate import V2_VESSEL_IDS as _FROZEN_VESSEL_IDS
from visualization.cpr import (
    CPR_CROSS_SECTION_RULE,
    CPR_FRAMING_RULE,
    CPR_SCHEMA,
    validate_cpr_payload,
)

#: The closed viewer CPR document schema (metadata + binding refs, no pixels).
CPR_VIEW_SCHEMA: Final[str] = "ccta.viewer.cpr_view.v1"

#: The closed typed-refusal vocabulary of the CPR read surface (Decision B;
#: mirrors the CENTERLINE_OVERLAY_* vocabulary).
CPR_REFUSAL_UNAVAILABLE: Final[str] = "CPR_VIEW_UNAVAILABLE"
CPR_REFUSAL_INVALID: Final[str] = "CPR_VIEW_INVALID"
CPR_REFUSAL_STALE: Final[str] = "CPR_VIEW_STALE"

_ULID_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_CPR_REF_RE: Final[re.Pattern[str]] = re.compile(
    rf"cpr_artifact_id:[{_ULID_CHARS}]{{26}}\Z"
)
_CENTERLINE_REF_RE: Final[re.Pattern[str]] = re.compile(
    rf"centerline_artifact_id:[{_ULID_CHARS}]{{26}}\Z"
)
_SOURCE_VOLUME_REF_RE: Final[re.Pattern[str]] = re.compile(
    rf"volume_id:[{_ULID_CHARS}]{{26}}\Z"
)
_PROVENANCE_REF_RE: Final[re.Pattern[str]] = re.compile(
    rf"provenance:[{_ULID_CHARS}]{{26}}\Z"
)

_CPR_VIEW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "session_id",
        "cpr_ref",
        "centerline_ref",
        "centerline_payload_sha256",
        "vessel_label",
        "centerline_status",
        "centerline_review_state",
        "centerline_generation",
        "review_state",
        "status",
        "matrix_size",
        "s_spacing_mm",
        "radial_spacing_mm",
        "output_extent_mm",
        "framing_rule",
        "cross_section_rule",
        "method",
        "scalar_domain",
        "sampling_parameters_sha256",
        "pixel_blob_ref",
        "source_volume_ref",
        "provenance_ref",
    }
)


class CPRRefusalError(ValueError):
    """A typed, fail-closed CPR read refusal (HTTP 409, never a fallback)."""

    http_status: Final[int] = 409

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")

    @property
    def payload(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": self.message}}


def _refuse_cpr(code: str, message: str) -> CPRRefusalError:
    return CPRRefusalError(code, message)


def _require_ulid(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise _refuse_cpr(CPR_REFUSAL_INVALID, f"{label} must be a ULID")
    try:
        validate_ulid(value)
    except (TypeError, ValueError):
        raise _refuse_cpr(CPR_REFUSAL_INVALID, f"{label} must be a ULID") from None
    return value


def _cpr_bound_label(
    artifact: Any,
    centerline_payload_documents: Mapping[str, Mapping[str, object]],
) -> str | None:
    """The sealed ``vessel_label`` of the centerline a CPR view is framed on.

    Resolved through the CPR payload's own bound ``centerline_ref`` (the
    same binding ``build_cpr_view_document`` serves).  Missing/invalid
    documents yield ``None`` (cannot match a vessel filter, never a match).
    """

    from core.coronary import validate_centerline_payload as _validate_cl
    from visualization.cpr import validate_cpr_payload as _validate_cpr

    artifact_id = str(getattr(artifact, "artifact_id", ""))
    cpr_document = centerline_payload_documents.get(artifact_id)
    if not isinstance(cpr_document, Mapping):
        return None
    try:
        cpr_validated = _validate_cpr(cpr_document)
    except Exception:  # noqa: BLE001 - an invalid payload cannot match
        return None
    centerline_ref = str(cpr_validated.get("centerline_ref", ""))
    if ":" not in centerline_ref:
        return None
    centerline_id = centerline_ref.split(":", 1)[1]
    cl_document = centerline_payload_documents.get(centerline_id)
    if not isinstance(cl_document, Mapping):
        return None
    try:
        label = _validate_cl(cl_document).get("vessel_label")
    except Exception:  # noqa: BLE001 - an invalid payload cannot match
        return None
    return label if isinstance(label, str) else None


def select_active_cpr_view(
    artifacts: Sequence[Any],
    vessel_label: str | None = None,
    centerline_payload_documents: Mapping[str, Mapping[str, object]] | None = None,
) -> Any:
    """The single ACTIVE CPR view artifact of the session, or a typed refusal.

    A CPR view is an ``MPR_VIEW`` whose outer ``payload_schema`` is exactly
    ``ccta.cpr.view.v1`` (never a fake orthogonal MPR).  Status decides
    candidacy exactly like the overlay seam: only ACTIVE artifacts are
    reviewable; zero candidates refuse UNAVAILABLE **only when the session
    never had a CPR view**.  Multiple ACTIVE CPR views refuse INVALID (the
    single-panel surface cannot silently pick one).  A session whose CPR
    view exists but is not ACTIVE (STALE after an EDIT/REJECT successor
    transition, SUPERSEDED, or INVALIDATED) refuses STALE — the predecessor
    bytes stay queryable but are never served as current (Decision B/C:
    predecessor CPR bytes stay queryable, never silently current).

    CORONARY_MULTI_VESSEL_NAV (Decision A): an optional ``vessel_label``
    filter narrows candidacy to views whose bound centerline's sealed
    payload carries that frozen ``VESSEL:*`` label (resolved through
    ``centerline_payload_documents``, which carries BOTH the CPR payloads
    keyed by view id AND the centerline payloads keyed by centerline id).
    ``None`` preserves the exact #28 single-panel behavior (including the
    exact refusal messages).
    """

    if vessel_label is not None and (
        not isinstance(vessel_label, str) or vessel_label not in _FROZEN_VESSEL_IDS
    ):
        raise _refuse_cpr(
            CPR_REFUSAL_INVALID,
            f"vessel_label {vessel_label!r} is outside the frozen VESSEL:* tokens",
        )
    cpr_views = [
        artifact
        for artifact in artifacts
        if getattr(artifact, "kind", None) is ArtifactKind.MPR_VIEW
        and getattr(getattr(artifact, "payload", None), "payload_schema", None)
        == CPR_SCHEMA
    ]
    if vessel_label is not None and centerline_payload_documents is not None:
        cpr_views = [
            artifact
            for artifact in cpr_views
            if _cpr_bound_label(artifact, centerline_payload_documents) == vessel_label
        ]
    candidates = [
        artifact
        for artifact in cpr_views
        if getattr(artifact, "status", None) is ArtifactStatus.ACTIVE
    ]
    if candidates:
        if len(candidates) > 1 and vessel_label is None:
            raise _refuse_cpr(
                CPR_REFUSAL_INVALID,
                f"{len(candidates)} ACTIVE CPR views are present; "
                "the single-panel CPR surface cannot choose one",
            )
        if len(candidates) > 1:
            raise _refuse_cpr(
                CPR_REFUSAL_INVALID,
                f"{len(candidates)} ACTIVE CPR views carry "
                f"vessel_label {vessel_label!r}; the CPR surface cannot choose one",
            )
        return candidates[0]
    if cpr_views:
        stale = sorted(
            (str(getattr(view, "artifact_id", "?")), getattr(view, "status", None))
            for view in cpr_views
        )
        detail = ", ".join(
            f"{artifact_id} is {getattr(status, 'value', status)}"
            for artifact_id, status in stale
        )
        if vessel_label is None:
            raise _refuse_cpr(
                CPR_REFUSAL_STALE,
                "no ACTIVE CPR view is present; the predecessor view is stale "
                f"(EDIT/REJECT successor is current): {detail}",
            )
        raise _refuse_cpr(
            CPR_REFUSAL_STALE,
            f"no ACTIVE CPR view carries vessel_label {vessel_label!r}; "
            f"the predecessor view is stale: {detail}",
        )
    if vessel_label is None:
        raise _refuse_cpr(
            CPR_REFUSAL_UNAVAILABLE,
            "no ACTIVE CPR view (ccta.cpr.view.v1) is present in this session",
        )
    raise _refuse_cpr(
        CPR_REFUSAL_UNAVAILABLE,
        f"no ACTIVE CPR view carries vessel_label {vessel_label!r}",
    )


def build_cpr_view_document(
    *,
    session_id: str,
    artifacts: Sequence[Any],
    payload_documents: Mapping[str, Mapping[str, object]],
    vessel_label: str | None = None,
) -> dict[str, object]:
    """Build the closed ``ccta.viewer.cpr_view.v1`` read document.

    The document serves the sealed CPR payload metadata (digest-bound pixel
    dimensions, spacing, framing rules, binding refs) plus the bound
    centerline generation (``centerline_ref`` + ``centerline_payload_sha256``
    + the live bound ``centerline_status``/``centerline_review_state``/
    ``centerline_generation``).  Pixel bytes travel via the frozen PNG
    encoder (see ``apps.viewer.cpr_render``); this document carries NO pixels
    and NO geometry of its own.  STALE is computed from the artifact STATUS
    (an EDIT/REJECT successor transition leaves the predecessor
    STALE/INVALIDATED/SUPERSEDED, which refuses here instead of serving stale
    bytes as current).
    """

    _require_ulid(session_id, "session_id")
    artifact = select_active_cpr_view(
        artifacts,
        vessel_label=vessel_label,
        centerline_payload_documents=payload_documents,
    )
    artifact_id = str(artifact.artifact_id)
    document = payload_documents.get(artifact_id)
    if document is None:
        raise _refuse_cpr(
            CPR_REFUSAL_INVALID,
            f"CPR view {artifact_id} has no payload document",
        )
    try:
        validated = validate_cpr_payload(document)
    except Exception as exc:  # noqa: BLE001 - a refused document is a typed refusal
        raise _refuse_cpr(
            CPR_REFUSAL_INVALID,
            f"CPR view {artifact_id} payload is invalid ({exc})",
        ) from None
    centerline_ref = str(validated["centerline_ref"])
    match = _CENTERLINE_REF_RE.fullmatch(centerline_ref)
    if match is None:
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "centerline_ref has the wrong canonical form")
    centerline_id = centerline_ref.split(":", 1)[1]
    centerline_status = "UNKNOWN"
    centerline_review = "UNKNOWN"
    bound_generation = "UNKNOWN"
    for candidate in artifacts:
        if str(getattr(candidate, "artifact_id", "")) != centerline_id:
            continue
        status = getattr(candidate, "status", None)
        centerline_status = str(getattr(status, "value", status))
        review = getattr(candidate, "review", None)
        review_state = getattr(review, "review_state", None)
        centerline_review = str(getattr(review_state, "value", review_state))
        lineage = getattr(candidate, "lineage", None)
        base = getattr(lineage, "geometry_base", None)
        geometry_id = getattr(base, "geometry_id", None)
        geometry_version = getattr(base, "geometry_version", None)
        if geometry_id == centerline_id and isinstance(geometry_version, int):
            bound_generation = f"{geometry_id}:v{geometry_version}"
        break
    else:
        raise _refuse_cpr(
            CPR_REFUSAL_INVALID,
            f"CPR view {artifact_id} names an unbound centerline",
        )
    if centerline_status != ArtifactStatus.ACTIVE.value:
        raise _refuse_cpr(
            CPR_REFUSAL_STALE,
            f"the bound centerline {centerline_id} is {centerline_status}; "
            "the CPR view is stale (EDIT/REJECT successor is current)",
        )
    if artifact.status is not ArtifactStatus.ACTIVE:
        raise _refuse_cpr(
            CPR_REFUSAL_STALE,
            f"CPR view {artifact_id} is {artifact.status.value}",
        )
    pixel_ref = validated["pixel_blob_ref"]
    assert isinstance(pixel_ref, dict)
    matrix = validated["matrix_size"]
    assert isinstance(matrix, (list, tuple)) and len(matrix) == 2
    s_spacing = validated["s_spacing_mm"]
    radial_spacing = validated["radial_spacing_mm"]
    output_extent = validated["output_extent_mm"]
    assert isinstance(s_spacing, (int, float)) and not isinstance(s_spacing, bool)
    assert isinstance(radial_spacing, (int, float)) and not isinstance(radial_spacing, bool)
    assert isinstance(output_extent, (list, tuple)) and len(output_extent) == 2
    record: dict[str, object] = {
        "schema_version": CPR_VIEW_SCHEMA,
        "session_id": session_id,
        "cpr_ref": f"cpr_artifact_id:{artifact_id}",
        "centerline_ref": centerline_ref,
        "centerline_payload_sha256": validated["centerline_payload_sha256"],
        "vessel_label": _cpr_bound_label(artifact, payload_documents),
        "centerline_status": centerline_status,
        "centerline_review_state": centerline_review,
        "centerline_generation": bound_generation,
        "review_state": artifact.review.review_state.value,
        "status": artifact.status.value,
        "matrix_size": [int(matrix[0]), int(matrix[1])],
        "s_spacing_mm": float(s_spacing),
        "radial_spacing_mm": float(radial_spacing),
        "output_extent_mm": list(output_extent),
        "framing_rule": validated["framing_rule"],
        "cross_section_rule": validated["cross_section_rule"],
        "method": validated["method"],
        "scalar_domain": validated["scalar_domain"],
        "sampling_parameters_sha256": validated["sampling_parameters_sha256"],
        "pixel_blob_ref": dict(pixel_ref),
        "source_volume_ref": validated["source_volume_ref"],
        "provenance_ref": validated["provenance_ref"],
    }
    return validate_cpr_view_document(record)


def validate_cpr_view_document(document: Mapping[str, object]) -> dict[str, object]:
    """Validate the closed ``ccta.viewer.cpr_view.v1`` document exactly."""

    if not isinstance(document, Mapping):
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "CPR view document must be an object")
    if any(not isinstance(key, str) for key in document):
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "CPR view document keys must be strings")
    keys = set(document)
    missing = sorted(_CPR_VIEW_FIELDS - keys)
    unknown = sorted(keys - _CPR_VIEW_FIELDS)
    if missing or unknown:
        raise _refuse_cpr(
            CPR_REFUSAL_INVALID,
            f"CPR view document key set is not closed (missing={missing}, unknown={unknown})",
        )
    if document["schema_version"] != CPR_VIEW_SCHEMA:
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "CPR view schema_version mismatch")
    session_id = document["session_id"]
    if not isinstance(session_id, str):
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "CPR view session_id must be a ULID")
    try:
        validate_ulid(session_id)
    except (TypeError, ValueError):
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "CPR view session_id must be a ULID") from None
    if (
        not isinstance(document["cpr_ref"], str)
        or _CPR_REF_RE.fullmatch(str(document["cpr_ref"])) is None
    ):
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "cpr_ref has the wrong canonical form")
    if (
        not isinstance(document["centerline_ref"], str)
        or _CENTERLINE_REF_RE.fullmatch(str(document["centerline_ref"])) is None
    ):
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "centerline_ref has the wrong canonical form")
    vessel_label = document["vessel_label"]
    if vessel_label is not None and (
        not isinstance(vessel_label, str) or vessel_label not in _FROZEN_VESSEL_IDS
    ):
        raise _refuse_cpr(
            CPR_REFUSAL_INVALID,
            "CPR view vessel_label is outside the frozen VESSEL:* tokens",
        )
    for digest_key in ("centerline_payload_sha256", "sampling_parameters_sha256"):
        if not is_sha256_hex(document.get(digest_key)):
            raise _refuse_cpr(
                CPR_REFUSAL_INVALID, f"{digest_key} must be lowercase sha256"
            )
    if document["framing_rule"] != CPR_FRAMING_RULE:
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "framing_rule is fixed")
    if document["cross_section_rule"] != CPR_CROSS_SECTION_RULE:
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "cross_section_rule is fixed")
    if document["method"] not in {"NEAREST", "TRILINEAR"}:
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "method is outside the closed enum")
    if document["scalar_domain"] not in {"STORED", "MODALITY_TRANSFORMED", "HU"}:
        raise _refuse_cpr(
            CPR_REFUSAL_INVALID, "scalar_domain is outside the closed vocabulary"
        )
    matrix = document["matrix_size"]
    if (
        not isinstance(matrix, (list, tuple))
        or len(matrix) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in matrix)
    ):
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "matrix_size must be two positive integers")
    for numeric_key in ("s_spacing_mm", "radial_spacing_mm"):
        value = document[numeric_key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise _refuse_cpr(
                CPR_REFUSAL_INVALID, f"{numeric_key} must be positive and finite"
            )
    extent = document["output_extent_mm"]
    if (
        not isinstance(extent, (list, tuple))
        or len(extent) != 2
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v))
            for v in extent
        )
    ):
        raise _refuse_cpr(
            CPR_REFUSAL_INVALID, "output_extent_mm must be a finite two-vector"
        )
    pixel_ref = document["pixel_blob_ref"]
    if not isinstance(pixel_ref, Mapping) or set(pixel_ref) != {
        "sha256",
        "codec",
        "byte_size",
    }:
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "pixel_blob_ref has an unexpected shape")
    if not is_sha256_hex(pixel_ref.get("sha256")):
        raise _refuse_cpr(
            CPR_REFUSAL_INVALID, "pixel_blob_ref.sha256 must be lowercase sha256"
        )
    if pixel_ref.get("codec") not in {"int16-le", "float32-le"}:
        raise _refuse_cpr(
            CPR_REFUSAL_INVALID, "pixel_blob_ref codec is outside the closed enum"
        )
    byte_size = pixel_ref.get("byte_size")
    rows, columns = int(matrix[0]), int(matrix[1])
    itemsize = 2 if pixel_ref.get("codec") == "int16-le" else 4
    if (
        isinstance(byte_size, bool)
        or not isinstance(byte_size, int)
        or byte_size != rows * columns * itemsize
    ):
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "pixel_blob_ref.byte_size is incorrect")
    source_ref = document["source_volume_ref"]
    if not isinstance(source_ref, str) or _SOURCE_VOLUME_REF_RE.fullmatch(source_ref) is None:
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "source_volume_ref has the wrong canonical form")
    provenance_ref = document["provenance_ref"]
    if not isinstance(provenance_ref, str) or _PROVENANCE_REF_RE.fullmatch(provenance_ref) is None:
        raise _refuse_cpr(CPR_REFUSAL_INVALID, "provenance_ref has the wrong canonical form")
    return dict(document)



__all__ = [
    "CPR_REFUSAL_INVALID",
    "CPR_REFUSAL_STALE",
    "CPR_REFUSAL_UNAVAILABLE",
    "CPR_VIEW_SCHEMA",
    "CPRRefusalError",
    "build_cpr_view_document",
    "select_active_cpr_view",
    "validate_cpr_view_document",
]
