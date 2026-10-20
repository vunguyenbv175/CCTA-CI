"""The read-only coronary tree schematic navigator document (display-only).

TREE_SCHEMATIC_NAV: a 2D schematic overview panel of the accepted
multi-vessel centerline paths.  The server derives one schematic document per
session from the ACTIVE CENTERLINE set by fixed-orientation projection +
polyline simplification of the sealed ``points_world`` polylines - a
read-only derivation (the canonical CENTERLINE payloads stay the source of
truth, Rule B).  The document is labeled SCHEMATIC everywhere, carries the
frozen vessel labels + assignment-source badges, and carries ZERO
quantitative readouts: the only numbers in the document are the simplified
2D schematic coordinates (Rule C/D: no measurement, no LLM near geometry, no
anatomy inference - AI never owns measurement logic and the schematic claims
no anatomical view).

Two honest coordinate statements are kept apart (overlay.py precedent):

* the sealed CENTERLINE payload geometry stays verbatim in its own payload;
* ``polyline`` entries here are DISPLAY-ONLY 2D coordinates under the fixed
  ``ORTHOGRAPHIC_WORLD_XY_V1`` rule (world-X -> schematic-u, world-Y ->
  schematic-v).  The rule names the math, never an anatomical plane: the
  schematic is explicitly NOT an axial/coronal/sagittal view.

Exactly the ACTIVE labeled CENTERLINE artifacts are shown (Rule F: a human
correction invalidates downstream derived results, so REJECTED/SUPERSEDED/
STALE predecessors are excluded by status).  Zero / stale-only / malformed
paths are typed, visible refusals - never a silent pick and never a stale
fallback:

* ``TREE_SCHEMATIC_UNAVAILABLE`` - no ACTIVE labeled CENTERLINE in session,
  or the ACTIVE set is only PARTIALLY navigable (one unlabeled ACTIVE path
  among labeled ones);
* ``TREE_SCHEMATIC_INVALID`` - malformed payload, a non-frozen label, a
  non-frozen selection, or a path the fixed projection cannot depict;
* ``TREE_SCHEMATIC_STALE`` - centerline paths exist but none is ACTIVE
  anymore (an EDIT/REJECT successor is current).

ALL-OR-ABSTAIN (Rule 10, and the F4 remediation): the derivation is
navigable as a whole or it is absent.  A schematic that shows only part of
the ACTIVE set, or that draws a path the ``ORTHOGRAPHIC_WORLD_XY_V1`` rule
collapses to fewer than two distinct positions, would render as a
current-looking diagram of something the server did not derive - so both
abstain with an explicit refusal instead.  No partial schematic, no invented
vessel shape, no silent perturbation of collapsed geometry.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Final, cast

from apps.viewer.schemas import ULID_RE
from core.artifacts import DerivedArtifact
from core.coronary import (
    VESSEL_ASSIGNMENT_SOURCES,
    VESSEL_LABELS,
    validate_centerline_payload,
    validate_vessel_assignment,
)
from core.digest import canonical_digest
from core.enums import ArtifactKind, ArtifactStatus, ReviewState

#: The closed tree-schematic document schema.
TREE_SCHEMATIC_SCHEMA: Final[str] = "ccta.viewer.tree_schematic.v1"
#: The mandatory document kind marker (every payload + caption is SCHEMATIC).
TREE_SCHEMATIC_KIND: Final[str] = "SCHEMATIC"
#: The mandatory visible caption (display-only; fixed so a schematic can
#: never be confused with anatomy).
TREE_SCHEMATIC_CAPTION: Final[str] = "SCHEMATIC \u2014 not anatomy"
#: The frozen fixed-orientation projection rule recorded in every document.
#: World-X maps to schematic-u, world-Y to schematic-v; the rule names the
#: math, never an anatomical plane.
TREE_SCHEMATIC_PROJECTION_RULE: Final[str] = "ORTHOGRAPHIC_WORLD_XY_V1"
#: The frozen simplification rule recorded in every document: deterministic
#: stride decimation keeping the first and last points, capped at
#: ``TREE_SCHEMATIC_MAX_POINTS`` points per vessel.
TREE_SCHEMATIC_SIMPLIFICATION_RULE: Final[str] = "STRIDE_DECIMATION_MAX_32_V1"
#: The simplification cap (display-only point budget per vessel polyline).
TREE_SCHEMATIC_MAX_POINTS: Final[int] = 32
#: The explicit badge when no bound vessel-assignment record is present in
#: the session.  A visible marker - never a fabricated source, never bare
#: anatomy (Decision B: the UI always badges source).
TREE_SCHEMATIC_BADGE_UNRECORDED: Final[str] = "UNRECORDED"
#: The canonical reference prefix of a CENTERLINE artifact.
CENTERLINE_REF_PREFIX: Final[str] = "centerline_artifact_id:"

_DOCUMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "session_id",
        "kind",
        "caption",
        "projection_rule",
        "simplification_rule",
        "selected_vessel",
        "vessels",
    }
)

_VESSEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "vessel_label",
        "assignment_source",
        "centerline_ref",
        "provenance_ref",
        "review_state",
        "status",
        "polyline",
    }
)

_ULID_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_CENTERLINE_REF_RE: Final[re.Pattern[str]] = re.compile(
    rf"centerline_artifact_id:[{_ULID_CHARS}]{{26}}\Z"
)
_PROVENANCE_REF_RE: Final[re.Pattern[str]] = re.compile(rf"provenance:[{_ULID_CHARS}]{{26}}\Z")

#: The closed typed-refusal vocabulary of the schematic read surface.
REFUSAL_UNAVAILABLE: Final[str] = "TREE_SCHEMATIC_UNAVAILABLE"
REFUSAL_INVALID: Final[str] = "TREE_SCHEMATIC_INVALID"
REFUSAL_STALE: Final[str] = "TREE_SCHEMATIC_STALE"


class TreeSchematicRefusalError(ValueError):
    """A typed, fail-closed schematic refusal (HTTP 409, never a fallback)."""

    http_status: Final[int] = 409

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")

    @property
    def payload(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": self.message}}


def _refuse(code: str, message: str) -> TreeSchematicRefusalError:
    return TreeSchematicRefusalError(code, message)


def project_to_schematic(
    points_world: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Project sealed world points to display-only schematic ``(u, v)`` pairs.

    The fixed-orientation rule is total and deterministic: ``u`` is world-X,
    ``v`` is world-Y (DICOM LPS millimetres, verbatim floats).  Same inputs
    => same outputs; no RNG, no view-state dependence.  Every value must be
    finite (a non-finite sealed coordinate is an INVALID payload upstream).
    """

    projected: list[list[float]] = []
    for index, point in enumerate(points_world):
        values = list(point)
        if len(values) != 3:
            raise _refuse(
                REFUSAL_INVALID,
                f"points_world[{index}] is not a 3-vector",
            )
        pair: list[float] = []
        for axis in (0, 1):
            value = values[axis]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise _refuse(
                    REFUSAL_INVALID,
                    f"points_world[{index}][{axis}] must be finite",
                )
            pair.append(float(value))
        projected.append(pair)
    return projected


def simplify_polyline(
    polyline: Sequence[Sequence[float]], max_points: int = TREE_SCHEMATIC_MAX_POINTS
) -> list[list[float]]:
    """Deterministically decimate one schematic polyline to ``max_points``.

    Stride decimation keeping the first and last points: ``n <= max_points``
    is verbatim; otherwise ``max_points`` evenly spaced indices
    (``round(i * (n - 1) / (max_points - 1))``, deduplicated, endpoints
    pinned).  Same inputs => same outputs; no RNG, no geometry inference.
    """

    points = [[float(value) for value in pair] for pair in polyline]
    if max_points < 2:
        raise _refuse(REFUSAL_INVALID, "max_points must be at least 2")
    count = len(points)
    if count <= max_points:
        return points
    indices: list[int] = []
    for position in range(max_points):
        index = int(round(position * (count - 1) / (max_points - 1)))
        index = min(max(index, 0), count - 1)
        if not indices or indices[-1] != index:
            indices.append(index)
    if indices[0] != 0:
        indices.insert(0, 0)
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return [points[index] for index in indices]


def resolve_assignment_source(
    payload_documents: Mapping[str, Mapping[str, object]],
    centerline_id: str,
    centerline_payload: Mapping[str, object],
) -> str:
    """The badge source of one centerline, read-only from sealed truth.

    A badge is a CLAIM about this centerline, so the WHOLE association is
    validated, never only the two identifiers that happen to be cheapest to
    compare.  A record binds this centerline only when all four hold:

    * ``centerline_ref`` names this artifact (the canonical reference);
    * ``centerline_payload_sha256`` equals the sealed payload digest (a
      superseded payload after EDIT unbinds its old record);
    * ``vessel_label`` equals the centerline's OWN frozen label - a record
      naming another vessel is a claim about that other vessel, whatever its
      digest says, so it can never badge this one;
    * ``source_volume_ref`` equals the centerline's own source volume - the
      assignment must be bound to the volume the geometry came from.

    Every fully matching record is collected before a badge is decided.  Two
    matching records that disagree about ``assignment_source`` are an
    unresolved conflict, not a licence to pick one: they abstain too.  The
    answer is therefore decided by the sealed record set alone, never by
    mapping iteration order.  Abstention is the explicit ``UNRECORDED``
    marker - never a fabricated source, never bare anatomy.
    """

    try:
        current_sha = canonical_digest(dict(centerline_payload))
    except Exception:  # noqa: BLE001 - an undigestable payload binds nothing
        return TREE_SCHEMATIC_BADGE_UNRECORDED
    wanted_label = centerline_payload.get("vessel_label")
    wanted_volume = centerline_payload.get("source_volume_ref")
    if not isinstance(wanted_label, str) or not isinstance(wanted_volume, str):
        # An unlabeled / unbound centerline carries no association to confirm.
        return TREE_SCHEMATIC_BADGE_UNRECORDED
    wanted_ref = f"{CENTERLINE_REF_PREFIX}{centerline_id}"
    sources: set[str] = set()
    for document in payload_documents.values():
        if not isinstance(document, Mapping):
            continue
        try:
            record = validate_vessel_assignment(document)
        except Exception:  # noqa: BLE001 - non-assignment documents cannot bind
            continue
        if record.get("centerline_ref") != wanted_ref:
            continue
        if record.get("centerline_payload_sha256") != current_sha:
            continue
        if record.get("vessel_label") != wanted_label:
            continue
        if record.get("source_volume_ref") != wanted_volume:
            continue
        source = record.get("assignment_source")
        if isinstance(source, str) and source in VESSEL_ASSIGNMENT_SOURCES:
            sources.add(source)
    if len(sources) != 1:
        # Zero matching records (nothing recorded) or several disagreeing ones
        # (an unresolved conflict): both abstain.  Guessing is not allowed.
        return TREE_SCHEMATIC_BADGE_UNRECORDED
    return sources.pop()


def _distinct_projected_positions(polyline: Sequence[Sequence[float]]) -> int:
    """How many DISTINCT ``(u, v)`` positions one polyline actually occupies.

    Point count and finiteness are not enough: a 3-D path can project onto a
    single schematic position under ``ORTHOGRAPHIC_WORLD_XY_V1`` and still
    satisfy both.  Such a polyline draws as a zero-length stroke, so the
    count is what decides whether the path can be depicted at all.
    """

    return len({(float(pair[0]), float(pair[1])) for pair in polyline})


def build_tree_schematic_document(
    *,
    session_id: str,
    artifacts: Sequence[DerivedArtifact],
    payload_documents: Mapping[str, Mapping[str, object]],
    selected_vessel: str | None = None,
) -> dict[str, object]:
    """Build the closed ``ccta.viewer.tree_schematic.v1`` document.

    Derived response-only from the session's ACTIVE CENTERLINE set: each
    labeled ACTIVE path contributes its frozen ``vessel_label``, its
    assignment-source badge, its sealed refs, and its simplified display-only
    polyline.  The result is ALL-OR-ABSTAIN: an unlabeled ACTIVE path cannot
    be navigated to (the closed SELECT_VESSEL grammar admits only frozen
    labels) and is therefore a hole in the navigable coverage, so ANY
    unlabeled ACTIVE path refuses the whole document; and a path the fixed
    projection collapses to fewer than two distinct positions cannot be
    depicted truthfully, so it refuses too.  Both name the cause explicitly
    in the refusal - never a partial document that reads as current, never an
    invented vessel shape.
    """

    if not isinstance(session_id, str) or ULID_RE.fullmatch(session_id) is None:
        raise _refuse(REFUSAL_INVALID, "schematic session_id must be a ULID")
    if selected_vessel is not None and (
        not isinstance(selected_vessel, str) or selected_vessel not in VESSEL_LABELS
    ):
        raise _refuse(
            REFUSAL_INVALID,
            f"selected_vessel {selected_vessel!r} is outside the frozen VESSEL:* tokens",
        )
    centerlines = [artifact for artifact in artifacts if artifact.kind is ArtifactKind.CENTERLINE]
    active = [artifact for artifact in centerlines if artifact.status is ArtifactStatus.ACTIVE]
    vessels: list[dict[str, object]] = []
    unlabeled: list[str] = []
    for artifact in active:
        document = payload_documents.get(artifact.artifact_id)
        if document is None:
            raise _refuse(
                REFUSAL_INVALID,
                f"CENTERLINE {artifact.artifact_id} has no payload document",
            )
        try:
            validated = validate_centerline_payload(document)
        except Exception as exc:  # noqa: BLE001 - a refused document is typed
            raise _refuse(
                REFUSAL_INVALID,
                f"CENTERLINE {artifact.artifact_id} payload is invalid ({exc})",
            ) from None
        label = validated.get("vessel_label")
        if label is None:
            unlabeled.append(artifact.artifact_id)
            continue
        if not isinstance(label, str) or label not in VESSEL_LABELS:
            raise _refuse(
                REFUSAL_INVALID,
                f"CENTERLINE {artifact.artifact_id} vessel_label is outside "
                "the frozen VESSEL:* tokens",
            )
        sealed_points = cast(list[list[float]], validated["points_world"])
        simplified = simplify_polyline(project_to_schematic(sealed_points))
        if _distinct_projected_positions(simplified) < 2:
            raise _refuse(
                REFUSAL_INVALID,
                f"CENTERLINE {artifact.artifact_id} projects to fewer than two distinct "
                f"positions under {TREE_SCHEMATIC_PROJECTION_RULE} and cannot be drawn "
                "truthfully as a schematic path",
            )
        vessels.append(
            {
                "vessel_label": label,
                "assignment_source": resolve_assignment_source(
                    payload_documents, artifact.artifact_id, validated
                ),
                "centerline_ref": f"{CENTERLINE_REF_PREFIX}{artifact.artifact_id}",
                "provenance_ref": str(validated["provenance_ref"]),
                "review_state": artifact.review.review_state.value,
                "status": artifact.status.value,
                "polyline": simplified,
            }
        )
    if unlabeled:
        raise _refuse(
            REFUSAL_UNAVAILABLE,
            "the ACTIVE CENTERLINE set is not fully navigable: "
            f"{len(unlabeled)} unlabeled ACTIVE path(s) carry no frozen VESSEL:* label "
            f"({', '.join(sorted(unlabeled))}); the closed SELECT_VESSEL grammar cannot "
            "reach them, so a partial schematic would read as complete",
        )
    if not vessels:
        if centerlines:
            stale = sorted(
                (
                    str(artifact.artifact_id),
                    getattr(artifact.status, "value", artifact.status),
                )
                for artifact in centerlines
            )
            detail = ", ".join(f"{artifact_id} is {status}" for artifact_id, status in stale)
            raise _refuse(
                REFUSAL_STALE,
                "no ACTIVE CENTERLINE artifact is present; the predecessor "
                f"path(s) are stale (EDIT/REJECT successor is current): {detail}",
            )
        raise _refuse(
            REFUSAL_UNAVAILABLE,
            "no ACTIVE CENTERLINE artifact is present in this session",
        )
    vessels.sort(key=lambda entry: str(entry["vessel_label"]))
    schematic: dict[str, object] = {
        "schema_version": TREE_SCHEMATIC_SCHEMA,
        "session_id": session_id,
        "kind": TREE_SCHEMATIC_KIND,
        "caption": TREE_SCHEMATIC_CAPTION,
        "projection_rule": TREE_SCHEMATIC_PROJECTION_RULE,
        "simplification_rule": TREE_SCHEMATIC_SIMPLIFICATION_RULE,
        "selected_vessel": selected_vessel,
        "vessels": vessels,
    }
    return validate_tree_schematic_document(schematic)


def validate_tree_schematic_document(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the closed schematic document exactly (fail-closed)."""

    if not isinstance(document, Mapping):
        raise _refuse(REFUSAL_INVALID, "schematic document must be an object")
    if any(not isinstance(key, str) for key in document):
        raise _refuse(REFUSAL_INVALID, "schematic document keys must be strings")
    keys = set(document)
    missing = sorted(_DOCUMENT_FIELDS - keys)
    unknown = sorted(keys - _DOCUMENT_FIELDS)
    if missing or unknown:
        raise _refuse(
            REFUSAL_INVALID,
            f"schematic document key set is not closed (missing={missing}, unknown={unknown})",
        )
    if document["schema_version"] != TREE_SCHEMATIC_SCHEMA:
        raise _refuse(REFUSAL_INVALID, "schematic schema_version mismatch")
    session_id = document["session_id"]
    if not isinstance(session_id, str) or ULID_RE.fullmatch(session_id) is None:
        raise _refuse(REFUSAL_INVALID, "schematic session_id must be a ULID")
    if document["kind"] != TREE_SCHEMATIC_KIND:
        raise _refuse(REFUSAL_INVALID, "schematic kind must be SCHEMATIC")
    if document["caption"] != TREE_SCHEMATIC_CAPTION:
        raise _refuse(REFUSAL_INVALID, "schematic caption is fixed")
    if document["projection_rule"] != TREE_SCHEMATIC_PROJECTION_RULE:
        raise _refuse(REFUSAL_INVALID, "schematic projection_rule is fixed")
    if document["simplification_rule"] != TREE_SCHEMATIC_SIMPLIFICATION_RULE:
        raise _refuse(REFUSAL_INVALID, "schematic simplification_rule is fixed")
    selected = document["selected_vessel"]
    if selected is not None and (not isinstance(selected, str) or selected not in VESSEL_LABELS):
        raise _refuse(
            REFUSAL_INVALID,
            "schematic selected_vessel is outside the frozen VESSEL:* tokens",
        )
    vessels = document["vessels"]
    if not isinstance(vessels, list) or not vessels:
        raise _refuse(REFUSAL_INVALID, "schematic vessels must be a non-empty list")
    seen: set[str] = set()
    for position, entry in enumerate(vessels):
        if not isinstance(entry, Mapping) or set(entry) != _VESSEL_FIELDS:
            raise _refuse(
                REFUSAL_INVALID,
                f"schematic vessels[{position}] is not a closed vessel entry",
            )
        label = entry["vessel_label"]
        if not isinstance(label, str) or label not in VESSEL_LABELS:
            raise _refuse(
                REFUSAL_INVALID,
                f"schematic vessels[{position}] vessel_label is outside the frozen VESSEL:* tokens",
            )
        if label in seen:
            raise _refuse(
                REFUSAL_INVALID,
                f"schematic vessels[{position}] duplicates vessel_label {label!r}",
            )
        seen.add(label)
        source = entry["assignment_source"]
        if not isinstance(source, str) or (
            source not in VESSEL_ASSIGNMENT_SOURCES and source != TREE_SCHEMATIC_BADGE_UNRECORDED
        ):
            raise _refuse(
                REFUSAL_INVALID,
                f"schematic vessels[{position}] assignment_source is outside "
                "the closed source vocabulary",
            )
        if (
            not isinstance(entry["centerline_ref"], str)
            or _CENTERLINE_REF_RE.fullmatch(str(entry["centerline_ref"])) is None
        ):
            raise _refuse(
                REFUSAL_INVALID,
                f"schematic vessels[{position}] centerline_ref has the wrong canonical form",
            )
        if (
            not isinstance(entry["provenance_ref"], str)
            or _PROVENANCE_REF_RE.fullmatch(str(entry["provenance_ref"])) is None
        ):
            raise _refuse(
                REFUSAL_INVALID,
                f"schematic vessels[{position}] provenance_ref has the wrong canonical form",
            )
        review_state = entry["review_state"]
        if not isinstance(review_state, str) or review_state not in {
            state.value for state in ReviewState
        }:
            raise _refuse(
                REFUSAL_INVALID,
                f"schematic vessels[{position}] review_state is outside the closed enum",
            )
        status = entry["status"]
        if not isinstance(status, str) or status not in {value.value for value in ArtifactStatus}:
            raise _refuse(
                REFUSAL_INVALID,
                f"schematic vessels[{position}] status is outside the closed enum",
            )
        polyline = entry["polyline"]
        if (
            not isinstance(polyline, list)
            or len(polyline) < 2
            or len(polyline) > TREE_SCHEMATIC_MAX_POINTS
        ):
            raise _refuse(
                REFUSAL_INVALID,
                f"schematic vessels[{position}] polyline must carry 2.."
                f"{TREE_SCHEMATIC_MAX_POINTS} simplified points",
            )
        for index, pair in enumerate(polyline):
            if not isinstance(pair, list) or len(pair) != 2:
                raise _refuse(
                    REFUSAL_INVALID,
                    f"schematic vessels[{position}] polyline[{index}] must be a pair",
                )
            for axis, value in enumerate(pair):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise _refuse(
                        REFUSAL_INVALID,
                        f"schematic vessels[{position}] polyline[{index}][{axis}] must be finite",
                    )
        if _distinct_projected_positions(polyline) < 2:
            raise _refuse(
                REFUSAL_INVALID,
                f"schematic vessels[{position}] polyline occupies fewer than two distinct "
                f"schematic positions under {TREE_SCHEMATIC_PROJECTION_RULE} and cannot be "
                "drawn truthfully",
            )
    ordered = sorted(seen)
    if [str(entry["vessel_label"]) for entry in vessels] != ordered:
        raise _refuse(REFUSAL_INVALID, "schematic vessels must be ordered by vessel_label")
    return dict(document)


__all__ = [
    "CENTERLINE_REF_PREFIX",
    "REFUSAL_INVALID",
    "REFUSAL_STALE",
    "REFUSAL_UNAVAILABLE",
    "TREE_SCHEMATIC_BADGE_UNRECORDED",
    "TREE_SCHEMATIC_CAPTION",
    "TREE_SCHEMATIC_KIND",
    "TREE_SCHEMATIC_MAX_POINTS",
    "TREE_SCHEMATIC_PROJECTION_RULE",
    "TREE_SCHEMATIC_SCHEMA",
    "TREE_SCHEMATIC_SIMPLIFICATION_RULE",
    "TreeSchematicRefusalError",
    "build_tree_schematic_document",
    "project_to_schematic",
    "resolve_assignment_source",
    "simplify_polyline",
    "validate_tree_schematic_document",
]
