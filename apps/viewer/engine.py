"""ViewerEngine: action execution, QC binding, and correction rebind.

Owns the viewer runtime state over a restored WS-02 session:

- LAYER-G frame lists built ONCE at open (immutable; no action mutates them);
- per-plane display state: current plane selection, current frame index, the
  current WL state (preset, custom v2, or raw identity mapping) with the
  current LAYER-R variant records minted under it, plus the display-only
  viewport;
- the append-only QC gate outcome ledger with ``qc_gate_state_ref``;
- nonce history (every seen nonce consumes it) and the accepted-action chain;
- one sealed audit event per action (accepted or rejected) into the WS-02
  ``PhiSafeAuditSink``.

Fail-closed session open (Part 6 table, 5 rows): unknown session id ->
``SESSION_NOT_FOUND`` with no state materialized; corrupt frame list (digest
mismatch vs ``frame_lists_ref``), dangling ``source_mpr_ref``, or missing
pixel blob -> ``SESSION_OPEN_FAILED`` with nothing rendered and zero partial
state; a stale/superseded QC ref -> open SUCCEEDS and the verbatim stale
outcome is displayed.  No open-failure path ever falls back to raw DICOM or
to a partial grid (Rule 10).

Correction routing (Part 6.1 (b), Rule F): ``REQUEST_CORRECTION`` enters the
QC_001-HUMAN append-only seam and NOTHING else.  On seam acceptance the
engine re-queries the consumer gate, advances ``qc_gate_state_ref``, and
marks the superseded outcome and its frame records STALE through the
append-only ledger (bytes retained, still queryable; prior frame records are
never rewritten).  Seam refusals surface as typed, visible failures.

Coronary single-path human seam (CORONARY_CENTERLINE_001 att0002, F5): the
SEPARATE ``REQUEST_CENTERLINE_DECISION`` action (ACCEPT / EDIT re-seed /
REJECT) routes into the coronary implementation's ``apply_coronary_decision``
over the working artifact store.  QC semantics are untouched - the QC action
keeps its own grammar and route byte-identical; a coronary EDIT mints a
HUMAN ``CORRECTION`` artifact plus a NEW ``CENTERLINE`` (never an in-place
edit) and its §44 provenance records.
"""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, cast

from apps.viewer.build_identity import (
    BuildIdentityError,
    repo_root_from_file,
    resolve_build_identity,
    validate_commit_sha,
    verify_commit_exists,
)
from apps.viewer.frames import (
    FrameListConstructionError,
    GeometryFrame,
    build_geometry_lists,
    identity_window_for_pixels,
    mint_render_variant,
    preset_window,
    render_png_for_record,
)
from apps.viewer.grammar import (
    NONCE_RE,
    ParsedAction,
    RejectionCode,
    SessionGrammarView,
    ViewerRejectionError,
    parse_action_body,
    validate_action,
)
from apps.viewer.overlay import (
    CORONARY_OVERLAY_ID,
    OVERLAY_DEFAULT_MODE,
    OVERLAY_IDS,
    OVERLAY_MODES,
    REFUSAL_UNAVAILABLE,
    CoronaryOverlayRefusalError,
    build_coronary_overlay_document,
    select_active_centerline,
)
from apps.viewer.presets import WL_PRESET_IDS, preset_by_id
from apps.viewer.schemas import (
    GATE_OUTCOME_SCHEMA,
    PLANES,
    SESSION_SCHEMA,
    STATE_SCHEMA,
    ViewerFrameRecord,
    ViewerSchemaError,
    ViewerSessionRecord,
    frame_lists_digest,
    validate_frame_document,
    validate_session_document,
)
from apps.viewer.transfer import to_binary32
from apps.workstation.session_store import (
    DerivedStateStoreManager,
    RestoredSession,
    SessionSnapshot,
)
from apps.workstation.volume_loader import CanonicalVolumeLoader
from core.artifacts import DerivedArtifact, DerivedArtifactStore, GeometryBase
from core.coordinates import CoordinateTransform, FrameSpec, invert_matrix
from core.coronary import (
    CORONARY_CLAIM_VERSION,
    CoronaryDecision,
    CoronaryRefusalCode,
    CoronaryValidationError,
    Seed,
    VesselnessParameters,
    apply_coronary_decision,
    build_centerline,
    centerline_provenance,
)
from core.dicom import read_dicom_instance
from core.dicom.qc import QCAssessmentArtifact, QCAssessmentPayload, QCScope
from core.digest import canonical_digest, canonical_json
from core.enums import (
    ActorType,
    ArtifactKind,
    ProducerType,
    ProvenanceEventType,
    QualityGrade,
    SpaceKind,
)
from core.errors import CanonicalValidationError, StateTransitionError
from core.events import EventLog, utc_now_iso
from core.ids import new_ulid
from core.provenance import (
    Actor,
    BlobRef,
    Environment,
    HumanEdit,
    InputRef,
    InvalidationDAG,
    NotApplicable,
    ProvenanceRecord,
    create_provenance,
)
from core.qc.gate import V2_VESSEL_IDS as _FROZEN_VESSEL_IDS
from core.qc.gate import real_qc_consumer_gate
from core.qc.human import (
    HumanEvidence,
    QCHumanRefusalError,
    create_human_correction,
    create_initial_human_assessment,
    select_effective_assessment,
)
from core.qc.types import QCInput
from visualization.mpr import decode_pixel_blob

_VIEWER_SOFTWARE: Final[str] = "ccta-viewer-0.1.0"
_VIEWER_CREATED_BY: Final[str] = "ccta_viewer"
_VIEWER_ACTOR_ID: Final[str] = "ccta-viewer"
_STATE_KEY_PREFIX: Final[str] = "viewer_state:"
_SESSION_KEY_PREFIX: Final[str] = "viewer_session:"
_AUDIT_ENTITY: Final[str] = "viewer_action"
_RAW_WL_PRESET_ID: Final[str | None] = None
#: The internal render-record archive document key (r2 R2-02): append-only,
#: implementation-internal state — NO frozen external frame/session schema
#: gains a key and the G/R identity model is unchanged.
_ARCHIVE_STATE_KEY: Final[str] = "render_archive"
#: F12 (spec Part 6.1 b.1, amendment V1.1): the response-only correction
#: capability disclosure — a SIBLING of ``outcome``, never modifying it.
_CORRECTION_CAPABILITIES_ACTION: Final[str] = "REQUEST_CORRECTION"
_CORRECTION_CAPABILITIES_ACTIONABLE: Final[tuple[str, ...]] = (
    "GOOD",
    "NOT_ANALYZABLE",
)
_CORRECTION_CAPABILITIES_DEFERRED: Final[tuple[str, ...]] = (
    "ACCEPTABLE",
    "LIMITED",
)
_CORRECTION_CAPABILITIES_DEFERRED_REASON: Final[str] = (
    "EVIDENCE_FLAGS_NOT_IN_V1_GRAMMAR"
)
#: T2 physician/manual seed route (Phase A, server side): the dedicated
#: ``POST /session/:id/seed`` envelope is exactly these keys — voxel plus
#: direction plus the nonce/lineage pair of the SELECT_VESSEL pattern.  It is
#: NOT a grammar action (``ACTIONS`` stays the closed 11).
_SEED_ENVELOPE_KEYS: Final[frozenset[str]] = frozenset(
    {"voxel", "direction", "client_nonce", "lineage"}
)
_SEED_LINEAGE_KEYS: Final[frozenset[str]] = frozenset(
    {"session_id", "parent_action_nonce"}
)
#: Evidence-ref prefix of the seed route's HUMAN correction lineage.  The
#: full ref carries the client nonce (``viewer-seed:<nonce>``), binding the
#: session/nonce lineage end to end beside the grammar decision prefix.
_SEED_EVIDENCE_PREFIX: Final[str] = "viewer-seed"
#: T2 Phase B: runtime marker of an INITIAL centerline build (first-build).
#: The reseed path records ``python-3.12 (viewer centerline decision)``; the
#: first-build path records this instead, so the §44 provenance of an initial
#: build is queryably distinct from a reseed successor (which additionally
#: carries a HUMAN CORRECTION artifact and a superseded predecessor).
_INITIAL_BUILD_RUNTIME: Final[str] = "python-3.12 (viewer centerline initial build)"

# ---------------------------------------------------------------------------
# F2_JOB_HANDLE_TRUTH: the seed-job lifecycle observation
# ---------------------------------------------------------------------------
#: F2_JOB_HANDLE_TRUTH: the served key of the lifecycle observation, a SIBLING
#: of the frozen ``ccta.viewer.state.v1`` keys.  It is ADDITIVE (a client that
#: does not read it behaves exactly as before) and it is served ONLY on the
#: state read route (:meth:`ViewerEngine.planes_state`) — never written into
#: the persisted state document, so no job survives a restart and there is no
#: job store, broker or queue subsystem behind it.
_SEED_JOBS_KEY: Final[str] = "seed_jobs"
#: The FROZEN lifecycle vocabulary: exactly these four states, named in the
#: F2_JOB_HANDLE_TRUTH contract.  No other state word may be emitted (no
#: "CANCELLED", "STOPPED", "FAILED", "DONE", "PARTIAL").  The vocabulary is
#: UNCHANGED by both F2_JOB_HANDLE_TRUTH narrowings: all four words are still
#: defined here and still the only words this module may ever emit.  What they
#: withdraw is the CLAIM, not the word — see the ``_SEED_JOB_COMPLETED`` entry
#: below: the server publishes only ``QUEUED``, ``RUNNING`` and ``REFUSED``, and
#: only while the execution that observed the fact is still running.
_SEED_JOB_QUEUED: Final[str] = "QUEUED"
_SEED_JOB_RUNNING: Final[str] = "RUNNING"
_SEED_JOB_COMPLETED: Final[str] = "COMPLETED"
_SEED_JOB_REFUSED: Final[str] = "REFUSED"
#: What a state MEANS, keyed to the observed server-side fact it is derived
#: from.  Every served word describes ONE EXECUTION and is served only while
#: that execution is still in flight: the record is created where the server
#: observes the execution queued, and it is REMOVED when that execution returns
#: — on success, refusal, cancellation or exception alike — so nothing
#: outlives its execution and no publication rule is needed to decide an
#: entry's fate.  There is deliberately no duration, percentage, ratio,
#: remaining-time or completion-estimate field: the server measures no such
#: quantity, so it emits none (Rule C/D, Rule G).
_SEED_JOB_STATE_FACTS: Final[dict[str, str]] = {
    _SEED_JOB_QUEUED: (
        "the seed request is validated and the server owns it; the governed "
        "build has not been entered yet"
    ),
    _SEED_JOB_RUNNING: (
        "the governed build has been entered for this job and has not returned"
    ),
    _SEED_JOB_COMPLETED: (
        "NEVER SERVED, and no longer derived.  This word is the fourth frozen "
        "lifecycle word of the F2_JOB_HANDLE_TRUTH contract and it is UNCHANGED "
        "in the vocabulary, but the CLAIM it used to carry is WITHDRAWN.  The "
        "first narrowing (manager decision 2026-09-27, after the "
        "contract-specified reviewer's CHANGES_REQUESTED verdict) removed the "
        "identity-level completion, because no truthful identity-level terminal "
        "publication can be established with the current coordination between "
        "nonce consumption, the live-execution registry and the serve path.  The "
        "SECOND narrowing (manager decision 2026-09-28) then removed the whole "
        "hazard class with it: a served entry now exists only while the "
        "execution that observed it is running and is removed when that "
        "execution returns, so no record can outlive its execution and NO "
        "identity-level outcome of any kind — completed, settled or withheld "
        "until settled — can ever be published.  A real acceptance still happens "
        "in full and is still observable in the frozen document's own "
        "``accepted_nonces`` and ``action_count``; the handle simply no longer "
        "restates it as a state, and a seeded identity is left unreported once "
        "its execution returns"
    ),
    _SEED_JOB_REFUSED: (
        "the governed seed route refused this job; refusal_code is the EXISTING "
        "closed code the server raised (F2 admission RejectionCode, a coronary "
        "seam refusal code, or a fail-closed ViewerOpenError RejectionCode), "
        "always the code of the very refusal this served state reports.  It is "
        "what ONE execution observed, and it is served for exactly as long as "
        "that execution is in flight: the record is removed when the execution "
        "returns, so the identity then serves no job entry at all — whether the "
        "refusal consumed its nonce, and whether a later attempt of the same "
        "nonce succeeds.  The attempt that raised the refusal always received "
        "that exact code in its own response body, so nothing is hidden from "
        "the execution that earned it"
    ),
}
#: The ONE frozen vocabulary, derived from the ONE derivation table: those
#: four lifecycle words, and nothing else, may ever be served.  The table is
#: still the single source of the vocabulary, so a fifth word cannot appear
#: without appearing here first.
_SEED_JOB_STATES: Final[tuple[str, ...]] = tuple(_SEED_JOB_STATE_FACTS)
#: Forward-only order of the OBSERVATIONS OF ONE EXECUTION.  An observation
#: never regresses, so every served state is a real forward move of the
#: execution that recorded it.  Publication needs no rule of its own: every
#: record in the served set belongs to an execution that has not returned yet,
#: because the route-exit hook removes the record of any execution that does
#: return (:meth:`ViewerEngine._seed_job_release`), so the served word is the
#: live execution's own latest observation and nothing more.  A completion is
#: NEVER published for any identity on any path (the withdrawn
#: identity-level claim — see the ``_SEED_JOB_COMPLETED`` entry).
_SEED_JOB_ORDER: Final[dict[str, int]] = {
    _SEED_JOB_QUEUED: 0,
    _SEED_JOB_RUNNING: 1,
    _SEED_JOB_COMPLETED: 2,
    _SEED_JOB_REFUSED: 2,
}


class ViewerOpenError(ValueError):
    """A fail-closed viewer session-open failure (Part 6 table)."""

    def __init__(self, code: RejectionCode, problems: list[str]) -> None:
        self.code = code
        self.problems = list(problems)
        super().__init__(f"{code.value}: {'; '.join(self.problems)}")

    @property
    def payload(self) -> dict[str, object]:
        return {"error": {"code": self.code.value, "message": "; ".join(self.problems)}}


class HumanSeamRefusalError(ValueError):
    """A typed refusal raised by the QC_001-HUMAN seam during routing."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")

    @property
    def payload(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": self.message}}


@dataclass(frozen=True, slots=True)
class SeedRequest:
    """One validated ``POST /session/:id/seed`` request (T2 Phase A).

    ``voxel`` is the physician-picked ``(i, j, k)`` index (i -> columns,
    j -> rows, k -> slices); ``direction`` is the physician-set world-frame
    direction hint.  Both are REQUIRED — the route mints no hidden defaults.
    """

    voxel: tuple[int, int, int]
    direction: tuple[float, float, float]
    client_nonce: str
    session_id: str
    parent_action_nonce: str | None


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """One bound consumer-gate outcome record (displayed VERBATIM)."""

    gate_id: str
    decision: str
    reason_code: str | None
    assessment_refs: tuple[str, ...]
    consumed_refs: tuple[str, ...]
    bound_at: str
    superseded_by: str | None = None

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": GATE_OUTCOME_SCHEMA,
            "gate_id": self.gate_id,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "assessment_refs": list(self.assessment_refs),
            "consumed_refs": list(self.consumed_refs),
            "bound_at": self.bound_at,
            "superseded_by": self.superseded_by,
        }

    @staticmethod
    def from_document(document: dict[str, object]) -> GateOutcome:
        if document.get("schema_version") != GATE_OUTCOME_SCHEMA:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED, ["gate outcome schema mismatch"]
            )
        return GateOutcome(
            gate_id=str(document.get("gate_id", "")),
            decision=str(document.get("decision", "")),
            reason_code=document.get("reason_code"),  # type: ignore[arg-type]
            assessment_refs=tuple(document.get("assessment_refs", ())),  # type: ignore[arg-type]
            consumed_refs=tuple(document.get("consumed_refs", ())),  # type: ignore[arg-type]
            bound_at=str(document.get("bound_at", "")),
            superseded_by=document.get("superseded_by"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class PlaneRenderState:
    """The current LAYER-R generation of one plane."""

    wl_preset_id: str | None
    records: tuple[ViewerFrameRecord, ...]

    def to_document(self) -> dict[str, object]:
        return {
            "wl_preset_id": self.wl_preset_id,
            "records": [record.to_document() for record in self.records],
        }


@dataclass(frozen=True, slots=True)
class SeedJobRecord:
    """One OBSERVED seed-job lifecycle state (F2_JOB_HANDLE_TRUTH).

    A record is the state ONE running seed execution has observed, and it
    lives exactly as long as that execution: it is created where the server
    observes the execution queued, and REMOVED by the route-exit hook when that
    execution returns — on success, refusal, cancellation or exception alike
    (:meth:`ViewerEngine._seed_job_release`).  Nothing outlives its execution,
    so the served set is bounded by the in-flight executions and no served
    entry can describe anything that has already finished.

    The identity is the EXISTING pair ``(session_id, client_nonce)`` of the
    seed request — no new id scheme.  ``state`` is one of the four frozen
    lifecycle words and every one of them is set at the point where the
    server OBSERVED the fact it names (``_SEED_JOB_STATE_FACTS``); none is
    inferred from elapsed time, a retry count, or a timeout.  ``refusal_code``
    is the EXISTING closed refusal code the server raised and is present only
    on ``REFUSED``.  The served document carries no percentage, progress
    ratio, ETA, remaining time or duration of any kind: the server measures
    none of those, so it asserts none of them.

    ``owner`` is the request EXECUTION that currently holds this public
    handle, held by object identity (``is``) and excluded from value
    comparison, from ``repr`` and from the served document.  It is what binds
    every observation and every release to the execution that actually owns the
    work: a DIFFERENT request that happens to carry the same
    ``client_nonce`` (a replay, a reuse, a second concurrent duplicate) is a
    DIFFERENT execution, so it can never transition, mutate or delete this
    record, and a record can never exist without an owner.  Ownership is
    therefore never forgeable or re-takable through a nonce: the ONE transfer is
    the admission award of ATT0003/ATT0004 below, which only a request the real
    F2 rule admitted can perform, and which hands over the HANDLE without
    handing over anything the former owner observed.

    ATT0003 (review finding R2-F1): a validated request holds the handle only
    PROVISIONALLY.  Creating the record and being admitted to run the governed
    build are two different events, so the request that created the ``QUEUED``
    record is not necessarily the one that really builds.  ``admitted`` records
    that ``owner`` is the execution the governed F2 admission really awarded
    the handle to: :meth:`ViewerEngine._seed_job_admit_owner` sets it, and
    moves ``owner`` to that execution, INSIDE the admission critical section,
    and it never clears.  ``admitted`` is an internal marker — NOT a lifecycle
    state, never served, never persisted — and it exists so that a request
    which merely created the record can never keep (or re-take) the public
    handle once a different execution really won the admission.

    F2_JOB_HANDLE_TRUTH, second narrowing (manager decision 2026-09-28): the
    served claim is now exactly "this execution's own observed state, while it
    is running".  The first narrowing had already withdrawn the identity-level
    completion claim, which required proving that a recorded outcome could not
    be overturned by another execution; the second removes the necessity
    instead of proving it.  A record is never retained after its execution
    returns, so there is no question of settling, finality or overturning: the
    per-execution record is deleted by its own owner's route exit and the
    identity then serves no job entry at all.  Consequently none of the
    machinery that existed only to decide a record's fate after its execution
    ended is load-bearing any more — no irreversibility predicate, no
    cross-execution withholding rule, no terminal-retention or supersession
    choice, and no acceptance or nonce-ledger read on the serve path.  The
    served ``refusal_code`` is still always the code of the refusal the served
    state reports, because it is still the very record the serving execution
    recorded.  Nothing was added by any of this — no served key, no new state
    word, no store, no dependency.
    """

    session_id: str
    client_nonce: str
    state: str
    owner: SeedRequest = field(compare=False, repr=False)
    admitted: bool = field(compare=False, repr=False, default=False)
    refusal_code: str | None = None

    def to_document(self) -> dict[str, object]:
        """The served per-job document (closed key set per state)."""

        document: dict[str, object] = {
            "session_id": self.session_id,
            "client_nonce": self.client_nonce,
            "state": self.state,
        }
        if self.state == _SEED_JOB_REFUSED:
            document["refusal_code"] = self.refusal_code
        return document


@dataclass
class ViewerRuntimeState:
    """The mutable-per-action viewer state (persisted as one document)."""

    session_id: str
    current_plane: str
    frame_index_by_plane: dict[str, int]
    viewport_by_plane: dict[str, dict[str, float]]
    renders: dict[str, PlaneRenderState]
    gate_outcomes: list[GateOutcome]
    used_nonces: list[str]
    accepted_nonces: list[str]
    stale_frame_ids: list[str]
    qc_ledger: list[dict[str, str]] = field(default_factory=list)
    action_count: int = 0
    qc_panel_open: bool = False
    #: R2-02 internal append-only archive: public provenance ULID -> the
    #: complete ten-key frame document plus the internal frame id.  Every
    #: minted record is archived BEFORE it replaces any plane's current
    #: tuple; superseded records stay queryable through it.
    render_archive: dict[str, ViewerFrameRecord] = field(default_factory=dict)
    #: CORONARY_CENTERLINE_001: closed overlay-visibility state of the one
    #: single-path coronary overlay (one entry per frozen overlay id; the
    #: default is visible).  Persisted with the viewer state document.
    overlay_visibility: dict[str, bool] = field(
        default_factory=lambda: {overlay_id: True for overlay_id in OVERLAY_IDS}
    )
    #: CORONARY_REVIEW_VIEW_V1 (Decision A): the closed display mode of the
    #: one overlay (``SLICE_LOCAL`` default; ``PROJECTION`` is the explicitly
    #: labeled localizer mode).  Persisted with the viewer state document.
    overlay_mode: str = OVERLAY_DEFAULT_MODE
    #: CORONARY_MULTI_VESSEL_NAV (Decision A): the selected vessel of the
    #: review surface (a frozen ``VESSEL:*`` token, or ``None`` for the
    #: legacy #28 single-path behavior in which the selectors pick the
    #: single ACTIVE path).  Persisted with the viewer state document.
    selected_vessel: str | None = None

    def grammar_view(self, study_id: str) -> SessionGrammarView:
        return SessionGrammarView(
            session_id=self.session_id,
            study_id=study_id,
            frame_count_by_plane={
                plane: len(self.renders[plane].records) for plane in PLANES
            },
            wl_presets_active=tuple(WL_PRESET_IDS),
            used_nonces=frozenset(self.used_nonces),
            accepted_nonces=frozenset(self.accepted_nonces),
        )


class ViewerEngine:
    """The viewer over one restored WS-02 session (no volume path of its own)."""

    def __init__(
        self,
        restored: RestoredSession,
        *,
        audit_sink: Any = None,
        id_factory: Any = None,
        now: Any = None,
        repo_root: Path | str | None = None,
    ) -> None:
        self._restored = restored
        self._sink = audit_sink
        self._id_factory = id_factory or new_ulid
        self._now = now or utc_now_iso
        self._repo_root = repo_root
        self._build_commit_sha: str | None = None
        self._study_id = f"study_id:{restored.study.study_id}"
        self._mpr_by_id: dict[str, DerivedArtifact] = {
            artifact.artifact_id: artifact
            for artifact in restored.artifacts
            if artifact.kind == ArtifactKind.MPR_VIEW
        }
        self._pixel_blobs: dict[str, bytes] = dict(restored.pixel_blobs)
        self._geometry: dict[str, list[GeometryFrame]] | None = None
        self._session_record: ViewerSessionRecord | None = None
        self._state: ViewerRuntimeState | None = None
        self._new_provenance: list[ProvenanceRecord] = []
        self._new_qc_artifacts: list[DerivedArtifact] = []
        self._new_qc_documents: dict[str, dict[str, object]] = {}
        #: CORONARY_CENTERLINE_001 (att0002, F5): CENTERLINE/CORRECTION
        #: artifacts minted by one REQUEST_CENTERLINE_DECISION EDIT, plus their
        #: sealed payload documents.  They live in the working correction store
        #: so the overlay read surface reflects the decision immediately.
        self._new_coronary_artifacts: list[str] = []
        self._new_coronary_documents: dict[str, dict[str, object]] = {}
        self._correction_store: DerivedArtifactStore | None = None
        #: F2_SEED_ADMIT (D1): per-engine first-build admission lock (stdlib
        #: only).  Serializes ONLY the ``validate + nonce-consume admit`` step
        #: of the first-build path (``apply_seed_bytes`` -> ``_execute_seed``
        #: when no ACTIVE centerline exists); the long compute
        #: (``_execute_seed_first_build`` derivation + ``store.add``) runs
        #: OUTSIDE the lock so ThreadingHTTPServer concurrency for distinct
        #: sessions and state polls is preserved.  A second concurrent
        #: same-session first-build is refused with the existing closed
        #: vocabulary (LINEAGE_GAP — its parent lineage no longer names the
        #: current session tip).  The reseed path is untouched.  Single-
        #: threaded behavior is unchanged (the lock is uncontended).
        self._seed_admit_lock = threading.Lock()
        #: F2_SEED_ADMIT (D1): the client_nonce of the currently ADMITTED but
        #: not-yet-built first-build on this session (``None`` when idle).
        #: Read/written ONLY under ``_seed_admit_lock``.  Lets a second
        #: concurrent first-build be refused as the existing LINEAGE_GAP
        #: while the first build's long compute runs OUTSIDE the lock (a
        #: plain "re-check accepted_nonces after compute" cannot see an
        #: in-flight admission because the accepted chain only advances at
        #: completion).  Released in a ``finally`` so a derivation refusal
        #: never wedges the session; single-threaded runs see ``None`` and
        #: behave byte-identically.
        self._seed_first_build_inflight: str | None = None
        #: F2_JOB_HANDLE_TRUTH: the OBSERVED seed-job states of the seed
        #: executions that are IN FLIGHT on this session, keyed by the EXISTING
        #: ``client_nonce`` (the identity pair is
        #: ``(session_id, client_nonce)``).  This is a runtime observation set,
        #: NOT a job store: it is never persisted, never rehydrated, and never
        #: outlives the engine.  Each entry is written only at a point where the
        #: server observed the fact its state names, and it is DELETED when that
        #: execution returns — on success, refusal, cancellation or exception
        #: alike (:meth:`_seed_job_release`) — so an entry always belongs to an
        #: execution that is still running and the set is bounded by the
        #: in-flight executions (no history, no broker, no queue subsystem).
        #: Nothing outlives its execution, so serving needs no rule at all:
        #: whatever is in this set is what the live executions have observed.
        #: Read/written under ``_seed_job_lock`` only.  Every record is OWNED by
        #: the request execution that holds its public handle
        #: (:class:`SeedJobRecord`), so a duplicate or replayed nonce can never
        #: transition, mutate or delete it.  A validated request holds the
        #: handle only until the governed F2 first-build admission awards it,
        #: inside the admission critical section
        #: (:meth:`_seed_job_admit_owner`); the record's internal ``admitted``
        #: marker records that award so it can never move again.
        self._seed_job_lock = threading.Lock()
        self._seed_jobs: dict[str, SeedJobRecord] = {}

    # ------------------------------------------------------------------
    # Governed implementation identity (F11, r2)
    # ------------------------------------------------------------------

    def _require_build_commit_sha(self) -> str:
        """The validated checkout identity; resolved at open, before state.

        Cached only within this initialized engine/build context — never a
        value inferred from a user workspace, request data, or environment.
        """

        if self._build_commit_sha is None:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED,
                ["governed build identity has not been resolved"],
            )
        return self._build_commit_sha

    def _verify_persisted_viewer_provenance(self) -> None:
        """Fail closed on zero/nonexistent identity in VIEWER-produced records.

        Viewer-created persisted provenance (session-open, frame renders,
        HUMAN-routed corrections) must carry a governed nonzero producing
        commit that exists in this checkout.  Historical records keep their
        ORIGINAL producing SHA — reopen never relabels them with the current
        build identity, and a later runtime HEAD is not required to equal a
        historical producing SHA.  A zero sentinel injected into a persisted
        viewer-produced record fails verification here.
        """

        resolved_root = repo_root_from_file()
        if self._repo_root is not None and Path(self._repo_root).resolve() != resolved_root:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED,
                [
                    f"supplied repo_root {self._repo_root} does not match "
                    f"executing code root {resolved_root}"
                ],
            )
        root = resolved_root
        for record in self._restored.provenance:
            is_viewer_record = (
                record.created_by == _VIEWER_CREATED_BY
                or record.subject_ref.startswith(
                    ("viewer_session:", "viewer_frame:")
                )
                or (
                    record.actor.actor_id == _VIEWER_ACTOR_ID
                    and record.actor.actor_type is ActorType.ENGINE
                )
            )
            if not is_viewer_record:
                continue
            try:
                validate_commit_sha(record.environment.code_commit_sha)
                verify_commit_exists(record.environment.code_commit_sha, repo_root=root)
            except BuildIdentityError as exc:
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED,
                    [
                        "persisted viewer provenance "
                        f"{record.provenance_id} failed the governed "
                        f"identity check: {exc}"
                    ],
                ) from None

    # ------------------------------------------------------------------
    # Open (Part 5.1 + Part 6 fail-closed table)
    # ------------------------------------------------------------------

    def open(self) -> tuple[ViewerSessionRecord, ViewerRuntimeState]:
        """Build LAYER-G, bind the gate outcome, and materialize raw renders.

        A previously persisted viewer session is rehydrated verbatim (the
        bound ``qc_gate_state_ref`` is NOT re-queried at open — a stale or
        superseded ref opens successfully and displays verbatim, Part 6 row
        5).  A fresh viewer session starts with the raw identity window per
        plane.  Every failure raises before any state is exposed.
        """

        if self._state is not None and self._session_record is not None:
            return (self._session_record, self._state)

        try:
            # F11 (r2) / R3-01 (r3): the governed implementation identity is resolved and
            # verified from the code checkout BEFORE any open state is
            # materialized.  Any unresolvable/dirty state fails closed with
            # zero partial state (Part 6 table discipline).
            resolved_root = repo_root_from_file()
            if self._repo_root is not None and Path(self._repo_root).resolve() != resolved_root:
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED,
                    [
                        f"supplied repo_root {self._repo_root} does not match "
                        f"executing code root {resolved_root}"
                    ],
                )
            if self._build_commit_sha is None:
                try:
                    self._build_commit_sha = resolve_build_identity(self._repo_root)
                except BuildIdentityError as exc:
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        [f"governed build identity resolution failed: {exc}"],
                    ) from None
            self._verify_persisted_viewer_provenance()

            geometry = self._build_geometry()
            self._verify_geometry_resolvable(geometry)
            self._geometry = geometry

            persisted_session = self._load_persisted_document(
                _SESSION_KEY_PREFIX, validate_session_document
            )
            persisted_state = self._load_persisted_state_document()

            if persisted_session is not None:
                session_id = persisted_session.session_id
                digest = self._geometry_digest(session_id, geometry)
                if digest != persisted_session.frame_lists_ref:
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        [
                            "corrupt frame list: digest mismatch vs frame_lists_ref; "
                            "nothing rendered"
                        ],
                    )
                if persisted_state is None:
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        ["persisted viewer session without its state document"],
                    )
                state = self._rehydrate_state(persisted_session, persisted_state)
                self._session_record = persisted_session
                self._state = state
                return (self._session_record, self._state)

            if persisted_state is not None:
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED,
                    ["persisted viewer state without its session record"],
                )

            session_id = str(self._id_factory())
            outcome = self._query_gate()
            record = ViewerSessionRecord(
                schema_version=SESSION_SCHEMA,
                session_id=session_id,
                study_id=self._study_id,
                planes=tuple(PLANES),
                frame_lists_ref=self._geometry_digest(session_id, geometry),
                qc_gate_state_ref=outcome.gate_id,
                wl_presets_active=tuple(WL_PRESET_IDS),
                opened_at=self._now(),
                provenance_ref="provenance:" + "0" * 26,
            )
            state = ViewerRuntimeState(
                session_id=session_id,
                current_plane=PLANES[0],
                frame_index_by_plane={plane: 0 for plane in PLANES},
                viewport_by_plane={
                    plane: {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0} for plane in PLANES
                },
                renders={},
                gate_outcomes=[outcome],
                used_nonces=[],
                accepted_nonces=[],
                stale_frame_ids=[],
            )
            self._state = state
            for plane in PLANES:
                state.renders[plane] = self._mint_plane(plane, _RAW_WL_PRESET_ID)
            provenance = self._mint_session_provenance(state)
            self._new_provenance.append(provenance)
            self._session_record = replace(
                record, provenance_ref=f"provenance:{provenance.provenance_id}"
            )
            return (self._session_record, self._state)
        except Exception:
            self._session_record = None
            self._state = None
            self._geometry = None
            raise

    def _volume_scalar_domains(self) -> dict[str, str]:
        """The RECORDED canonical scalar domain of every restored volume.

        Read from the envelope each volume actually reopens with (the
        content-addressed canonical envelope, never from the blob and never
        defaulted).  A volume whose envelope carries no usable
        ``scalar_domain`` is deliberately ABSENT from the result: a
        record-free domain is its own state and the F18 gate refuses it, it
        is never treated as HU.
        """

        from apps.workstation.volume_loader import CanonicalVolumeLoader

        loader = CanonicalVolumeLoader(Path(self._restored.paths.blobs))
        domains: dict[str, str] = {}
        for volume in self._restored.volumes:
            try:
                envelope = loader.read_envelope(volume.blob_ref.sha256)
            except (OSError, ValueError, KeyError):
                continue
            domain = envelope.get("scalar_domain") if isinstance(envelope, dict) else None
            if isinstance(domain, str) and not isinstance(domain, bool) and domain:
                domains[volume.volume_id] = domain
        return domains

    def _build_geometry(self) -> dict[str, list[GeometryFrame]]:
        try:
            return build_geometry_lists(
                session_id=self._restored.study.study_id,
                artifacts=self._restored.artifacts,
                payload_documents=self._restored.payload_documents,
                volume_ids=tuple(volume.volume_id for volume in self._restored.volumes),
                volume_scalar_domains=self._volume_scalar_domains(),
            )
        except FrameListConstructionError as exc:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED, [str(exc)]
            ) from None

    def _geometry_digest(
        self, session_id: str, geometry: dict[str, list[GeometryFrame]]
    ) -> str:
        return frame_lists_digest(
            session_id,
            {
                plane: [(f.frame_index, f.mpr_artifact_id) for f in geometry[plane]]
                for plane in PLANES
            },
        )

    def _verify_geometry_resolvable(
        self, geometry: dict[str, list[GeometryFrame]]
    ) -> None:
        problems: list[str] = []
        for plane in PLANES:
            for frame in geometry[plane]:
                artifact = self._mpr_by_id.get(frame.mpr_artifact_id)
                if artifact is None:
                    problems.append(f"dangling source_mpr_ref {frame.mpr_artifact_id}")
                    continue
                payload = self._restored.payload_documents.get(artifact.artifact_id)
                if payload is None:
                    problems.append(
                        f"missing payload document for MPR {artifact.artifact_id}"
                    )
                    continue
                pixel_ref = payload.get("pixel_blob_ref")
                digest = pixel_ref.get("sha256") if isinstance(pixel_ref, dict) else None
                if not isinstance(digest, str) or digest not in self._pixel_blobs:
                    problems.append(
                        f"missing pixel blob for listed MPR {artifact.artifact_id}"
                    )
        if problems:
            raise ViewerOpenError(RejectionCode.SESSION_OPEN_FAILED, problems)

    def _load_persisted_document(
        self, prefix: str, validator: Any
    ) -> Any:
        found: list[Any] = []
        for key, document in self._restored.payload_documents.items():
            if key.startswith(prefix):
                found.append(validator(document))
        if len(found) > 1:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED,
                [f"multiple persisted documents under {prefix!r}"],
            )
        return found[0] if found else None

    def _load_persisted_state_document(self) -> dict[str, object] | None:
        documents = [
            document
            for key, document in self._restored.payload_documents.items()
            if key.startswith(_STATE_KEY_PREFIX)
        ]
        if len(documents) > 1:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED,
                ["multiple persisted viewer state documents"],
            )
        return documents[0] if documents else None

    def _rehydrate_state(
        self,
        record: ViewerSessionRecord,
        document: dict[str, object],
    ) -> ViewerRuntimeState:
        if document.get("schema_version") != STATE_SCHEMA:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED, ["viewer state schema mismatch"]
            )
        if document.get("session_id") != record.session_id:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED,
                ["viewer state session binding mismatch"],
            )
        raw_renders = document.get("renders")
        if not isinstance(raw_renders, dict):
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED, ["viewer state renders missing"]
            )
        unknown_planes = sorted(set(raw_renders) - set(PLANES))
        if unknown_planes:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED,
                [
                    "viewer state renders contain unknown extra planes "
                    f"{unknown_planes}"
                ],
            )
        renders: dict[str, PlaneRenderState] = {}
        for plane in PLANES:
            raw = raw_renders.get(plane)
            if not isinstance(raw, dict):
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED,
                    [f"viewer state renders[{plane}] missing or malformed"],
                )
            raw_records = raw.get("records")
            if not isinstance(raw_records, list):
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED,
                    [f"viewer state renders[{plane}].records missing or not a list"],
                )
            validated_records: list[ViewerFrameRecord] = []
            for item in raw_records:
                if not isinstance(item, dict):
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        [f"viewer state renders[{plane}] record is not an object"],
                    )
                try:
                    frame_record = validate_frame_document(item)
                except ViewerSchemaError as exc:
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        [f"viewer state renders[{plane}] record fails schema: {exc}"],
                    ) from None
                validated_records.append(frame_record)
            renders[plane] = PlaneRenderState(
                wl_preset_id=raw.get("wl_preset_id"), records=tuple(validated_records)
            )
        archive = self._rehydrate_render_archive(document, renders)
        outcomes: list[GateOutcome] = []
        raw_outcomes = document.get("gate_outcomes", [])
        if not isinstance(raw_outcomes, list):
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED, ["gate outcome ledger malformed"]
            )
        for raw_outcome in raw_outcomes:
            if not isinstance(raw_outcome, dict):
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED, ["gate outcome ledger malformed"]
                )
            outcomes.append(GateOutcome.from_document(raw_outcome))
        if not outcomes:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED, ["gate outcome ledger is empty"]
            )
        raw_indexes = document.get("frame_index_by_plane", {})
        if not isinstance(raw_indexes, dict):
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED,
                ["viewer state frame_index_by_plane malformed"],
            )
        frame_index_by_plane = {plane: int(values) for plane, values in raw_indexes.items()}
        raw_viewports = document.get("viewport_by_plane", {})
        if not isinstance(raw_viewports, dict):
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED,
                ["viewer state viewport_by_plane malformed"],
            )
        viewport_by_plane: dict[str, dict[str, float]] = {}
        for plane, values in raw_viewports.items():
            if not isinstance(values, dict):
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED,
                    ["viewer state viewport_by_plane malformed"],
                )
            viewport_by_plane[str(plane)] = {
                str(key): float(value) for key, value in values.items()
            }
        raw_used = document.get("used_nonces", [])
        raw_accepted = document.get("accepted_nonces", [])
        raw_stale = document.get("stale_frame_ids", [])
        raw_ledger = document.get("qc_ledger", [])
        used_nonces: list[str] = []
        accepted_nonces: list[str] = []
        stale_frame_ids: list[str] = []
        qc_ledger: list[dict[str, str]] = []
        if not isinstance(raw_used, list):
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED, ["viewer state nonce ledger malformed"]
            )
        if not isinstance(raw_accepted, list):
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED, ["viewer state nonce ledger malformed"]
            )
        if not isinstance(raw_stale, list):
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED, ["viewer state stale ledger malformed"]
            )
        if not isinstance(raw_ledger, list):
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED, ["viewer state qc ledger malformed"]
            )
        for item in raw_used:
            if not isinstance(item, str):
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED, ["viewer state nonce malformed"]
                )
            used_nonces.append(item)
        for item in raw_accepted:
            if not isinstance(item, str):
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED, ["viewer state nonce malformed"]
                )
            accepted_nonces.append(item)
        for item in raw_stale:
            if not isinstance(item, str):
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED, ["viewer state stale malformed"]
                )
            stale_frame_ids.append(item)
        for entry in raw_ledger:
            if not isinstance(entry, dict):
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED, ["viewer state qc ledger malformed"]
                )
            qc_ledger.append(
                {
                    "artifact_id": str(entry.get("artifact_id")),
                    "status": str(entry.get("status")),
                }
            )
        raw_action_count = document.get("action_count", 0)
        action_count = (
            int(raw_action_count)
            if isinstance(raw_action_count, int) and not isinstance(raw_action_count, bool)
            else 0
        )
        # CORONARY_CENTERLINE_001: overlay visibility is closed state.  A state
        # document written before the overlay existed (no key at all) reopens
        # with the default (visible); an unknown overlay id or a non-boolean
        # value fails closed BEFORE any reopened state is exposed.
        overlay_visibility = {overlay_id: True for overlay_id in OVERLAY_IDS}
        raw_overlays = document.get("overlay_visibility")
        if raw_overlays is not None:
            if not isinstance(raw_overlays, dict):
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED,
                    ["viewer state overlay_visibility malformed"],
                )
            unknown_overlays = sorted(set(raw_overlays) - set(OVERLAY_IDS))
            if unknown_overlays:
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED,
                    [
                        "viewer state overlay_visibility contains unknown overlay "
                        f"ids {unknown_overlays}"
                    ],
                )
            for overlay_id in OVERLAY_IDS:
                if overlay_id not in raw_overlays:
                    continue
                value = raw_overlays[overlay_id]
                if not isinstance(value, bool):
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        [
                            "viewer state overlay_visibility value for "
                            f"{overlay_id} is not a boolean"
                        ],
                    )
                overlay_visibility[overlay_id] = value
        # CORONARY_REVIEW_VIEW_V1 (Decision A): overlay display mode is
        # closed state.  A state document written before the mode existed
        # (no key at all) reopens with the default (SLICE_LOCAL); an unknown
        # mode fails closed BEFORE any reopened state is exposed.
        overlay_mode = OVERLAY_DEFAULT_MODE
        raw_mode = document.get("overlay_mode")
        if raw_mode is not None:
            if not isinstance(raw_mode, str) or raw_mode not in OVERLAY_MODES:
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED,
                    ["viewer state overlay_mode is outside the closed vocabulary"],
                )
            overlay_mode = raw_mode
        # CORONARY_MULTI_VESSEL_NAV (Decision A): selected vessel is closed
        # state.  A state document written before selection existed (no key
        # at all) reopens with None (legacy #28 single-path behavior); an
        # unknown token fails closed BEFORE any reopened state is exposed.
        selected_vessel: str | None = None
        raw_vessel = document.get("selected_vessel")
        if raw_vessel is not None:
            if not isinstance(raw_vessel, str) or raw_vessel not in _FROZEN_VESSEL_IDS:
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED,
                    ["viewer state selected_vessel is outside the frozen VESSEL:* tokens"],
                )
            selected_vessel = raw_vessel
        # R2-02/R2-03 (r2): fail closed BEFORE any reopened state is exposed
        # — complete geometry/render bijection, archive integrity, governed
        # historical-retention verification, and archived-record PNG
        # binding.  Zero partial state on any failure (Part 6 discipline).
        self._attach_internal_frame_ids(renders, archive)
        problems = validate_render_geometry_bijection(
            session_id=str(document.get("session_id")),
            geometry=self._require_geometry(),
            renders=renders,
            archive=archive,
        )
        problems.extend(_archive_integrity_problems(stale_frame_ids, renders, archive))
        if problems:
            raise ViewerOpenError(RejectionCode.SESSION_OPEN_FAILED, problems)
        self._verify_archive_png_binding(archive)
        return ViewerRuntimeState(
            session_id=str(document.get("session_id")),
            current_plane=str(document.get("current_plane", PLANES[0])),
            frame_index_by_plane=frame_index_by_plane,
            viewport_by_plane=viewport_by_plane,
            renders=renders,
            gate_outcomes=outcomes,
            used_nonces=used_nonces,
            accepted_nonces=accepted_nonces,
            stale_frame_ids=stale_frame_ids,
            qc_ledger=qc_ledger,
            action_count=action_count,
            qc_panel_open=bool(document.get("qc_panel_open", False)),
            render_archive=archive,
            overlay_visibility=overlay_visibility,
            overlay_mode=overlay_mode,
            selected_vessel=selected_vessel,
        )

    def _rehydrate_render_archive(
        self,
        document: dict[str, object],
        renders: dict[str, PlaneRenderState],
    ) -> dict[str, ViewerFrameRecord]:
        """Rehydrate the internal render-record archive (r2 R2-02).

        A persisted archive document (identified solely by key presence
        ``render_archive in document``) requires a dictionary and is validated
        key-by-key (keyed by the PUBLIC provenance ULID; each entry must carry
        exactly its ten-key frame document plus its internal ``frame_id`` and
        must not collide); present non-dictionary values (including null) fail
        closed with SESSION_OPEN_FAILED before legacy recovery or state exposure.
        Genuinely old state documents with the archive key absent initialize
        the archive from current records only (resolving internal frame IDs from
        retained provenance) — already-lost history is never fabricated.
        Archive membership is append-only across rehydration: current records
        not already present are added; persisted archived entries always win.
        """

        state_session_id = str(document.get("session_id"))
        raw_archive = document.get(_ARCHIVE_STATE_KEY)
        archive: dict[str, ViewerFrameRecord] = {}
        has_persisted_archive = _ARCHIVE_STATE_KEY in document
        if has_persisted_archive:
            if not isinstance(raw_archive, dict):
                raise ViewerOpenError(
                    RejectionCode.SESSION_OPEN_FAILED,
                    ["viewer state render archive malformed"],
                )
            for public_id, entry in raw_archive.items():
                if not isinstance(public_id, str) or not public_id:
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        ["viewer state render archive key malformed"],
                    )
                if not isinstance(entry, dict):
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        [f"archived render record {public_id} is not an object"],
                    )
                frame_id = entry.get("frame_id")
                if not isinstance(frame_id, str) or not frame_id:
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        [
                            f"archived render record {public_id} is missing "
                            "its internal frame_id"
                        ],
                    )
                frame_document = entry.get("frame")
                if not isinstance(frame_document, dict):
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        [
                            f"archived render record {public_id} is missing "
                            "its frame document"
                        ],
                    )
                try:
                    record = validate_frame_document(frame_document)
                except ViewerSchemaError as exc:
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        [
                            f"archived render record {public_id} fails the "
                            f"closed frame schema: {exc}"
                        ],
                    ) from None
                if record.provenance_ref.split(":", 1)[1] != public_id:
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        [
                            f"archived render record {public_id} carries a "
                            "mismatched provenance_ref"
                        ],
                    )
                if record.session_id != state_session_id:
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        [
                            f"archived render record {public_id} is bound to "
                            "a different session"
                        ],
                    )
                if public_id in archive:
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        [
                            f"viewer state render archive has a duplicate "
                            f"key {public_id}"
                        ],
                    )
                # Validate internal identity against retained provenance where recovered (r3 R3-03)
                matching_prov = next(
                    (p for p in self._restored.provenance if p.provenance_id == public_id),
                    None,
                )
                if (
                    matching_prov is not None
                    and matching_prov.subject_ref.startswith("viewer_frame:")
                ):
                    prov_frame_id = matching_prov.subject_ref.split(":", 1)[1]
                    if frame_id != prov_frame_id:
                        raise ViewerOpenError(
                            RejectionCode.SESSION_OPEN_FAILED,
                            [
                                f"archived render record {public_id} internal frame_id "
                                f"{frame_id} does not match retained provenance {prov_frame_id}"
                            ],
                        )
                archive[public_id] = replace(record, frame_id=frame_id)

            # R3-03: For an existing archive, reject missing current membership
            # rather than silently repairing, and require complete ten-key document
            # equality whenever current and archive share a public ID.
            for plane in PLANES:
                for record in renders[plane].records:
                    public_id = record.provenance_ref.split(":", 1)[1]
                    existing = archive.get(public_id)
                    if existing is None:
                        raise ViewerOpenError(
                            RejectionCode.SESSION_OPEN_FAILED,
                            [
                                f"current render record {public_id} is missing "
                                "from the existing render archive"
                            ],
                        )
                    if record.to_document() != existing.to_document():
                        raise ViewerOpenError(
                            RejectionCode.SESSION_OPEN_FAILED,
                            [
                                f"current render record {public_id} conflicts "
                                "with archived record (ten-key document equality required)"
                            ],
                        )
                    if (
                        record.frame_id
                        and existing.frame_id
                        and record.frame_id != existing.frame_id
                    ):
                        raise ViewerOpenError(
                            RejectionCode.SESSION_OPEN_FAILED,
                            [
                                f"archived render record {public_id} collides "
                                "with a current record under a different "
                                "internal frame id"
                            ],
                        )
        else:
            # R3-04: Initialization-from-current ONLY for genuinely old documents
            # lacking the archive field. Resolve each internal frame ID from the retained
            # provenance record named by its public provenance_ref and its viewer_frame
            # subject; verify the binding; fail typed when unresolvable (never fabricate).
            for plane in PLANES:
                for record in renders[plane].records:
                    public_id = record.provenance_ref.split(":", 1)[1]
                    matching_prov = next(
                        (p for p in self._restored.provenance if p.provenance_id == public_id),
                        None,
                    )
                    if matching_prov is None:
                        raise ViewerOpenError(
                            RejectionCode.SESSION_OPEN_FAILED,
                            [
                                f"old state document current record {public_id} "
                                "cannot resolve internal frame id: retained provenance missing"
                            ],
                        )
                    if not matching_prov.subject_ref.startswith("viewer_frame:"):
                        raise ViewerOpenError(
                            RejectionCode.SESSION_OPEN_FAILED,
                            [
                                f"old state document current record {public_id} provenance "
                                f"subject {matching_prov.subject_ref} is not viewer_frame"
                            ],
                        )
                    recovered_frame_id = matching_prov.subject_ref.split(":", 1)[1]
                    if not recovered_frame_id:
                        raise ViewerOpenError(
                            RejectionCode.SESSION_OPEN_FAILED,
                            [
                                f"old state document current record {public_id} "
                                "has empty recovered internal frame id"
                            ],
                        )
                    archive[public_id] = replace(record, frame_id=recovered_frame_id)
        return archive

    def _attach_internal_frame_ids(
        self,
        renders: dict[str, PlaneRenderState],
        archive: dict[str, ViewerFrameRecord],
    ) -> None:
        """Restore each rehydrated record's internal frame id from the archive.

        The serialized ten-key frame document excludes the internal frame id
        (record-identifier metadata, Part 5.1); the archive is its durable
        carrier.  Rehydrated CURRENT records get their ids back so later
        corrections mark STALE exactly the superseded records (r2 R2-02, r3 R3-04).
        """

        for plane in PLANES:
            current = renders[plane]
            rebuilt: list[ViewerFrameRecord] = []
            changed = False
            for record in current.records:
                public_id = record.provenance_ref.split(":", 1)[1]
                archived = archive.get(public_id)
                if (
                    archived is not None
                    and archived.frame_id
                    and record.frame_id != archived.frame_id
                ):
                    rebuilt.append(replace(record, frame_id=archived.frame_id))
                    changed = True
                else:
                    rebuilt.append(record)
            if changed:
                renders[plane] = PlaneRenderState(
                    wl_preset_id=current.wl_preset_id, records=tuple(rebuilt)
                )

    def _verify_archive_png_binding(
        self, archive: dict[str, ViewerFrameRecord]
    ) -> None:
        """Verify every archived record's PNG binding at open (r2 R2-02).

        The archived bytes are reconstructed from the retained immutable
        source pixels plus the stored WL and must match the record's
        ``png_sha256``/``png_byte_size`` before anything is exposed or
        served; archived records are never rewritten or fabricated.
        """

        for public_id, record in sorted(archive.items()):
            png = self._render_archived_png(public_id, record)
            del png  # _render_archived_png raises on any binding failure

    def _mint_session_provenance(self, state: ViewerRuntimeState) -> ProvenanceRecord:
        """The §44 record of the viewer session open (LAYER-G construction)."""

        geometry = self._require_geometry()
        mpr_inputs: list[InputRef] = []
        mpr_upstreams: list[str] = []
        for plane in PLANES:
            for frame in geometry[plane]:
                artifact = self._mpr_by_id[frame.mpr_artifact_id]
                mpr_inputs.append(
                    InputRef(
                        ref=f"mpr_artifact_id:{artifact.artifact_id}",
                        kind="artifact",
                        sha256_at_use=artifact.payload.blob_sha256,
                    )
                )
                if artifact.provenance_ref not in mpr_upstreams:
                    mpr_upstreams.append(artifact.provenance_ref)
        if not mpr_inputs:
            series = self._restored.series[0]
            mpr_inputs.append(
                InputRef(
                    ref=series.dicom_series_instance_uid,
                    kind="series",
                    sha256_at_use=series.original_dicom_ref.manifest_sha256,
                )
            )
        record = create_provenance(
            subject_ref=f"viewer_session:{state.session_id}",
            event_type=ProvenanceEventType.DERIVED,
            actor=Actor(actor_type=ActorType.ENGINE, actor_id=_VIEWER_ACTOR_ID),
            inputs=tuple(mpr_inputs),
            preprocessing="viewer-session-open-layer-g-construction",
            parameters_ref=BlobRef(
                blob_ref="viewer-frame-lists",
                sha256=(
                    self._session_record.frame_lists_ref
                    if self._session_record
                    else "0" * 64
                ),
            ),
            human_edit=NotApplicable(
                value=None, reason="automatic viewer session open"
            ),
            upstream_refs=tuple(mpr_upstreams)
            or NotApplicable(
                value=None, reason="empty LAYER-G: no accepted MPR provenance bound"
            ),
            environment=Environment(
                software_version=_VIEWER_SOFTWARE,
                code_commit_sha=self._require_build_commit_sha(),
                runtime="python-stdlib-viewer",
                deterministic_seed=None,
            ),
            created_by=_VIEWER_CREATED_BY,
        )
        return record

    # ------------------------------------------------------------------
    # QC consumer gate (Part 6.1)
    # ------------------------------------------------------------------

    def _qc_candidates(
        self, state: ViewerRuntimeState | None = None
    ) -> tuple[QCAssessmentArtifact, ...]:
        """The gate's candidate set: every QC assessment visible to the session.

        Restored QC artifacts (kind ``QC_ASSESSMENT``) enter with their true
        record status; viewer-routed corrections enter through the persisted
        payload documents (``qc_artifact:<ULID>`` keys) with the status from
        the viewer's own append-only qc ledger (in-memory corrections carry
        their live record status).
        """

        candidates: list[QCAssessmentArtifact] = []
        seen_refs: set[str] = set()
        for artifact in self._restored.artifacts:
            if artifact.kind != ArtifactKind.QC_ASSESSMENT:
                continue
            document = self._restored.payload_documents.get(artifact.artifact_id)
            if document is None:
                continue
            view = _assessment_view(artifact, document)
            seen_refs.add(view.artifact_ref)
            candidates.append(view)
        resolved_state = state if state is not None else self._state
        ledger_statuses = (
            {entry["artifact_id"]: str(entry["status"]) for entry in resolved_state.qc_ledger}
            if resolved_state is not None
            else {}
        )
        for key, document in self._restored.payload_documents.items():
            if not key.startswith("qc_artifact:") or not isinstance(document, dict):
                continue
            artifact_id = key.split(":", 1)[1]
            status = ledger_statuses.get(artifact_id, "ACTIVE")
            candidates.append(_document_view(artifact_id, document, status))
        for artifact in self._new_qc_artifacts:
            document = self._new_qc_documents.get(artifact.artifact_id)
            if document is None:
                continue
            view = _assessment_view(artifact, document)
            if view.artifact_ref in seen_refs:
                continue
            candidates.append(view)
        return tuple(candidates)

    def _gate_lineage(self) -> dict[str, object]:
        lineage: dict[str, object] = {}
        for artifact in self._restored.artifacts:
            if artifact.kind != ArtifactKind.MPR_VIEW:
                continue
            document = self._restored.payload_documents.get(artifact.artifact_id)
            if document is None:
                continue
            lineage[f"mpr_artifact_id:{artifact.artifact_id}"] = {
                "source_volume_ref": document.get("source_volume_ref")
            }
        for volume in self._restored.volumes:
            lineage[f"volume_id:{volume.volume_id}"] = {"study_ref": self._study_id}
        return lineage

    def _query_gate(self, state: ViewerRuntimeState | None = None) -> GateOutcome:
        mpr_refs = tuple(
            f"mpr_artifact_id:{artifact.artifact_id}"
            for artifact in self._restored.artifacts
            if artifact.kind == ArtifactKind.MPR_VIEW
        )
        consumed = (self._study_id, *mpr_refs)
        result = real_qc_consumer_gate(
            consumed,
            candidates=self._qc_candidates(state),
            consumer="ccta-viewer",
            subject_lineage=self._gate_lineage(),
        )
        return GateOutcome(
            gate_id=f"gate:{self._id_factory()}",
            decision=result.decision.value,
            reason_code=result.reason_code.value if result.reason_code else None,
            assessment_refs=tuple(result.assessment_refs),
            consumed_refs=tuple(result.consumed_refs),
            bound_at=self._now(),
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _mint_plane(
        self,
        plane: str,
        wl_preset_id: str | None,
        *,
        custom_window: tuple[float, float] | None = None,
    ) -> PlaneRenderState:
        """Mint LAYER-R records for one plane; archive each BEFORE replacing.

        The caller replaces the plane's current tuple only after every
        record exists in the archive (r2 R2-02 insert-before-replace).

        ``custom_window`` (v2, spec Part 4.2) carries the user-applied
        ``(center, width)`` binary32 values of a custom WL submit; it selects
        the IDENTICAL frozen Part 4.1 transfer path as presets/raw, differing
        only in the effective window inputs.  It is only meaningful when
        ``wl_preset_id`` is ``None`` (the custom LAYER-R variant class), and
        the recorded ``wl_applied`` then carries exactly these values.
        """

        geometry = self._require_geometry()
        state = self._require_state()
        records: list[ViewerFrameRecord] = []
        for frame in geometry[plane]:
            mpr = self._mpr_by_id[frame.mpr_artifact_id]
            payload = self._restored.payload_documents[frame.mpr_artifact_id]
            blob_ref = payload["pixel_blob_ref"]
            digest = str(blob_ref["sha256"])  # type: ignore[index]
            pixels = decode_pixel_blob(payload, self._pixel_blobs[digest])
            if custom_window is not None:
                center, width = custom_window
            elif wl_preset_id is None:
                center, width = identity_window_for_pixels(pixels)
            else:
                center, width = preset_window(preset_by_id(wl_preset_id))
            mint = mint_render_variant(
                session_id=state.session_id,
                geometry=frame,
                wl_preset_id=wl_preset_id,
                center=center,
                width=width,
                mpr_artifact=mpr,
                mpr_payload=payload,
                pixel_blob=self._pixel_blobs[digest],
                provenance_upstream=(mpr.provenance_ref,),
                code_commit_sha=self._require_build_commit_sha(),
                repo_root=self._repo_root,
            )
            records.append(mint.record)
            self._archive_record(mint.record)
            self._new_provenance.append(mint.provenance)
        return PlaneRenderState(wl_preset_id=wl_preset_id, records=tuple(records))

    def _archive_record(self, record: ViewerFrameRecord) -> None:
        """Append one minted record to the internal archive (r2 R2-02).

        Every minted record is archived BEFORE it can replace any plane's
        current tuple: the archive is keyed by the public provenance ULID and
        retains the complete ten-key frame document plus the internal frame
        id (the STALE-lookup key).  Append-only: existing entries are never
        rewritten.
        """

        state = self._require_state()
        public_id = record.provenance_ref.split(":", 1)[1]
        if public_id not in state.render_archive:
            state.render_archive[public_id] = record

    def png_for_frame(self, frame_id: str) -> tuple[bytes, ViewerFrameRecord] | None:
        """Re-derive the frozen PNG bytes for one frame record by its id.

        The public record id is the ``provenance_ref`` ULID (unique per
        record creation, present in the serialized 10-key document); the
        internal ``frame_id`` (the provenance subject) resolves equally.
        Both CURRENT and ARCHIVED (superseded, r2 R2-02) records resolve:
        archived records are reconstructed from the retained immutable
        source pixels plus the stored WL and are digest/size checked
        against their record BEFORE serving — bytes are retained and still
        queryable, never rewritten.
        """

        state = self._require_state()
        for plane in PLANES:
            for record in state.renders[plane].records:
                public_id = record.provenance_ref.split(":", 1)[1]
                if frame_id not in (public_id, record.frame_id):
                    continue
                png = self._render_record_png(record)
                actual_sha = hashlib.sha256(png).hexdigest()
                if actual_sha != record.png_sha256 or len(png) != record.png_byte_size:
                    raise ViewerOpenError(
                        RejectionCode.SESSION_OPEN_FAILED,
                        [
                            f"current render record {public_id} fails the "
                            "digest/size check before serving (retained bytes are "
                            "never rewritten or fabricated)"
                        ],
                    )
                return (png, record)
        for public_id, archived in state.render_archive.items():
            if frame_id != public_id and (
                not archived.frame_id or frame_id != archived.frame_id
            ):
                continue
            png = self._render_archived_png(public_id, archived)
            return (png, archived)
        return None

    def _render_record_png(self, record: ViewerFrameRecord) -> bytes:
        payload = self._restored.payload_documents[
            record.source_mpr_ref.split(":", 1)[1]
        ]
        blob_ref = payload["pixel_blob_ref"]
        digest = str(blob_ref["sha256"])  # type: ignore[index]
        return render_png_for_record(
            payload,
            self._pixel_blobs[digest],
            record.wl_applied["center"],
            record.wl_applied["width"],
        )

    def _render_archived_png(
        self, public_id: str, record: ViewerFrameRecord
    ) -> bytes:
        """Reconstruct an archived record's PNG with a pre-serve check."""

        png = self._render_record_png(record)
        actual_sha = hashlib.sha256(png).hexdigest()
        if actual_sha != record.png_sha256 or len(png) != record.png_byte_size:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED,
                [
                    f"archived render record {public_id} fails the "
                    "digest/size check before serving (retained bytes are "
                    "never rewritten or fabricated)"
                ],
            )
        return png

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def apply_action_bytes(
        self, body: bytes, *, path_session_id: str
    ) -> dict[str, object]:
        """Validate + apply one action POST; returns the state document.

        Every action — accepted or rejected — appends exactly one sealed
        audit event (Part 6.1 (b); E2: every rejection path is visible).
        A rejected attempt's ``client_nonce`` is consumed for the session's
        lifetime (uniqueness over every SEEN nonce, fail-closed); this
        includes QC_001-HUMAN seam refusals (r2 R2-01): the refused nonce is
        consumed exactly once, without ever entering the accepted chain.
        """

        self._require_state()
        try:
            parsed = validate_action(
                body,
                path_session_id=path_session_id,
                session=self._require_state().grammar_view(self._study_id),
            )
        except ViewerRejectionError as rejection:
            self._consume_nonce(body)
            self._audit_rejection(rejection)
            raise
        return self._execute(parsed)

    def _consume_nonce(self, body: bytes) -> None:
        """Record the attempt's client_nonce (if parseable) as used."""

        import json as _json

        try:
            parsed = _json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, _json.JSONDecodeError):
            return
        if isinstance(parsed, dict) and isinstance(parsed.get("client_nonce"), str):
            nonce = parsed["client_nonce"]
            if nonce not in self._require_state().used_nonces:
                self._require_state().used_nonces.append(nonce)

    def _execute(self, parsed: ParsedAction) -> dict[str, object]:
        state = self._require_state()
        plane = parsed.view_plane
        if parsed.action == "SET_SLICE":
            state.frame_index_by_plane[str(parsed.params["plane"])] = int(
                str(parsed.params["frame_index"])
            )
        elif parsed.action == "SET_PLANE":
            state.current_plane = str(parsed.params["plane"])
        elif parsed.action == "SET_WL_PRESET":
            # WL mints LAYER-R variants for the CURRENT plane ONLY; the G
            # list is untouched (frame_index and frame_lists_ref never move).
            state.renders[plane] = self._mint_plane(
                plane, str(parsed.params["preset_id"])
            )
        elif parsed.action == "SET_WL_CUSTOM":
            # v2 custom WL (spec Part 4.2): the validated numerics are cast
            # ONCE to their binary32 values here, then flow through the
            # IDENTICAL frozen Part 4.1 transfer as presets/raw; the record
            # carries wl_preset_id=None and wl_applied = the applied values.
            state.renders[plane] = self._mint_plane(
                plane,
                None,
                custom_window=(
                    float(to_binary32(float(str(parsed.params["center"])))),
                    float(to_binary32(float(str(parsed.params["width"])))),
                ),
            )
        elif parsed.action == "CLEAR_WL":
            state.renders[plane] = self._mint_plane(plane, _RAW_WL_PRESET_ID)
        elif parsed.action == "SET_VIEWPORT_MATRIX":
            state.viewport_by_plane[plane] = {
                "zoom": float(str(parsed.params["zoom"])),
                "pan_x": float(str(parsed.params["pan_x"])),
                "pan_y": float(str(parsed.params["pan_y"])),
            }
        elif parsed.action == "OPEN_QC_PANEL":
            state.qc_panel_open = True
        elif parsed.action == "REQUEST_CORRECTION":
            try:
                self._route_correction(parsed)
            except HumanSeamRefusalError as refusal:
                # R2-01 (r2): a seam-refused attempt's client_nonce is
                # consumed EXACTLY ONCE for the session's lifetime — it
                # enters used_nonces (seen -> NONCE_REUSE on reuse) but is
                # NOT added to accepted_nonces, the accepted-action count
                # is NOT incremented, and no gate/render state changes.
                # Exactly one sealed rejection audit is emitted.
                if parsed.client_nonce not in state.used_nonces:
                    state.used_nonces.append(parsed.client_nonce)
                self._audit_seam_refusal(refusal)
                raise
        elif parsed.action == "REQUEST_CENTERLINE_DECISION":
            # CORONARY_CENTERLINE_001 att0002 (F5): the single-path human seam.
            # A refused decision consumes its nonce exactly like a refused
            # REQUEST_CORRECTION (fail-closed, visible, one audit event).
            try:
                self._route_centerline_decision(parsed)
            except HumanSeamRefusalError as refusal:
                if parsed.client_nonce not in state.used_nonces:
                    state.used_nonces.append(parsed.client_nonce)
                self._audit_seam_refusal(refusal)
                raise
        elif parsed.action == "SET_OVERLAY_VISIBILITY":
            # CORONARY_CENTERLINE_001: the one overlay-visibility toggle.  The
            # allowlist and boolean type were closed by the grammar; the
            # engine only records the closed state (never touches LAYER-G).
            # CORONARY_REVIEW_VIEW_V1 (Decision A): the closed display mode
            # rides the same action (SLICE_LOCAL default, PROJECTION localizer).
            state.overlay_visibility[str(parsed.params["overlay_id"])] = bool(
                parsed.params["visible"]
            )
            state.overlay_mode = str(parsed.params["mode"])
        elif parsed.action == "SELECT_VESSEL":
            # CORONARY_MULTI_VESSEL_NAV (Decision A): the ONLY mutation path
            # for the stored vessel selection.  The frozen VESSEL:* membership
            # was closed by the grammar; the engine only records the closed
            # state (never touches geometry, never infers anatomy).
            state.selected_vessel = str(parsed.params["vessel_label"])
        state.used_nonces.append(parsed.client_nonce)
        state.accepted_nonces.append(parsed.client_nonce)
        state.action_count += 1
        self._audit_action(parsed)
        return self.state_document()

    def _route_correction(self, parsed: ParsedAction) -> None:
        """Route REQUEST_CORRECTION ONLY into the QC_001-HUMAN seam."""

        state = self._require_state()
        qc_input = self._build_qc_input()
        grade = QualityGrade(str(parsed.params["proposed_grade"]))
        # The closed v1 grammar carries NO flag vocabulary; the flags below
        # are the ONLY matrix-determined values.  ACCEPTABLE/LIMITED require
        # evidence flags the grammar cannot express, so the seam refuses them
        # fail-closed (never a fabricated finding, never a silent fallback).
        flags = ("ABSTAINED",) if grade is QualityGrade.NOT_ANALYZABLE else ()
        evidence = HumanEvidence(
            edit_summary="viewer REQUEST_CORRECTION routed human assessment",
            edited_at=self._now(),
        )
        store = self._get_correction_store()
        effective = select_effective_assessment(
            self._study_id, self._qc_candidates(self._require_state())
        )
        common: dict[str, Any] = dict(
            quality_grade=grade,
            flags=list(flags),
            human_evidence=evidence,
            # Prior records minted by THIS engine (earlier corrections) are
            # supplied so the seam's chain verifies across corrections — but
            # only the CLOSED human-ledger records: every supplied record's
            # upstream must resolve within the supplied set plus the seam's
            # own volume/transform roots.  Frame-render provenance references
            # MPR provenance outside that universe, so those records are
            # excluded here (they ride the session snapshot, not the seam).
            provenance_records=_closed_provenance_prefix(tuple(self._new_provenance)),
            artifact_store=store,
            created_by=_VIEWER_CREATED_BY,
            created_at=self._now(),
            code_commit_sha=self._require_build_commit_sha(),
            software_version=_VIEWER_SOFTWARE,
            id_factory=self._id_factory,
        )
        try:
            if effective is None:
                result = create_initial_human_assessment(qc_input, **common)
            else:
                result = create_human_correction(
                    qc_input, supersedes_ref=effective.artifact_ref, **common
                )
        except QCHumanRefusalError as exc:
            raise HumanSeamRefusalError(exc.code.value, str(exc)) from None
        except ValueError as exc:
            raise HumanSeamRefusalError("QC_HUMAN_MATRIX_REFUSAL", str(exc)) from None
        self._new_qc_artifacts.append(result.artifact)
        self._new_qc_documents[result.artifact.artifact_id] = dict(
            result.payload_document
        )
        self._new_provenance.extend(result.provenance_records)
        state.qc_ledger.append(
            {"artifact_id": result.artifact.artifact_id, "status": "ACTIVE"}
        )
        if effective is not None:
            # The superseded predecessor keeps its payload document bytes
            # (append-only); the viewer qc ledger records the transition.
            predecessor_id = str(effective.artifact_ref).split(":", 1)[1]
            state.qc_ledger.append(
                {"artifact_id": predecessor_id, "status": "SUPERSEDED"}
            )
        # Rule F rebind: re-query the gate, advance qc_gate_state_ref, mark
        # the superseded outcome + its frames STALE (append-only ledger; old
        # outcome bytes retained and still queryable).
        outcome = self._query_gate()
        if state.gate_outcomes:
            superseded = state.gate_outcomes[-1]
            if superseded.superseded_by is None:
                state.gate_outcomes[-1] = replace(superseded, superseded_by=outcome.gate_id)
        state.gate_outcomes.append(outcome)
        record = self._session_record
        if record is not None:
            self._session_record = replace(record, qc_gate_state_ref=outcome.gate_id)
        for plane in PLANES:
            current = state.renders[plane]
            state.stale_frame_ids.extend(
                r.frame_id for r in current.records if r.frame_id
            )
            state.renders[plane] = self._mint_plane(
                plane,
                current.wl_preset_id,
                custom_window=_uniform_applied_window(current),
            )

    # ------------------------------------------------------------------
    # Coronary single-path human seam (att0002, F5)
    # ------------------------------------------------------------------

    def _working_artifact_ids(self) -> tuple[str, ...]:
        """Restored artifact ids plus the artifacts minted by this session."""

        restored = [artifact.artifact_id for artifact in self._restored.artifacts]
        return tuple(
            restored
            + [
                artifact_id
                for artifact_id in self._new_coronary_artifacts
                if artifact_id not in restored
            ]
        )

    def _working_artifacts(self) -> tuple[DerivedArtifact, ...]:
        """The current artifact view with the working statuses applied.

        Restored records resolve through the working correction store (so an
        ACCEPT/REJECT/EDIT status change is immediately visible); EDIT
        successors are appended.  Before any correction route runs the
        restored tuple is returned unchanged.
        """

        if self._correction_store is None:
            return self._restored.artifacts
        return tuple(
            self._correction_store.get(artifact_id)
            for artifact_id in self._working_artifact_ids()
        )

    def _working_payload_documents(self) -> dict[str, dict[str, object]]:
        documents = dict(self._restored.payload_documents)
        documents.update(self._new_coronary_documents)
        return documents

    def _route_centerline_decision(self, parsed: ParsedAction) -> None:
        """Route ONE ``REQUEST_CENTERLINE_DECISION`` into the coronary seam.

        This is a route SEPARATE from ``REQUEST_CORRECTION``: the QC route
        keeps its semantics byte-identical.  The decision applies to the
        single ACTIVE CENTERLINE named by ``centerline_ref``; zero, multiple
        or mismatched targets are typed, visible refusals.  ACCEPT/REJECT
        call :func:`apply_coronary_decision` directly; EDIT mints a HUMAN
        ``CORRECTION`` artifact plus the NEW re-seeded centerline through the
        same helper (never an in-place edit).
        """

        store = self._get_correction_store()
        state = self._require_state()
        served_vessel = state.selected_vessel
        try:
            artifact = select_active_centerline(
                self._working_artifacts(),
                vessel_label=served_vessel,
                payload_documents=self._working_payload_documents(),
            )
        except CoronaryOverlayRefusalError as refusal:
            raise HumanSeamRefusalError(refusal.code, refusal.message) from None
        requested_ref = str(parsed.params["centerline_ref"])
        if requested_ref != f"centerline_artifact_id:{artifact.artifact_id}":
            raise HumanSeamRefusalError(
                "CENTERLINE_DECISION_TARGET_INVALID",
                "centerline_ref does not name the single ACTIVE centerline of this session",
            )
        decision = CoronaryDecision(str(parsed.params["decision"]))
        evidence_ref = f"viewer-centerline-decision:{parsed.client_nonce}"
        if decision is CoronaryDecision.EDIT:
            self._apply_centerline_reseed(store, artifact, parsed, evidence_ref)
            return
        try:
            apply_coronary_decision(
                store,
                centerline_id=artifact.artifact_id,
                decision=decision,
                actor_role=ActorType.HUMAN,
                reviewed_by=_VIEWER_CREATED_BY,
                evidence_ref=evidence_ref,
                at=self._now(),
            )
        except (StateTransitionError, CoronaryValidationError) as exc:
            raise HumanSeamRefusalError("CENTERLINE_DECISION_REFUSED", str(exc)) from None

    def _apply_centerline_reseed(
        self,
        store: DerivedArtifactStore,
        artifact: DerivedArtifact,
        parsed: ParsedAction,
        evidence_ref: str,
    ) -> None:
        """Human EDIT: re-seed -> GOVERNED new derivation -> NEW artifacts."""

        seed_values = cast(list[int], parsed.params["seed_voxel"])
        direction_values = cast(list[float], parsed.params["seed_direction_world"])
        self._reseed_centerline(
            store,
            artifact,
            i=int(seed_values[0]),
            j=int(seed_values[1]),
            k=int(seed_values[2]),
            direction=(
                float(direction_values[0]),
                float(direction_values[1]),
                float(direction_values[2]),
            ),
            evidence_ref=evidence_ref,
            reason="viewer re-seed (REQUEST_CENTERLINE_DECISION)",
            preprocessing=(
                "viewer REQUEST_CENTERLINE_DECISION:EDIT "
                "(human re-seed -> NEW centerline)"
            ),
        )

    def _reseed_centerline(
        self,
        store: DerivedArtifactStore,
        artifact: DerivedArtifact,
        *,
        i: int,
        j: int,
        k: int,
        direction: tuple[float, float, float],
        evidence_ref: str,
        reason: str,
        preprocessing: str,
    ) -> Seed:
        """Shared governed re-seed: typed Seed -> NEW derivation -> NEW artifacts.

        Both the ``REQUEST_CENTERLINE_DECISION`` EDIT path and the dedicated
        T2 ``POST /session/:id/seed`` route run through here (reuse, not a
        parallel implementation): explicit server-side bounds validation
        against the session volume dimensions with typed refusals
        (``SEED_OUT_OF_BOUNDS`` / ``SEED_DIRECTION_INVALID``), ``Seed``
        creation, the governed ``build_centerline`` derivation, HUMAN
        correction + ENGINE derivation provenance, the EDIT supersede with
        STALE fan-out, and the EDIT-successor CPR framing.  Reselect
        supersedes: the predecessor is marked superseded, never edited in
        place.  Returns the created ``Seed``.
        """

        if not self._restored.volumes:
            raise HumanSeamRefusalError(
                "CENTERLINE_DECISION_REFUSED",
                "the re-seed route requires a canonical volume in the session",
            )
        # att0003 (F5-repeat): resolve the ACTIVE centerline's sealed payload
        # through the working documents so an in-memory EDIT successor (not
        # yet persisted) is found, not only the restored map.
        payload = self._working_payload_documents().get(artifact.artifact_id)
        if not isinstance(payload, dict):
            raise HumanSeamRefusalError(
                "CENTERLINE_DECISION_REFUSED",
                "the ACTIVE centerline has no sealed payload document",
            )
        source_ref = payload.get("source_volume_ref")
        volume = next(
            (
                candidate
                for candidate in self._restored.volumes
                if source_ref == f"volume_id:{candidate.volume_id}"
            ),
            None,
        )
        if volume is None:
            raise HumanSeamRefusalError(
                "CENTERLINE_DECISION_REFUSED",
                "the ACTIVE centerline's source volume is not present in the session",
            )
        transform, transform_ref, transform_provenance_ref = self._persisted_transform(
            frame=volume.coordinate_frame
        )
        frame = volume.coordinate_frame
        loader = CanonicalVolumeLoader(self._restored.paths.blobs)
        array = loader.read_array(
            volume.blob_ref.sha256,
            envelope=loader.read_envelope(volume.blob_ref.sha256),
        )
        scales = cast(list[float], payload["scales_mm"])
        parameters = VesselnessParameters(
            scales_mm=tuple(float(value) for value in scales),
            threshold=float(cast(float, payload["threshold"])),
        )
        if parameters.sha256 != payload["parameters_sha256"]:
            raise HumanSeamRefusalError(
                "CENTERLINE_DECISION_REFUSED",
                "the sealed derivation parameters cannot be faithfully "
                "reconstructed (digest mismatch); refusing a silent "
                "re-parameterization",
            )
        # Explicit server-side bounds validation against the session volume
        # dimensions (T2 Phase A): 0 <= i < columns, 0 <= j < rows,
        # 0 <= k < slices where dimensions are (slices, rows, columns) =
        # (K, J, I).  Out-of-volume fails explicitly with the typed
        # SEED_OUT_OF_BOUNDS refusal (never a silent clamp, never a default).
        dimensions = tuple(int(value) for value in frame.dimensions)
        if not (
            0 <= i < dimensions[2]
            and 0 <= j < dimensions[1]
            and 0 <= k < dimensions[0]
        ):
            raise HumanSeamRefusalError(
                CoronaryRefusalCode.SEED_OUT_OF_BOUNDS.value,
                f"seed voxel {(i, j, k)} is outside the session volume "
                f"(columns={dimensions[2]}, rows={dimensions[1]}, "
                f"slices={dimensions[0]}): 0 <= i < columns, 0 <= j < rows, "
                "0 <= k < slices is required",
            )
        # Explicit server-side direction validation (finite non-zero world
        # hint): violations fail with the typed SEED_DIRECTION_INVALID
        # refusal.  ``Seed`` re-checks the same domain at construction.
        if (
            len(direction) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in direction
            )
            or any(not math.isfinite(float(value)) for value in direction)
        ):
            raise HumanSeamRefusalError(
                CoronaryRefusalCode.SEED_DIRECTION_INVALID.value,
                "seed direction must be a finite 3-vector",
            )
        if math.sqrt(sum(float(value) ** 2 for value in direction)) <= 0.0:
            raise HumanSeamRefusalError(
                CoronaryRefusalCode.SEED_DIRECTION_INVALID.value,
                "seed direction must be non-zero",
            )
        try:
            seed = Seed(i=i, j=j, k=k, direction=direction)
        except CoronaryValidationError as exc:
            raise HumanSeamRefusalError(exc.code.value, str(exc)) from None
        new_centerline_id = str(self._id_factory())
        new_provenance_id = str(self._id_factory())
        correction_id = str(self._id_factory())
        correction_provenance_id = str(self._id_factory())
        seed_digest = canonical_digest(seed.to_document())
        correction_payload: dict[str, object] = {
            "schema_version": "ccta.payload.coronary_correction.v1",
            "decision": "EDIT",
            "seed_voxel": [seed.i, seed.j, seed.k],
            "seed_direction_world": list(seed.unit_direction()),
            "reason": reason,
        }
        try:
            rebuilt = build_centerline(
                volume,
                transform,
                frame,
                loader=loader,
                seed=seed,
                parameters=parameters,
                provenance_ref=f"provenance:{new_provenance_id}",
                transform_artifact_ref=transform_ref,
                study_instance_uid=str(payload["study_instance_uid"]),
                frame_of_reference_uid=frame.frame_of_reference_uid,
                scalar_domain=str(payload["scalar_domain"]),
                source_array=array,
                vessel_label=cast(
                    str | None, payload.get("vessel_label")
                ),
                artifact_id=new_centerline_id,
                created_by=_VIEWER_CREATED_BY,
            )
        except CoronaryValidationError as exc:
            raise HumanSeamRefusalError("CENTERLINE_DECISION_REFUSED", str(exc)) from None
        mask_blob = cast(dict[str, object], rebuilt.payload["vessel_mask_blob_ref"])
        centerline_record = centerline_provenance(
            subject_ref=f"centerline_artifact_id:{new_centerline_id}",
            volume=volume,
            seed=seed,
            parameters=parameters,
            mask_sha256=rebuilt.path.diagnostics.mask_sha256,
            mask_blob_sha256=str(mask_blob["sha256"]),
            response_sha256=rebuilt.path.diagnostics.response_sha256,
            transform_artifact_ref=transform_ref,
            transform_digest=canonical_digest(transform),
            volume_provenance_ref=volume.provenance_ref,
            transform_provenance_ref=transform_provenance_ref,
            software_version=_VIEWER_SOFTWARE,
            code_commit_sha=self._require_build_commit_sha(),
            runtime="python-3.12 (viewer centerline decision)",
            created_by=_VIEWER_CREATED_BY,
            at=self._now(),
            provenance_id=new_provenance_id,
        )
        correction_record = create_provenance(
            subject_ref=f"correction_artifact_id:{correction_id}",
            event_type=ProvenanceEventType.REVIEWED,
            actor=Actor(actor_type=ActorType.HUMAN, actor_id=_VIEWER_CREATED_BY),
            inputs=(
                InputRef(
                    ref=f"centerline_artifact_id:{artifact.artifact_id}",
                    kind="artifact",
                    sha256_at_use=artifact.payload.blob_sha256,
                ),
            ),
            preprocessing=preprocessing,
            parameters_ref=BlobRef(
                blob_ref=f"coronary-centerline-correction:EDIT:seed-{seed_digest}",
                sha256=seed_digest,
            ),
            human_edit=HumanEdit(
                editor=_VIEWER_CREATED_BY,
                edit_summary=(
                    "re-seed the single-path centerline (viewer human seam)"
                ),
                at=self._now(),
            ),
            upstream_refs=(centerline_record.provenance_id,),
            environment=Environment(
                software_version=_VIEWER_SOFTWARE,
                code_commit_sha=self._require_build_commit_sha(),
                runtime="python-3.12 (viewer centerline decision)",
                deterministic_seed=None,
            ),
            created_by=_VIEWER_CREATED_BY,
            at=self._now(),
            provenance_id=correction_provenance_id,
        )
        try:
            apply_coronary_decision(
                store,
                centerline_id=artifact.artifact_id,
                decision=CoronaryDecision.EDIT,
                actor_role=ActorType.HUMAN,
                reviewed_by=_VIEWER_CREATED_BY,
                evidence_ref=evidence_ref,
                at=self._now(),
                correction_payload=correction_payload,
                rebuild=lambda: rebuilt,
                correction_provenance_ref=f"provenance:{correction_provenance_id}",
                correction_study_ref=self._study_id,
                correction_claim_version=CORONARY_CLAIM_VERSION,
                correction_release_lane=artifact.release_lane,
                correction_created_by=_VIEWER_CREATED_BY,
                correction_artifact_id=correction_id,
            )
        except (StateTransitionError, CoronaryValidationError) as exc:
            raise HumanSeamRefusalError("CENTERLINE_DECISION_REFUSED", str(exc)) from None
        self._new_coronary_artifacts.extend([correction_id, new_centerline_id])
        self._new_coronary_documents[new_centerline_id] = dict(rebuilt.payload)
        self._new_coronary_documents[correction_id] = dict(correction_payload)
        self._new_provenance.extend([centerline_record, correction_record])
        # CORONARY_REVIEW_VIEW_V1 att0002 (F1-PREFERRED): frame a successor
        # CPR inside the EDIT path.  The successor centerline rebuild pattern
        # above already resolved the volume/transform/array; the successor CPR
        # reuses the predecessor CPR framing params (s/radial spacing, count,
        # method, fill) + the same source volume/transform/provenance refs,
        # mirroring the centerline rebuild.  The successor CPR binds the NEW
        # centerline generation (geometry_base pins the successor), so both
        # panels serve the successor generation with the same centerline_ref
        # (Decision C).  Pixel bytes land in the governed blob store through
        # the frozen CPR build entry point (no new numerics).
        self._frame_successor_cpr_view(
            store,
            predecessor_centerline_id=artifact.artifact_id,
            successor_centerline=rebuilt,
            successor_provenance=centerline_record,
            volume=volume,
            transform=transform,
            transform_ref=transform_ref,
            source_array=array,
        )
        return seed

    # ------------------------------------------------------------------
    # T2 physician/manual seed route, Phase A (server side)
    # ------------------------------------------------------------------

    def current_seed(self) -> dict[str, object] | None:
        """The ACTIVE path's physician seed, or None when no seed exists.

        Derived read-only from the ACTIVE CENTERLINE's sealed payload (the
        canonical data model stays the single center of truth; no module
        keeps its own seed copy).  A fresh session with no ACTIVE centerline
        has NO seed — the route mints no hidden defaults.
        """

        state = self._require_state()
        try:
            artifact = select_active_centerline(
                self._working_artifacts(),
                vessel_label=state.selected_vessel,
                payload_documents=self._working_payload_documents(),
            )
        except CoronaryOverlayRefusalError:
            return None
        payload = self._working_payload_documents().get(artifact.artifact_id)
        if not isinstance(payload, dict):
            return None
        voxel = payload.get("seed_voxel")
        direction = payload.get("seed_direction_world")
        if (
            not isinstance(voxel, list)
            or len(voxel) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in voxel
            )
            or not isinstance(direction, list)
            or len(direction) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in direction
            )
        ):
            return None
        return {
            "voxel": [int(voxel[0]), int(voxel[1]), int(voxel[2])],
            "direction": [
                float(direction[0]),
                float(direction[1]),
                float(direction[2]),
            ],
            "centerline_ref": f"centerline_artifact_id:{artifact.artifact_id}",
        }

    def apply_seed_bytes(
        self, body: bytes, *, path_session_id: str
    ) -> dict[str, object]:
        """Validate + apply one ``POST /session/:id/seed`` body.

        Byte gates (duplicate keys, canonical encoding) are the shared
        ``parse_action_body`` gates; envelope/nonce/lineage failures are the
        pinned grammar rejections (the 16-code vocabulary gains nothing);
        bounds/direction failures are the typed coronary refusals.  Accepted
        and seam-refused attempts consume the nonce exactly like the action
        route (fail-closed, one sealed audit event each).

        F2_JOB_HANDLE_TRUTH: the validated request is where the job EXISTS.
        A request refused by the byte/envelope/lineage gates above never
        becomes a job (there is no build to own and the synchronous typed
        rejection is the answer).

        F2_JOB_HANDLE_TRUTH, second narrowing (manager decision 2026-09-28):
        a served entry lives exactly as long as the execution that produced it.
        The ``QUEUED`` observation and the whole governed build therefore run
        INSIDE the one scope whose ``finally`` calls ``_seed_job_release``,
        which removes that execution's record whatever happened.  So every
        path out of this method — normal return, refusal, raised — is paired
        with exactly one removal, and when the request has returned its
        identity serves no job entry at all.  That is the whole of the
        registration/retirement pairing the earlier rounds needed a separate
        live registry for: with no record outliving its execution there is
        nothing left to withhold and no cross-execution question to answer.
        The registration is bookkeeping only: which request is accepted or
        refused, what it is refused with, the consumed-nonce accounting, the
        sealed audits and the returned document are exactly what they were,
        because the real rules below run unchanged.
        """

        self._require_state()
        try:
            parsed = parse_action_body(body)
        except ViewerRejectionError as rejection:
            self._consume_nonce(body)
            self._audit_rejection(rejection)
            raise
        try:
            request = self._validate_seed_request(
                parsed, path_session_id=path_session_id
            )
        except ViewerRejectionError as rejection:
            self._consume_nonce(body)
            self._audit_rejection(rejection)
            raise
        # F2_JOB_HANDLE_TRUTH, second narrowing: the ``QUEUED`` observation AND
        # the whole governed build run inside the scope whose ``finally``
        # removes this execution's record (:meth:`_seed_job_release`).  Every
        # path out of this block — normal, refused, raised — is therefore paired
        # with exactly one removal, so a record can never outlive the route that
        # created it and a later poll can never see a word for an execution
        # that has already returned.
        try:
            self._seed_job_queued(request)
            return self._execute_seed(request)
        finally:
            self._seed_job_release(request)

    def apply_seed(
        self,
        *,
        voxel: tuple[int, int, int] | list[int],
        direction: tuple[float, float, float] | list[float],
        client_nonce: str,
        parent_action_nonce: str | None,
        path_session_id: str,
    ) -> dict[str, object]:
        """Apply one physician seed through the governed reseed seam.

        Typed convenience over :meth:`apply_seed_bytes` (identical
        validation): the caller supplies the picked voxel, the set
        direction, and the nonce/lineage pair; this method encodes the
        canonical request bytes and runs the shared byte path.
        """

        body = canonical_json(
            {
                "client_nonce": client_nonce,
                "direction": list(direction),
                "lineage": {
                    "parent_action_nonce": parent_action_nonce,
                    "session_id": path_session_id,
                },
                "voxel": list(voxel),
            }
        ).encode("utf-8")
        return self.apply_seed_bytes(body, path_session_id=path_session_id)

    def _validate_seed_request(
        self, parsed: dict[str, object], *, path_session_id: str
    ) -> SeedRequest:
        """Validate one parsed seed envelope (grammar precedence, reused)."""

        state = self._require_state()
        if state.session_id != path_session_id:
            raise ViewerRejectionError(
                RejectionCode.SESSION_NOT_FOUND,
                "no open viewer session for this id",
            )
        lineage_raw = parsed.get("lineage")
        if not isinstance(lineage_raw, dict) or set(lineage_raw) != _SEED_LINEAGE_KEYS:
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "lineage must be exactly {session_id, parent_action_nonce}",
            )
        lineage: dict[str, object] = lineage_raw
        if lineage["session_id"] != state.session_id:
            raise ViewerRejectionError(
                RejectionCode.SESSION_NOT_FOUND,
                "lineage.session_id must equal the session id in the request path",
            )
        if set(parsed) != _SEED_ENVELOPE_KEYS:
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "seed envelope must be exactly {voxel, direction, client_nonce, lineage}",
            )
        client_nonce = parsed.get("client_nonce")
        if not isinstance(client_nonce, str) or NONCE_RE.fullmatch(client_nonce) is None:
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "client_nonce must be 16-64 characters from [A-Za-z0-9_-]",
            )
        parent = lineage["parent_action_nonce"]
        if parent is not None and (
            not isinstance(parent, str) or NONCE_RE.fullmatch(parent) is None
        ):
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "parent_action_nonce must be a client_nonce or None",
            )
        if client_nonce in state.used_nonces:
            raise ViewerRejectionError(
                RejectionCode.NONCE_REUSE,
                "client_nonce was already seen in this session's lifetime",
            )
        if not state.accepted_nonces:
            if parent is not None:
                raise ViewerRejectionError(
                    RejectionCode.LINEAGE_GAP,
                    "the first action of a session carries parent_action_nonce=None",
                )
        elif parent is None or parent not in state.accepted_nonces:
            raise ViewerRejectionError(
                RejectionCode.LINEAGE_GAP,
                "parent_action_nonce does not reference an accepted action",
            )
        voxel_raw = parsed.get("voxel")
        if (
            not isinstance(voxel_raw, list)
            or len(voxel_raw) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in voxel_raw
            )
        ):
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "voxel must be three integers [i, j, k] (no hidden defaults)",
            )
        direction_raw = parsed.get("direction")
        if (
            not isinstance(direction_raw, list)
            or len(direction_raw) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in direction_raw
            )
        ):
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM,
                "direction must be three numbers (no hidden defaults)",
            )
        return SeedRequest(
            voxel=(int(voxel_raw[0]), int(voxel_raw[1]), int(voxel_raw[2])),
            direction=(
                float(direction_raw[0]),
                float(direction_raw[1]),
                float(direction_raw[2]),
            ),
            client_nonce=client_nonce,
            session_id=str(lineage["session_id"]),
            parent_action_nonce=cast("str | None", parent),
        )

    def _execute_seed(self, request: SeedRequest) -> dict[str, object]:
        """Run one validated seed through the shared reseed seam.

        Phase B first-build: when NO ACTIVE centerline exists at all (the
        real T2 case — a fresh session), the seed builds the INITIAL
        centerline via :meth:`_execute_seed_first_build` instead of refusing.
        Any other overlay state (AMBIGUOUS / INVALID, or a vessel-filtered
        miss while another path is ACTIVE) still refuses fail-closed — the
        first-build path never silently chooses between existing paths.

        F2_SEED_ADMIT (D1): the first-build admission (validate lineage
        against the CURRENT tip + consume the nonce) is atomic under the
        per-engine ``_seed_admit_lock``; the long compute
        (``_execute_seed_first_build`` derivation + ``store.add``) runs
        OUTSIDE the lock.  The reseed path is untouched.  A second
        concurrent same-session first-build is refused with the existing
        closed vocabulary — ``LINEAGE_GAP`` (``parent_action_nonce`` no
        longer names the current session tip) — never silently forked,
        queued, or merged.  No new rejection code.

        F2_JOB_HANDLE_TRUTH: this route OBSERVES the job lifecycle for the
        served state document — ``QUEUED`` on entry, ``RUNNING`` where the
        governed build is entered, and ``REFUSED`` beside the existing typed-
        refusal bookkeeping.  NO completion is observed here: the served
        guarantee is what ONE running execution observed, and it says nothing
        about any identity-level outcome (the acceptance below is still
        performed in full and is still reported by the returned frozen
        document's own ``accepted_nonces`` and ``action_count``, but the handle
        does not restate it as a state).  The observations are pure additions:
        the returned document, the persisted state document, the admission rule
        and every clinical computation are byte-identical to the
        pre-F2_JOB_HANDLE_TRUTH route.

        F2_SEED_NONCE_IRREVERSIBILITY: the re-seed acceptance path below
        re-checks the client nonce against the CURRENT ``used_nonces`` before
        applying anything, so one ``(session_id, client_nonce)`` identity is
        applied at most once.  This is the EXISTING ``_admit_seed_first_build``
        decision reused verbatim, with the EXISTING one-sealed-refusal
        bookkeeping this path already uses for its other refusals — no new
        rejection code, no new status word, no new served key.  A nonce that
        was never used behaves exactly as before on every path (success,
        ``SEED_OUT_OF_BOUNDS``, ``SESSION_OPEN_FAILED``, the first-build
        branch above included).

        Both corrections coexist on this rule.  Note for the reader that the
        ``used_nonces`` bookkeeping further down is the pre-existing behaviour
        and is unchanged; the re-seed re-check above is a separate, later
        guard.  Neither one is read by the serve path: what the handle reports
        is only what this execution observed while it was running.
        """

        state = self._require_state()
        store = self._get_correction_store()
        try:
            artifact = select_active_centerline(
                self._working_artifacts(),
                vessel_label=state.selected_vessel,
                payload_documents=self._working_payload_documents(),
            )
        except CoronaryOverlayRefusalError as refusal:
            if (
                refusal.code == REFUSAL_UNAVAILABLE
                and self._has_no_active_centerline()
            ):
                return self._execute_seed_first_build_serialized(request, store)
            refusal_error = HumanSeamRefusalError(refusal.code, refusal.message)
            if request.client_nonce not in state.used_nonces:
                state.used_nonces.append(request.client_nonce)
            self._audit_seam_refusal(refusal_error)
            self._seed_job_refused(request, refusal_error.code)
            raise refusal_error from None
        # F2_SEED_NONCE_IRREVERSIBILITY: the re-seed acceptance path re-checks
        # the client nonce against the CURRENT ``used_nonces`` BEFORE applying
        # anything.  Both earlier checks — request validation
        # (``_validate_seed_request``) and the first-build admission
        # (``_admit_seed_first_build``) — run BEFORE this point, so a nonce
        # consumed by ANY other request in between (a concurrent seed, a
        # malformed attempt, an ordinary action) is still UNSEEN at validation
        # while already SEEN here.  Without this re-check the re-seed path
        # appended the SAME nonce to ``used_nonces`` AND to
        # ``accepted_nonces`` a second time and advanced ``action_count``
        # twice, so one identity was accepted twice and any terminal record
        # published for it could be overturned (R6-F1 / R6-F2).
        #
        # Fail-closed, and reusing the EXISTING closed vocabulary: the EXISTING
        # ``NONCE_REUSE`` code with the same message the first-build admission
        # uses, raised BEFORE any artifact, payload document, provenance
        # record, store write, served-state movement or acceptance
        # bookkeeping, so the duplicate applies NOTHING.  The nonce is not
        # appended a second time — it is already in ``used_nonces`` by
        # definition here, which is exactly why this refusal needs no consume
        # step of its own — and the one sealed refusal record is the same
        # ``VIEWER_ACTION_REJECTED``/``{"code": "NONCE_REUSE"}`` event the
        # validation-time refusal of the same identity already emits.  A
        # nonce that was never used is unaffected: the first-build branch
        # above returns first, so no first-seed decision changes.
        if request.client_nonce in state.used_nonces:
            duplicate_refusal = ViewerRejectionError(
                RejectionCode.NONCE_REUSE,
                "client_nonce was already seen in this session's lifetime",
            )
            self._audit_rejection(duplicate_refusal)
            raise duplicate_refusal
        evidence_ref = f"{_SEED_EVIDENCE_PREFIX}:{request.client_nonce}"
        self._seed_job_running(request)
        try:
            seed = self._reseed_centerline(
                store,
                artifact,
                i=request.voxel[0],
                j=request.voxel[1],
                k=request.voxel[2],
                direction=request.direction,
                evidence_ref=evidence_ref,
                reason="viewer seed (physician-selected seed -> NEW centerline)",
                preprocessing=(
                    "viewer seed:EDIT (physician-selected seed -> NEW centerline)"
                ),
            )
        except HumanSeamRefusalError as refusal:
            if request.client_nonce not in state.used_nonces:
                state.used_nonces.append(request.client_nonce)
            self._audit_seam_refusal(refusal)
            self._seed_job_refused(request, refusal.code)
            raise
        except ViewerOpenError as open_refusal:
            # Fail-closed session refusal (governed build identity, state
            # access, ...): a REAL refusal the server raised and answered, so
            # the job is REFUSED with that exact existing code.  Note the
            # nonce is NOT consumed here, so this identity can legitimately be
            # retried — which is exactly why the handle says nothing about the
            # identity either way: the recorded refusal is removed with this
            # execution's own record when the route returns, and a later
            # execution of the same nonce records its own observation.  The
            # refusal is recorded either way and is still answered to the
            # client that raised it with its exact real code.
            self._seed_job_refused(request, open_refusal.code.value)
            raise
        state.used_nonces.append(request.client_nonce)
        state.accepted_nonces.append(request.client_nonce)
        state.action_count += 1
        self._audit_seed(request, seed=seed)
        # F2_JOB_HANDLE_TRUTH, second narrowing: NO lifecycle observation is
        # recorded here.  This statement really accepts that identity, and the
        # acceptance is fully performed and fully reported by the frozen
        # document it returns; the handle simply never restates it.  The route's
        # ``finally`` then removes the record, so the identity is not reported
        # at all once this execution has returned — which is abstention, never a
        # fabricated word.
        return self.state_document()

    def _has_no_active_centerline(self) -> bool:
        """True only when NO ACTIVE CENTERLINE exists at all (unfiltered).

        The filtered miss (a vessel selection with no matching path while
        another path is ACTIVE) is NOT an empty session: building beside the
        existing path would silently fork the single-path review surface, so
        it refuses instead of first-building.
        """

        try:
            select_active_centerline(
                self._working_artifacts(),
                vessel_label=None,
                payload_documents=self._working_payload_documents(),
            )
        except CoronaryOverlayRefusalError as refusal:
            return refusal.code == REFUSAL_UNAVAILABLE
        return False

    def _resolve_first_build_volume(self, i: int, j: int, k: int) -> Any:
        """The single session volume containing the seed voxel (Phase B).

        The seed selects its volume by containment (``0 <= i < columns``,
        ``0 <= j < rows``, ``0 <= k < slices`` over each volume's
        ``(slices, rows, columns)`` frame dimensions): no volume at all is a
        seam refusal; containment in none is the typed ``SEED_OUT_OF_BOUNDS``
        refusal (never a silent clamp); containment in several is a seam
        refusal (never a silent pick, Rule 10).
        """

        volumes = list(self._restored.volumes)
        if not volumes:
            raise HumanSeamRefusalError(
                "CENTERLINE_DECISION_REFUSED",
                "the first-build seed route requires a canonical volume in the session",
            )
        containing = [
            volume
            for volume in volumes
            if 0 <= i < int(volume.coordinate_frame.dimensions[2])
            and 0 <= j < int(volume.coordinate_frame.dimensions[1])
            and 0 <= k < int(volume.coordinate_frame.dimensions[0])
        ]
        if not containing:
            dims = "; ".join(
                f"{volume.volume_id}: "
                f"(columns={int(volume.coordinate_frame.dimensions[2])}, "
                f"rows={int(volume.coordinate_frame.dimensions[1])}, "
                f"slices={int(volume.coordinate_frame.dimensions[0])})"
                for volume in volumes
            )
            raise HumanSeamRefusalError(
                CoronaryRefusalCode.SEED_OUT_OF_BOUNDS.value,
                f"seed voxel {(i, j, k)} is outside every session volume "
                f"({dims}): 0 <= i < columns, 0 <= j < rows, "
                "0 <= k < slices is required",
            )
        if len(containing) > 1:
            refs = ", ".join(f"volume_id:{volume.volume_id}" for volume in containing)
            raise HumanSeamRefusalError(
                "CENTERLINE_DECISION_REFUSED",
                f"seed voxel {(i, j, k)} is contained in {len(containing)} "
                f"session volumes ({refs}); the first-build route cannot "
                "choose one",
            )
        return containing[0]

    # ------------------------------------------------------------------
    # F2_JOB_HANDLE_TRUTH: observed seed-job lifecycle of a RUNNING execution
    # ------------------------------------------------------------------

    def _seed_job_queued(self, request: SeedRequest) -> None:
        """The server OBSERVES a validated seed request awaiting its build.

        Written at the top of :meth:`_execute_seed`: the envelope, nonce and
        lineage validated, the server now owns this job, and the governed
        build has not been entered.  This is the RECORD-CREATING observation —
        the only one that may create a record.  The request that creates it
        holds the public handle only PROVISIONALLY, because it has not reached
        the governed admission yet: if a different, already-validated request
        carrying the same ``client_nonce`` really wins that admission, the
        handle is handed to it (see :meth:`_seed_job_admit_owner`) — restarted
        at this same provisional ``QUEUED`` word, because a word this request
        has earned is not this request's to give away — and every observation of
        the creating request becomes a no-op from then on.

        A request refused by envelope/lineage validation before the route is
        entered (a grammar rejection with no build to own) produces NO job: the
        synchronous typed rejection is the answer and the served document makes
        no claim about it.

        F2_JOB_HANDLE_TRUTH, second narrowing: this call and the whole governed
        build run inside the scope whose ``finally`` calls
        :meth:`_seed_job_release`, so the record this creates cannot outlive
        the execution that created it — not on a normal return, not on a
        refusal, and not if anything below raises.
        """

        self._seed_job_observe(request, _SEED_JOB_QUEUED)

    def _seed_job_admit_owner(self, request: SeedRequest) -> None:
        """Hand the public handle to the execution that WON the F2 admission.

        ATT0003 (review finding R2-F1): record creation and first-build
        admission are two different events, so the request that created the
        ``QUEUED`` record is not necessarily the request that really runs the
        governed build.  This method is called by
        :meth:`_execute_seed_first_build_serialized` INSIDE the same
        ``_seed_admit_lock`` hold that just admitted ``request`` — that is,
        only after the unchanged F2 rule (:meth:`_admit_seed_first_build`)
        returned WITHOUT refusing.  It decides nothing about admission: which
        request is admitted, and what is accepted or refused, is exactly what
        the frozen rule decided a statement earlier.

        WHAT THE AWARD GUARANTEES, AND WHAT IT DOES NOT CLAIM: the admitted
        execution can never be excluded from its own public handle, because
        from this statement on it is the owner, so every observation it goes on
        to make is recorded rather than dropped as a non-owner's.  That is the
        whole of what the award decides: WHICH EXECUTION may speak for the
        handle.  It decides nothing about WHAT is said, and in particular it
        never carries a word forward.

        ATT0004 (review finding F1, P1 — the admission award must not inherit
        another execution's observation).  The award is a nested lock
        acquisition and the thread really does hold both locks for its duration:
        the caller holds ``_seed_admit_lock`` and this method takes
        ``_seed_job_lock`` while holding it, releasing only the inner one on
        return.  The order is always ``_seed_admit_lock`` -> ``_seed_job_lock``
        and never the reverse (no path acquires the admission lock while
        holding the job lock), so no deadlock is possible.  That lock order is
        the whole of the atomicity, and it does NOT serialise a competitor's
        refusal: a request that was already refused in an EARLIER hold consumes
        its nonce, audits the refusal and records ``REFUSED`` in
        :meth:`_execute_seed_first_build_serialized`'s ``except`` clause, which
        runs OUTSIDE ``_seed_admit_lock``.  Such a request can therefore be
        running its refusal handler at the very moment this award runs, and its
        recorded ``REFUSED``/``RUNNING`` word is still on the record this award
        is about to take over.

        THE ABSTENTION, WHICH IS THE PRODUCT'S OWN RULE: do not publish a word
        that no live execution earned.  ``request`` is live (its own route is
        inside the same ``try/finally`` as this award), its envelope, nonce and
        lineage really validated, and the governed build has not been entered
        yet — which is exactly and completely what ``QUEUED`` says, and the
        only served word that makes no claim about a particular execution's
        progress.  So the award RESTARTS the handle at that provisional word
        and clears the former owner's ``refusal_code``: no ``REFUSED`` and no
        refusal code of one execution can ever be published for another, and the
        admitted execution's own forward-only ``RUNNING`` observation (rank 1 >
        rank 0) is accepted on top of it.  Nothing is loosened in
        :meth:`_seed_job_observe` to achieve this — merely allowing a backwards
        order would still publish the former execution's refusal before the new
        owner's first observation.  The alternative the review allowed —
        abstaining and letting the record go — was rejected because the record's
        capacity to inform has NOT run out here: the record still belongs to a
        live execution, and dropping it would make the admitted execution's own
        whole build unreportable, which is the one thing this capability exists
        to deliver (reports/q1/F2_PROGRESS_CANCEL_MEASURE_20260923.md).

        The award never moves a handle that was already awarded, never fabricates
        a record that does not exist, and never outlives the execution: the
        record it marks is removed by that execution's own route exit
        (:meth:`_seed_job_release`).  Ownership is therefore not forgeable or
        re-takable by a replayed or reused nonce — only an execution the real
        F2 rule admitted can hold the public handle, and a nonce that was
        already used can never be admitted — but ownership IS transferred here,
        once, by the admission and by nothing else.  The award is not itself a
        lifecycle observation: it never advances a state, it only refuses to
        inherit one.
        """

        with self._seed_job_lock:
            record = self._seed_jobs.get(request.client_nonce)
            if record is None or record.admitted:
                return
            self._seed_jobs[record.client_nonce] = replace(
                record,
                state=_SEED_JOB_QUEUED,
                owner=request,
                admitted=True,
                refusal_code=None,
            )

    def _seed_job_running(self, request: SeedRequest) -> None:
        """The governed build has been ENTERED for this job (F2).

        Written where the long derivation starts, i.e. immediately before the
        first-build compute (post-admission) or the reseed compute.  This is
        the server's own claim that it still owns the work.
        """

        self._seed_job_observe(request, _SEED_JOB_RUNNING)

    def _seed_job_refused(self, request: SeedRequest, refusal_code: str) -> None:
        """A REAL refusal the governed route raised for this job (F2).

        ``refusal_code`` is always the code the server actually raised, from
        the EXISTING closed vocabularies: the F2 admission ``RejectionCode``
        (e.g. ``LINEAGE_GAP``, ``NONCE_REUSE``), a coronary seam refusal code
        (e.g. ``SEED_OUT_OF_BOUNDS``), or a fail-closed ``ViewerOpenError``
        ``RejectionCode``.  Never a heuristic, never a timeout, never an idle
        or retry inference.  A refusal is a fact about the execution that
        RAISED it: if this request does not own the record (a duplicate or
        replayed nonce, refused for its own valid reason), the observation is
        dropped and no other execution's job is touched — a real refusal is
        never attributed to the wrong job.

        The word is served only for as long as THIS execution is in flight.  The
        route-exit hook (:meth:`_seed_job_release`) then removes the record, so
        the identity serves no job entry at all once the refusal's own
        execution has returned — whether or not that refusal consumed the
        nonce, and whatever a later attempt of the same nonce really does.  A
        later execution of the same identity that is really refused with a
        DIFFERENT code records its OWN observation, and the served
        ``refusal_code`` is always the code of the refusal the served state
        reports: it is read straight off the record the serving execution
        recorded, never borrowed from another attempt.
        """

        self._seed_job_observe(request, _SEED_JOB_REFUSED, refusal_code=refusal_code)

    def _seed_job_release(self, request: SeedRequest) -> None:
        """THE ROUTE-EXIT HOOK: nothing outlives its execution (F2 truthfulness).

        Called from the ``finally`` of :meth:`apply_seed_bytes`, so it runs on
        EVERY path out of that route: a real success, a real refusal, a
        cancelled connection and an escaping exception alike.  It REMOVES the
        record of the execution that owns it, whatever state that record holds.
        That is the whole of the guarantee: an execution's observations are
        served while it is running and are gone once it returns, so the server
        can neither publish a stale ``QUEUED``/``RUNNING`` long after it
        stopped holding the work, nor keep a refusal around to be read as the
        identity's outcome after a later attempt of the same nonce has run.

        It can never imply the work stopped either: a job whose execution has
        returned is simply not reported.  And it claims nothing about the
        identity's final outcome — the acceptance of a really successful seed
        is untouched and remains fully reported by the frozen document's
        ``accepted_nonces`` and ``action_count``; only the handle's per-execution
        entry is withdrawn.

        The removal is OWNERSHIP-BOUND like every other observation: a request
        that does not hold the record's public handle removes nothing, so a
        refused duplicate can never delete the record of the execution that
        still owns the work.  It is idempotent (a repeated release, or a release
        by a request that never created a record, removes nothing), and it needs
        no de-registration of any kind: with no live registry left, the record's
        own existence IS the registration, and it is gone exactly when the route
        that created it has returned.
        """

        with self._seed_job_lock:
            record = self._seed_jobs.get(request.client_nonce)
            if record is None or record.owner is not request:
                return
            del self._seed_jobs[request.client_nonce]

    def _seed_job_observe(
        self, request: SeedRequest, state: str, *, refusal_code: str | None = None
    ) -> None:
        """Record one observed lifecycle state, keyed by client_nonce.

        OWNERSHIP FIRST (F2_JOB_HANDLE_TRUTH ATT0002): every transition is
        bound to the request EXECUTION that holds the record's public handle.
        If a record already exists for this ``client_nonce`` and it belongs to a
        different request, the observation is DROPPED — a non-holder never
        transitions, mutates or deletes another execution's record, and the
        holder keeps the truthful state its own facts earned.  Identity is
        object identity (``record.owner is request``), so a replayed or reused
        nonce cannot forge or inherit ownership.  The ONE exception is the
        admission award (:meth:`_seed_job_admit_owner`), which is not an
        observation but a transfer: it moves ``owner`` to the admitted execution
        and RESTARTS the record at the provisional ``QUEUED`` word, so a former
        owner's observed state and its refusal code are never carried into the
        new owner's handle — and the forward-only order below then accepts that
        execution's own ``RUNNING`` observation on top of it.  The internal
        ``admitted`` marker travels with the record on every transition, so the
        admission award of ATT0003 can never be dropped by a later observation.

        The state words are the frozen four; a lifecycle state is only ever
        recorded for a job the server already OBSERVED as ``QUEUED``, and only
        ever forward, so the served document can never show an orphan or a
        skipped predecessor.  Only THREE of the four are ever RECORDED —
        ``QUEUED``, ``RUNNING`` and ``REFUSED``: no production path observes a
        completion (the withdrawn identity-level claim).  No retention rule
        exists here either: the record is removed by its own owner's route exit
        (:meth:`_seed_job_release`), so the served set is bounded by the
        in-flight executions and needs no eviction.  No I/O under the lock.
        """

        if state not in _SEED_JOB_STATES:  # pragma: no cover - frozen vocabulary
            raise ValueError(f"seed job state {state!r} is outside the frozen four")
        with self._seed_job_lock:
            known = self._seed_jobs.get(request.client_nonce)
            if known is not None and known.owner is not request:
                return  # a non-owner never touches another execution's record
            if state == _SEED_JOB_QUEUED:
                if known is not None:
                    return  # a job already exists for this client_nonce
            elif known is None or _SEED_JOB_ORDER[state] <= _SEED_JOB_ORDER[known.state]:
                return
            self._seed_jobs[request.client_nonce] = SeedJobRecord(
                session_id=request.session_id,
                client_nonce=request.client_nonce,
                state=state,
                owner=request,
                admitted=known.admitted if known is not None else False,
                refusal_code=refusal_code if state == _SEED_JOB_REFUSED else None,
            )

    def _seed_jobs_document(self) -> list[dict[str, object]]:
        """The observed seed jobs of the executions that are RUNNING right now.

        Truthful by construction, and rule-free by construction: every entry
        names the job it reports, the frozen lifecycle words are the only words
        emitted, and the keys per state are closed (``session_id``,
        ``client_nonce``, ``state``, plus ``refusal_code`` on a refusal).  No
        percentage, ratio, ETA, remaining time, duration or estimate appears
        here or anywhere in the served document.

        F2_JOB_HANDLE_TRUTH, second narrowing (manager decision 2026-09-28):
        there is no decision left to make here.  A record exists only while the
        execution that owns it is still running — the route-exit hook
        (:meth:`_seed_job_release`) removes it on every path out of the
        route — so every record in this set belongs to an execution that has
        not returned, and the served document is simply that execution's own
        latest observation.  Nothing is withheld, nothing is retained after an
        execution ends, and no execution other than the owner can speak for a
        record, so the serve path reads no other state at all: no irreversibility
        predicate, no acceptance evidence, no nonce ledger, no live registry and
        no second snapshot pass.  The record is serialized exactly as observed
        (:meth:`SeedJobRecord.to_document`), so the served ``refusal_code`` is
        always the code of the very refusal the served state reports.

        ATT0004 (review finding F1, P1) closes the last gap in that sentence:
        "that execution's own latest observation" is now true even across the
        admission award, because the award restarts the handle at the new
        owner's own provisional ``QUEUED`` word instead of inheriting the former
        owner's ``RUNNING`` or ``REFUSED``.  So every word served here, for every
        nonce, at every instant, is a word the execution that currently owns the
        record earned for itself.

        One ``_seed_job_lock`` hold, no I/O, no rule evaluation: it is a pure
        read of what the running executions have already observed.
        """

        with self._seed_job_lock:
            return [record.to_document() for record in self._seed_jobs.values()]

    def _admit_seed_first_build(self, request: SeedRequest) -> None:
        """Atomically validate + consume-nonce one first-build admission.

        F2_SEED_ADMIT (D1): runs UNDER the per-engine ``_seed_admit_lock``
        (called only from :meth:`_execute_seed_first_build_serialized`) while
        the long compute runs OUTSIDE the lock.  The admission re-checks,
        against the CURRENT tip at admission time:

        * the client nonce is still unseen (else ``NONCE_REUSE`` — the
          fail-closed duplicate refusal, unchanged from the single-threaded
          path);
        * the lineage still names the current tip, with the exact
          single-threaded rules: no accepted action yet implies the parent
          must be ``None``; otherwise the parent must name an accepted nonce
          (else the EXISTING closed-vocabulary ``LINEAGE_GAP``);
        * no ACTIVE centerline appeared since the pre-lock check (else
          ``LINEAGE_GAP`` — the ``parent_action_nonce=None`` first-build
          claim no longer names the current session tip);
        * no other first-build admission is already in flight on this
          session (else ``LINEAGE_GAP`` — the second concurrent
          first-build's parent lineage no longer names the current tip,
          which is now the admitted-but-not-yet-built nonce; never silently
          forked, queued, or merged; no new rejection code).

        On success the nonce is consumed into ``used_nonces`` and recorded
        as the in-flight admission.  The accepted-chain append plus the
        ``action_count`` increment still happen in
        :meth:`_execute_seed_first_build` after the compute completes,
        exactly as in the single-threaded path.  A derivation refusal later
        leaves the admitted nonce consumed — fail-closed, identical to the
        pre-existing single-threaded behavior where refusal paths append to
        ``used_nonces`` without touching ``accepted_nonces``.
        """

        state = self._require_state()
        if request.client_nonce in state.used_nonces:
            raise ViewerRejectionError(
                RejectionCode.NONCE_REUSE,
                "client_nonce was already seen in this session's lifetime",
            )
        if not state.accepted_nonces:
            if request.parent_action_nonce is not None:
                raise ViewerRejectionError(
                    RejectionCode.LINEAGE_GAP,
                    "the first action of a session carries parent_action_nonce=None",
                )
        elif (
            request.parent_action_nonce is None
            or request.parent_action_nonce not in state.accepted_nonces
        ):
            raise ViewerRejectionError(
                RejectionCode.LINEAGE_GAP,
                "parent_action_nonce does not reference an accepted action",
            )
        if not self._has_no_active_centerline():
            raise ViewerRejectionError(
                RejectionCode.LINEAGE_GAP,
                "a centerline is already present: the first-build lineage "
                "no longer names the current session tip",
            )
        if self._seed_first_build_inflight is not None:
            raise ViewerRejectionError(
                RejectionCode.LINEAGE_GAP,
                "a first-build seed admission is already in flight on this "
                "session: parent_action_nonce no longer names the current tip",
            )
        state.used_nonces.append(request.client_nonce)
        self._seed_first_build_inflight = request.client_nonce

    def _execute_seed_first_build_serialized(
        self, request: SeedRequest, store: DerivedArtifactStore
    ) -> dict[str, object]:
        """Serialize first-build admission; compute outside the lock.

        F2_SEED_ADMIT (D1): validate + nonce-consume admission is atomic
        under ``_seed_admit_lock``; the long ``_execute_seed_first_build``
        derivation + ``store.add`` runs OUTSIDE the lock so
        ThreadingHTTPServer concurrency for distinct sessions and state
        polls is preserved.  A second concurrent same-session first-build is
        refused with the existing closed vocabulary (``LINEAGE_GAP`` —
        preferred: its parent lineage no longer names the current session
        tip; ``NONCE_REUSE`` for a true duplicate nonce).  A refused second
        admission consumes its seen nonce and emits exactly one sealed
        rejection audit — matching the ``apply_seed_bytes`` fail-closed
        contract for rejected attempts (seen nonces are consumed;
        one sealed event per attempt).  The reseed path is untouched.
        First-build derivation refusals (``HumanSeamRefusalError`` from
        ``_execute_seed_first_build``) keep the exact pre-existing behavior:
        the admitted nonce stays consumed in ``used_nonces`` and exactly one
        sealed refusal audit is emitted.  The in-flight marker is always
        released, so a later retry with a fresh nonce follows the exact
        single-threaded rules.

        F2_JOB_HANDLE_TRUTH: the lifecycle observation rides the same
        bookkeeping without touching the rule — a refused F2 admission is
        RECORDED as the job's ``REFUSED`` state (with the EXACT code this
        function raises), and the admission lock is acquired exactly as often
        as before.  It is held for one extra in-memory handle award after a
        SUCCESSFUL admission (:meth:`_seed_job_admit_owner`): one dict lookup
        and at most one record replacement, no I/O, no derivation, no rule
        evaluation, and no effect on which request is admitted.  That award is
        what makes the served handle belong to the execution that really
        builds, even when a different request created the record first — and it
        is load-bearing for the served guarantee, because the observations the
        admitted execution goes on to make are recorded against the awarded
        handle and would otherwise be dropped as a non-owner's.  An admission
        that CONSUMED the nonce is still not an acceptance: an admitted
        execution that then really fails is served ``REFUSED`` with its real
        code only while it is running, and its own record is removed by its
        route exit like every other execution's.

        ATT0004 (review finding F1, P1): the ``except ViewerRejectionError``
        clause below runs OUTSIDE ``_seed_admit_lock``, and that is exactly why
        the award is not allowed to inherit a word.  A request refused in an
        EARLIER hold can still be consuming its nonce, auditing and recording
        its ``REFUSED`` while a later request is being admitted here, so its
        record may still be sitting on this nonce when
        :meth:`_seed_job_admit_owner` runs.  The award therefore restarts the
        handle at the provisional ``QUEUED`` word and clears that ``refusal_code``
        (see its docstring): the admitted execution's own ``RUNNING``
        observation, one line below, is then accepted on top of it instead of
        being rejected by the forward-only order.
        """

        state = self._require_state()
        try:
            with self._seed_admit_lock:
                self._admit_seed_first_build(request)
                # ATT0003 (review finding R2-F1): the public handle follows the
                # ADMISSION, not the record creation, and the award is made in
                # this same critical section — so a competing request that is
                # refused after this hold can never speak for the handle this
                # execution was really awarded.  ATT0004 (F1, P1) completes that
                # claim: a request refused in an EARLIER hold reaches its own
                # refusal handler outside this lock, so the award abstains from
                # inheriting any word such a request left behind.  No rule is
                # evaluated here and no decision is changed: the award is one
                # in-memory dict lookup and at most one record replacement.
                self._seed_job_admit_owner(request)
        except ViewerRejectionError as admission_refusal:
            if request.client_nonce not in state.used_nonces:
                state.used_nonces.append(request.client_nonce)
            self._audit_rejection(admission_refusal)
            self._seed_job_refused(request, admission_refusal.code.value)
            raise
        try:
            self._seed_job_running(request)
            return self._execute_seed_first_build(request, store)
        except HumanSeamRefusalError as first_refusal:
            if request.client_nonce not in state.used_nonces:
                state.used_nonces.append(request.client_nonce)
            self._audit_seam_refusal(first_refusal)
            self._seed_job_refused(request, first_refusal.code)
            raise
        except ViewerOpenError as open_refusal:
            # Fail-closed session refusal (governed build identity, state
            # access, ...).  It is a REAL refusal the server raised and
            # answered, so the job is REFUSED with that exact existing code.
            self._seed_job_refused(request, open_refusal.code.value)
            raise
        finally:
            with self._seed_admit_lock:
                if self._seed_first_build_inflight == request.client_nonce:
                    self._seed_first_build_inflight = None

    def _execute_seed_first_build(
        self, request: SeedRequest, store: DerivedArtifactStore
    ) -> dict[str, object]:
        """Build the INITIAL centerline from a physician seed (Phase B).

        The governed ``build_centerline`` derivation runs over the
        seed-selected session volume with the frozen default
        ``VesselnessParameters`` (no per-request parameters exist on the
        route, so nothing is defaulted silently — the recorded
        ``parameters_sha256`` names exactly what ran).  The NEW artifact is
        registered directly (``store.add``): there is no predecessor, so no
        CORRECTION artifact, no supersede, and no STALE fan-out — nothing to
        stale.  Provenance is the full §44 centerline record with the
        initial-build runtime marker (distinct from the reseed decision
        runtime).  No CPR view is framed: its framing parameters come from a
        predecessor view, and fabricating them would be a hidden default.
        Derivation failures (including the ``NOT_ANALYZABLE`` abstention)
        surface as typed seam refusals, never as fabricated paths.
        """

        state = self._require_state()
        volume = self._resolve_first_build_volume(
            request.voxel[0], request.voxel[1], request.voxel[2]
        )
        frame = volume.coordinate_frame
        if (
            len(request.direction) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in request.direction
            )
            or any(not math.isfinite(float(value)) for value in request.direction)
        ):
            raise HumanSeamRefusalError(
                CoronaryRefusalCode.SEED_DIRECTION_INVALID.value,
                "seed direction must be a finite 3-vector",
            )
        if math.sqrt(sum(float(value) ** 2 for value in request.direction)) <= 0.0:
            raise HumanSeamRefusalError(
                CoronaryRefusalCode.SEED_DIRECTION_INVALID.value,
                "seed direction must be non-zero",
            )
        try:
            seed = Seed(
                i=request.voxel[0],
                j=request.voxel[1],
                k=request.voxel[2],
                direction=request.direction,
            )
        except CoronaryValidationError as exc:
            raise HumanSeamRefusalError(exc.code.value, str(exc)) from None
        transform, transform_ref, transform_provenance_ref = self._persisted_transform(frame=frame)
        loader = CanonicalVolumeLoader(self._restored.paths.blobs)
        try:
            envelope = loader.read_envelope(volume.blob_ref.sha256)
            array = loader.read_array(
                volume.blob_ref.sha256,
                envelope=envelope,
            )
        except (OSError, ValueError, KeyError) as exc:
            raise HumanSeamRefusalError(
                "CENTERLINE_DECISION_REFUSED",
                f"the source volume blob is unreadable ({exc})",
            ) from None
        scalar_domain = envelope.get("scalar_domain") if isinstance(envelope, dict) else None
        if scalar_domain not in {"STORED", "MODALITY_TRANSFORMED", "HU"}:
            raise HumanSeamRefusalError(
                "CENTERLINE_DECISION_REFUSED",
                "the source volume envelope carries no usable scalar_domain; "
                "refusing a silent domain default",
            )
        parameters = VesselnessParameters()
        new_centerline_id = str(self._id_factory())
        new_provenance_id = str(self._id_factory())
        try:
            built = build_centerline(
                volume,
                transform,
                frame,
                loader=loader,
                seed=seed,
                parameters=parameters,
                provenance_ref=f"provenance:{new_provenance_id}",
                transform_artifact_ref=transform_ref,
                study_instance_uid=self._restored.study.dicom_study_instance_uid,
                frame_of_reference_uid=frame.frame_of_reference_uid,
                scalar_domain=str(scalar_domain),
                source_array=array,
                vessel_label=state.selected_vessel,
                artifact_id=new_centerline_id,
                created_by=_VIEWER_CREATED_BY,
            )
        except CoronaryValidationError as exc:
            raise HumanSeamRefusalError("CENTERLINE_DECISION_REFUSED", str(exc)) from None
        mask_blob = cast(dict[str, object], built.payload["vessel_mask_blob_ref"])
        centerline_record = centerline_provenance(
            subject_ref=f"centerline_artifact_id:{new_centerline_id}",
            volume=volume,
            seed=seed,
            parameters=parameters,
            mask_sha256=built.path.diagnostics.mask_sha256,
            mask_blob_sha256=str(mask_blob["sha256"]),
            response_sha256=built.path.diagnostics.response_sha256,
            transform_artifact_ref=transform_ref,
            transform_digest=canonical_digest(transform),
            volume_provenance_ref=volume.provenance_ref,
            transform_provenance_ref=transform_provenance_ref,
            software_version=_VIEWER_SOFTWARE,
            code_commit_sha=self._require_build_commit_sha(),
            runtime=_INITIAL_BUILD_RUNTIME,
            created_by=_VIEWER_CREATED_BY,
            at=self._now(),
            provenance_id=new_provenance_id,
        )
        try:
            store.add(built.artifact)
        except (StateTransitionError, CanonicalValidationError) as exc:
            raise HumanSeamRefusalError("CENTERLINE_DECISION_REFUSED", str(exc)) from None
        self._new_coronary_artifacts.append(new_centerline_id)
        self._new_coronary_documents[new_centerline_id] = dict(built.payload)
        self._new_provenance.append(centerline_record)
        # F2_SEED_ADMIT: the admission already consumed this nonce into
        # ``used_nonces`` (fail-closed duplicate accounting is admission
        # state, not completion state) — append here ONLY if admission did
        # not (keeps the pre-existing non-admission callers exact).
        if request.client_nonce not in state.used_nonces:
            state.used_nonces.append(request.client_nonce)
        state.accepted_nonces.append(request.client_nonce)
        state.action_count += 1
        self._audit_seed(request, seed=seed)
        # F2_JOB_HANDLE_TRUTH: NO lifecycle observation is recorded at this
        # acceptance statement.  The artifact is registered, the nonce is in the
        # accepted chain and the sealed accepted audit is out — never inferred
        # from a timer or a heuristic — and all three are exactly what they
        # always were; the job handle simply publishes no completion, so the
        # route's ``finally`` removes the record with this execution.
        return self.state_document()

    def _audit_seed(self, request: SeedRequest, *, seed: Seed) -> None:
        """One sealed audit event for an accepted physician seed."""

        if self._sink is None:
            return
        self._sink.record(
            entity=_AUDIT_ENTITY,
            entity_id=request.client_nonce,
            event_type="VIEWER_ACTION_ACCEPTED",
            to_value="SEED",
            from_value=None,
            evidence_ref=self._require_state().session_id,
            payload={
                "action": "SEED",
                "voxel": [seed.i, seed.j, seed.k],
                "direction": list(seed.unit_direction()),
                "parent_action_nonce": request.parent_action_nonce,
            },
            actor_role="VIEWER",
        )

    def _frame_successor_cpr_view(
        self,
        store: DerivedArtifactStore,
        *,
        predecessor_centerline_id: str,
        successor_centerline: Any,
        successor_provenance: Any,
        volume: Any,
        transform: CoordinateTransform,
        transform_ref: str,
        source_array: Any,
    ) -> None:
        """Frame the EDIT-successor CPR view (F1-PREFERRED, Decision B/C).

        Reads the single ACTIVE predecessor ``ccta.cpr.view.v1`` payload from
        the working documents and reuses its framing params verbatim
        (``s_spacing_mm`` / ``radial_spacing_mm`` / ``radial_count`` /
        ``method`` / ``out_of_bounds_fill_value``) plus its source
        volume/transform/provenance refs — mirroring the centerline rebuild
        pattern.  A session without an ACTIVE predecessor CPR simply keeps
        its previous CPR state (no CPR is fabricated for a session that never
        had one); any framing failure is a typed seam refusal (never a
        silent skip, never stale bytes served as current).
        """

        from visualization.cpr import build_cpr_view, cpr_provenance

        predecessor_id: str | None = None
        predecessor_payload: dict[str, object] | None = None
        for artifact_id in self._working_artifact_ids():
            try:
                candidate = store.get(artifact_id)
            except KeyError:
                continue
            if (
                candidate.kind is not ArtifactKind.MPR_VIEW
                or candidate.payload.payload_schema != "ccta.cpr.view.v1"
            ):
                continue
            document = self._working_payload_documents().get(artifact_id)
            if isinstance(document, dict):
                predecessor_id = artifact_id
                predecessor_payload = document
                break
        if predecessor_id is None or predecessor_payload is None:
            return
        try:
            s_spacing = float(predecessor_payload["s_spacing_mm"])  # type: ignore[arg-type]
            radial_spacing = float(predecessor_payload["radial_spacing_mm"])  # type: ignore[arg-type]
            radial_count_raw = predecessor_payload["radial_count"]
            method = str(predecessor_payload["method"])
            fill = float(predecessor_payload["out_of_bounds_fill_value"])  # type: ignore[arg-type]
            study_uid = str(predecessor_payload["study_instance_uid"])
            series_uid_raw = predecessor_payload.get("series_instance_uid")
            frame_uid_raw = predecessor_payload.get("frame_of_reference_uid")
        except (KeyError, TypeError, ValueError) as exc:
            raise HumanSeamRefusalError(
                "CENTERLINE_DECISION_REFUSED",
                f"the predecessor CPR framing params are unreadable ({exc})",
            ) from None
        if (
            isinstance(radial_count_raw, bool)
            or not isinstance(radial_count_raw, int)
            or radial_count_raw < 1
        ):
            raise HumanSeamRefusalError(
                "CENTERLINE_DECISION_REFUSED",
                "the predecessor CPR radial_count is unreadable",
            )
        new_cpr_id = str(self._id_factory())
        new_cpr_provenance_id = str(self._id_factory())
        successor_artifact = successor_centerline.artifact
        try:
            framed = build_cpr_view(
                volume,
                transform,
                centerline=successor_centerline,
                s_spacing_mm=s_spacing,
                radial_spacing_mm=radial_spacing,
                radial_count=int(radial_count_raw),
                method=method,
                out_of_bounds_fill_value=fill,
                loader=CanonicalVolumeLoader(self._restored.paths.blobs),
                source_array=source_array,
                transform_artifact_ref=transform_ref,
                provenance_ref=f"provenance:{new_cpr_provenance_id}",
                study_instance_uid=study_uid,
                series_instance_uid=(
                    str(series_uid_raw) if series_uid_raw is not None else None
                ),
                frame_of_reference_uid=(
                    str(frame_uid_raw) if frame_uid_raw is not None else None
                ),
                release_lane=successor_artifact.release_lane,
                created_by=_VIEWER_CREATED_BY,
                artifact_id=new_cpr_id,
            )
        except Exception as exc:  # noqa: BLE001 - a failed recompute is a typed refusal
            raise HumanSeamRefusalError(
                "CENTERLINE_DECISION_REFUSED",
                f"the EDIT-successor CPR recompute failed ({exc})",
            ) from None
        cpr_record = cpr_provenance(
            subject_ref=f"mpr_artifact_id:{new_cpr_id}",
            volume=volume,
            centerline_artifact=successor_artifact,
            sampling_parameters_sha256=str(
                framed.payload["sampling_parameters_sha256"]
            ),
            volume_provenance_ref=successor_provenance.provenance_id,
            centerline_provenance_ref=successor_provenance.provenance_id,
            software_version=_VIEWER_SOFTWARE,
            code_commit_sha=self._require_build_commit_sha(),
            runtime="python-3.12 (viewer centerline decision)",
            created_by=_VIEWER_CREATED_BY,
            at=self._now(),
            provenance_id=new_cpr_provenance_id,
        )
        store.add(framed.artifact)
        self._pixel_blobs[str(framed.pixel_blob_ref["sha256"])] = bytes(
            framed.pixels.tobytes(order="C")
        )
        self._new_coronary_artifacts.append(new_cpr_id)
        self._new_coronary_documents[new_cpr_id] = dict(framed.payload)
        self._new_provenance.append(cpr_record)

    def _get_correction_store(self) -> DerivedArtifactStore:
        """The append-only correction store over a working journal clone.

        The store is seeded from the session's artifact records; the seam's
        new events land on the working log and are merged into the saved
        journal at persist time (the restored log itself is never rewritten).
        """

        if self._correction_store is not None:
            return self._correction_store
        store = DerivedArtifactStore(EventLog(), InvalidationDAG())
        for artifact in self._restored.artifacts:
            if artifact.artifact_id in store:
                continue
            store.add(artifact)
        self._correction_store = store
        return store

    def _persisted_transform(
        self, frame: FrameSpec | None = None
    ) -> tuple[CoordinateTransform, str, str]:
        """The restored session's persisted versioned transform.

        Returns ``(transform, coordinate_transform:<id>, provenance_ref)``;
        an absent/unresolvable transform is a session-open failure.  When a
        ``frame`` is supplied, the transform describing exactly that frame's
        affine is preferred (a session can carry several persisted grids).
        """

        candidates: list[tuple[CoordinateTransform, str, str]] = []
        for artifact in self._restored.artifacts:
            if artifact.kind != ArtifactKind.COORDINATE_TRANSFORM:
                continue
            document = self._restored.payload_documents.get(artifact.artifact_id)
            if document is None:
                continue
            from apps.workstation.session_store import decode_value

            decoded = decode_value(document)
            if not isinstance(decoded, CoordinateTransform):
                continue
            candidates.append(
                (
                    decoded,
                    f"coordinate_transform:{artifact.artifact_id}",
                    artifact.provenance_ref,
                )
            )
        if frame is not None:
            expected = invert_matrix(frame.voxel_to_world_matrix())
            for candidate, ref, provenance_ref in candidates:
                try:
                    world_to_voxel = (
                        candidate
                        if candidate.from_space is SpaceKind.WORLD
                        and candidate.to_space is SpaceKind.VOXEL
                        else candidate.inverse
                    )
                except (CanonicalValidationError, ValueError):
                    continue
                if world_to_voxel.matrix == expected:
                    return (candidate, ref, provenance_ref)
        if candidates:
            return candidates[0]
        raise ViewerOpenError(
            RejectionCode.SESSION_OPEN_FAILED,
            ["the correction route requires the persisted versioned transform"],
        )

    def _build_qc_input(self) -> QCInput:
        """Build the canonical QCInput from the restored session (read-only)."""

        restored = self._restored
        if not restored.volumes or not restored.series:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED,
                ["the correction route requires a canonical volume in the session"],
            )
        volume = restored.volumes[0]
        series_uids = set(volume.source_series_refs)
        series = next(
            (item for item in restored.series if item.series_id in series_uids),
            restored.series[0],
        )
        blob_loader = CanonicalVolumeLoader(restored.paths.blobs)
        array = blob_loader.read_array(
            volume.blob_ref.sha256,
            envelope=blob_loader.read_envelope(volume.blob_ref.sha256),
        )
        readonly = array.copy(order="C")
        readonly.setflags(write=False)
        source_root = Path(series.original_dicom_ref.root_path)
        instances = tuple(
            read_dicom_instance(source_root / entry.file_path)
            for entry in series.manifest.instances
        )
        transform, transform_ref, transform_provenance_ref = self._persisted_transform()
        return QCInput(
            volume=volume,
            array=readonly,
            source_manifest_sha256=series.original_dicom_ref.manifest_sha256,
            instances=instances,
            instance_digests={
                instance.sop_instance_uid: hashlib.sha256(
                    instance.pixel_data
                ).hexdigest()
                for instance in instances
            },
            scalar_domain="STORED",
            intensity_unit="STORED",
            transform=transform,
            transform_ref=transform_ref,
            transform_sha256=canonical_digest(transform),
            transform_provenance_ref=transform_provenance_ref,
            geometry_base=GeometryBase(f"volume_geometry:{volume.volume_id}", 1),
            subject_refs=(self._study_id,),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, store: DerivedStateStoreManager) -> str:
        """Save the WS-02 session including the viewer documents (total-or-fail).

        QC assessment records follow the accepted 001-C persistence pattern:
        the sealed 13-key payload documents ride ``payload_documents`` under
        their ``qc_artifact:<ULID>`` refs; the ``DerivedArtifact`` records do
        NOT ride the snapshot (a QC record's ``provenance_ref`` is the
        prefixed ``provenance:<ULID>`` aggregate ref, which the snapshot's
        raw-id provenance index cannot resolve — the accepted 001-C flow
        persists exactly the same way).
        """

        self._require_state()
        record = self._require_session_record()
        documents = dict(self._restored.payload_documents)
        for artifact_id, document in self._new_qc_documents.items():
            documents[f"qc_artifact:{artifact_id}"] = document
        for artifact_id, document in self._new_coronary_documents.items():
            documents[artifact_id] = document
        documents[self.session_key()] = record.to_document()
        documents[self.state_key()] = self.state_document()
        artifacts: list[DerivedArtifact] = []
        for artifact in self._restored.artifacts:
            if (
                self._correction_store is not None
                and artifact.artifact_id in self._correction_store
            ):
                artifacts.append(self._correction_store.get(artifact.artifact_id))
            else:
                artifacts.append(artifact)
        if self._correction_store is not None:
            restored_ids = {artifact.artifact_id for artifact in self._restored.artifacts}
            for artifact_id in self._new_coronary_artifacts:
                if artifact_id in restored_ids:
                    continue
                artifacts.append(self._correction_store.get(artifact_id))
        existing_provenance = {
            record_id.provenance_id for record_id in self._restored.provenance
        }
        provenance = list(self._restored.provenance)
        for new_record in self._new_provenance:
            if new_record.provenance_id in existing_provenance:
                continue
            provenance.append(new_record)
        snapshot = SessionSnapshot(
            study=self._restored.study,
            series=self._restored.series,
            volumes=self._restored.volumes,
            artifacts=tuple(artifacts),
            payload_documents=documents,
            provenance=tuple(provenance),
            log=self._merged_log(),
            skipped_instances=self._restored.skipped_instances,
            pixel_blobs=dict(self._pixel_blobs),
        )
        return store.save(snapshot)

    def _merged_log(self) -> EventLog:
        """The restored journal plus this engine's new seeds/events (append-only)."""

        log = EventLog()
        seeded: set[tuple[str, str]] = set()
        for (entity, entity_id), fields in self._restored.log.initial_states():
            log.seed(entity, entity_id, dict(fields))
            seeded.add((entity, entity_id))
        for event in self._restored.log.events:
            log.append_event(event)
        if self._correction_store is not None:
            working = self._correction_store.event_log
            for (entity, entity_id), fields in working.initial_states():
                if (entity, entity_id) in seeded:
                    continue
                log.seed(entity, entity_id, dict(fields))
                seeded.add((entity, entity_id))
            replayed = {
                (event.entity, event.entity_id, event.event_type, event.to_value, event.at)
                for event in self._restored.log.events
            }
            for event in working.events:
                key = (event.entity, event.entity_id, event.event_type, event.to_value, event.at)
                if key in replayed:
                    continue
                log.append_event(event)
        return log

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _audit_action(self, parsed: ParsedAction) -> None:
        if self._sink is None:
            return
        self._sink.record(
            entity=_AUDIT_ENTITY,
            entity_id=parsed.client_nonce,
            event_type="VIEWER_ACTION_ACCEPTED",
            to_value=parsed.action,
            from_value=None,
            evidence_ref=self._require_state().session_id,
            payload={
                "action": parsed.action,
                "plane": parsed.view_plane,
                "frame_index": parsed.view_frame_index,
                "parent_action_nonce": parsed.parent_action_nonce,
            },
            actor_role="VIEWER",
        )

    def _audit_rejection(self, rejection: ViewerRejectionError) -> None:
        if self._sink is None:
            return
        state = self._state
        self._sink.record(
            entity=_AUDIT_ENTITY,
            entity_id=rejection.code.value,
            event_type="VIEWER_ACTION_REJECTED",
            to_value=rejection.code.value,
            from_value=None,
            evidence_ref=state.session_id if state else None,
            payload={"code": rejection.code.value},
            actor_role="VIEWER",
        )

    def _audit_seam_refusal(self, refusal: HumanSeamRefusalError) -> None:
        if self._sink is None:
            return
        state = self._state
        self._sink.record(
            entity=_AUDIT_ENTITY,
            entity_id=refusal.code,
            event_type="VIEWER_ACTION_REJECTED",
            to_value=refusal.code,
            from_value=None,
            evidence_ref=state.session_id if state else None,
            payload={"code": refusal.code},
            actor_role="VIEWER",
        )

    # ------------------------------------------------------------------
    # Read surfaces
    # ------------------------------------------------------------------

    def _require_state(self) -> ViewerRuntimeState:
        if self._state is None:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED, ["viewer session is not open"]
            )
        return self._state

    def _require_session_record(self) -> ViewerSessionRecord:
        if self._session_record is None:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED, ["viewer session is not open"]
            )
        return self._session_record

    def _require_geometry(self) -> dict[str, list[GeometryFrame]]:
        if self._geometry is None:
            raise ViewerOpenError(
                RejectionCode.SESSION_OPEN_FAILED, ["viewer session is not open"]
            )
        return self._geometry

    def state_document(self) -> dict[str, object]:
        """The closed ``ccta.viewer.state.v1`` persistence document."""

        state = self._require_state()
        return {
            "schema_version": STATE_SCHEMA,
            "session_id": state.session_id,
            "current_plane": state.current_plane,
            "frame_index_by_plane": dict(state.frame_index_by_plane),
            "viewport_by_plane": {
                plane: dict(values) for plane, values in state.viewport_by_plane.items()
            },
            "renders": {
                plane: render.to_document() for plane, render in state.renders.items()
            },
            "render_archive": {
                public_id: {
                    "frame_id": record.frame_id,
                    "frame": record.to_document(),
                }
                for public_id, record in state.render_archive.items()
            },
            "gate_outcomes": [outcome.to_document() for outcome in state.gate_outcomes],
            "used_nonces": list(state.used_nonces),
            "accepted_nonces": list(state.accepted_nonces),
            "stale_frame_ids": list(state.stale_frame_ids),
            "qc_ledger": [dict(entry) for entry in state.qc_ledger],
            "action_count": state.action_count,
            "qc_panel_open": state.qc_panel_open,
            "overlay_visibility": {
                overlay_id: state.overlay_visibility[overlay_id]
                for overlay_id in OVERLAY_IDS
            },
            "overlay_mode": state.overlay_mode,
            "selected_vessel": state.selected_vessel,
        }

    def session_document(self) -> dict[str, object]:
        return self._require_session_record().to_document()

    def session_key(self) -> str:
        return f"{_SESSION_KEY_PREFIX}{self._require_state().session_id}"

    def state_key(self) -> str:
        return f"{_STATE_KEY_PREFIX}{self._require_state().session_id}"

    def session_id(self) -> str:
        return self._require_state().session_id

    def new_provenance(self) -> tuple[ProvenanceRecord, ...]:
        return tuple(self._new_provenance)

    def new_qc_artifacts(self) -> tuple[DerivedArtifact, ...]:
        return tuple(self._new_qc_artifacts)

    def new_qc_documents(self) -> dict[str, dict[str, object]]:
        return dict(self._new_qc_documents)

    def planes_state(self) -> dict[str, object]:
        """The SERVED state document: the frozen state plus seed-job truth.

        F2_JOB_HANDLE_TRUTH: this is the document ``GET /session/:id/state``
        returns, so a polling client can tell a running from a queued build —
        the exact blindness measured in
        ``reports/q1/F2_PROGRESS_CANCEL_MEASURE_20260923.md`` (row 2:
        ``poll_exposes_inflight_status = false``).

        The ``seed_jobs`` key is ADDITIVE: it is a sibling of the frozen
        ``ccta.viewer.state.v1`` keys, so a client that does not read it
        behaves exactly as before (the whole rest of the document is
        unchanged), and it is emitted only once the server has OBSERVED a
        seed job for this session.  It is NOT part of
        :meth:`state_document`, so the persisted state document, the
        ``POST``/``/action`` responses and a reopened session are
        byte-identical to before (no job store, nothing to rehydrate).

        THE RETAINED GUARANTEE (manager decisions 2026-09-27 and 2026-09-28,
        after the contract-specified reviewer's CHANGES_REQUESTED verdict).
        This document reports what ONE currently running seed EXECUTION has
        observed, and it stops reporting it the moment that execution returns.
        It does NOT report, and must not be read as, an outcome of the
        ``(session_id, client_nonce)`` identity: no ``COMPLETED`` is published
        for any identity, from any source, on any path, and no ``REFUSED``
        outlives the execution that raised it.  A seed whose execution has
        returned — whether it was really accepted or really refused — is
        therefore ABSENT from ``seed_jobs``: the server says nothing rather
        than something it cannot keep supporting.  The real acceptance of a
        seed remains fully visible where it always was, in this same
        document's ``accepted_nonces`` and ``action_count``.

        Detach semantics (client-side "cancel"): a client that stops waiting
        for its seed POST and polls instead observes ``QUEUED``/``RUNNING``
        and may conclude the job is STILL OWNED BY THE SERVER while the
        execution is still running; a really refused
        execution is reported ``REFUSED`` with the exact code the server
        raised, for as long as that execution has not returned.  No state word
        ever claims the work stopped, and no state word ever claims the work
        finished.

        Every served entry is one running execution's own observation, not an
        identity's verdict: a request that does not hold the record's public
        handle contributes no observation to it, and the ownership binding and
        the ``admitted`` award are never served.  Three further guarantees hold
        on every code path, and they are stated exactly as the code guarantees
        them (ATT0004, review findings R3-F1/R3-F2; ATT0005 finding F1;
        ATT0006 finding R5-F1):

        * a served entry exists only while its own execution is in flight, and
          is removed when that execution returns — on success, refusal,
          cancellation or exception alike — so nothing is ever served for a
          finished execution and no stale ``QUEUED``/``RUNNING`` can linger;
        * the served ``refusal_code`` is always the code of the very refusal
          the served state reports, because it is read straight off the record
          that execution recorded; a later same-nonce attempt records its own
          observation and never borrows a code;
        * no ``COMPLETED`` is ever served — not from the session-wide
          ``accepted_nonces`` ledger (an all-action one that ordinary grammar
          actions append to, so it says nothing about a seed), not from any
          seed-scoped evidence, and not from any record.  The frozen vocabulary
          keeps all four words, and the server publishes three of them.

        There is no publication rule left to bypass: the served set is exactly
        the records the running executions hold (:meth:`_seed_jobs_document`),
        so no read order, no nonce ledger and no cross-execution check can
        change what a poll sees.
        """

        document = self.state_document()
        jobs = self._seed_jobs_document()
        if jobs:
            document[_SEED_JOBS_KEY] = jobs
        return document

    def frames_document(self, plane: str) -> dict[str, object]:
        """The ordered current frame records of one plane (LAYER-G order)."""

        state = self._require_state()
        if plane not in PLANES:
            raise ViewerRejectionError(
                RejectionCode.UNKNOWN_PARAM, "plane is outside the closed enum"
            )
        return {
            "plane": plane,
            "frames": [
                record.to_document() for record in state.renders[plane].records
            ],
        }

    def qc_panel_document(self) -> dict[str, object]:
        """The verbatim bound gate outcome + append-only ledger (Part 6.1).

        F12 (spec Part 6.1 b.1, amendment V1.1): the response carries the
        ``correction_capabilities`` disclosure as a SIBLING object derived
        response-only — it NEVER modifies ``outcome``, is not persisted
        state, and infers/synthesizes no evidence flags.  The mandatory
        NOT_ANALYZABLE abstention marker in the correction route is
        unchanged.
        """

        state = self._require_state()
        record = self._session_record
        current = state.gate_outcomes[-1] if state.gate_outcomes else None
        return {
            "qc_gate_state_ref": record.qc_gate_state_ref if record else None,
            "outcome": current.to_document() if current else None,
            "outcome_ledger": [outcome.to_document() for outcome in state.gate_outcomes],
            "stale_frame_ids": list(state.stale_frame_ids),
            "correction_capabilities": {
                "action": _CORRECTION_CAPABILITIES_ACTION,
                "actionable_grades": list(_CORRECTION_CAPABILITIES_ACTIONABLE),
                "deferred_grades": list(_CORRECTION_CAPABILITIES_DEFERRED),
                "deferred_reason": _CORRECTION_CAPABILITIES_DEFERRED_REASON,
            },
        }

    def coronary_overlay_document(
        self, *, mode: str | None = None, vessel_label: str | None = None
    ) -> dict[str, object]:
        """The closed ``ccta.viewer.coronary_overlay.v1`` read surface.

        Derived response-only from the restored session (read-only): the single
        ACTIVE CENTERLINE artifact's sealed payload geometry is projected into
        the current LAYER-G frames' pixel index space for display.  Zero or
        multiple ACTIVE centerlines (or a missing/invalid payload) raise the
        typed :class:`apps.viewer.overlay.CoronaryOverlayRefusalError`, which
        the server surfaces as a visible 409 - the overlay is never fabricated
        and the path is never silently chosen (Rule 10).

        Decision A: the served subset is slice-local by default
        (``SLICE_LOCAL``); an explicit ``mode="PROJECTION"`` serves the full
        path with the localizer caption contract.  An unknown mode is a typed
        refusal.  A request-time mode never mutates the stored visibility
        state.
        """

        state = self._require_state()
        served_mode = state.overlay_mode if mode is None else mode
        served_vessel = state.selected_vessel if vessel_label is None else vessel_label
        if served_mode not in OVERLAY_MODES:
            from apps.viewer.overlay import CoronaryOverlayRefusalError as _Refusal

            raise _Refusal(
                "CENTERLINE_OVERLAY_INVALID",
                f"mode {mode!r} is outside the closed SLICE_LOCAL | PROJECTION vocabulary",
            )
        return build_coronary_overlay_document(
            session_id=state.session_id,
            artifacts=self._working_artifacts(),
            payload_documents=self._working_payload_documents(),
            geometry=self._require_geometry(),
            visible=state.overlay_visibility[CORONARY_OVERLAY_ID],
            mode=served_mode,
            vessel_label=served_vessel,
        )

    # ------------------------------------------------------------------
    # Tree-schematic navigator read surface (TREE_SCHEMATIC_NAV)
    # ------------------------------------------------------------------

    def tree_schematic_document(self) -> dict[str, object]:
        """The closed ``ccta.viewer.tree_schematic.v1`` read surface.

        Derived response-only from the working artifacts (restored + the
        in-memory EDIT successors, so an EDIT successor is immediately
        visible): one display-only schematic entry per ACTIVE labeled
        CENTERLINE path (fixed-orientation projection + simplification of
        the sealed ``points_world`` polylines, labeled SCHEMATIC, zero
        quantitative readouts).  Zero / stale-only / malformed paths raise
        the typed :class:`apps.viewer.tree_schematic.TreeSchematicRefusalError`,
        which the server surfaces as a visible 409 - a schematic is never
        fabricated and a stale path is never served as current (Rule 10).
        The served ``selected_vessel`` is the session's stored selection
        (the closed SELECT_VESSEL grammar owns the only mutation path).
        """

        from apps.viewer.tree_schematic import build_tree_schematic_document

        state = self._require_state()
        return build_tree_schematic_document(
            session_id=state.session_id,
            artifacts=self._working_artifacts(),
            payload_documents=self._working_payload_documents(),
            selected_vessel=state.selected_vessel,
        )

    # ------------------------------------------------------------------
    # Synchronized MPR crosshair cursor read surface (MPR_SYNC_CURSOR)
    # ------------------------------------------------------------------

    def sync_point_document(
        self,
        *,
        plane: str,
        frame_index: int,
        row: float,
        column: float,
    ) -> dict[str, object]:
        """The closed ``ccta.viewer.sync_point.v1`` read surface.

        Derived response-only from the served LAYER-G geometry plus the
        persisted MPR payloads (read-only): one ``(plane, frame_index, row,
        column)`` pick maps forward to a shared world coordinate (the owned
        MPR lattice arithmetic, reused as-is) and back into the nearest
        frame of every plane (the overlay parallel projection, reused
        as-is).  Touches no runtime state — the cursor is ephemeral, so a
        reselect simply derives again and navigation refresh re-derives from
        the stored pick.  Out-of-matrix picks project to the typed
        :class:`apps.viewer.sync_point.SyncPointRefusalError`, which the
        server surfaces as a visible 409 — a partial cursor is never served
        (Rule 10).  No grammar action is involved (``ACTIONS`` stays closed).
        """

        from apps.viewer.sync_point import build_sync_point_document

        state = self._require_state()
        return build_sync_point_document(
            session_id=state.session_id,
            plane=plane,
            frame_index=frame_index,
            row=row,
            column=column,
            geometry=self._require_geometry(),
            payload_documents=self._working_payload_documents(),
        )

    # ------------------------------------------------------------------
    # CPR straightened-view read surface (Decision B/C)
    # ------------------------------------------------------------------

    def cpr_view_document(
        self, vessel_label: str | None = None
    ) -> dict[str, object]:
        """The closed ``ccta.viewer.cpr_view.v1`` read surface.

        Derived response-only from the working artifacts (restored + the
        in-memory EDIT successors, so an EDIT successor is immediately
        visible): the single ACTIVE ``ccta.cpr.view.v1`` view plus its bound
        centerline generation.  Zero / multiple / malformed / stale views
        raise the typed :class:`apps.viewer.cpr_view.CPRRefusalError`, which
        the server surfaces as a visible 409 - CPR bytes are never fabricated
        and a stale view is never served as current (Rule 10).
        """

        from apps.viewer.cpr_view import build_cpr_view_document

        state = self._require_state()
        served_vessel = (
            state.selected_vessel if vessel_label is None else vessel_label
        )
        return build_cpr_view_document(
            session_id=state.session_id,
            artifacts=self._working_artifacts(),
            payload_documents=self._working_payload_documents(),
            vessel_label=served_vessel,
        )

    def cpr_png_bytes(
        self, vessel_label: str | None = None
    ) -> tuple[bytes, dict[str, object]]:
        """The CPR PNG bytes + the view document they were rendered from.

        WL is server-authoritative (Decision C): the effective window is the
        session's current WL state - the focused plane's current
        ``wl_applied`` (preset / custom / raw-identity), resolved read-only.
        Both panels always show the SAME bound centerline generation: the
        view document carries ``centerline_ref`` +
        ``centerline_payload_sha256`` + ``centerline_generation``, matching
        the overlay document's ``centerline_ref`` of the same session.
        ``cpr_view_document()`` already selected the single ACTIVE view, so
        no re-selection happens here.
        """

        from apps.viewer.cpr_render import cpr_window_for_pixels, render_cpr_png
        from apps.viewer.presets import preset_by_id
        from visualization.cpr import decode_cpr_pixel_blob

        state = self._require_state()
        served_vessel = (
            state.selected_vessel if vessel_label is None else vessel_label
        )
        view = self.cpr_view_document(vessel_label=served_vessel)
        cpr_id = str(view["cpr_ref"]).split(":", 1)[1]
        cpr_payload = self._working_payload_documents()[cpr_id]
        pixel_ref = view["pixel_blob_ref"]
        assert isinstance(pixel_ref, dict)
        assert isinstance(pixel_ref["sha256"], str)
        digest = str(pixel_ref["sha256"])
        blob = self._pixel_blobs.get(digest)
        if blob is None:
            from apps.viewer.cpr_view import CPRRefusalError as _CPRRefusal

            raise _CPRRefusal(
                "CPR_VIEW_UNAVAILABLE",
                f"CPR view {cpr_id} pixel blob is not present in this session",
            )
        pixels = decode_cpr_pixel_blob(cpr_payload, blob)
        # Server WL authority: the focused plane's current record window.
        # The CPR view has no LAYER-R of its own; the window is read from the
        # current MPR renders (preset/custom/raw), never from client state.
        plane = state.current_plane
        records = state.renders[plane].records if plane in state.renders else ()
        if records:
            current = records[state.frame_index_by_plane.get(plane, 0) % len(records)]
            center = float(current.wl_applied["center"])
            width = float(current.wl_applied["width"])
            preset_id = current.wl_preset_id
        else:
            center, width = cpr_window_for_pixels(
                pixels, wl_preset_id=None, custom_window=None
            )
            preset_id = None
        # Re-resolve through the single frozen path for determinism: preset
        # windows come from the stored preset, custom/raw from wl_applied.
        if preset_id is not None:
            preset = preset_by_id(preset_id)
            center, width = cpr_window_for_pixels(
                pixels, wl_preset_id=preset_id, preset_window=(preset.center, preset.width)
            )
        else:
            center, width = cpr_window_for_pixels(
                pixels, wl_preset_id=None, custom_window=(center, width)
            )
        png, _canvas_digest = render_cpr_png(pixels, center, width)
        view_with_wl = dict(view)
        view_with_wl["wl_preset_id"] = preset_id
        view_with_wl["wl_applied"] = {"center": center, "width": width}
        return (png, view_with_wl)

    # ------------------------------------------------------------------
    # Seventh check: reopen identity (Part 5.1 / Part 2)
    # ------------------------------------------------------------------

    def verify_reopen_identity(self) -> list[str]:
        """Re-render every current AND archived record; compare tuples+digests.

        Frame ULIDs are excluded per Part 5.1 (record-identifier metadata
        only); the identity tuple + ``png_sha256`` + ``png_byte_size`` must
        be bit-identical, and the LAYER-G digest must never have moved.
        R2-03 (r2): the seventh check reuses the read-only COMPLETE
        geometry/render bijection validator, so an omissions-only snapshot
        (e.g. a deleted plane render list) can no longer pass vacuously.
        R2-02 (r2): archived records are validated against their
        geometry/WL/PNG binding too, and governed historical-retention
        verification must resolve every STALE id against retained records.
        """

        state = self._require_state()
        geometry = self._require_geometry()
        record = self._require_session_record()
        problems: list[str] = []
        digest = self._geometry_digest(state.session_id, geometry)
        if digest != record.frame_lists_ref:
            problems.append("LAYER-G frame_lists_ref digest moved")
        problems.extend(
            validate_render_geometry_bijection(
                session_id=state.session_id,
                geometry=geometry,
                renders=state.renders,
                archive=state.render_archive,
            )
        )
        problems.extend(
            _archive_integrity_problems(
                state.stale_frame_ids,
                state.renders,
                state.render_archive,
                provenance=self._restored.provenance,
            )
        )
        for plane in PLANES:
            for stored in state.renders[plane].records:
                if (
                    isinstance(stored.frame_index, bool)
                    or not isinstance(stored.frame_index, int)
                    or not 0 <= stored.frame_index < len(geometry[plane])
                ):
                    # Already reported by the bijection validator; never an
                    # uncaught IndexError or TypeError (r2 R2-03, r3 R3-05).
                    continue
                frame = geometry[plane][stored.frame_index]
                payload = self._restored.payload_documents[frame.mpr_artifact_id]
                blob_ref = payload["pixel_blob_ref"]
                digest_key = str(blob_ref["sha256"])  # type: ignore[index]
                png = render_png_for_record(
                    payload,
                    self._pixel_blobs[digest_key],
                    stored.wl_applied["center"],
                    stored.wl_applied["width"],
                )
                if hashlib.sha256(png).hexdigest() != stored.png_sha256:
                    problems.append(
                        f"{plane}/{stored.frame_index}: png_sha256 diverges after reopen"
                    )
                if len(png) != stored.png_byte_size:
                    problems.append(
                        f"{plane}/{stored.frame_index}: png_byte_size diverges after reopen"
                    )
                expected_tuple = (
                    state.session_id,
                    plane,
                    stored.frame_index,
                    f"mpr_artifact_id:{frame.mpr_artifact_id}",
                    stored.wl_preset_id,
                    dict(stored.wl_applied),
                )
                actual_tuple = (
                    stored.session_id,
                    stored.plane,
                    stored.frame_index,
                    stored.source_mpr_ref,
                    stored.wl_preset_id,
                    dict(stored.wl_applied),
                )
                if actual_tuple != expected_tuple:
                    problems.append(
                        f"{plane}/{stored.frame_index}: identity tuple diverges"
                    )
        for public_id, archived in sorted(state.render_archive.items()):
            png = self._render_record_png(archived)
            if hashlib.sha256(png).hexdigest() != archived.png_sha256:
                problems.append(
                    f"archive/{public_id}: png_sha256 diverges after reopen"
                )
            if len(png) != archived.png_byte_size:
                problems.append(
                    f"archive/{public_id}: png_byte_size diverges after reopen"
                )
        return problems


def validate_render_geometry_bijection(
    *,
    session_id: str,
    geometry: dict[str, list[GeometryFrame]],
    renders: dict[str, PlaneRenderState],
    archive: dict[str, ViewerFrameRecord] | None = None,
) -> list[str]:
    """The read-only COMPLETE geometry/render bijection validator (r2 R2-03).

    Used by session-open rehydration (fail-closed) and reused by the seventh
    check ``VIEWER_FRAME_IDENTITY``.  For every plane the render-record count
    must equal the geometry count, the record indices must be exactly the
    ordered sequence ``0..n-1`` (no duplicates, no reordering), and every
    record's session/plane/source binding must match its corresponding
    geometry frame.  Unknown extra planes and malformed/out-of-range indices
    are reported as problems (typed ``SESSION_OPEN_FAILED`` upstream) — the
    validator never mutates its inputs, never raises IndexError, and keeps
    genuinely empty geometry planes legal (empty list == empty records).
    When *archive* is supplied, every archived record is validated against
    its geometry binding as well (r2 R2-02).
    """

    problems: list[str] = []
    extra_planes = sorted(set(renders) - set(PLANES))
    if extra_planes:
        problems.append(
            f"viewer state renders contain unknown extra planes {extra_planes}"
        )
    for plane in PLANES:
        plane_state = renders.get(plane)
        if plane_state is None:
            problems.append(f"viewer state renders[{plane}] missing")
            continue
        geometry_frames = geometry.get(plane, [])
        records = plane_state.records
        if len(records) != len(geometry_frames):
            problems.append(
                f"{plane}: render record count {len(records)} does not equal "
                f"the geometry frame count {len(geometry_frames)}"
            )
        for position, record in enumerate(records):
            if (
                isinstance(record.frame_index, bool)
                or not isinstance(record.frame_index, int)
                or record.frame_index < 0
            ):
                problems.append(
                    f"{plane}: record at position {position} carries malformed "
                    f"frame_index {record.frame_index!r} (nonnegative integer required)"
                )
                continue
            if record.frame_index != position:
                problems.append(
                    f"{plane}: record at position {position} carries "
                    f"frame_index {record.frame_index} (exact ordered "
                    "indices 0..n-1 required)"
                )
            if record.session_id != session_id:
                problems.append(
                    f"{plane}/{position}: record session binding "
                    f"{record.session_id!r} does not match the open session"
                )
            if record.plane != plane:
                problems.append(
                    f"{plane}/{position}: record plane binding "
                    f"{record.plane!r} does not match its plane list"
                )
            if position < len(geometry_frames) and record.source_mpr_ref != (
                f"mpr_artifact_id:{geometry_frames[position].mpr_artifact_id}"
            ):
                problems.append(
                    f"{plane}/{position}: record source binding "
                    f"{record.source_mpr_ref} does not match its geometry "
                    f"frame "
                    f"mpr_artifact_id:{geometry_frames[position].mpr_artifact_id}"
                )
    if archive is not None:
        for public_id, record in sorted(archive.items()):
            label = f"archive/{public_id}"
            if record.provenance_ref.split(":", 1)[1] != public_id:
                problems.append(
                    f"{label}: archive key does not match the record's "
                    f"public provenance id {record.provenance_ref}"
                )
            if record.session_id != session_id:
                problems.append(
                    f"{label}: archived record session binding mismatch"
                )
            if record.plane not in PLANES:
                problems.append(
                    f"{label}: archived record plane {record.plane!r} is "
                    "outside the closed plane enum"
                )
                continue
            geometry_frames = geometry.get(record.plane, [])
            if not 0 <= record.frame_index < len(geometry_frames):
                problems.append(
                    f"{label}: archived frame_index {record.frame_index} is "
                    f"out of range for {record.plane} geometry "
                    f"({len(geometry_frames)} frames)"
                )
                continue
            expected_source = (
                f"mpr_artifact_id:{geometry_frames[record.frame_index].mpr_artifact_id}"
            )
            if record.source_mpr_ref != expected_source:
                problems.append(
                    f"{label}: archived source binding {record.source_mpr_ref} "
                    f"does not match its geometry frame {expected_source}"
                )
    return problems


def _uniform_applied_window(
    current: PlaneRenderState,
) -> tuple[float, float] | None:
    """The re-applied custom window of a non-preset plane, or None for raw.

    The custom LAYER-R variant class (spec Part 4.2) records the SAME
    user-applied ``(center, width)`` on every record of its plane, while the
    raw identity mapping records each frame's own identity window.  A
    re-mint (the correction rebind re-mints every plane under its CURRENT
    WL) therefore carries the uniform applied window forward when there is
    exactly one, and falls back to raw identity when windows differ.  A raw
    plane whose frames happen to share one identity window is
    indistinguishable from custom AND byte-equivalent: the render is a pure
    function of the pixel bytes plus the applied window.
    """

    if current.wl_preset_id is not None or not current.records:
        return None
    windows = {
        (record.wl_applied["center"], record.wl_applied["width"])
        for record in current.records
    }
    if len(windows) != 1:
        return None
    return next(iter(windows))


def _archive_integrity_problems(
    stale_frame_ids: list[str],
    renders: dict[str, PlaneRenderState],
    archive: dict[str, ViewerFrameRecord],
    provenance: tuple[ProvenanceRecord, ...] | None = None,
) -> list[str]:
    """Governed historical-retention verification (r2 R2-02, r3 R3-03).

    Every STALE frame id must resolve to a retained archived record (never
    fabricated), every current record must be retained in the archive,
    retained internal frame ids must be unambiguous, complete ten-key document
    equality is required between current and archived copies sharing a public ID,
    and internal frame IDs must agree with retained provenance where present.
    """

    problems: list[str] = []
    retained_frame_ids: set[str] = set()
    for public_id, record in sorted(archive.items()):
        if record.frame_id:
            if record.frame_id in retained_frame_ids:
                problems.append(
                    f"archive/{public_id}: internal frame id "
                    f"{record.frame_id} is retained under multiple records"
                )
            retained_frame_ids.add(record.frame_id)
            if provenance is not None:
                matching_prov = next(
                    (p for p in provenance if p.provenance_id == public_id), None
                )
                if (
                    matching_prov is not None
                    and matching_prov.subject_ref.startswith("viewer_frame:")
                ):
                    prov_frame_id = matching_prov.subject_ref.split(":", 1)[1]
                    if record.frame_id != prov_frame_id:
                        problems.append(
                            f"archive/{public_id}: internal frame id {record.frame_id} "
                            f"does not match retained provenance {prov_frame_id}"
                        )
    for stale_id in stale_frame_ids:
        if stale_id not in retained_frame_ids:
            problems.append(
                "governed historical-retention verification failed: STALE "
                f"frame id {stale_id} does not resolve to a retained render "
                "record (records are never fabricated)"
            )
    for plane in PLANES:
        for record in renders.get(plane, PlaneRenderState(None, ())).records:
            public_id = record.provenance_ref.split(":", 1)[1]
            if public_id not in archive:
                problems.append(
                    f"{plane}/{record.frame_index}: current record {public_id} "
                    "is not retained in the render archive"
                )
            else:
                archived = archive[public_id]
                if record.to_document() != archived.to_document():
                    problems.append(
                        f"{plane}/{record.frame_index}: current record {public_id} "
                        "conflicts with archived record (ten-key document equality required)"
                    )
                if (
                    record.frame_id
                    and archived.frame_id
                    and record.frame_id != archived.frame_id
                ):
                    problems.append(
                        f"{plane}/{record.frame_index}: current record {public_id} "
                        f"internal frame id {record.frame_id} does not match "
                        f"archived {archived.frame_id}"
                    )
    return problems


def _closed_provenance_prefix(
    records: tuple[ProvenanceRecord, ...],
) -> tuple[ProvenanceRecord, ...]:
    """The subset of *records* whose upstream closure is self-contained.

    A record is included when every upstream ref resolves within the
    included set itself (the seam adds its own source-manifest/volume/
    transform roots for the remaining roots).  The selection is deterministic
    and monotone: pass one keeps everything the previous pass kept.
    """

    included: dict[str, ProvenanceRecord] = {}
    changed = True
    while changed:
        changed = False
        for record in records:
            if record.provenance_id in included:
                continue
            upstreams = (
                ()
                if isinstance(record.upstream_refs, NotApplicable)
                else tuple(record.upstream_refs)
            )
            if all(up in included for up in upstreams):
                included[record.provenance_id] = record
                changed = True
    return tuple(included[record.provenance_id] for record in records
                 if record.provenance_id in included)


def _document_view(
    artifact_id: str, document: dict[str, object], status: str
) -> QCAssessmentArtifact:
    """Rebuild the closed gate view from a persisted ``qc_artifact:`` doc."""

    payload = QCAssessmentPayload(
        schema_version=str(document["schema_version"]),
        scope=QCScope(str(document["scope"])),
        subject_refs=tuple(document["subject_refs"]),  # type: ignore[arg-type]
        quality_grade=QualityGrade(str(document["quality_grade"])),
        confidence=None,
        flags=tuple(document["flags"]),  # type: ignore[arg-type]
        vessel_id=None,
        region_id=None,
        producer=ProducerType(str(document["producer"])),
        produced_by_ref=str(document["produced_by_ref"]),
        supersedes_ref=document.get("supersedes_ref"),  # type: ignore[arg-type]
        justification_code=str(document["justification_code"]),
        evidence_refs=tuple(document["evidence_refs"]),  # type: ignore[arg-type]
    )
    return QCAssessmentArtifact(
        artifact_ref=f"qc_artifact:{artifact_id}",
        payload=payload,
        producer=payload.producer,
        status=status,
    )


def _assessment_view(
    artifact: DerivedArtifact, document: dict[str, object]
) -> QCAssessmentArtifact:
    """Rebuild the closed gate view of a persisted QC artifact (read-only)."""

    payload = QCAssessmentPayload(
        schema_version=str(document["schema_version"]),
        scope=QCScope(str(document["scope"])),
        subject_refs=tuple(document["subject_refs"]),  # type: ignore[arg-type]
        quality_grade=QualityGrade(str(document["quality_grade"])),
        confidence=None,
        flags=tuple(document["flags"]),  # type: ignore[arg-type]
        vessel_id=None,
        region_id=None,
        producer=ProducerType(str(document["producer"])),
        produced_by_ref=str(document["produced_by_ref"]),
        supersedes_ref=document.get("supersedes_ref"),  # type: ignore[arg-type]
        justification_code=str(document["justification_code"]),
        evidence_refs=tuple(document["evidence_refs"]),  # type: ignore[arg-type]
    )
    return QCAssessmentArtifact(
        artifact_ref=f"qc_artifact:{artifact.artifact_id}",
        payload=payload,
        producer=payload.producer,
        status=artifact.status,
    )


__all__ = [
    "GateOutcome",
    "HumanSeamRefusalError",
    "PlaneRenderState",
    "SeedJobRecord",
    "ViewerEngine",
    "ViewerOpenError",
    "ViewerRuntimeState",
    "validate_render_geometry_bijection",
]
