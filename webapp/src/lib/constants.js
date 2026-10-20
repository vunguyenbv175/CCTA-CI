/**
 * Frozen client-side mirrors of the viewer grammar surface.
 *
 * The values below are display/interaction mirrors of the frozen server
 * contract (apps/viewer/grammar.py + apps/viewer/presets.py).  The server
 * remains the single authority: every envelope built from these constants is
 * re-validated server-side and a mismatch surfaces as a typed rejection.
 * The mirrors exist so the client can refuse malformed envelopes BEFORE the
 * network boundary (never silently, never clamped server-side).
 */

/** The closed plane enum, in frozen order. */
export const PLANES = Object.freeze(["AXIAL", "CORONAL", "SAGITTAL"]);

/** The closed v1 action set (grammar.ACTIONS). */
export const ACTIONS = Object.freeze([
  "SET_SLICE",
  "SET_PLANE",
  "SET_WL_PRESET",
  "CLEAR_WL",
  "SET_VIEWPORT_MATRIX",
  "OPEN_QC_PANEL",
  "REQUEST_CORRECTION",
]);

/** The closed per-action param key sets (grammar.ACTION_PARAM_KEYS). */
export const ACTION_PARAM_KEYS = Object.freeze({
  SET_SLICE: Object.freeze(["plane", "frame_index"]),
  SET_PLANE: Object.freeze(["plane"]),
  SET_WL_PRESET: Object.freeze(["preset_id"]),
  CLEAR_WL: Object.freeze([]),
  SET_VIEWPORT_MATRIX: Object.freeze(["zoom", "pan_x", "pan_y"]),
  OPEN_QC_PANEL: Object.freeze([]),
  REQUEST_CORRECTION: Object.freeze([
    "subject_ref",
    "proposed_grade",
    "justification_code",
  ]),
});

/** The closed envelope key set (grammar.ENVELOPE_KEYS). */
export const ENVELOPE_KEYS = Object.freeze([
  "action",
  "params",
  "client_nonce",
  "view",
  "lineage",
]);

/** The accepted QualityGrade enum (grammar.QUALITY_GRADES). */
export const QUALITY_GRADES = Object.freeze([
  "GOOD",
  "ACCEPTABLE",
  "LIMITED",
  "NOT_ANALYZABLE",
]);

/** The ONLY accepted correction justification in v1. */
export const JUSTIFICATION_HUMAN_EVIDENCE = "HUMAN_EVIDENCE";

/** The frozen v1 WL preset ids, in frozen preset order (presets.WL_PRESET_IDS). */
export const WL_PRESET_IDS = Object.freeze([
  "CT_MEDIASTINUM",
  "CT_VESSEL_CONTRAST",
  "CT_LUNG",
  "CT_BONE",
  "CT_SOFT_TISSUE",
]);

/** grammar.ZOOM_MAX: the largest accepted matrix zoom. */
export const ZOOM_MAX = 64.0;

/** grammar.PAN_EXTENT_LIMIT: the largest accepted |pan| per axis. */
export const PAN_EXTENT_LIMIT = 2.0;

/** grammar.NONCE_RE: 16-64 chars from [A-Za-z0-9_-]. */
export const NONCE_RE = /^[A-Za-z0-9_-]{16,64}$/;

/** schemas.ULID_RE: the Crockford base32 ULID alphabet. */
export const ULID_RE = /^[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}$/;

/** grammar study reference form: exactly ``study_id:<ULID>``. */
export const STUDY_REF_RE = /^study_id:[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}$/;

/** The frozen grammar's matrix decimal precision used before submit. */
export const MATRIX_DECIMALS = 6;
