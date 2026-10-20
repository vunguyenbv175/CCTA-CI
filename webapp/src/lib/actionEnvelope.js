/**
 * Action envelope construction + closed-envelope validation (client side).
 *
 * The v0 client already implemented the generation-bound POST discipline
 * (apps/viewer/static/app.js, preserved under webapp/v0-fallback/); this
 * module is its strict successor: the SAME frozen envelope shape, plus a
 * client-side closed-key validation so a malformed envelope is refused
 * before it reaches the network.
 *
 * The enforcement boundary that matters most (constraint E5): every action
 * carries a CLOSED param set.  ``SET_WL_PRESET`` is the only action carrying
 * a ``preset_id`` and its value must be one of the frozen presets.  The v2
 * successor (spec Part 4.2) extends the closed set by exactly ONE action:
 * ``SET_WL_CUSTOM`` with exactly ``(center, width)`` finite binary32
 * numerics and ``width > 0`` — the same closed domain the server enforces.
 * Every other action keeps its frozen param set.  CORONARY_CENTERLINE_001
 * extends it by exactly one more closed action, ``SET_OVERLAY_VISIBILITY``
 * with ``(overlay_id, visible)`` from the frozen overlay allowlist and a
 * strict boolean — the same closed domain the server enforces.
 * CORONARY_REVIEW_VIEW_V1 (Decision A) extends that ONE action with exactly
 * one further closed key, ``mode`` in ``(SLICE_LOCAL, PROJECTION)``.
 */
import {
  ACTION_PARAM_KEYS,
  ACTIONS,
  JUSTIFICATION_HUMAN_EVIDENCE,
  MATRIX_DECIMALS,
  NONCE_RE,
  PAN_EXTENT_LIMIT,
  PLANES,
  QUALITY_GRADES,
  STUDY_REF_RE,
  WL_PRESET_IDS,
  ZOOM_MAX,
} from "./constants.js";
import { OVERLAY_IDS, OVERLAY_MODES } from "./coronaryOverlay.js";
import { ViewerClientError } from "./errors.js";

/** The v2 custom-WL action (VIEWER_001_SPEC Part 4.2). */
export const WL_CUSTOM_ACTION = "SET_WL_CUSTOM";

/** The closed param key set of ``SET_WL_CUSTOM``. */
export const WL_CUSTOM_PARAM_KEYS = Object.freeze(["center", "width"]);

/** The coronary overlay-visibility action (CORONARY_CENTERLINE_001). */
export const OVERLAY_VISIBILITY_ACTION = "SET_OVERLAY_VISIBILITY";

/** The closed param key set of ``SET_OVERLAY_VISIBILITY``. */
export const OVERLAY_VISIBILITY_PARAM_KEYS = Object.freeze([
  "overlay_id",
  "visible",
  "mode",
]);

/** The single-path coronary human decision action (att0002, F5). */
export const CENTERLINE_DECISION_ACTION = "REQUEST_CENTERLINE_DECISION";

/** The explicit vessel-selection action (CORONARY_MULTI_VESSEL_NAV, Decision A). */
export const SELECT_VESSEL_ACTION = "SELECT_VESSEL";

/** The closed param key set of ``SELECT_VESSEL``. */
export const SELECT_VESSEL_PARAM_KEYS = Object.freeze(["vessel_label"]);

/** The frozen vessel-label tokens (grammar VESSEL_LABELS). */
export const VESSEL_LABELS = Object.freeze(["VESSEL:LAD", "VESSEL:LCX", "VESSEL:RCA"]);

/** The closed param key set of ``REQUEST_CENTERLINE_DECISION``. */
export const CENTERLINE_DECISION_PARAM_KEYS = Object.freeze([
  "subject_ref",
  "centerline_ref",
  "decision",
  "seed_voxel",
  "seed_direction_world",
]);

/** The closed decision vocabulary (grammar.CENTERLINE_DECISIONS). */
export const CENTERLINE_DECISIONS = Object.freeze(["ACCEPT", "EDIT", "REJECT"]);

/** grammar centerline reference form: exactly ``centerline_artifact_id:<ULID>``. */
export const CENTERLINE_REF_RE =
  /^centerline_artifact_id:[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}$/;

/** The largest finite IEEE-754 binary32 magnitude (grammar.FLOAT32_MAX). */
export const FLOAT32_MAX = 3.4028234663852886e38;

const EXTENDED_PARAM_KEYS = Object.freeze({
  ...ACTION_PARAM_KEYS,
  [WL_CUSTOM_ACTION]: WL_CUSTOM_PARAM_KEYS,
  [OVERLAY_VISIBILITY_ACTION]: OVERLAY_VISIBILITY_PARAM_KEYS,
  [CENTERLINE_DECISION_ACTION]: CENTERLINE_DECISION_PARAM_KEYS,
  [SELECT_VESSEL_ACTION]: SELECT_VESSEL_PARAM_KEYS,
});

/**
 * Validate the closed ``SET_WL_CUSTOM`` numeric domain WITHOUT clamping.
 *
 * Both numerics must be finite binary32 values and ``width`` must stay a
 * positive binary32 value; a width that rounds to binary32 zero is refused
 * exactly as the server refuses it.  Nothing here is ever nudged.
 *
 * @param {{center: number, width: number}} params
 */
export function assertCustomWl(params) {
  for (const key of ["center", "width"]) {
    const value = params[key];
    if (
      typeof value !== "number" ||
      !Number.isFinite(value) ||
      Math.abs(value) > FLOAT32_MAX ||
      !Number.isFinite(Math.fround(value))
    ) {
      throw new ViewerClientError(
        "UNKNOWN_PARAM",
        `${key} must be a finite binary32 number`
      );
    }
  }
  if (!(Math.fround(params.width) > 0)) {
    throw new ViewerClientError(
      "UNKNOWN_PARAM",
      "width must be a positive binary32 value (the server never clamps)"
    );
  }
}

/**
 * Validate one action/params pair against the closed client-side mirror of
 * the frozen grammar.  Throws ViewerClientError on any violation; NEVER
 * clamps, never drops keys, never substitutes defaults.
 *
 * @param {string} action
 * @param {object} params
 * @param {{activePresetIds?: readonly string[]}} [options]
 */
export function validateActionPayload(action, params, options = {}) {
  if (
    !ACTIONS.includes(action) &&
    action !== WL_CUSTOM_ACTION &&
    action !== OVERLAY_VISIBILITY_ACTION &&
    action !== CENTERLINE_DECISION_ACTION &&
    action !== SELECT_VESSEL_ACTION
  ) {
    throw new ViewerClientError("UNKNOWN_ACTION", `action ${action} is not in the closed set`);
  }
  if (params === null || typeof params !== "object" || Array.isArray(params)) {
    throw new ViewerClientError("UNKNOWN_PARAM", "params must be an object");
  }
  const expected = EXTENDED_PARAM_KEYS[action];
  const actual = Object.keys(params).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    throw new ViewerClientError(
      "UNKNOWN_PARAM",
      `${action} params must be exactly [${wanted.join(",")}]`
    );
  }
  if (action === "SET_SLICE") {
    assertPlane(params.plane, "params.plane");
    assertFrameIndex(params.frame_index);
  } else if (action === "SET_PLANE") {
    assertPlane(params.plane, "params.plane");
  } else if (action === "SET_WL_PRESET") {
    const activePresets = options.activePresetIds || WL_PRESET_IDS;
    if (typeof params.preset_id !== "string" || !activePresets.includes(params.preset_id)) {
      throw new ViewerClientError(
        "UNKNOWN_PRESET",
        "preset_id is outside the frozen active preset set"
      );
    }
  } else if (action === WL_CUSTOM_ACTION) {
    assertCustomWl(params);
  } else if (action === "SET_VIEWPORT_MATRIX") {
    assertMatrix(params);
  } else if (action === "REQUEST_CORRECTION") {
    if (typeof params.subject_ref !== "string" || !STUDY_REF_RE.test(params.subject_ref)) {
      throw new ViewerClientError(
        "SUBJECT_REF_INVALID",
        "subject_ref must be exactly this session's study_id:<ULID>"
      );
    }
    if (typeof params.proposed_grade !== "string" || !QUALITY_GRADES.includes(params.proposed_grade)) {
      throw new ViewerClientError(
        "GRADE_NOT_IN_ENUM",
        "proposed_grade is outside the accepted enum"
      );
    }
    if (params.justification_code !== JUSTIFICATION_HUMAN_EVIDENCE) {
      throw new ViewerClientError(
        "JUSTIFICATION_NOT_IN_ENUM",
        "justification_code must be exactly HUMAN_EVIDENCE"
      );
    }
  } else if (action === OVERLAY_VISIBILITY_ACTION) {
    if (
      typeof params.overlay_id !== "string" ||
      !OVERLAY_IDS.includes(params.overlay_id)
    ) {
      throw new ViewerClientError(
        "UNKNOWN_PARAM",
        "overlay_id is outside the closed overlay allowlist"
      );
    }
    if (typeof params.visible !== "boolean") {
      throw new ViewerClientError("UNKNOWN_PARAM", "visible must be a boolean");
    }
    if (typeof params.mode !== "string" || !OVERLAY_MODES.includes(params.mode)) {
      throw new ViewerClientError(
        "UNKNOWN_PARAM",
        "mode is outside the closed SLICE_LOCAL | PROJECTION vocabulary"
      );
    }
  } else if (action === SELECT_VESSEL_ACTION) {
    if (
      typeof params.vessel_label !== "string" ||
      !VESSEL_LABELS.includes(params.vessel_label)
    ) {
      throw new ViewerClientError(
        "UNKNOWN_PARAM",
        "vessel_label is outside the frozen VESSEL:* tokens"
      );
    }
  } else if (action === CENTERLINE_DECISION_ACTION) {
    if (typeof params.subject_ref !== "string" || !STUDY_REF_RE.test(params.subject_ref)) {
      throw new ViewerClientError(
        "SUBJECT_REF_INVALID",
        "subject_ref must be exactly study_id:<ULID>"
      );
    }
    if (
      typeof params.centerline_ref !== "string" ||
      !CENTERLINE_REF_RE.test(params.centerline_ref)
    ) {
      throw new ViewerClientError(
        "UNKNOWN_PARAM",
        "centerline_ref must be exactly centerline_artifact_id:<ULID>"
      );
    }
    if (
      typeof params.decision !== "string" ||
      !CENTERLINE_DECISIONS.includes(params.decision)
    ) {
      throw new ViewerClientError(
        "UNKNOWN_PARAM",
        "decision is outside the closed ACCEPT/EDIT/REJECT vocabulary"
      );
    }
    if (params.decision === "EDIT") {
      if (
        !Array.isArray(params.seed_voxel) ||
        params.seed_voxel.length !== 3 ||
        !params.seed_voxel.every(
          (value) => Number.isInteger(value) && value >= 0
        )
      ) {
        throw new ViewerClientError(
          "UNKNOWN_PARAM",
          "EDIT requires seed_voxel as three non-negative integers"
        );
      }
      if (
        !Array.isArray(params.seed_direction_world) ||
        params.seed_direction_world.length !== 3 ||
        !params.seed_direction_world.every(
          (value) => typeof value === "number" && Number.isFinite(value)
        ) ||
        params.seed_direction_world.every((value) => value === 0)
      ) {
        throw new ViewerClientError(
          "UNKNOWN_PARAM",
          "EDIT requires seed_direction_world as a finite non-zero 3-vector"
        );
      }
    } else if (params.seed_voxel != null || params.seed_direction_world != null) {
      throw new ViewerClientError(
        "UNKNOWN_PARAM",
        "ACCEPT/REJECT must carry null seed fields"
      );
    }
  }
}

/**
 * Build one canonical envelope (no serialization here; see canonicalJson).
 *
 * @param {{
 *   action: string, params: object, clientNonce: string, sessionId: string,
 *   parentActionNonce: string | null, plane: string, frameIndex: number,
 *   activePresetIds?: readonly string[],
 * }} input
 * @returns {{action: string, params: object, client_nonce: string, view: object, lineage: object}}
 */
export function buildActionEnvelope(input) {
  const {
    action,
    params,
    clientNonce,
    sessionId,
    parentActionNonce,
    plane,
    frameIndex,
    activePresetIds,
  } = input;
  validateActionPayload(action, params, { activePresetIds });
  // The matrix is the only action whose values are decimals; normalize them
  // to the frozen precision so the canonical request bytes are identical
  // between the JavaScript and Python serializers.  Values outside bounds
  // were already refused above (never clamped).
  const finalParams =
    action === "SET_VIEWPORT_MATRIX" ? assertMatrix(params) : params;
  if (typeof sessionId !== "string" || !sessionId) {
    throw new ViewerClientError("SESSION_INVALID", "session id is required");
  }
  if (typeof clientNonce !== "string" || !NONCE_RE.test(clientNonce)) {
    throw new ViewerClientError("NONCE_INVALID", "client_nonce must match the frozen grammar");
  }
  if (parentActionNonce !== null && parentActionNonce !== undefined) {
    if (typeof parentActionNonce !== "string" || !NONCE_RE.test(parentActionNonce)) {
      throw new ViewerClientError("NONCE_INVALID", "parent_action_nonce must match the grammar");
    }
  }
  assertPlane(plane, "view.plane");
  assertFrameIndex(frameIndex, "view.frame_index");
  return {
    action,
    params: finalParams,
    client_nonce: clientNonce,
    view: { plane, frame_index: frameIndex },
    lineage: {
      session_id: sessionId,
      parent_action_nonce:
        parentActionNonce === undefined ? null : parentActionNonce,
    },
  };
}

/**
 * Quantize one matrix component to the frozen decimal precision.
 *
 * This is a serialization transform (so JS/Python canonical bytes agree), not
 * a silent clamp: out-of-range values are REFUSED by assertMatrix, and the
 * quantized result is re-validated against the closed domain (a zoom that
 * would quantize to 0 is refused, never nudged positive).
 */
export function quantizeMatrixValue(value) {
  const rounded = Number(value.toFixed(MATRIX_DECIMALS));
  return Object.is(rounded, -0) ? 0 : rounded;
}

function assertPlane(plane, label) {
  if (typeof plane !== "string" || !PLANES.includes(plane)) {
    throw new ViewerClientError("UNKNOWN_PARAM", `${label} is outside the closed plane enum`);
  }
}

function assertFrameIndex(value, label = "frame_index") {
  if (!Number.isInteger(value) || value < 0) {
    throw new ViewerClientError("INDEX_OUT_OF_RANGE", `${label} must be a nonnegative integer`);
  }
}

/**
 * Validate a viewport matrix against the frozen bounds WITHOUT clamping.
 *
 * The server never clamps (Rule 10); the client therefore refuses to build an
 * out-of-range envelope and the UI reports the refusal instead of persisting
 * a display state different from what the physician sees.
 *
 * The closed domain is re-validated AFTER quantization: a zoom in
 * ``(0, 5e-7)`` passes the raw check but quantizes to ``0``, which the frozen
 * server rejects with ``ZOOM_NON_POSITIVE``.  Such a value must be refused
 * here (never clamped to a positive epsilon) so the client can never build an
 * envelope the frozen seam will reject.
 */
export function assertMatrix(params) {
  const zoom = params.zoom;
  if (typeof zoom !== "number" || !Number.isFinite(zoom)) {
    throw new ViewerClientError("UNKNOWN_PARAM", "zoom must be a finite number");
  }
  if (zoom <= 0) {
    throw new ViewerClientError("ZOOM_NON_POSITIVE", "zoom must be > 0 (never clamped)");
  }
  for (const key of ["pan_x", "pan_y"]) {
    const value = params[key];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new ViewerClientError("UNKNOWN_PARAM", `${key} must be a finite number`);
    }
  }
  if (zoom > ZOOM_MAX || Math.abs(params.pan_x) > PAN_EXTENT_LIMIT || Math.abs(params.pan_y) > PAN_EXTENT_LIMIT) {
    throw new ViewerClientError(
      "MATRIX_NON_FINITE",
      `zoom must be <= ${ZOOM_MAX} and |pan| <= ${PAN_EXTENT_LIMIT}`
    );
  }
  const quantized = {
    zoom: quantizeMatrixValue(zoom),
    pan_x: quantizeMatrixValue(params.pan_x),
    pan_y: quantizeMatrixValue(params.pan_y),
  };
  if (!Number.isFinite(quantized.zoom) || quantized.zoom <= 0) {
    throw new ViewerClientError(
      "ZOOM_NON_POSITIVE",
      "zoom quantizes to 0 at the frozen precision (never clamped)"
    );
  }
  if (
    quantized.zoom > ZOOM_MAX ||
    Math.abs(quantized.pan_x) > PAN_EXTENT_LIMIT ||
    Math.abs(quantized.pan_y) > PAN_EXTENT_LIMIT
  ) {
    throw new ViewerClientError(
      "MATRIX_NON_FINITE",
      `quantized zoom must be <= ${ZOOM_MAX} and |pan| <= ${PAN_EXTENT_LIMIT}`
    );
  }
  return quantized;
}

export default {
  CENTERLINE_DECISION_ACTION,
  CENTERLINE_DECISION_PARAM_KEYS,
  CENTERLINE_DECISIONS,
  OVERLAY_VISIBILITY_ACTION,
  OVERLAY_VISIBILITY_PARAM_KEYS,
  SELECT_VESSEL_ACTION,
  SELECT_VESSEL_PARAM_KEYS,
  VESSEL_LABELS,
  buildActionEnvelope,
  validateActionPayload,
  quantizeMatrixValue,
};
