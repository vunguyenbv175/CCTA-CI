/**
 * Display-state projection of the frozen ``ccta.viewer.state.v1`` document.
 *
 * The server's state document is the single source of session display state
 * (current plane, per-plane frame index, per-plane viewport matrix, accepted
 * nonce chain).  This projection is READ-ONLY and display-only: it never
 * mutates the document and never invents values for missing planes.
 */
import { PLANES } from "./constants.js";
import { ViewerClientError } from "./errors.js";

/**
 * @param {object} document frozen state document returned by GET /state or POST /action
 * @returns {{
 *   sessionId: string,
 *   currentPlane: string,
 *   frameIndexByPlane: Record<string, number>,
 *   viewportByPlane: Record<string, {zoom: number, pan_x: number, pan_y: number}>,
 *   recordsByPlane: Record<string, object[]>,
 *   wlPresetIdByPlane: Record<string, string | null>,
 *   acceptedNonces: string[],
 *   lastAcceptedNonce: string | null,
 *   qcPanelOpen: boolean,
 *   actionCount: number,
 * }}
 */
export function displayStateFrom(document) {
  if (document === null || typeof document !== "object") {
    throw new ViewerClientError("STATE_INVALID", "state document must be an object");
  }
  const sessionId = document.session_id;
  if (typeof sessionId !== "string" || !sessionId) {
    throw new ViewerClientError("STATE_INVALID", "state document carries no session_id");
  }
  const currentPlane = PLANES.includes(document.current_plane)
    ? document.current_plane
    : PLANES[0];
  const frameIndexByPlane = {};
  const viewportByPlane = {};
  const recordsByPlane = {};
  const wlPresetIdByPlane = {};
  const rawIndexes = document.frame_index_by_plane || {};
  const rawViewports = document.viewport_by_plane || {};
  const rawRenders = document.renders || {};
  for (const plane of PLANES) {
    const index = rawIndexes[plane];
    frameIndexByPlane[plane] = Number.isInteger(index) && index >= 0 ? index : 0;
    const matrix = rawViewports[plane];
    viewportByPlane[plane] =
      matrix && typeof matrix === "object"
        ? {
            zoom: Number(matrix.zoom),
            pan_x: Number(matrix.pan_x),
            pan_y: Number(matrix.pan_y),
          }
        : { zoom: 1, pan_x: 0, pan_y: 0 };
    const render = rawRenders[plane];
    if (render && typeof render === "object") {
      recordsByPlane[plane] = Array.isArray(render.records) ? render.records.slice() : [];
      wlPresetIdByPlane[plane] =
        render.wl_preset_id === null || render.wl_preset_id === undefined
          ? null
          : String(render.wl_preset_id);
    } else {
      recordsByPlane[plane] = [];
      wlPresetIdByPlane[plane] = null;
    }
  }
  const acceptedNonces = Array.isArray(document.accepted_nonces)
    ? document.accepted_nonces.filter((item) => typeof item === "string")
    : [];
  return {
    sessionId,
    currentPlane,
    frameIndexByPlane,
    viewportByPlane,
    recordsByPlane,
    wlPresetIdByPlane,
    acceptedNonces,
    lastAcceptedNonce: acceptedNonces.length
      ? acceptedNonces[acceptedNonces.length - 1]
      : null,
    qcPanelOpen: document.qc_panel_open === true,
    actionCount: Number.isInteger(document.action_count) ? document.action_count : 0,
  };
}

/**
 * Clamp a desired frame index into [0, frameCount-1] for display selection.
 *
 * This is DISPLAY selection only (which record to show).  It never builds an
 * out-of-range SET_SLICE envelope: the caller must still validate the value
 * against the frozen frame list.
 */
export function clampFrameIndex(index, frameCount) {
  if (!Number.isInteger(frameCount) || frameCount <= 0) {
    return 0;
  }
  if (!Number.isInteger(index)) {
    return 0;
  }
  return Math.min(Math.max(index, 0), frameCount - 1);
}

export default { displayStateFrom, clampFrameIndex };
