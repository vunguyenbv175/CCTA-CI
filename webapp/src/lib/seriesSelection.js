/**
 * Deterministic selection heuristics for the physician reading loop.
 *
 * ADAPT of the CMR ``seriesSelection`` preference shape (preference rules,
 * not code): a physician's explicit choice always wins within the page
 * lifetime (sticky selection), otherwise a documented deterministic default
 * is used.  All functions are pure: no DOM, no network, no library state.
 */
import { PLANES } from "./constants.js";
import { clampFrameIndex } from "./sessionState.js";

/**
 * Pick the session to open: sticky physician choice first, else the first
 * server-listed session; an empty list yields ``null`` (empty state, no throw).
 *
 * @param {readonly string[]} sessions
 * @param {{stickySessionId?: string | null}} [options]
 * @returns {string | null}
 */
export function pickPreferredSession(sessions, options = {}) {
  const list = Array.isArray(sessions) ? sessions : [];
  const sticky = options.stickySessionId ?? null;
  if (sticky && list.includes(sticky)) {
    return sticky;
  }
  return list.length ? list[0] : null;
}

/**
 * Pick the focused plane: sticky physician choice first, else the server's
 * current plane, else the first plane that carries frames, else AXIAL.
 *
 * @param {{currentPlane?: string, recordsByPlane?: Record<string, unknown[]>}} state
 * @param {{stickyPlane?: string | null}} [options]
 * @returns {string}
 */
export function pickFocusedPlane(state, options = {}) {
  const sticky = options.stickyPlane ?? null;
  if (sticky && PLANES.includes(sticky)) {
    return sticky;
  }
  if (state && PLANES.includes(state.currentPlane)) {
    return state.currentPlane;
  }
  if (state && state.recordsByPlane) {
    const withFrames = PLANES.find(
      (plane) => (state.recordsByPlane[plane] || []).length > 0
    );
    if (withFrames) {
      return withFrames;
    }
  }
  return PLANES[0];
}

/**
 * Pick the frame index to display for one plane: the server value clamped
 * into the plane's frame list (display selection only).
 *
 * @param {Record<string, number>} frameIndexByPlane
 * @param {string} plane
 * @param {number} frameCount
 * @returns {number}
 */
export function pickFrameIndex(frameIndexByPlane, plane, frameCount) {
  const desired = frameIndexByPlane ? frameIndexByPlane[plane] : 0;
  return clampFrameIndex(desired, frameCount);
}

/**
 * Pick the plane an action's ``view`` should target: the focused plane when it
 * has frames, else the first plane that carries frames, else the focused plane
 * (an action against an empty plane would be a typed VIEW_MISMATCH; callers
 * surface that refusal instead of fabricating a view).
 */
export function pickViewPlane(focusedPlane, recordsByPlane) {
  if ((recordsByPlane[focusedPlane] || []).length > 0) {
    return focusedPlane;
  }
  return PLANES.find((plane) => (recordsByPlane[plane] || []).length > 0) || focusedPlane;
}

export default { pickFocusedPlane, pickFrameIndex, pickPreferredSession, pickViewPlane };
