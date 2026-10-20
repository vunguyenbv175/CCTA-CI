/**
 * Coronary centerline overlay (display-only).
 *
 * The frozen client metadata contract deliberately carries NO patient geometry
 * (see ``metadata.js``): the served stack viewports address the served PNG in
 * pixel index space.  The server therefore serves the closed
 * ``ccta.viewer.coronary_overlay.v1`` document, whose ``plane_projections``
 * map the sealed centerline world polyline into each LAYER-G frame's
 * ``(column, row)`` pixel indices under the explicit, display-only
 * ``PARALLEL_IN_PLANE_PIXEL_INDEX_V1`` rule.  This module only maps those
 * indices through the viewport's own world<->canvas transform and strokes the
 * polyline: it invents no geometry, performs no measurement, and never talks
 * to the network itself (the frameSource boundary does).
 */
import { ViewerClientError } from "./errors.js";

/** The closed overlay allowlist (apps/viewer/overlay.py OVERLAY_IDS). */
export const OVERLAY_IDS = Object.freeze(["CENTERLINE"]);

/** The single v1 coronary overlay id. */
export const CORONARY_OVERLAY_ID = "CENTERLINE";

/** The closed overlay document schema. */
export const CORONARY_OVERLAY_SCHEMA = "ccta.viewer.coronary_overlay.v1";

/** The frozen display-projection rule of the served document. */
export const CORONARY_OVERLAY_PROJECTION_RULE =
  "PARALLEL_IN_PLANE_PIXEL_INDEX_V1";

/** The frozen slice-local slab rule (Decision A). */
export const CORONARY_OVERLAY_SLAB_RULE =
  "SLICE_LOCAL_HALF_CELL_OF_FRAME_SPACING_V1";

/** The closed display-mode vocabulary (Decision A). */
export const OVERLAY_MODES = Object.freeze(["SLICE_LOCAL", "PROJECTION"]);

/** The default display mode (the normal overlay is slice-local). */
export const OVERLAY_DEFAULT_MODE = "SLICE_LOCAL";

/** The visible localizer caption of the PROJECTION mode. */
export const OVERLAY_PROJECTION_CAPTION =
  "PROJECTION — not same-slice anatomy";

/** The stroke colour/width of the display-only slice-local path. */
export const OVERLAY_STROKE = "#ffd23f";
export const OVERLAY_LINE_WIDTH = 1.5;

/** The visually distinct style of the PROJECTION localizer mode. */
export const OVERLAY_PROJECTION_STROKE = "#7dd3fc";
export const OVERLAY_PROJECTION_LINE_WIDTH = 1.0;
export const OVERLAY_PROJECTION_DASH = Object.freeze([6, 4]);

/**
 * Fetch the overlay document.
 *
 * F8: only ``CENTERLINE_OVERLAY_UNAVAILABLE`` (no ACTIVE centerline) is the
 * ordinary "no overlay" case and returns ``null``; malformed/ambiguous
 * overlay states (``CENTERLINE_OVERLAY_INVALID`` / ``CENTERLINE_OVERLAY_AMBIGUOUS``)
 * are rethrown so the caller keeps a VISIBLE typed failure.  Any other
 * failure is rethrown as well (never swallowed).
 *
 * @param {{getCoronary: (sessionId: string) => Promise<object>}} source
 * @param {string} sessionId
 * @returns {Promise<object | null>}
 */
export async function loadCoronaryOverlay(source, sessionId) {
  if (!source || typeof source.getCoronary !== "function" || !sessionId) {
    return null;
  }
  try {
    return await source.getCoronary(sessionId);
  } catch (error) {
    if (
      error instanceof ViewerClientError &&
      String(error.code || "") === "CENTERLINE_OVERLAY_UNAVAILABLE"
    ) {
      return null;
    }
    throw error;
  }
}

/**
 * The display projection of one displayed frame, or null.
 *
 * @param {object} document closed overlay document
 * @param {string} plane
 * @param {number} frameIndex
 */
export function projectionForFrame(document, plane, frameIndex) {
  const byPlane =
    document && typeof document === "object" ? document.plane_projections : null;
  const frames = byPlane && typeof byPlane === "object" ? byPlane[plane] : null;
  if (!Array.isArray(frames) || !frames.length) {
    return null;
  }
  const wanted = Number(frameIndex);
  const match = frames.find((entry) => Number(entry.frame_index) === wanted);
  return match || frames[0];
}

/**
 * The display runs of one displayed frame, or null.
 *
 * Decision A: the server computes the slab-filtered draw subset per frame
 * (``in_slab_runs``); the client draws exactly the served subset and never
 * re-derives clipping.  In ``PROJECTION`` mode the served runs cover the
 * full path (the explicitly labeled localizer); otherwise they are the
 * slice-local runs (possibly empty: nothing is drawn, never a fallback).
 *
 * @param {object} document closed overlay document
 * @param {string} plane
 * @param {number} frameIndex
 */
export function runsForFrame(document, plane, frameIndex) {
  const projection = projectionForFrame(document, plane, frameIndex);
  if (!projection || !Array.isArray(projection.in_slab_runs)) {
    return null;
  }
  return projection.in_slab_runs;
}

/**
 * Draw the overlay of one plane on its dedicated canvas.
 *
 * Canvas sizing follows the host element; the mapping uses the viewport's own
 * ``worldToCanvas`` with the served ``(column, row, 0)`` index-space point.
 * Every failure to map a point simply drops that point (never a fabricated
 * position).  ``canvas.dataset.overlayDrawn`` records how many points mapped
 * and were stroked (0 when the overlay is hidden/unavailable).
 *
 * Decision A: draws exactly the served ``in_slab_runs`` subset.  In
 * ``PROJECTION`` mode the path is dashed thin with the localizer caption
 * contract (``canvas.dataset.overlayMode``); unknown modes are never drawn.
 *
 * @returns {number} mapped point count
 */
export function drawCoronaryOverlay({ canvas, viewport, document, plane, frameIndex }) {
  if (!canvas) {
    return 0;
  }
  const host = canvas.parentElement;
  const width = Math.max(
    1,
    Math.round((host && host.clientWidth) || canvas.width || 1)
  );
  const height = Math.max(
    1,
    Math.round((host && host.clientHeight) || canvas.height || 1)
  );
  if (canvas.width !== width) {
    canvas.width = width;
  }
  if (canvas.height !== height) {
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  if (context) {
    context.clearRect(0, 0, canvas.width, canvas.height);
  }
  canvas.dataset.overlayDrawn = "0";
  if (!document || typeof document !== "object" || !viewport) {
    canvas.hidden = true;
    return 0;
  }
  const visible = document.visible !== false;
  canvas.dataset.overlayId = String(document.overlay_id || "");
  canvas.dataset.overlayVisible = visible ? "true" : "false";
  canvas.dataset.overlayPoints = String(document.point_count || 0);
  const mode = String(document.mode || OVERLAY_DEFAULT_MODE);
  canvas.dataset.overlayMode = mode;
  canvas.hidden = !visible;
  if (!visible) {
    return 0;
  }
  if (mode !== "SLICE_LOCAL" && mode !== "PROJECTION") {
    return 0;
  }
  const runs = runsForFrame(document, plane, frameIndex);
  if (!runs) {
    return 0;
  }
  const isProjection = mode === "PROJECTION";
  let drawn = 0;
  const stroked = [];
  for (const run of runs) {
    const pairs = run && Array.isArray(run.columns_rows) ? run.columns_rows : [];
    const path = [];
    for (const pair of pairs) {
      if (!Array.isArray(pair) || pair.length !== 2) {
        continue;
      }
      const column = Number(pair[0]);
      const row = Number(pair[1]);
      if (!Number.isFinite(column) || !Number.isFinite(row)) {
        continue;
      }
      let mapped = null;
      try {
        mapped = viewport.worldToCanvas([column, row, 0]);
      } catch {
        mapped = null;
      }
      if (
        !mapped ||
        !Number.isFinite(Number(mapped[0])) ||
        !Number.isFinite(Number(mapped[1]))
      ) {
        continue;
      }
      path.push([Number(mapped[0]), Number(mapped[1])]);
    }
    drawn += path.length;
    if (path.length >= 2) {
      stroked.push(path);
    }
  }
  canvas.dataset.overlayDrawn = String(drawn);
  canvas.dataset.overlayRuns = String(stroked.length);
  canvas.dataset.overlayProjection = isProjection ? "true" : "false";
  if (!context || !stroked.length) {
    return drawn;
  }
  context.strokeStyle = isProjection ? OVERLAY_PROJECTION_STROKE : OVERLAY_STROKE;
  context.lineWidth = isProjection ? OVERLAY_PROJECTION_LINE_WIDTH : OVERLAY_LINE_WIDTH;
  context.setLineDash(isProjection ? [...OVERLAY_PROJECTION_DASH] : []);
  for (const path of stroked) {
    context.beginPath();
    context.moveTo(path[0][0], path[0][1]);
    for (let index = 1; index < path.length; index += 1) {
      context.lineTo(path[index][0], path[index][1]);
    }
    context.stroke();
  }
  context.setLineDash([]);
  return drawn;
}

export default {
  CORONARY_OVERLAY_ID,
  CORONARY_OVERLAY_PROJECTION_RULE,
  CORONARY_OVERLAY_SCHEMA,
  CORONARY_OVERLAY_SLAB_RULE,
  OVERLAY_DEFAULT_MODE,
  OVERLAY_IDS,
  OVERLAY_LINE_WIDTH,
  OVERLAY_MODES,
  OVERLAY_PROJECTION_CAPTION,
  OVERLAY_PROJECTION_DASH,
  OVERLAY_PROJECTION_LINE_WIDTH,
  OVERLAY_PROJECTION_STROKE,
  OVERLAY_STROKE,
  drawCoronaryOverlay,
  loadCoronaryOverlay,
  projectionForFrame,
  runsForFrame,
};
