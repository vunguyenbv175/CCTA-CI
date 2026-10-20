/**
 * Synchronized MPR crosshair cursor (ephemeral view aid).
 *
 * MPR_SYNC_CURSOR: one shared world-space cursor across the AXIAL/CORONAL/
 * SAGITTAL panels.  The server derives the closed
 * ``ccta.viewer.sync_point.v1`` document from one (plane, frame, row,
 * column) pick (forward lattice math + parallel in-plane projection,
 * server-side only); this module only fetches that document and strokes
 * crosshair LINES through the served per-plane (column, row) positions.
 * It invents no geometry, renders no numbers, persists nothing, and never
 * talks to the network itself (the frameSource boundary does).
 */
import { ViewerClientError } from "./errors.js";

/** The closed sync-point document schema. */
export const SYNC_POINT_SCHEMA = "ccta.viewer.sync_point.v1";

/** The mandatory document kind marker (a cursor is a view aid, never a finding). */
export const SYNC_CURSOR_KIND = "CURSOR";

/** The mandatory visible caption (fixed; the cursor is never a finding). */
export const SYNC_CURSOR_CAPTION = "CURSOR — view aid, not a finding";

/** The frozen forward-derivation rule named by the served document. */
export const SYNC_FORWARD_RULE = "MPR_WORLD_PIXEL_CENTER_V1";

/** The frozen inverse-projection rule named by the served document. */
export const SYNC_PROJECTION_RULE = "PARALLEL_IN_PLANE_PIXEL_INDEX_V1";

/** The stroke colour/width of the display-only crosshair lines. */
export const SYNC_CURSOR_STROKE = "#f472b6";
export const SYNC_CURSOR_LINE_WIDTH = 1;

/**
 * Fetch the sync-point document for one cursor pick.
 *
 * Only ``SYNC_POINT_UNAVAILABLE`` (a plane carries no served frame) is the
 * ordinary "no cursor" case and returns ``null``; malformed/out-of-volume
 * picks (``SYNC_POINT_INVALID`` / ``SYNC_POINT_OUT_OF_VOLUME``) are rethrown
 * so the caller keeps a VISIBLE typed failure.  Any other failure is
 * rethrown as well (never swallowed).
 *
 * @param {{getSyncPoint: (sessionId: string, pick: object) => Promise<object>}} source
 * @param {string} sessionId
 * @param {{plane: string, frameIndex: number, row: number, column: number}} pick
 * @returns {Promise<object | null>}
 */
export async function loadSyncPoint(source, sessionId, pick) {
  if (!source || typeof source.getSyncPoint !== "function" || !sessionId) {
    return null;
  }
  assertSyncPick(pick);
  try {
    const document = await source.getSyncPoint(sessionId, pick);
    return validateSyncPointDocument(document);
  } catch (error) {
    if (
      error instanceof ViewerClientError &&
      String(error.code || "") === "SYNC_POINT_UNAVAILABLE"
    ) {
      return null;
    }
    throw error;
  }
}

/**
 * Validate one cursor pick before the network (never clamped).
 *
 * @param {{plane: string, frameIndex: number, row: number, column: number}} pick
 */
export function assertSyncPick(pick) {
  if (!pick || typeof pick !== "object") {
    throw new ViewerClientError("SYNC_PICK_INVALID", "cursor pick must be an object");
  }
  if (typeof pick.plane !== "string" || !pick.plane) {
    throw new ViewerClientError("SYNC_PICK_INVALID", "cursor pick plane must be a string");
  }
  if (!Number.isInteger(pick.frameIndex) || pick.frameIndex < 0) {
    throw new ViewerClientError(
      "SYNC_PICK_INVALID",
      "cursor pick frameIndex must be a nonnegative integer"
    );
  }
  for (const key of ["row", "column"]) {
    if (typeof pick[key] !== "number" || !Number.isFinite(pick[key])) {
      throw new ViewerClientError(
        "SYNC_PICK_INVALID",
        `cursor pick ${key} must be a finite number`
      );
    }
  }
}

/**
 * Validate the closed sync-point document (fail-closed, client side).
 *
 * @param {object} document served document
 * @returns {object} the same document
 */
export function validateSyncPointDocument(document) {
  if (!document || typeof document !== "object") {
    throw new ViewerClientError("SYNC_POINT_INVALID", "sync-point document must be an object");
  }
  if (document.schema_version !== SYNC_POINT_SCHEMA) {
    throw new ViewerClientError("SYNC_POINT_INVALID", "sync-point schema_version mismatch");
  }
  if (document.kind !== SYNC_CURSOR_KIND) {
    throw new ViewerClientError("SYNC_POINT_INVALID", "sync-point kind must be CURSOR");
  }
  if (document.caption !== SYNC_CURSOR_CAPTION) {
    throw new ViewerClientError("SYNC_POINT_INVALID", "sync-point caption is fixed");
  }
  if (document.derivation_rule !== SYNC_FORWARD_RULE) {
    throw new ViewerClientError("SYNC_POINT_INVALID", "sync-point derivation_rule is fixed");
  }
  if (document.projection_rule !== SYNC_PROJECTION_RULE) {
    throw new ViewerClientError("SYNC_POINT_INVALID", "sync-point projection_rule is fixed");
  }
  if (!document.planes || typeof document.planes !== "object") {
    throw new ViewerClientError("SYNC_POINT_INVALID", "sync-point planes must be an object");
  }
  return document;
}

/**
 * The served in-plane cursor position of one plane, or null.
 *
 * @param {object} document closed sync-point document
 * @param {string} plane
 */
export function cursorForPlane(document, plane) {
  const byPlane =
    document && typeof document === "object" ? document.planes : null;
  const entry = byPlane && typeof byPlane === "object" ? byPlane[plane] : null;
  if (!entry || typeof entry !== "object") {
    return null;
  }
  const row = Number(entry.row);
  const column = Number(entry.column);
  if (!Number.isFinite(row) || !Number.isFinite(column)) {
    return null;
  }
  return { row, column, frameIndex: Number(entry.frame_index) };
}

/**
 * Draw the crosshair lines of one plane on its dedicated canvas.
 *
 * Full-width horizontal + full-height vertical lines through the served
 * (column, row) position, mapped through the viewport's own
 * ``worldToCanvas`` (the same index-space mapping the overlay draw uses).
 * Lines only: no numbers, no labels, no readouts.  Every mapping failure
 * simply draws nothing (never a fabricated position).
 * ``canvas.dataset.syncDrawn`` records whether lines were stroked ("1"/"0");
 * ``canvas.dataset.syncSuppressed`` records frame-mismatch suppression
 * ("1" when the displayed frame differs from the served plane entry).
 *
 * Must run INSIDE the shared overlay-canvas pass after the overlay/seed
 * draws (the pass owner clears once); this function never clears.
 *
 * FINDING 1 (frame binding): the served document names the exact LAYER-G
 * frame of every plane (``planes[plane].frame_index``).  When the panel
 * displays any other frame, the three crosshairs would not represent the
 * same world point, so this function draws NOTHING on the mismatched plane
 * (suppression, never a fabricated position).  Suppression was chosen over
 * auto-synchronizing the displayed stacks because it is the smaller change
 * consistent with the existing panel idioms in ``webapp/src/ui/app.js``:
 * the pending-seed marker (``drawSeedMarker``) likewise hides when
 * ``anchor.frameIndex !== overlayFrameIndex(plane)`` instead of moving the
 * stack, and programmatic stack moves here would fight the wheel/scroll
 * ownership (SET_SLICE debounce) and the binding-checked refresh.  The
 * caller passes the displayed frame as ``frameIndex`` (the same
 * ``overlayFrameIndex(plane)`` the overlay draw uses); an omitted or
 * non-finite ``frameIndex`` disables the guard for backward compatibility.
 * The visible reason for the suppression lives in the app layer
 * (``#sync-status`` off-frame note); this module only records the datasets.
 *
 * @returns {number} 1 when lines were stroked, else 0
 */
export function drawSyncCursor({ canvas, viewport, document, plane, frameIndex }) {
  if (!canvas) {
    return 0;
  }
  canvas.dataset.syncDrawn = "0";
  canvas.dataset.syncSuppressed = "0";
  if (!document || typeof document !== "object" || !viewport) {
    return 0;
  }
  if (typeof viewport.worldToCanvas !== "function") {
    return 0;
  }
  const cursor = cursorForPlane(document, plane);
  if (!cursor) {
    return 0;
  }
  const displayed = Number(frameIndex);
  const served = Number(cursor.frameIndex);
  if (Number.isFinite(displayed) && Number.isFinite(served) && displayed !== served) {
    canvas.dataset.syncDrawn = "0";
    canvas.dataset.syncSuppressed = "1";
    return 0;
  }
  let mapped = null;
  try {
    mapped = viewport.worldToCanvas([cursor.column, cursor.row, 0]);
  } catch {
    mapped = null;
  }
  if (
    !mapped ||
    !Number.isFinite(Number(mapped[0])) ||
    !Number.isFinite(Number(mapped[1]))
  ) {
    return 0;
  }
  const x = Number(mapped[0]);
  const y = Number(mapped[1]);
  const context = canvas.getContext("2d");
  canvas.dataset.syncDrawn = "1";
  canvas.hidden = false;
  if (!context) {
    return 1;
  }
  context.strokeStyle = SYNC_CURSOR_STROKE;
  context.lineWidth = SYNC_CURSOR_LINE_WIDTH;
  context.setLineDash([]);
  context.beginPath();
  context.moveTo(0, y);
  context.lineTo(canvas.width, y);
  context.moveTo(x, 0);
  context.lineTo(x, canvas.height);
  context.stroke();
  return 1;
}

export default {
  SYNC_CURSOR_CAPTION,
  SYNC_CURSOR_KIND,
  SYNC_CURSOR_LINE_WIDTH,
  SYNC_CURSOR_STROKE,
  SYNC_FORWARD_RULE,
  SYNC_POINT_SCHEMA,
  SYNC_PROJECTION_RULE,
  assertSyncPick,
  cursorForPlane,
  drawSyncCursor,
  loadSyncPoint,
  validateSyncPointDocument,
};
