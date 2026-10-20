/**
 * Straightened CPR panel (display-only, Decision B/C).
 *
 * The panel renders the accepted ``ccta.cpr.view.v1`` computation through
 * the dedicated server seam (``GET /session/:id/cpr`` for the closed view
 * document + ``GET /session/:id/cpr.png`` for the frozen PNG bytes).  The
 * client performs NO geometry, NO windowing and NO measurement: the PNG
 * bytes are server-rendered under server WL authority and are drawn 1:1
 * onto a plain 2D canvas.  Typed CPR refusals (``CPR_VIEW_UNAVAILABLE`` /
 * ``CPR_VIEW_INVALID`` / ``CPR_VIEW_STALE``) are surfaced visibly through
 * the sibling typed-failure element, mirroring the overlay F8 contract.
 */
import { ViewerClientError } from "./errors.js";

/** The closed CPR view document schema. */
export const CPR_VIEW_SCHEMA = "ccta.viewer.cpr_view.v1";

/** The typed CPR refusal codes (closed vocabulary). */
export const CPR_REFUSALS = Object.freeze([
  "CPR_VIEW_UNAVAILABLE",
  "CPR_VIEW_INVALID",
  "CPR_VIEW_STALE",
]);

/**
 * Load the CPR view document.
 *
 * Only ``CPR_VIEW_UNAVAILABLE`` (no ACTIVE CPR view) is the ordinary "no
 * panel" case and returns ``null``; malformed/stale views
 * (``CPR_VIEW_INVALID`` / ``CPR_VIEW_STALE``) are rethrown so the caller
 * keeps a VISIBLE typed failure.  Any other failure is rethrown as well.
 */
export async function loadCprView(source, sessionId) {
  if (!source || typeof source.getCpr !== "function" || !sessionId) {
    return null;
  }
  try {
    return await source.getCpr(sessionId);
  } catch (error) {
    if (
      error instanceof ViewerClientError &&
      String(error.code || "") === "CPR_VIEW_UNAVAILABLE"
    ) {
      return null;
    }
    throw error;
  }
}

/**
 * Draw the CPR PNG bytes onto a plain 2D canvas (1:1, no resampling).
 *
 * @param {{canvas: HTMLCanvasElement, pngBytes: Uint8Array}} options
 * @returns {Promise<number>} drawn pixel count (rows * columns), 0 when hidden
 */
export async function drawCprPanel({ canvas, pngBytes }) {
  if (!canvas) {
    return 0;
  }
  const context = canvas.getContext("2d");
  if (context) {
    context.clearRect(0, 0, canvas.width, canvas.height);
  }
  canvas.dataset.cprDrawn = "0";
  if (!pngBytes || !pngBytes.length) {
    canvas.hidden = true;
    return 0;
  }
  const blob = new Blob([pngBytes], { type: "image/png" });
  const bitmap = await createImageBitmap(blob);
  try {
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    if (context) {
      context.drawImage(bitmap, 0, 0);
    }
    canvas.dataset.cprDrawn = String(bitmap.width * bitmap.height);
    canvas.dataset.cprRows = String(bitmap.height);
    canvas.dataset.cprColumns = String(bitmap.width);
    canvas.hidden = false;
    return bitmap.width * bitmap.height;
  } finally {
    if (bitmap && typeof bitmap.close === "function") {
      bitmap.close();
    }
  }
}

export default { CPR_VIEW_SCHEMA, CPR_REFUSALS, drawCprPanel, loadCprView };
