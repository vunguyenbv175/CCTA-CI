/**
 * ``ccta-viewer://`` image id grammar (ADAPT of the CMR ``cmr-study://``
 * pattern; reimplemented against the CCTA server, no CMR code copied).
 *
 * Form: ``ccta-viewer://<session_id>/<plane>/<frame_index>/<record_id>``
 *   - session_id: the governed viewer session ULID
 *   - plane: one of the closed PLANES enum
 *   - frame_index: nonnegative integer position in the plane's frozen order
 *   - record_id: the frame record's PUBLIC id (the provenance ULID) — it is
 *     part of the image id so a WL re-mint (new record, new PNG bytes) is a
 *     NEW image id for the rendering cache, never a stale hit.
 *
 * The image id carries DISPLAY sequencing only.  It encodes no patient
 * geometry, no pixel spacing and no calibration claim (Rule B: the client
 * never sees raw DICOM; geometry is not exposed by the frozen API).
 */
import { PLANES } from "./constants.js";
import { ViewerClientError } from "./errors.js";

export const CCTA_VIEWER_SCHEME = "ccta-viewer";

/**
 * @param {string} sessionId governed viewer session id
 * @param {string} plane closed plane enum member
 * @param {number} frameIndex nonnegative integer frame index
 * @param {string} recordId public frame record id (provenance ULID)
 * @returns {string} the canonical image id
 */
export function makeImageId(sessionId, plane, frameIndex, recordId) {
  if (typeof sessionId !== "string" || !sessionId) {
    throw new ViewerClientError("IMAGE_ID_INVALID", "session id is required");
  }
  if (!PLANES.includes(plane)) {
    throw new ViewerClientError("IMAGE_ID_INVALID", `unknown plane ${plane}`);
  }
  if (!Number.isInteger(frameIndex) || frameIndex < 0) {
    throw new ViewerClientError(
      "IMAGE_ID_INVALID",
      "frame index must be a nonnegative integer"
    );
  }
  if (typeof recordId !== "string" || !recordId) {
    throw new ViewerClientError("IMAGE_ID_INVALID", "record id is required");
  }
  return (
    `${CCTA_VIEWER_SCHEME}://${encodeURIComponent(sessionId)}` +
    `/${plane}/${frameIndex}/${encodeURIComponent(recordId)}`
  );
}

/**
 * Strictly parse one ``ccta-viewer://`` image id.
 *
 * @param {string} imageId
 * @returns {{sessionId: string, plane: string, frameIndex: number, recordId: string}}
 */
export function parseImageId(imageId) {
  if (typeof imageId !== "string" || !imageId.startsWith(`${CCTA_VIEWER_SCHEME}://`)) {
    throw new ViewerClientError(
      "IMAGE_ID_INVALID",
      `not a ${CCTA_VIEWER_SCHEME} image id`
    );
  }
  const rest = imageId.slice(`${CCTA_VIEWER_SCHEME}://`.length);
  const parts = rest.split("/");
  if (parts.length !== 4) {
    throw new ViewerClientError(
      "IMAGE_ID_INVALID",
      "expected <session>/<plane>/<frame_index>/<record_id>"
    );
  }
  const [sessionRaw, plane, indexRaw, recordRaw] = parts;
  let sessionId;
  let recordId;
  try {
    sessionId = decodeURIComponent(sessionRaw);
    recordId = decodeURIComponent(recordRaw);
  } catch (error) {
    throw new ViewerClientError("IMAGE_ID_INVALID", `URI decode failed: ${error}`);
  }
  if (!sessionId || !recordId) {
    throw new ViewerClientError("IMAGE_ID_INVALID", "empty session or record id");
  }
  if (!PLANES.includes(plane)) {
    throw new ViewerClientError("IMAGE_ID_INVALID", `unknown plane ${plane}`);
  }
  if (!/^[0-9]+$/.test(indexRaw)) {
    throw new ViewerClientError("IMAGE_ID_INVALID", "frame index is not a digit string");
  }
  const frameIndex = Number(indexRaw);
  if (!Number.isSafeInteger(frameIndex)) {
    throw new ViewerClientError("IMAGE_ID_INVALID", "frame index is out of range");
  }
  return { sessionId, plane, frameIndex, recordId };
}

/**
 * The public frame id of one frozen frame record (``provenance:<ULID>``).
 *
 * @param {{provenance_ref?: string}} record
 * @returns {string}
 */
export function publicFrameId(record) {
  const ref = record && record.provenance_ref;
  if (typeof ref !== "string" || !ref.startsWith("provenance:")) {
    throw new ViewerClientError(
      "FRAME_RECORD_INVALID",
      "frame record carries no provenance_ref"
    );
  }
  return ref.slice("provenance:".length);
}

export default { CCTA_VIEWER_SCHEME, makeImageId, parseImageId, publicFrameId };
