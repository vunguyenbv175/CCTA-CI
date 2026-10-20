/**
 * DISPLAY-ONLY metadata for the Cornerstone stack viewports (Rule B).
 *
 * Field mapping is derived strictly from the frozen frame document fields
 * (schemas.py:101-108): source_mpr_ref, plane, frame_index, wl_applied,
 * provenance_ref.  The frozen API exposes NO patient geometry, NO pixel
 * spacing, NO anatomical calibration — therefore:
 *
 *   - imagePlaneModule carries dimensions only (rows/columns).  Cornerstone's
 *     own ``getImagePlaneModule`` then flags ``usingDefaultValues`` and keeps
 *     uncalibrated display coordinates; nothing in this client implies
 *     measurement validity.
 *   - imagePixelModule describes the transport PNG (RGB24, 8-bit).
 *   - the ONLY numeric range the renderer receives is the identity transport
 *     range [0, 255] (see IDENTITY_DISPLAY_VOI): the frozen server-side
 *     transfer function VIEWER_BINARY32_WL_V1 has ALREADY produced these
 *     8-bit channel values, so the client maps them 1:1 to screen.  No
 *     client-side WL LUT over the PNG exists anywhere; ``wl_applied`` and
 *     ``wl_preset_id`` are shown as captions and are never re-applied.
 */
import { ViewerClientError } from "./errors.js";
import { publicFrameId } from "./imageIds.js";

/**
 * The identity transport display range applied to the ALREADY-windowed RGB24
 * PNG.  center=127.5, width=255 covers exactly [0, 255] (getVOIRange semantics:
 * lower = center - width/2, upper = center + width/2).  This is a display
 * identity, NOT a window/level: it never depends on wl_applied.
 */
export const IDENTITY_DISPLAY_VOI = Object.freeze({ center: 127.5, width: 255.0 });

/**
 * Map one frozen frame document to the display facts of one image.
 *
 * @param {object} record closed ``ccta.viewer.frame.v1`` document
 * @returns {{recordId: string, sessionId: string, plane: string, frameIndex: number,
 *            sourceMprRef: string, wlPresetId: string | null,
 *            wlApplied: {center: number, width: number}, provenanceRef: string}}
 */
export function frameDisplayFromRecord(record) {
  if (record === null || typeof record !== "object") {
    throw new ViewerClientError("FRAME_RECORD_INVALID", "frame record must be an object");
  }
  const wlApplied = record.wl_applied;
  if (
    wlApplied === null ||
    typeof wlApplied !== "object" ||
    typeof wlApplied.center !== "number" ||
    typeof wlApplied.width !== "number"
  ) {
    throw new ViewerClientError(
      "FRAME_RECORD_INVALID",
      "frame record wl_applied must be {center, width}"
    );
  }
  return {
    recordId: publicFrameId(record),
    sessionId: String(record.session_id),
    plane: String(record.plane),
    frameIndex: Number(record.frame_index),
    sourceMprRef: String(record.source_mpr_ref),
    wlPresetId: record.wl_preset_id === null ? null : String(record.wl_preset_id),
    wlApplied: { center: wlApplied.center, width: wlApplied.width },
    provenanceRef: String(record.provenance_ref),
  };
}

/**
 * One human-readable, dependency-free caption for a frame display.
 *
 * @param {ReturnType<typeof frameDisplayFromRecord>} display
 * @returns {string}
 */
export function frameCaption(display) {
  const wl = display.wlPresetId === null ? "raw" : display.wlPresetId;
  const center = display.wlApplied.center;
  const width = display.wlApplied.width;
  return `${display.plane} · frame ${display.frameIndex} · WL ${wl} (C=${center}, W=${width})`;
}

/**
 * The per-image display store used by the metadata provider.
 *
 * The provider is asked for metadata BY imageId; entries are registered when
 * the loader decodes an image (rows/columns come from the decoded transport
 * PNG, never from a geometry claim).
 */
export function createFrameDisplayStore() {
  const entries = new Map();
  return {
    set(imageId, display) {
      entries.set(imageId, display);
    },
    get(imageId) {
      return entries.get(imageId);
    },
    delete(imageId) {
      entries.delete(imageId);
    },
    clear() {
      entries.clear();
    },
  };
}

/**
 * Build the display metadata entry for one loaded image.
 *
 * @param {object} record frozen frame document
 * @param {{width: number, height: number}} decoded decoded transport size
 */
export function buildFrameDisplay(record, decoded) {
  const base = frameDisplayFromRecord(record);
  if (
    !Number.isInteger(decoded.width) ||
    !Number.isInteger(decoded.height) ||
    decoded.width <= 0 ||
    decoded.height <= 0
  ) {
    throw new ViewerClientError("FRAME_DECODE_INVALID", "decoded PNG size is invalid");
  }
  return Object.freeze({ ...base, rows: decoded.height, columns: decoded.width });
}

/**
 * The Cornerstone metadata provider callback (registered at priority 10000).
 *
 * Returns DISPLAY-ONLY modules for ccta-viewer image ids and ``undefined`` for
 * everything else.  It never fabricates imagePositionPatient / cosines /
 * pixel spacing / frameOfReferenceUID.
 *
 * @param {ReturnType<typeof createFrameDisplayStore>} store
 * @returns {(type: string, imageId: string) => object | undefined}
 */
export function createMetadataProvider(store) {
  return function cctaMetadataProvider(type, imageId) {
    if (typeof imageId !== "string" || !imageId.startsWith("ccta-viewer://")) {
      return undefined;
    }
    const display = store.get(imageId);
    if (!display) {
      return undefined;
    }
    if (type === "imagePlaneModule") {
      // Dimensions only.  Deliberately NO rowPixelSpacing/columnPixelSpacing,
      // NO rowCosines/columnCosines, NO imagePositionPatient, NO
      // imageOrientationPatient, NO frameOfReferenceUID.
      return { rows: display.rows, columns: display.columns };
    }
    if (type === "imagePixelModule") {
      return {
        samplesPerPixel: 3,
        photometricInterpretation: "RGB",
        rows: display.rows,
        columns: display.columns,
        bitsAllocated: 8,
        bitsStored: 8,
        highBit: 7,
        pixelRepresentation: 0,
      };
    }
    if (type === "generalSeriesModule") {
      // The displayed artifact is a transport RGB rendering, not a modality
      // dataset; no clinical modality/series claim is made.
      return { modality: "OT" };
    }
    return undefined;
  };
}

export default {
  IDENTITY_DISPLAY_VOI,
  buildFrameDisplay,
  createFrameDisplayStore,
  createMetadataProvider,
  frameCaption,
  frameDisplayFromRecord,
};
