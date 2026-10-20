/**
 * The ``ccta-viewer://`` frame loader (ADAPT of the CMR imageLoader pattern).
 *
 * A load consists of exactly:
 *   1. parse the image id (closed form, imageIds.js);
 *   2. resolve the frozen frame record for (session, plane, frame_index);
 *   3. verify the record's PUBLIC id equals the id encoded in the image id
 *      (a WL re-mint changes the public id; a mismatch is refused, never
 *      silently served);
 *   4. GET ``/frame/<record_id>.png`` — the frozen RGB24 transport bytes;
 *   5. decode and hand back a Cornerstone IImage of the ALREADY-WINDOWED
 *      transport pixels.  No raw DICOM, no dicom-image-loader, no client-side
 *      window/level math touches these bytes.
 *
 * All effects are injected so the loader logic is testable and so the bundle
 * keeps exactly one fetch path (frameSource.js).
 */
import { IDENTITY_DISPLAY_VOI, buildFrameDisplay } from "./metadata.js";
import { ViewerClientError } from "./errors.js";
import { parseImageId, publicFrameId } from "./imageIds.js";

/**
 * @param {{
 *   resolveRecord: (parsed: object) => Promise<object>,
 *   fetchPngBytes: (recordId: string) => Promise<ArrayBuffer|Uint8Array>,
 *   decodePng: (bytes: Uint8Array, record: object) => Promise<{width: number, height: number, rgb: Uint8Array}>,
 *   buildImage: (input: object) => object,
 *   onDisplay?: (imageId: string, display: object) => void,
 * }} dependencies
 * @returns {(imageId: string) => Promise<object>} the Cornerstone loader handler
 */
export function createFrameLoader(dependencies) {
  const { resolveRecord, fetchPngBytes, decodePng, buildImage, onDisplay } = dependencies;
  return async function loadCctaViewerImage(imageId) {
    const parsed = parseImageId(imageId);
    let record;
    try {
      record = await resolveRecord(parsed);
    } catch (error) {
      if (error instanceof ViewerClientError) {
        throw error;
      }
      throw new ViewerClientError(
        "FRAME_LOOKUP_FAILED",
        `frame lookup failed for ${imageId}: ${error}`
      );
    }
    if (!record) {
      throw new ViewerClientError(
        "FRAME_NOT_FOUND",
        `no frame record for ${parsed.plane}[${parsed.frameIndex}]`
      );
    }
    const recordId = publicFrameId(record);
    if (recordId !== parsed.recordId) {
      throw new ViewerClientError(
        "FRAME_BINDING_MISMATCH",
        `image id record ${parsed.recordId} does not match the served record ${recordId}`
      );
    }
    const bytes = await fetchPngBytes(recordId);
    const decoded = await decodePng(toUint8(bytes), record);
    const display = buildFrameDisplay(record, decoded);
    const image = buildImage({ imageId, record, decoded, display });
    if (onDisplay) {
      onDisplay(imageId, display);
    }
    return image;
  };
}

/**
 * Build one Cornerstone IImage over the decoded transport RGB bytes.
 *
 * @param {{
 *   imageId: string, record: object,
 *   decoded: {width: number, height: number, rgb: Uint8Array},
 *   createVoxelManager?: (input: object) => object,
 * }} input
 */
export function buildTransportImage(input) {
  const { imageId, decoded, createVoxelManager } = input;
  const rgb = decoded.rgb;
  if (!(rgb instanceof Uint8Array) || rgb.length !== decoded.width * decoded.height * 3) {
    throw new ViewerClientError(
      "FRAME_DECODE_INVALID",
      "decoded RGB buffer does not match its dimensions"
    );
  }
  const image = {
    imageId,
    minPixelValue: 0,
    maxPixelValue: 255,
    slope: 1,
    intercept: 0,
    // Identity transport display range over ALREADY-WINDOWED RGB24 bytes.
    windowCenter: IDENTITY_DISPLAY_VOI.center,
    windowWidth: IDENTITY_DISPLAY_VOI.width,
    getPixelData: () => rgb,
    rows: decoded.height,
    columns: decoded.width,
    height: decoded.height,
    width: decoded.width,
    color: true,
    rgba: false,
    numberOfComponents: 3,
    invert: false,
    photometricInterpretation: "RGB",
    sizeInBytes: rgb.byteLength,
    dataType: "Uint8Array",
  };
  if (createVoxelManager) {
    image.voxelManager = createVoxelManager({
      scalarData: rgb,
      width: decoded.width,
      height: decoded.height,
      numberOfComponents: 3,
      id: imageId,
    });
  }
  return image;
}

function toUint8(bytes) {
  if (bytes instanceof Uint8Array) {
    return bytes;
  }
  if (bytes instanceof ArrayBuffer) {
    return new Uint8Array(bytes);
  }
  if (ArrayBuffer.isView(bytes)) {
    return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  throw new ViewerClientError("FRAME_BYTES_INVALID", "frame bytes are not a byte buffer");
}

export default { createFrameLoader, buildTransportImage };
