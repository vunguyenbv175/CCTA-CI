/**
 * Transport PNG decode (browser-only adapter).
 *
 * The frozen server PNG is RGB24 (grayscale-replicated, ALREADY windowed by
 * VIEWER_BINARY32_WL_V1).  Decoding uses the platform image decoder
 * (createImageBitmap) and extracts the RGB channels 1:1.  No window/level
 * arithmetic is performed anywhere in this path.
 */
import { ViewerClientError } from "../lib/errors.js";

/**
 * @param {Uint8Array} pngBytes frozen transport PNG bytes
 * @returns {Promise<{width: number, height: number, rgb: Uint8Array}>}
 */
export async function decodeTransportPng(pngBytes) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(
      new Blob([pngBytes], { type: "image/png" })
    );
  } catch (error) {
    throw new ViewerClientError(
      "FRAME_DECODE_FAILED",
      `transport PNG could not be decoded: ${error}`
    );
  }
  try {
    const width = bitmap.width;
    const height = bitmap.height;
    if (!width || !height) {
      throw new ViewerClientError("FRAME_DECODE_FAILED", "transport PNG has zero size");
    }
    const canvas = createCanvas(width, height);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) {
      throw new ViewerClientError("FRAME_DECODE_FAILED", "no 2D context available");
    }
    context.drawImage(bitmap, 0, 0);
    const rgba = context.getImageData(0, 0, width, height).data;
    const rgb = new Uint8Array(width * height * 3);
    for (let source = 0, target = 0; source < rgba.length; source += 4, target += 3) {
      rgb[target] = rgba[source];
      rgb[target + 1] = rgba[source + 1];
      rgb[target + 2] = rgba[source + 2];
    }
    return { width, height, rgb };
  } finally {
    if (typeof bitmap.close === "function") {
      bitmap.close();
    }
  }
}

function createCanvas(width, height) {
  if (typeof OffscreenCanvas === "function") {
    return new OffscreenCanvas(width, height);
  }
  if (typeof document !== "undefined") {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }
  throw new ViewerClientError("FRAME_DECODE_FAILED", "no canvas implementation available");
}

export default { decodeTransportPng };
