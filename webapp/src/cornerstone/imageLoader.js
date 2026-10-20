/**
 * Register the ``ccta-viewer`` image loader with the Cornerstone imageLoader
 * (core dependency, admitted; no dicom-image-loader anywhere).
 */
import { imageLoader } from "@cornerstonejs/core";
import { VoxelManager } from "@cornerstonejs/core/utilities";
import { CCTA_VIEWER_SCHEME } from "../lib/imageIds.js";
import { buildTransportImage, createFrameLoader } from "../lib/frameLoader.js";
import { decodeTransportPng } from "./transport.js";

/**
 * @param {{
 *   resolveRecord: (parsed: object) => Promise<object>,
 *   fetchPngBytes: (recordId: string) => Promise<Uint8Array>,
 *   onDisplay?: (imageId: string, display: object) => void,
 * }} dependencies
 */
export function registerCctaImageLoader(dependencies) {
  const handler = createFrameLoader({
    resolveRecord: dependencies.resolveRecord,
    fetchPngBytes: dependencies.fetchPngBytes,
    decodePng: (bytes) => decodeTransportPng(bytes),
    buildImage: (input) =>
      buildTransportImage({
        ...input,
        createVoxelManager: (voxelInput) =>
          VoxelManager.createImageVoxelManager(voxelInput),
      }),
    onDisplay: dependencies.onDisplay,
  });
  imageLoader.registerImageLoader(CCTA_VIEWER_SCHEME, (imageId) => ({
    promise: handler(imageId),
    cancelFn: () => {},
  }));
  return handler;
}

export default { registerCctaImageLoader };
