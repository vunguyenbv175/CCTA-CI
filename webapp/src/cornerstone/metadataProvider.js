/**
 * Register the DISPLAY-ONLY CCTA metadata provider with Cornerstone
 * (priority 10000, same provider-priority pattern as the CMR source).
 */
import { metaData } from "@cornerstonejs/core";
import { createMetadataProvider } from "../lib/metadata.js";

export const METADATA_PROVIDER_PRIORITY = 10000;

/**
 * @param {ReturnType<import("../lib/metadata.js").createFrameDisplayStore>} store
 */
export function registerCctaMetadataProvider(store) {
  metaData.addProvider(createMetadataProvider(store), METADATA_PROVIDER_PRIORITY);
}

export default { registerCctaMetadataProvider, METADATA_PROVIDER_PRIORITY };
