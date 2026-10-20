/**
 * CCTA physician-viewer bundle entry.
 *
 * Boot order: Cornerstone core + tools init, display-metadata provider
 * registration, ccta-viewer:// loader registration, then the application
 * controller over the unchanged viewer server.
 */
import { init as initCornerstone } from "@cornerstonejs/core";
import { init as initCornerstoneTools } from "@cornerstonejs/tools";
import { createFrameSource } from "./lib/frameSource.js";
import { createFrameDisplayStore } from "./lib/metadata.js";
import { registerCctaImageLoader } from "./cornerstone/imageLoader.js";
import { registerCctaMetadataProvider } from "./cornerstone/metadataProvider.js";
import { createClientApp } from "./ui/app.js";

function boot() {
  initCornerstone();
  initCornerstoneTools();

  const source = createFrameSource();
  const displayStore = createFrameDisplayStore();
  const app = createClientApp({ source });

  registerCctaMetadataProvider(displayStore);
  registerCctaImageLoader({
    resolveRecord: (parsed) => app.registry.resolveRecord(parsed),
    fetchPngBytes: (recordId) => source.getPngBytes(recordId),
    onDisplay: (imageId, display) => displayStore.set(imageId, display),
  });

  void app.init();
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
}

export { boot };
