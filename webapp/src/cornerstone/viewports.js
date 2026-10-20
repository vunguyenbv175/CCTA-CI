/**
 * RenderingEngine + per-plane STACK viewport management.
 *
 * Multi-STACK design (prior art E3): one Cornerstone RenderingEngine owns
 * three STACK viewports (AXIAL/CORONAL/SAGITTAL), each displaying the frozen
 * per-plane frame list.  Wheel = StackScroll is the CCTA adaptation.
 */
import { Enums, RenderingEngine } from "@cornerstonejs/core";
import { PLANES } from "../lib/constants.js";
import { ViewerClientError } from "../lib/errors.js";

export const RENDERING_ENGINE_ID = "ccta-physician-viewer-engine";

export function viewportIdFor(plane) {
  return `ccta-viewport-${plane}`;
}

export function createViewportManager(options = {}) {
  const renderingEngineId = options.renderingEngineId || RENDERING_ENGINE_ID;
  const engine = new RenderingEngine(renderingEngineId);
  const viewportIds = new Map();
  const viewportElements = new Map();

  return {
    renderingEngineId,
    engine,

    /** Enable (or reuse) the STACK viewport of one plane over its element. */
    ensureViewport(plane, element) {
      if (!PLANES.includes(plane)) {
        throw new ViewerClientError("PLANE_INVALID", `unknown plane ${plane}`);
      }
      const viewportId = viewportIdFor(plane);
      if (viewportIds.has(plane)) {
        return engine.getViewport(viewportId);
      }
      viewportElements.set(plane, element);
      engine.enableElement({
        viewportId,
        type: Enums.ViewportType.STACK,
        element,
        defaultOptions: { background: [0, 0, 0] },
      });
      viewportIds.set(plane, viewportId);
      return engine.getViewport(viewportId);
    },

    viewportFor(plane) {
      const viewportId = viewportIds.get(plane);
      return viewportId ? engine.getViewport(viewportId) : undefined;
    },

    /** Replace the stack of one plane; returns the actually displayed index. */
    async setStack(plane, imageIds, index) {
      const viewport = this.viewportFor(plane);
      if (!viewport) {
        throw new ViewerClientError("VIEWPORT_MISSING", `no viewport for ${plane}`);
      }
      if (!imageIds.length) {
        // A plane with no frozen frames is presented as an empty viewport,
        // never as a stale stack from a previous session.
        const viewportId = viewportIds.get(plane);
        const element = viewportElements.get(plane);
        engine.disableElement(viewportId);
        engine.enableElement({
          viewportId,
          type: Enums.ViewportType.STACK,
          element,
          defaultOptions: { background: [0, 0, 0] },
        });
        return 0;
      }
      const safeIndex = Math.min(Math.max(Number(index) || 0, 0), imageIds.length - 1);
      await viewport.setStack(imageIds, safeIndex);
      viewport.render();
      return viewport.getCurrentImageIdIndex();
    },

    /** Move one plane's stack to an index (no-op when the stack is empty). */
    async setIndex(plane, index) {
      const viewport = this.viewportFor(plane);
      if (!viewport) {
        return 0;
      }
      const imageIds = viewport.getImageIds();
      if (!imageIds || !imageIds.length) {
        return 0;
      }
      const safeIndex = Math.min(Math.max(Number(index) || 0, 0), imageIds.length - 1);
      await viewport.setImageIdIndex(safeIndex);
      viewport.render();
      return viewport.getCurrentImageIdIndex();
    },

    /**
     * Apply one frozen viewport matrix to a plane's camera.
     *
     * pan_x/pan_y are normalized viewport extents (the frozen grammar's unit);
     * they are converted to canvas pixels for the Cornerstone camera.  Values
     * are validated BEFORE this call (actionEnvelope.assertMatrix) — this
     * function never clamps silently.
     */
    applyMatrix(plane, matrix, element) {
      const viewport = this.viewportFor(plane);
      if (!viewport) {
        return;
      }
      const width = element && element.clientWidth ? element.clientWidth : 1;
      const height = element && element.clientHeight ? element.clientHeight : 1;
      viewport.setZoom(matrix.zoom);
      viewport.setPan([matrix.pan_x * width, matrix.pan_y * height]);
      viewport.render();
    },

    /** Read the plane's camera as normalized frozen-matrix components. */
    readMatrix(plane, element) {
      const viewport = this.viewportFor(plane);
      if (!viewport) {
        return { zoom: 1, pan_x: 0, pan_y: 0 };
      }
      const width = element && element.clientWidth ? element.clientWidth : 1;
      const height = element && element.clientHeight ? element.clientHeight : 1;
      const pan = viewport.getPan();
      return {
        zoom: viewport.getZoom(),
        pan_x: pan[0] / width,
        pan_y: pan[1] / height,
      };
    },

    reset() {
      for (const plane of PLANES) {
        const viewportId = viewportIds.get(plane);
        if (viewportId) {
          try {
            engine.disableElement(viewportId);
          } catch {
            // Element may already be gone during teardown.
          }
        }
      }
      viewportIds.clear();
      viewportElements.clear();
    },

    destroy() {
      try {
        engine.destroy();
      } catch {
        // Engine teardown is best-effort; the page is going away.
      }
      viewportIds.clear();
    },
  };
}

export default { createViewportManager, viewportIdFor, RENDERING_ENGINE_ID };
