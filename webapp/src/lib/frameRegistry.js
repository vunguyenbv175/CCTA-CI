/**
 * The per-session frame registry: imageId <-> frozen frame record resolution.
 *
 * The registry is a display cache over the per-plane frame documents.  It can
 * always resolve a record on demand (one GET per plane lookup miss) so a
 * Cornerstone cache miss never degrades to a fabricated image.
 */
import { ViewerClientError } from "./errors.js";
import { makeImageId } from "./imageIds.js";

/**
 * @param {{getFrames: (sessionId: string, plane: string) => Promise<{frames: object[]}>}} dependencies
 */
export function createFrameRegistry(dependencies) {
  const { getFrames } = dependencies;
  const byPlane = new Map();
  const inFlight = new Map();

  const keyFor = (sessionId, plane) => `${sessionId}|${plane}`;

  const storeRecords = (sessionId, plane, records) => {
    byPlane.set(
      keyFor(sessionId, plane),
      records.filter((record) => record && typeof record === "object")
    );
  };

  const loadPlane = async (sessionId, plane) => {
    const key = keyFor(sessionId, plane);
    if (byPlane.has(key)) {
      return byPlane.get(key);
    }
    if (!inFlight.has(key)) {
      inFlight.set(
        key,
        getFrames(sessionId, plane)
          .then((document) => {
            storeRecords(sessionId, plane, document.frames || []);
            inFlight.delete(key);
            return byPlane.get(key);
          })
          .catch((error) => {
            inFlight.delete(key);
            throw error;
          })
      );
    }
    return inFlight.get(key);
  };

  return {
    /** Replace the cached records of one plane (after GET frames / an action). */
    setPlaneRecords(sessionId, plane, records) {
      storeRecords(sessionId, plane, records);
    },

    /** The cached records of one plane (never triggers a fetch). */
    planeRecords(sessionId, plane) {
      return byPlane.get(keyFor(sessionId, plane)) || [];
    },

    /**
     * Resolve the record for one parsed image id; fetches the plane document
     * once on a cold cache.
     */
    async resolveRecord(parsed) {
      const records = await loadPlane(parsed.sessionId, parsed.plane);
      const record = records.find(
        (item) => Number(item.frame_index) === parsed.frameIndex
      );
      if (!record) {
        throw new ViewerClientError(
          "FRAME_NOT_FOUND",
          `no record at ${parsed.plane}[${parsed.frameIndex}]`
        );
      }
      return record;
    },

    /**
     * The image ids of one plane's cached records, in frozen order.  Each id
     * carries the record's public id, so a re-minted PNG is a new image id.
     */
    imageIds(sessionId, plane) {
      const sorted = [...(byPlane.get(keyFor(sessionId, plane)) || [])].sort(
        (a, b) => Number(a.frame_index) - Number(b.frame_index)
      );
      return sorted.map((record) => {
        const publicId = String(record.provenance_ref).split(":").pop();
        return makeImageId(sessionId, plane, Number(record.frame_index), publicId);
      });
    },

    /** Drop every cached plane (session switch / teardown). */
    clear() {
      byPlane.clear();
      inFlight.clear();
    },
  };
}

export default { createFrameRegistry };
