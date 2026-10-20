/**
 * The one HTTP boundary of the client (same-origin loopback only).
 *
 * Every request is relative (same origin), credentialed with
 * ``credentials: "same-origin"`` and performs no third-party/network egress:
 * the loopback server is the only peer.  Typed server rejections are rethrown
 * as ViewerClientError carrying the SERVER code (never swallowed).
 */
import { ViewerClientError } from "./errors.js";

/**
 * @param {{fetchImpl?: typeof fetch}} [options]
 */
export function createFrameSource(options = {}) {
  const fetchImpl = options.fetchImpl || ((...args) => fetch(...args));

  const getJson = async (path) => {
    const response = await fetchImpl(path, { credentials: "same-origin" });
    const document = await response.json();
    if (!response.ok) {
      throw serverRefusal(document, response.status);
    }
    return document;
  };

  return {
    /** GET /sessions -> string[] */
    async getSessions() {
      const document = await getJson("/sessions");
      const sessions = Array.isArray(document.sessions) ? document.sessions : [];
      return sessions.filter((item) => typeof item === "string");
    },

    /** GET /session/:id/state -> frozen state document */
    async getState(sessionId) {
      return getJson(sessionPath(sessionId, "/state"));
    },

    /** GET /session/:id/frames/:plane -> {plane, frames} */
    async getFrames(sessionId, plane) {
      const document = await getJson(sessionPath(sessionId, `/frames/${plane}`));
      const frames = Array.isArray(document.frames) ? document.frames : [];
      return { plane: String(document.plane || plane), frames };
    },

    /** GET /session/:id/qc -> frozen QC document (verbatim) */
    async getQc(sessionId) {
      return getJson(sessionPath(sessionId, "/qc"));
    },

    /** GET /session/:id/coronary -> closed coronary overlay document */
    async getCoronary(sessionId, options = {}) {
      const mode = options && typeof options.mode === "string" ? options.mode : "";
      const vessel =
        options && typeof options.vessel === "string" ? options.vessel : "";
      const query = [
        mode ? `mode=${encodeURIComponent(mode)}` : "",
        vessel ? `vessel=${encodeURIComponent(vessel)}` : "",
      ]
        .filter(Boolean)
        .join("&");
      const suffix = query ? `/coronary?${query}` : "/coronary";
      return getJson(sessionPath(sessionId, suffix));
    },

    /** GET /session/:id/cpr -> closed CPR view document (metadata, no pixels) */
    async getCpr(sessionId, options = {}) {
      const vessel =
        options && typeof options.vessel === "string" ? options.vessel : "";
      const suffix = vessel ? `/cpr?vessel=${encodeURIComponent(vessel)}` : "/cpr";
      return getJson(sessionPath(sessionId, suffix));
    },

    /** GET /session/:id/cpr.png -> the frozen CPR PNG transport bytes */
    async getCprPngBytes(sessionId, options = {}) {
      const vessel =
        options && typeof options.vessel === "string" ? options.vessel : "";
      const suffix = vessel ? `/cpr.png?vessel=${encodeURIComponent(vessel)}` : "/cpr.png";
      const response = await fetchImpl(sessionPath(sessionId, suffix), {
        credentials: "same-origin",
      });
      if (!response.ok) {
        let document = null;
        try {
          document = await response.json();
        } catch {
          document = null;
        }
        throw serverRefusal(document, response.status);
      }
      return new Uint8Array(await response.arrayBuffer());
    },

    /** GET /session/:id/tree-schematic -> closed navigator document */
    async getTreeSchematic(sessionId) {
      return getJson(sessionPath(sessionId, "/tree-schematic"));
    },

    /** GET /session/:id/sync-point -> closed cursor derivation (no stored state) */
    async getSyncPoint(sessionId, pick) {
      const plane = encodeURIComponent(String(pick.plane));
      const frame = encodeURIComponent(String(pick.frameIndex));
      const row = encodeURIComponent(String(pick.row));
      const column = encodeURIComponent(String(pick.column));
      return getJson(
        sessionPath(sessionId, `/sync-point?plane=${plane}&frame=${frame}&row=${row}&column=${column}`)
      );
    },

    /** GET /frame/:id.png -> the frozen PNG transport bytes */
    async getPngBytes(recordId) {
      const path = `/frame/${encodeURIComponent(recordId)}.png`;
      const response = await fetchImpl(path, { credentials: "same-origin" });
      if (!response.ok) {
        let document = null;
        try {
          document = await response.json();
        } catch {
          document = null;
        }
        throw serverRefusal(document, response.status);
      }
      return new Uint8Array(await response.arrayBuffer());
    },

    /** POST /session/:id/action with canonical bytes -> state document */
    async postAction(sessionId, envelope, canonicalBody) {
      const response = await fetchImpl(sessionPath(sessionId, "/action"), {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: canonicalBody,
      });
      const document = await response.json();
      if (!response.ok) {
        throw serverRefusal(document, response.status);
      }
      return document;
    },

    /** POST /session/:id/seed with canonical bytes -> state document.
     *
     * F2 truthful-wait (waiter-detach only): accepts an optional trailing
     * ``{signal}``-style options object (a client ``fetch`` option, never a
     * wire/server key).  Only the seed POST fetch is ever abortable — every
     * other route (``getState``/``postAction``/reads) never receives a signal.
     */
    async postSeed(sessionId, envelope, canonicalBody, options = {}) {
      const fetchOptions = {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: canonicalBody,
      };
      if (options && options.signal !== undefined && options.signal !== null) {
        fetchOptions.signal = options.signal;
      }
      const response = await fetchImpl(sessionPath(sessionId, "/seed"), fetchOptions);
      const document = await response.json();
      if (!response.ok) {
        throw serverRefusal(document, response.status);
      }
      return document;
    },
  };
}

function sessionPath(sessionId, suffix) {
  return `/session/${encodeURIComponent(sessionId)}${suffix}`;
}

function serverRefusal(document, status) {
  const error = document && typeof document === "object" ? document.error : null;
  if (error && typeof error.code === "string") {
    return new ViewerClientError(error.code, String(error.message || "server refusal"));
  }
  return new ViewerClientError(`HTTP_${status}`, "request rejected");
}

export default { createFrameSource };
