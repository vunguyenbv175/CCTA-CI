/* Closed, same-origin physician client: presentation only, no fetch/import/CDN. */
(() => {
  "use strict";
  const planes = ["AXIAL", "CORONAL", "SAGITTAL"];
  const presetIds = ["CT_MEDIASTINUM", "CT_VESSEL_CONTRAST", "CT_LUNG", "CT_BONE", "CT_SOFT_TISSUE"];
  const freshViewport = () => ({ zoom: 1, pan_x: 0, pan_y: 0 });
  const state = {
    sessionId: null,
    plane: "AXIAL",
    frames: [],
    frameIndex: 0,
    parent: null,
    nonce: 0,
    viewportByPlane: Object.fromEntries(planes.map((plane) => [plane, freshViewport()])),
    displayToken: 0,
    sessionGeneration: 0,
  };
  const $ = (id) => document.getElementById(id);
  const status = (text) => { $("status").textContent = text; };
  const nextNonce = (prefix) => `${prefix}-${Date.now().toString(36)}-${(++state.nonce).toString(36).padStart(12, "0")}`;
  const canonical = (value) => {
    if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
    if (value && typeof value === "object") return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
    return JSON.stringify(value);
  };
  const errorText = (error) => error instanceof Error ? error.message : String(error);
  const currentBinding = () => ({ sessionId: state.sessionId, generation: state.sessionGeneration });
  const bindingIsCurrent = (binding) => Boolean(binding)
    && binding.sessionId === state.sessionId
    && binding.generation === state.sessionGeneration;
  const sessionPath = (sessionId, suffix) => `/session/${encodeURIComponent(sessionId)}${suffix}`;
  const isCurrent = (token, binding) => token === state.displayToken && bindingIsCurrent(binding);

  async function getJson(path) {
    const response = await fetch(path, { credentials: "same-origin" });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error ? `${body.error.code}: ${body.error.message}` : `HTTP ${response.status}`);
    return body;
  }

  function acceptedState(document) {
    if (!document || typeof document !== "object") return;
    const indexes = document.frame_index_by_plane || {};
    const viewports = document.viewport_by_plane || {};
    state.frameIndexByPlane = {};
    state.viewportByPlane = Object.fromEntries(planes.map((plane) => [plane, freshViewport()]));
    state.plane = planes.includes(document.current_plane) ? document.current_plane : "AXIAL";
    state.parent = Array.isArray(document.accepted_nonces) && document.accepted_nonces.length
      ? document.accepted_nonces[document.accepted_nonces.length - 1]
      : null;
    planes.forEach((plane) => {
      if (Number.isInteger(indexes[plane])) state.frameIndexByPlane = state.frameIndexByPlane || {};
      if (Number.isInteger(indexes[plane])) state.frameIndexByPlane[plane] = indexes[plane];
      if (viewports[plane] && typeof viewports[plane] === "object") {
        state.viewportByPlane[plane] = {
          zoom: Number(viewports[plane].zoom),
          pan_x: Number(viewports[plane].pan_x),
          pan_y: Number(viewports[plane].pan_y),
        };
      }
    });
    state.frameIndex = state.frameIndexByPlane && Number.isInteger(state.frameIndexByPlane[state.plane])
      ? state.frameIndexByPlane[state.plane] : 0;
    state.viewport = state.viewportByPlane[state.plane] || freshViewport();
    $("plane-select").value = state.plane;
    $("zoom").value = state.viewport.zoom;
    $("pan-x").value = state.viewport.pan_x;
    $("pan-y").value = state.viewport.pan_y;
  }

  function invalidateDisplay() {
    state.displayToken += 1;
    $("frame").removeAttribute("src");
    $("frame").onerror = null;
    $("frame-caption").textContent = "No frame selected";
    $("qc-panel").hidden = true;
    $("qc-text").textContent = "";
  }

  async function postAction(action, params) {
    const binding = currentBinding();
    const sessionId = binding.sessionId;
    const body = {
      action,
      params,
      client_nonce: nextNonce(action),
      view: { plane: state.plane, frame_index: state.frameIndex },
      lineage: { session_id: sessionId, parent_action_nonce: state.parent },
    };
    const response = await fetch(`/session/${encodeURIComponent(sessionId)}/action`, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: canonical(body),
    });
    const document = await response.json();
    if (!bindingIsCurrent(binding)) return null;
    if (!response.ok) {
      const error = document.error || { code: `HTTP_${response.status}`, message: "request rejected" };
      throw new Error(`${error.code}: ${error.message}`);
    }
    acceptedState(document);
    return { document, binding };
  }

  async function loadQc(token, binding) {
    try {
      const document = await getJson(sessionPath(binding.sessionId, "/qc"));
      if (!isCurrent(token, binding)) return;
      $("qc-panel").hidden = false;
      $("qc-text").textContent = JSON.stringify(document, null, 2);
    } catch (error) {
      if (!isCurrent(token, binding)) return;
      const message = errorText(error);
      $("qc-panel").hidden = false;
      $("qc-text").textContent = message;
      status(message);
    }
  }

  async function loadFrame(record, token, binding) {
    const id = record.provenance_ref.split(":")[1];
    const url = `/frame/${encodeURIComponent(id)}.png`;
    let response;
    try {
      // This request preserves the server's typed rejection document.  The
      // actual displayed image is still a same-origin URL below, allowing the
      // browser to perform normal image caching and If-None-Match revalidation.
      response = await fetch(url, { credentials: "same-origin" });
    } catch (error) {
      if (isCurrent(token, binding)) {
        $("frame").removeAttribute("src");
        throw error;
      }
      return;
    }
    if (response.status !== 304 && !response.ok) {
      let document = null;
      try { document = await response.json(); } catch (_error) { /* typed fallback below */ }
      const error = document && document.error
        ? `${document.error.code}: ${document.error.message}`
        : `FRAME_${response.status}: frame request rejected`;
      if (isCurrent(token, binding)) {
        $("frame").removeAttribute("src");
        throw new Error(error);
      }
      return;
    }
    if (!isCurrent(token, binding)) return;
    $("frame").removeAttribute("src");
    $("frame").onerror = () => {
      if (isCurrent(token, binding)) {
        $("frame").removeAttribute("src");
        status("FRAME_IMAGE_LOAD_FAILED: frame image could not be displayed");
      }
    };
    // Use the PNG endpoint directly: CSP default-src 'self' permits it and
    // the browser owns its ETag/304 cache negotiation.
    $("frame").src = url;
    applyDisplayState();
    $("frame-caption").textContent = `${state.plane} · frame ${record.frame_index} · WL ${record.wl_preset_id || "raw"}`;
  }

  async function loadFrames(binding = currentBinding()) {
    if (!bindingIsCurrent(binding)) return;
    const token = ++state.displayToken;
    $("qc-panel").hidden = true;
    $("qc-text").textContent = "";
    const document = await getJson(sessionPath(binding.sessionId, `/frames/${state.plane}`));
    if (!isCurrent(token, binding)) return;
    state.frames = document.frames || [];
    const desiredIndex = state.frameIndex;
    const current = state.frames.find((frame) => frame.frame_index === desiredIndex) || state.frames[0];
    if (current) state.frameIndex = current.frame_index;
    $("slice-label").textContent = state.frames.length ? `Slice ${state.frameIndex + 1} / ${state.frames.length}` : "No frames";
    const qcPromise = loadQc(token, binding);
    try {
      if (current) await loadFrame(current, token, binding);
      else {
        $("frame").removeAttribute("src");
        $("frame-caption").textContent = "No frame available";
      }
    } finally {
      await qcPromise;
    }
  }

  function applyDisplayState() {
    state.viewport = state.viewportByPlane[state.plane] || freshViewport();
    $("zoom").value = state.viewport.zoom;
    $("pan-x").value = state.viewport.pan_x;
    $("pan-y").value = state.viewport.pan_y;
    $("frame").style.transform = `scale(${state.viewport.zoom}) translate(${state.viewport.pan_x * 10}px, ${state.viewport.pan_y * 10}px)`;
  }

  async function refresh(binding = currentBinding()) {
    if (!bindingIsCurrent(binding)) return;
    await loadFrames(binding);
  }

  async function actionAndRefresh(action, params) {
    const binding = currentBinding();
    try {
      const result = await postAction(action, params);
      if (!result) return;
      await refresh(binding);
    } catch (error) {
      if (!bindingIsCurrent(binding)) return;
      applyDisplayState();
      status(errorText(error));
    }
  }

  async function selectPlane(plane) {
    const binding = currentBinding();
    try {
      const result = await postAction("SET_PLANE", { plane });
      if (!result) return;
      await refresh(binding);
    } catch (error) {
      if (!bindingIsCurrent(binding)) return;
      applyDisplayState();
      status(errorText(error));
    }
  }

  async function scroll(delta) {
    const index = state.frameIndex + delta;
    if (index < 0 || index >= state.frames.length) return;
    await actionAndRefresh("SET_SLICE", { plane: state.plane, frame_index: index });
  }

  async function switchSession(sessionId) {
    invalidateDisplay();
    state.sessionGeneration += 1;
    state.sessionId = sessionId;
    const binding = currentBinding();
    state.parent = null;
    state.frames = [];
    state.frameIndex = 0;
    try {
      const document = await getJson(sessionPath(binding.sessionId, "/state"));
      if (!bindingIsCurrent(binding)) return;
      acceptedState(document);
      await refresh(binding);
    } catch (error) {
      if (bindingIsCurrent(binding)) status(errorText(error));
    }
  }

  async function init() {
    try {
      const listing = await getJson("/sessions");
      const select = $("session-select");
      select.replaceChildren();
      (listing.sessions || []).forEach((id) => {
        const option = document.createElement("option");
        option.value = id;
        option.textContent = id;
        select.appendChild(option);
      });
      if (!listing.sessions || !listing.sessions.length) {
        status("No governed sessions are available.");
        return;
      }
      await switchSession(listing.sessions[0]);
      select.value = state.sessionId;
    } catch (error) {
      status(errorText(error));
    }
  }

  $("session-select").addEventListener("change", (event) => { void switchSession(event.target.value); });
  $("plane-select").addEventListener("change", (event) => { void selectPlane(event.target.value); });
  $("previous").addEventListener("click", () => { void scroll(-1); });
  $("next").addEventListener("click", () => { void scroll(1); });
  presetIds.forEach((id) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = id.replace("CT_", "");
    button.addEventListener("click", () => { void actionAndRefresh("SET_WL_PRESET", { preset_id: id }); });
    $("presets").appendChild(button);
  });
  $("clear-wl").addEventListener("click", () => { void actionAndRefresh("CLEAR_WL", {}); });
  $("apply-viewport").addEventListener("click", () => {
    const viewport = { zoom: Number($("zoom").value), pan_x: Number($("pan-x").value), pan_y: Number($("pan-y").value) };
    void actionAndRefresh("SET_VIEWPORT_MATRIX", viewport);
  });
  $("open-qc").addEventListener("click", () => { void loadQc(state.displayToken, currentBinding()); });
  void init();
})();
