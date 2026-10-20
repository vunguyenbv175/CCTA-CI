/**
 * CCTA physician-viewer application controller.
 *
 * Consumes the UNCHANGED CCTA viewer server (session list, per-plane frame
 * documents, frame PNGs, QC document, action seam).  The v0 generation-bound
 * async discipline (apps/viewer/static/app.js, preserved under
 * webapp/v0-fallback/) is the interaction contract: every async continuation
 * checks its binding and display token before touching the DOM or state.
 */
import { Enums } from "@cornerstonejs/core";
import { createLinearRGBTransferFunction } from "@cornerstonejs/core/utilities";
import { PLANES, QUALITY_GRADES, WL_PRESET_IDS } from "../lib/constants.js";
import { IDENTITY_DISPLAY_VOI } from "../lib/metadata.js";
import { canonicalJson } from "../lib/canonical.js";
import { assertMatrix, buildActionEnvelope } from "../lib/actionEnvelope.js";
import { createFrameSource } from "../lib/frameSource.js";
import { createFrameRegistry } from "../lib/frameRegistry.js";
import {
  CORONARY_OVERLAY_ID,
  OVERLAY_DEFAULT_MODE,
  OVERLAY_MODES,
  OVERLAY_PROJECTION_CAPTION,
  drawCoronaryOverlay,
  loadCoronaryOverlay,
} from "../lib/coronaryOverlay.js";
import { CPR_VIEW_SCHEMA, loadCprView } from "../lib/cprPanel.js";
import {
  TREE_SCHEMATIC_CAPTION,
  loadTreeSchematic,
  schematicVesselLabels,
  schematicViewBox,
} from "../lib/treeSchematic.js";
import {
  SYNC_CURSOR_CAPTION,
  drawSyncCursor,
  loadSyncPoint,
} from "../lib/syncCursor.js";
import { displayStateFrom } from "../lib/sessionState.js";
import {
  pickFocusedPlane,
  pickFrameIndex,
  pickPreferredSession,
  pickViewPlane,
} from "../lib/seriesSelection.js";
import {
  correctionCapabilities,
  createQcController,
  deriveStudyRef,
} from "../lib/qcSeam.js";
import { ViewerClientError } from "../lib/errors.js";
import {
  assertSeedDirection,
  assertSeedVoxel,
  buildSeedEnvelope,
} from "../lib/seedEnvelope.js";
import {
  SEED_PROVENANCE_UNAVAILABLE,
  captureAcceptedSeedBinding,
  seedProvenanceText,
} from "../lib/seedProvenance.js";
import { registerCctaImageLoader } from "../cornerstone/imageLoader.js";
import { registerCctaMetadataProvider } from "../cornerstone/metadataProvider.js";
import { ensureCctaToolGroup } from "../cornerstone/tools.js";
import { createViewportManager } from "../cornerstone/viewports.js";

/**
 * @param {{document?: Document, fetchImpl?: typeof fetch, source?: object}} [environment]
 */
export function createClientApp(environment = {}) {
  const doc = environment.document || document;
  const source = environment.source || createFrameSource({ fetchImpl: environment.fetchImpl });
  const registry = createFrameRegistry({
    getFrames: (sessionId, plane) => source.getFrames(sessionId, plane),
  });
  const viewports = createViewportManager();

  const state = {
    sessionId: null,
    generation: 0,
    displayToken: 0,
    parentActionNonce: null,
    focusedPlane: PLANES[0],
    stickySessionId: null,
    stickyPlane: null,
    model: null,
    qcDocument: null,
    overlayDocument: null,
    // Decision A: the closed display mode of the overlay (SLICE_LOCAL
    // default; PROJECTION is the explicitly labeled localizer mode).
    overlayMode: OVERLAY_DEFAULT_MODE,
    // F8: the typed failure of the overlay read surface.  UNAVAILABLE means
    // "no ACTIVE centerline" (the control stays hidden); AMBIGUOUS/INVALID
    // are surfaced VISIBLY instead of silently removing the overlay.
    overlayError: null,
    // Decision B: the straightened CPR panel state (view document + typed
    // failure; UNAVAILABLE hides the panel, INVALID/STALE stay visible).
    cprDocument: null,
    cprError: null,
    // TREE_SCHEMATIC_NAV: the read-only navigator panel state (schematic
    // document + typed failure; UNAVAILABLE hides the panel, INVALID/STALE
    // stay visible).  Clicks submit the EXISTING SELECT_VESSEL action.
    schematicDocument: null,
    schematicError: null,
    // MPR_SYNC_CURSOR: the ephemeral shared crosshair pick (never persisted,
    // never a finding).  syncSource is the last-placed (plane, frame, row,
    // column) pick; syncDocument is the served derivation; reselect replaces
    // the pick and navigation refresh re-derives from it.  syncSeq is the
    // cursor-specific request generation: it changes on clear AND reselect
    // (FINDING 2) so a cleared-while-pending GET or an older out-of-order
    // pick can never resurrect/overwrite/erase the latest cursor — the
    // session+generation binding alone cannot see those races.
    syncSource: null,
    syncDocument: null,
    syncError: null,
    syncSeq: 0,
    // CORONARY_MULTI_VESSEL_NAV (Decision A): the selected vessel of the
    // review surface (a frozen VESSEL:* token, or null for the legacy #28
    // single-path behavior).  Persisted server-side via SELECT_VESSEL.
    selectedVessel: null,
    // TSN_R4_F1 (selection authority correction): the optimistic click is
    // tracked separately, displayed ONLY as pending, never as accepted;
    // reconciled at settle.
    pendingSelection: null,
    // Q3-SEED-PROV F1-race (R2/R3): monotonic vessel-navigation sequence.  A
    // seed submission records it BEFORE posting; the accept-time snapshot is
    // captured only while it is unchanged, so a vessel switch accepted while
    // the /seed response is in flight fails the capture closed (typed
    // unavailable).  R3: panel loads tag themselves with it (drop stale ones).
    vesselNavSeq: 0,
    // TREE_SCHEMATIC_NAV F2: monotonic ACTIVE-CENTERLINE-SET revision.  It
    // advances ONLY when the canonical set this client mutated is replaced -
    // today that is an accepted reseed (POST /session/:id/seed mints a NEW
    // centerline through the governed reseed seam).  A centerline correction
    // (REQUEST_CENTERLINE_DECISION ACCEPT/REJECT/EDIT) changes the same set,
    // but the client has no call site for that action, so nothing here can
    // observe it yet: whoever adds one MUST advance this revision on
    // acceptance, or that path silently keeps displaying superseded geometry.
    // Every derived navigation panel load is bound to the revision it started
    // under, so a load issued before the set was replaced is dropped instead
    // of repainting geometry the server has already superseded.  Distinct
    // from ``vesselNavSeq``: that one says WHICH vessel is displayed, this one
    // says WHICH geometry generation is displayed - a reseed changes the
    // geometry WITHOUT changing the selected vessel, which is exactly the
    // case the navigation guard deliberately cannot see.
    activeSetSeq: 0,
    // TREE_SCHEMATIC_NAV F2: per-panel request ordinals.  ``vesselNavSeq``
    // and ``activeSetSeq`` both describe WHAT is displayed; these describe
    // WHICH load produced it.  Two loads can share one navigation and one
    // set revision (the physician pressed Refresh twice) - without an ordinal
    // the slower earlier response still passes every binding check and
    // overwrites the newer one, including overwriting a typed refusal with
    // the geometry that preceded it.  Each panel counts its own requests so
    // the sequential overlay -> CPR -> schematic chain never cancels itself.
    overlayReqSeq: 0,
    cprReqSeq: 0,
    schematicReqSeq: 0,
    // TSN_R4_F1 (selection authority correction): monotonic refusal-recovery
    // fence.  Bumped synchronously at refusal-recovery entry before the first
    // await; carried in every binding as binding.recoverySeq; loads whose
    // binding.recoverySeq is older than state.selectionRecoverySeq must never
    // write state.selectedVessel nor apply documents.
    selectionRecoverySeq: 0,
    // CORONARY_MULTI_VESSEL_NAV (Decision B): assignment-source badge of
    // the served vessel (USER_ASSIGNED | FIXTURE_DERIVED | ANATOMY_VERIFIED).
    vesselAssignmentSource: null,
    actionQueue: Promise.resolve(),
    nonceCounter: 0,
    signatures: {},
    applyingProgrammatic: false,
    dragPlane: null,
    // T2 physician/manual seed (Phase B): the pending pick is explicit state
    // only — voxel + direction stay EMPTY until the physician picks or types
    // them (no hidden defaults); the last accepted seed of the session is
    // shown beside it.  Reselect replaces the pending pick; submit posts the
    // dedicated seed route through the same action queue (lineage-safe).
    seedArmed: false,
    seedAnchor: null,
    seedDrag: null,
    seedCurrent: null,
    // Q3-SEED-PROV F1: the served centerline binding observed when the seed
    // was ACCEPTED (snapshot, never re-captured on panel reloads) — the
    // displayed overlay/CPR documents must still match it or the summary
    // renders the typed unavailable state.
    seedBinding: null,
    seedError: null,
    // WL_V2 (spec Part 4.2) drag state: one primary-drag gesture at a time.
    wlDrag: null,
    wlSubmitTimers: new Map(),
  };
  const elements = {};
  const sliceTimers = new Map();
  const syncTimers = new Map();
  const identityTransferFunctions = new WeakMap();
  const WL_SUBMIT_DEBOUNCE_MS = 150;
  const qcController = createQcController({
    postAction: (action, params) => postAction(action, params),
    getQc: () => source.getQc(state.sessionId),
  });

  // ---------------------------------------------------------------- helpers

  const $ = (id) => doc.getElementById(id);
  const setStatus = (text) => {
    const status = $("status");
    if (status) {
      status.textContent = text;
    }
  };
  const errorText = (error) => (error instanceof Error ? error.message : String(error));
  // Q3-SEED-PROV F1-race (R3): every binding also carries the vessel-navigation
  // sequence it was created under.  ``bindingIsCurrent`` keeps its
  // session/generation meaning for the action/QC/sync flows; the panel loads
  // (overlay/CPR/schematic) additionally require that sequence through
  // ``panelLoadIsCurrent``, so a STALE navigation response — an older
  // navigation released after a newer one — is dropped instead of mutating
  // ``state.selectedVessel`` / the displayed documents.
  const currentBinding = () => ({
    sessionId: state.sessionId,
    generation: state.generation,
    vesselNavSeq: state.vesselNavSeq,
    // TREE_SCHEMATIC_NAV F2: the ACTIVE-centerline-set revision this load was
    // issued under.  Carried additively; the Q3 guard below is untouched and
    // keeps exactly its navigation meaning.
    activeSetSeq: state.activeSetSeq,
    recoverySeq: state.selectionRecoverySeq,
  });
  const bindingIsCurrent = (binding) =>
    Boolean(binding) &&
    binding.sessionId === state.sessionId &&
    binding.generation === state.generation;
  // Q3-SEED-PROV F1-race (R3): the panel-load guard (same idiom as the
  // sync-cursor ``syncSeq`` guard).  A panel load tagged with an older
  // navigation sequence than the current one is a stale navigation response
  // and must not mutate selection or documents; the R3 variant was exactly an
  // old LAD ``/coronary``+``/cpr`` response released while an RCA seed
  // submission was in flight, which rebound the accepted RCA seed to LAD.
  //
  // TREE_SCHEMATIC_NAV F2: this guard DELIBERATELY does not see set changes.
  // It answers "is the physician still navigating the same vessel?", which a
  // reseed or a correction does not change - the same vessel keeps being
  // selected while its geometry is replaced underneath.  Reusing it for that
  // question would be the wrong discrimination, so the missing one is added
  // separately below rather than folded in here.
  const panelLoadIsCurrent = (binding) =>
    bindingIsCurrent(binding) && binding.vesselNavSeq === state.vesselNavSeq;
  // TREE_SCHEMATIC_NAV F2: the ACTIVE-set guard the navigation guard cannot
  // be.  A load issued under set revision N is dropped once the canonical set
  // has been replaced (revision N+1), so geometry derived from superseded
  // paths can never be painted over their successors — and never repainted
  // after a newer refresh has already displayed a refusal.  Combined with the
  // per-panel request ordinal below, a derived panel load is applied only if
  // it is BOTH the newest request of its panel AND still bound to the
  // navigation and the set revision it was issued under.
  const panelSetIsCurrent = (binding) =>
    panelLoadIsCurrent(binding) && binding.activeSetSeq === state.activeSetSeq;
  // Q3-SEED-PROV F1-race (R2): a seed submission's navigation identity is
  // {sessionId, generation, vesselNavSeq}, recorded BEFORE the submission's
  // first await (``submitSeed``).  ``selectVessel`` bumps ``vesselNavSeq``
  // synchronously at the click — before its queued SELECT_VESSEL action lands
  // — so a vessel switch accepted while the /seed response is in flight is
  // visible here and the accept-time capture fails closed.
  const submissionIsCurrent = (submission) =>
    Boolean(submission) &&
    submission.sessionId === state.sessionId &&
    submission.generation === state.generation &&
    submission.vesselNavSeq === state.vesselNavSeq;
  const nextNonce = (action) => {
    state.nonceCounter += 1;
    return `${action.toLowerCase()}-${Date.now().toString(36)}-${state.nonceCounter
      .toString(36)
      .padStart(12, "0")}`;
  };

  function renderViewportShells() {
    const container = $("viewports");
    if (!container || container.children.length) {
      return;
    }
    for (const plane of PLANES) {
      const figure = doc.createElement("figure");
      figure.className = "viewport";
      figure.id = `viewport-${plane}`;
      const canvasHost = doc.createElement("div");
      canvasHost.className = "viewport-canvas";
      canvasHost.id = `viewport-canvas-${plane}`;
      const caption = doc.createElement("figcaption");
      caption.id = `viewport-caption-${plane}`;
      caption.textContent = `${plane}: loading…`;
      // Display-only coronary overlay canvas: a dedicated 2D layer above the
      // Cornerstone canvas (never a Cornerstone render pass, never a
      // measurement surface).
      const overlayCanvas = doc.createElement("canvas");
      overlayCanvas.className = "overlay-canvas";
      overlayCanvas.id = `overlay-canvas-${plane}`;
      overlayCanvas.dataset.overlayDrawn = "0";
      overlayCanvas.setAttribute("aria-hidden", "true");
      overlayCanvas.hidden = true;
      figure.appendChild(canvasHost);
      figure.appendChild(caption);
      container.appendChild(figure);
      elements[plane] = { host: canvasHost, caption, overlay: overlayCanvas };
      canvasHost.appendChild(overlayCanvas);
    }
  }

  function renderPresetButtons() {
    const presets = $("presets");
    if (!presets || presets.children.length) {
      return;
    }
    for (const presetId of WL_PRESET_IDS) {
      const button = doc.createElement("button");
      button.type = "button";
      button.dataset.presetId = presetId;
      button.textContent = presetId.replace("CT_", "");
      button.addEventListener("click", () => {
        void applyPreset(presetId);
      });
      presets.appendChild(button);
    }
  }

  /** The v2 numeric-entry controls for custom window/level (spec 4.2). */
  function renderWlCustomControls() {
    const toolbar = doc.querySelector(".toolbar");
    if (!toolbar || $("wl-custom")) {
      return;
    }
    const group = doc.createElement("span");
    group.id = "wl-custom";
    group.className = "presets";
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", "Custom window/level");
    const centerInput = doc.createElement("input");
    centerInput.id = "wl-center-input";
    centerInput.type = "number";
    centerInput.step = "any";
    centerInput.size = 6;
    centerInput.setAttribute("aria-label", "window center (HU)");
    const widthInput = doc.createElement("input");
    widthInput.id = "wl-width-input";
    widthInput.type = "number";
    widthInput.step = "any";
    widthInput.size = 6;
    widthInput.setAttribute("aria-label", "window width (HU)");
    const centerLabel = doc.createElement("label");
    centerLabel.textContent = "C ";
    centerLabel.appendChild(centerInput);
    const widthLabel = doc.createElement("label");
    widthLabel.textContent = "W ";
    widthLabel.appendChild(widthInput);
    const submit = doc.createElement("button");
    submit.id = "wl-custom-submit";
    submit.type = "button";
    submit.textContent = "Apply C/W";
    submit.addEventListener("click", () => {
      void applyCustomWlFromInputs();
    });
    group.appendChild(centerLabel);
    group.appendChild(widthLabel);
    group.appendChild(submit);
    toolbar.insertBefore(group, $("clear-wl") || null);
  }

  function updateFocusClasses() {
    for (const plane of PLANES) {
      const figure = $(`viewport-${plane}`);
      if (figure) {
        figure.classList.toggle("focused", plane === state.focusedPlane);
      }
    }
  }

  function updateCaption(plane) {
    const info = elements[plane];
    if (!info) {
      return;
    }
    const records = registry.planeRecords(state.sessionId, plane);
    const viewport = viewports.viewportFor(plane);
    const index = viewport ? viewport.getCurrentImageIdIndex() : 0;
    const record = records.find((item) => Number(item.frame_index) === index);
    if (!record) {
      info.caption.textContent = `${plane}: no frames`;
      return;
    }
    // The frozen frame schema does not distinguish an identity/raw mapping
    // frame from a custom-WL frame: both carry wl_preset_id=None.  Label
    // both with the neutral "C/W" (the exact applied center/width are shown
    // beside it), never "raw" — a custom submit must not be mislabeled as
    // the identity mapping.
    const wl = record.wl_preset_id === null || record.wl_preset_id === undefined
      ? "C/W"
      : String(record.wl_preset_id);
    const matrix = state.model
      ? state.model.viewportByPlane[plane]
      : { zoom: 1, pan_x: 0, pan_y: 0 };
    const provenanceId = String(record.provenance_ref).split(":").pop();
    info.caption.textContent =
      `${plane} · frame ${index + 1}/${records.length} · WL ${wl}` +
      ` (C=${record.wl_applied.center}, W=${record.wl_applied.width})` +
      ` · zoom ${matrix.zoom.toFixed(2)} · pan ${matrix.pan_x.toFixed(2)},${matrix.pan_y.toFixed(2)}` +
      ` · record ${provenanceId}`;
    if (plane === state.focusedPlane) {
      syncWlInputs();
    }
  }

  function currentView() {
    const recordsByPlane = state.model ? state.model.recordsByPlane : {};
    const plane = pickViewPlane(state.focusedPlane, recordsByPlane);
    const records = recordsByPlane[plane] || [];
    const viewport = viewports.viewportFor(plane);
    const index = viewport
      ? viewport.getCurrentImageIdIndex()
      : state.model
        ? state.model.frameIndexByPlane[plane] || 0
        : 0;
    return { plane, frameIndex: pickFrameIndex({ [plane]: index }, plane, records.length) };
  }

  // ------------------------------------------------------- custom WL (v2)

  function currentPlaneRecord(plane) {
    const records = registry.planeRecords(state.sessionId, plane);
    const viewport = viewports.viewportFor(plane);
    const index = viewport ? viewport.getCurrentImageIdIndex() : 0;
    const record = records.find((item) => Number(item.frame_index) === index);
    return record || records[0] || null;
  }

  /** The frozen identity transport range over the 8-bit RGB channels. */
  function identityTransportRange() {
    return {
      lower: IDENTITY_DISPLAY_VOI.center - IDENTITY_DISPLAY_VOI.width / 2,
      upper: IDENTITY_DISPLAY_VOI.center + IDENTITY_DISPLAY_VOI.width / 2,
    };
  }

  /**
   * Pin one actor's RGB transfer function to the identity transport ramp.
   *
   * The served PNG bytes are ALREADY windowed by the frozen server-side
   * VIEWER_BINARY32_WL_V1 transfer; the admitted WindowLevel tool supplies
   * drag arithmetic only.  A fixed linear ramp over [0, 255] makes every
   * viewport VOI range map the 8-bit transport bytes 1:1 to screen, so no
   * client-side LUT is ever applied over the served PNGs (spec Part 4.2).
   *
   * Cornerstone's setVOI copies the viewport's WL interaction range onto
   * this transfer function, so the identity range is re-asserted on EVERY
   * call.  VOI_MODIFIED dispatches synchronously from inside setVOI, i.e.
   * before the tool's render(), which keeps the transport LUT at [0, 255]
   * for every rendered frame.
   */
  function pinIdentityTransferFunction(viewport) {
    if (!viewport || typeof viewport.getDefaultActor !== "function") {
      return;
    }
    const entry = viewport.getDefaultActor();
    const actor = entry && entry.actor ? entry.actor : null;
    if (!actor || typeof actor.getProperty !== "function") {
      return;
    }
    const property = actor.getProperty();
    if (!property || typeof property.setRGBTransferFunction !== "function") {
      return;
    }
    const range = identityTransportRange();
    let identity = identityTransferFunctions.get(actor);
    if (!identity) {
      // The frozen identity transport range: [0, 255] over 8-bit channels.
      identity = createLinearRGBTransferFunction({
        lower: range.lower,
        upper: range.upper,
      });
      identityTransferFunctions.set(actor, identity);
    }
    identity.setRange(range.lower, range.upper);
    if (property.getRGBTransferFunction(0) !== identity) {
      property.setUseLookupTableScalarRange(true);
      property.setRGBTransferFunction(0, identity);
    }
  }

  /**
   * One VOI_MODIFIED step: keep the transport LUT neutral, then translate
   * the tool's interaction-scratch change into a pending server submit.
   *
   * The viewport's VOI range is the admitted WindowLevel tool's arithmetic
   * scratch (WL interaction state); the actor's transfer function is the
   * transport LUT and must never render anything but the [0, 255] identity
   * ramp.  Cornerstone couples them inside setVOI; re-pinning here runs
   * synchronously before the tool's render().
   */
  function onVoiModified(plane) {
    pinIdentityTransferFunction(viewports.viewportFor(plane));
    const drag = state.wlDrag;
    if (!drag || drag.plane !== plane || state.applyingProgrammatic) {
      return;
    }
    const viewport = viewports.viewportFor(plane);
    const properties =
      viewport && typeof viewport.getProperties === "function"
        ? viewport.getProperties()
        : null;
    const range = properties ? properties.voiRange : null;
    if (!range) {
      return;
    }
    // target = applied window + (interaction scratch - scratch at drag start)
    const center =
      drag.base.center + ((range.lower + range.upper) / 2 - drag.baseRange.center);
    const width = drag.base.width + (range.upper - range.lower - drag.baseRange.width);
    if (!Number.isFinite(center) || !Number.isFinite(width) || !(width > 0)) {
      return;
    }
    drag.pending = { center, width };
    scheduleWlSubmit(plane);
  }

  function scheduleWlSubmit(plane) {
    const existing = state.wlSubmitTimers.get(plane);
    if (existing) {
      clearTimeout(existing);
    }
    // Trailing-edge debounce: a continuous drag posts at most once per
    // quiet period, so the session endpoint can never be flooded.
    state.wlSubmitTimers.set(
      plane,
      setTimeout(() => {
        state.wlSubmitTimers.delete(plane);
        const drag = state.wlDrag;
        if (!drag || drag.plane !== plane || !drag.pending) {
          return;
        }
        const pending = drag.pending;
        drag.pending = null;
        void submitWlDrag(plane, pending);
      }, WL_SUBMIT_DEBOUNCE_MS)
    );
  }

  function flushWlSubmits() {
    const drag = state.wlDrag;
    state.wlDrag = null;
    const planes = [...state.wlSubmitTimers.keys()];
    for (const plane of planes) {
      const timer = state.wlSubmitTimers.get(plane);
      if (timer) {
        clearTimeout(timer);
      }
      state.wlSubmitTimers.delete(plane);
    }
    if (drag && drag.pending) {
      void submitWlDrag(drag.plane, drag.pending);
    }
  }

  /**
   * Submit one pending drag window/level for a plane (spec Part 4.2).
   *
   * The numerics are validated by the closed envelope BEFORE the network
   * (never clamped); the server re-renders and the fresh LAYER-R records
   * are reloaded, so the displayed bytes are always server-rendered.
   */
  async function submitWlDrag(plane, pending) {
    if (!state.model || !pending) {
      return;
    }
    if (
      !Number.isFinite(pending.center) ||
      !Number.isFinite(pending.width) ||
      !(pending.width > 0)
    ) {
      return;
    }
    const record = currentPlaneRecord(plane);
    if (
      record &&
      record.wl_applied &&
      record.wl_applied.center === pending.center &&
      record.wl_applied.width === pending.width
    ) {
      // The drag returned to the applied values: nothing to submit.
      return;
    }
    await applyCustomWl(plane, pending.center, pending.width);
  }

  function beginWlDrag(plane) {
    if (!state.model || state.applyingProgrammatic) {
      return;
    }
    const record = currentPlaneRecord(plane);
    const viewport = viewports.viewportFor(plane);
    if (
      !record ||
      !record.wl_applied ||
      !viewport ||
      typeof viewport.getProperties !== "function"
    ) {
      return;
    }
    const properties = viewport.getProperties();
    const range = properties ? properties.voiRange : null;
    // Capture the interaction baseline: the applied window plus the range
    // the tool's scratch currently holds.  The tool adds its per-event
    // deltas to the scratch; onVoiModified maps them back onto the applied
    // window.  The transport LUT is not part of this state.
    state.wlDrag = {
      plane,
      base: { center: record.wl_applied.center, width: record.wl_applied.width },
      baseRange: range
        ? {
            center: (range.lower + range.upper) / 2,
            width: range.upper - range.lower,
          }
        : { center: IDENTITY_DISPLAY_VOI.center, width: IDENTITY_DISPLAY_VOI.width },
      pending: null,
    };
    pinIdentityTransferFunction(viewport);
  }

  // ------------------------------------------------------------ action path

  function postAction(action, params, options = {}) {
    const binding = currentBinding();
    const run = state.actionQueue.then(async () => {
      if (!bindingIsCurrent(binding)) {
        return null;
      }
      const view = options.view || currentView();
      const envelope = buildActionEnvelope({
        action,
        params,
        clientNonce: options.nonce || nextNonce(action),
        sessionId: state.sessionId,
        parentActionNonce: state.parentActionNonce,
        plane: view.plane,
        frameIndex: view.frameIndex,
      });
      let document;
      try {
        document = await source.postAction(
          state.sessionId,
          envelope,
          canonicalJson(envelope)
        );
      } catch (error) {
        if (bindingIsCurrent(binding)) {
          setStatus(errorText(error));
        }
        throw error;
      }
      if (!bindingIsCurrent(binding)) {
        return null;
      }
      const model = displayStateFrom(document);
      state.parentActionNonce = model.lastAcceptedNonce;
      await applyModel(model, { restore: false });
      return document;
    });
    // Keep the queue alive on refusal; callers still observe the rejection.
    state.actionQueue = run.then(
      () => undefined,
      () => undefined
    );
    return run;
  }

  async function actionAndStatus(action, params, options) {
    try {
      return await postAction(action, params, options);
    } catch (error) {
      return null;
    }
  }

  async function loadPlaneRecords(sessionId, plane) {
    const document = await source.getFrames(sessionId, plane);
    registry.setPlaneRecords(sessionId, plane, document.frames || []);
    return document.frames || [];
  }

  async function applyModel(model, options = {}) {
    const binding = currentBinding();
    if (!bindingIsCurrent(binding)) {
      return;
    }
    state.model = model;
    for (const plane of PLANES) {
      const records = model.recordsByPlane[plane] || [];
      registry.setPlaneRecords(model.sessionId, plane, records);
      const imageIds = registry.imageIds(model.sessionId, plane);
      const viewport = viewports.viewportFor(plane);
      if (!viewport) {
        continue;
      }
      const desired = pickFrameIndex(model.frameIndexByPlane, plane, records.length);
      const signature = imageIds.join("|");
      state.applyingProgrammatic = true;
      try {
        if (state.signatures[plane] !== signature) {
          state.signatures[plane] = signature;
          await viewports.setStack(plane, imageIds, desired);
        } else if (viewport.getCurrentImageIdIndex() !== desired) {
          await viewports.setIndex(plane, desired);
        }
        if (options.restore) {
          viewports.applyMatrix(plane, model.viewportByPlane[plane], elements[plane].host);
        }
        // Keep the actor's transport LUT neutral after every stack/property
        // change; the WL interaction state lives in state.wlDrag, never in
        // the transfer function.
        pinIdentityTransferFunction(viewport);
      } catch (error) {
        if (bindingIsCurrent(binding)) {
          setStatus(errorText(error));
        }
      } finally {
        state.applyingProgrammatic = false;
      }
      updateCaption(plane);
      redrawOverlay(plane);
    }
    state.focusedPlane = pickFocusedPlane(model, { stickyPlane: state.stickyPlane });
    updateFocusClasses();
    updateSeedControls();
  }

  // ------------------------------------------------------------- selection

  async function openSession(sessionId) {
    state.generation += 1;
    state.sessionId = sessionId;
    state.parentActionNonce = null;
    // A new session owns its own QC presentation: hide and reset the prior
    // session's panel before any of its state is cleared or loaded.
    hideQcPanel();
    state.model = null;
    state.overlayDocument = null;
    state.overlayError = null;
    state.overlayMode = OVERLAY_DEFAULT_MODE;
    state.selectedVessel = null;
    state.vesselAssignmentSource = null;
    state.cprDocument = null;
    state.cprError = null;
    state.schematicDocument = null;
    state.schematicError = null;
    // TREE_SCHEMATIC_NAV F3: clearing the two state words is NOT enough.  The
    // schematic lives in the DOM, and the DOM survives the session switch: the
    // predecessor session's paths, caption, badges and click/keydown listeners
    // would stay visible AND clickable, so clicking one would submit a
    // SELECT_VESSEL for a vessel of session A against session B.  Disarm the
    // rendered panel synchronously here, before the first await below, so no
    // clickable predecessor geometry can survive a session switch.
    disarmSchematicPanel();
    // MPR_SYNC_CURSOR: the cursor is ephemeral — a new session starts with
    // no pick, no document, and no error (nothing persists across sessions).
    // The cursor generation resets too (the session+generation binding
    // already invalidates prior-session flights; the reset keeps the
    // cursor-specific counter coherent from a known base).
    state.syncSource = null;
    state.syncDocument = null;
    state.syncError = null;
    state.syncSeq = 0;
    for (const timer of syncTimers.values()) {
      clearTimeout(timer);
    }
    syncTimers.clear();
    updateSyncControls();
    // A new session owns its own seed presentation: disarm, drop the pending
    // pick and the last accepted seed, and clear the entry fields.  F2
    // truthful-wait: the session switch is the guard-fires path (§3.2-3) —
    // stop the poll silently (no wait write; the newer session owns the
    // display) and drop the per-submission controller without aborting any
    // fetch that no longer belongs to this session's display.
    stopSeedWait();
    clearSeedWaitController();
    state.seedArmed = false;
    state.seedAnchor = null;
    state.seedDrag = null;
    state.seedCurrent = null;
    state.seedBinding = null;
    state.seedError = null;
    clearSeedInputs();
    updateSeedControls();
    updateOverlayToggle();
    updateCprPanel();
    state.signatures = {};
    state.stickyPlane = null;
    registry.clear();
    for (const timer of sliceTimers.values()) {
      clearTimeout(timer);
    }
    sliceTimers.clear();
    viewports.reset();
    for (const plane of PLANES) {
      if (elements[plane]) {
        elements[plane].host.removeAttribute("data-rendered");
        elements[plane].caption.textContent = `${plane}: loading…`;
      }
    }
    // Q3-SEED-PROV F1-race (R4): the initialization's own navigation tag —
    // captured here after the synchronous session switch but before the
    // first await (same pre-await idiom as selectVessel's navigation tag).  A
    // newer vessel navigation (or session switch) started while
    // initialization is in flight supersedes it, so the async selection
    // effects below (served-state assignment, default SELECT_VESSEL RCA
    // response) use this tag's sequence guard instead of the bare
    // session/generation check.
    const binding = currentBinding();
    doc.body.dataset.clientState = "loading";
    setStatus("Loading session…");
    try {
      const stateDocument = await source.getState(sessionId);
      if (!bindingIsCurrent(binding)) {
        return;
      }
      const records = await Promise.all(
        PLANES.map((plane) => loadPlaneRecords(sessionId, plane))
      );
      if (!bindingIsCurrent(binding)) {
        return;
      }
      const model = displayStateFrom(stateDocument);
      PLANES.forEach((plane, index) => {
        model.recordsByPlane[plane] = records[index];
      });
      // Q3-SEED-PROV F1-race (R5): essential viewport initialization ALWAYS
      // runs, even when this initialization is superseded by a newer vessel
      // navigation started while it was in flight (the R5 P2 regression: the
      // R4 guard returned from ALL initialization here, skipping
      // ensureViewports()/applyModel(), so a mid-init LAD click left the page
      // `loading` + LINEAGE_GAP + 0 viewports vs the R4 baseline `ready` + 3
      // viewports).  The sequence guard below scopes to the SELECTION
      // mutations only (served-state assignment + default SELECT_VESSEL RCA):
      // viewports/model always render from THIS init's own fetched documents,
      // while a superseded init never rebinds the physician's selection or
      // posts a stale default selection.
      state.parentActionNonce = model.lastAcceptedNonce;
      ensureViewports();
      await applyModel(model, { restore: true });
      if (!bindingIsCurrent(binding)) {
        return;
      }
      // CORONARY_MULTI_VESSEL_NAV: the served state owns the selection —
      // but only while THIS initialization is still the current navigation
      // (R4: a newer vessel navigation started while initialization was in
      // flight supersedes it — applying the served/stale selection here
      // would undo the physician's navigation; same sequence idiom as the
      // panel loads, which receive this same binding below).
      if (panelLoadIsCurrent(binding)) {
        const servedVessel = stateDocument.selected_vessel;
        state.selectedVessel =
          typeof servedVessel === "string" && servedVessel.startsWith("VESSEL:")
            ? servedVessel
            : null;
      }
      const select = $("session-select");
      if (select) {
        select.value = sessionId;
      }
      if (model.qcPanelOpen) {
        await loadQcPanel(binding);
      }
      await loadOverlay(binding);
      await loadCpr(binding);
      await loadSchematic(binding);
      // CORONARY_MULTI_VESSEL_NAV: when the session carries several ACTIVE
      // paths, the unfiltered read is AMBIGUOUS by design — persist the RCA
      // path (the #28 legacy path) via the closed SELECT_VESSEL grammar so
      // the physician lands on a usable review surface instead of a refusal.
      // Single-path sessions never take this branch (their unfiltered read
      // succeeds).  The persisted selection also keeps follow-up actions'
      // lineage chainable (parent_action_nonce).
      if (
        panelLoadIsCurrent(binding) &&
        state.overlayError &&
        state.overlayError.code === "CENTERLINE_OVERLAY_AMBIGUOUS" &&
        !state.selectedVessel
      ) {
        // The action-queue view snapshot predates the plane records here;
        // post the closed envelope directly (same seam, explicit frame 0).
        // NOTE: the HTTP body MUST be the canonical JSON of the envelope
        // (server compares body bytes to the canonical form; see the
        // postAction envelope/canonicalBody split in frameSource).
        const envelope = buildActionEnvelope({
          action: "SELECT_VESSEL",
          params: { vessel_label: "VESSEL:RCA" },
          clientNonce: nextNonce("SELECT_VESSEL"),
          sessionId: state.sessionId,
          parentActionNonce: state.parentActionNonce,
          plane: PLANES[0],
          frameIndex: 0,
        });
        let defaulted = null;
        try {
          defaulted = await source.postAction(
            state.sessionId,
            envelope,
            canonicalJson(envelope)
          );
        } catch (error) {
          if (bindingIsCurrent(binding)) {
            setStatus(errorText(error));
          }
        }
        if (defaulted && bindingIsCurrent(binding)) {
          const model = displayStateFrom(defaulted);
          state.parentActionNonce = model.lastAcceptedNonce;
        }
        // Q3-SEED-PROV F1-race (R4): the default-selection response is an
        // async initialization response like the panel loads — navigation-guard
        // it BEFORE it mutates state.  A newer vessel navigation (LAD clicked
        // while this response was held) supersedes this initialization, so the
        // RCA assignment below must not run: the R4 variant was exactly this
        // held default SELECT_VESSEL RCA response released after an LAD
        // navigation settled, rebinding an accepted LAD seed to RCA.  The
        // parent-nonce update above is lineage bookkeeping (same as the panel
        // loads' document updates); the SELECTION mutation is what the guard
        // protects.
        if (!panelLoadIsCurrent(binding)) {
          return;
        }
        if (defaulted) {
          state.selectedVessel = "VESSEL:RCA";
          await loadOverlay(binding);
          await loadCpr(binding);
          await loadSchematic(binding);
        } else if (!state.overlayError) {
          await loadOverlay(binding);
          await loadCpr(binding);
          await loadSchematic(binding);
        }
      }
      if (!bindingIsCurrent(binding)) {
        return;
      }
      doc.body.dataset.clientState = "ready";
      setStatus("");
    } catch (error) {
      if (!bindingIsCurrent(binding)) {
        return;
      }
      // TREE_SCHEMATIC_NAV F3: a failed initialization must leave the SAME
      // cleared state a successful switch leaves.  Otherwise session A's
      // schematic stays on screen under session B's error state, still
      // clickable, for the whole time B fails to load.
      disarmSchematicPanel();
      doc.body.dataset.clientState = "error";
      setStatus(errorText(error));
    }
  }

  function ensureViewports() {
    for (const plane of PLANES) {
      const info = elements[plane];
      if (!info) {
        continue;
      }
      viewports.ensureViewport(plane, info.host);
      ensureCctaToolGroup({
        viewportId: viewports.viewportFor(plane).id,
        renderingEngineId: viewports.renderingEngineId,
      });
    }
    attachViewportEvents();
  }

  function attachViewportEvents() {
    for (const plane of PLANES) {
      const info = elements[plane];
      if (!info || info.attached) {
        continue;
      }
      info.attached = true;
      info.host.addEventListener("mousedown", (event) => {
        state.dragPlane = plane;
        if (event.button === 0) {
          if (state.seedArmed) {
            beginSeedPick(plane, event);
          } else {
            beginWlDrag(plane);
          }
        }
        void focusPlane(plane);
      });
      info.host.addEventListener("mousemove", (event) => {
        if (state.seedArmed) {
          updateSeedDrag(plane, event);
        }
      });
      info.host.addEventListener(Enums.Events.VOI_MODIFIED, () => {
        onVoiModified(plane);
      });
      info.host.addEventListener(Enums.Events.STACK_NEW_IMAGE, (event) => {
        onStackNewImage(plane, event);
      });
      // MPR_SYNC_CURSOR: double-click places (reselects) the ephemeral
      // crosshair pick on the displayed frame; the server derives the shared
      // world point and the per-plane lines (no client-side geometry).
      info.host.addEventListener("dblclick", (event) => {
        void placeSyncCursor(plane, event);
      });
      info.host.addEventListener(Enums.Events.IMAGE_RENDERED, () => {
        info.host.dataset.rendered = "true";
        redrawOverlay(plane);
      });
    }
    if (!state.documentListenersAttached) {
      state.documentListenersAttached = true;
      doc.addEventListener("mouseup", () => {
        flushWlSubmits();
        void persistViewportMatrix();
      });
    }
  }

  async function focusPlane(plane) {
    const wasFocused = state.focusedPlane === plane;
    state.focusedPlane = plane;
    state.stickyPlane = plane;
    updateFocusClasses();
    if (wasFocused || !state.model || state.model.currentPlane === plane) {
      return;
    }
    await actionAndStatus("SET_PLANE", { plane });
    // MPR_SYNC_CURSOR: the focused plane changed — re-derive from the stored
    // pick so the crosshair refreshes on plane change (no-op without a pick).
    await refreshSyncCursor();
  }

  function onStackNewImage(plane, event) {
    updateCaption(plane);
    redrawOverlay(plane);
    // MPR_SYNC_CURSOR: the displayed image changed — redraw from the cached
    // document now and re-derive (debounced) from the stored pick, so the
    // crosshair refreshes on scroll without any client-side geometry.
    if (state.syncSource) {
      const pendingSync = syncTimers.get(plane);
      if (pendingSync) {
        clearTimeout(pendingSync);
      }
      syncTimers.set(
        plane,
        setTimeout(() => {
          syncTimers.delete(plane);
          void refreshSyncCursor();
        }, 200)
      );
    }
    if (state.applyingProgrammatic || !state.model) {
      return;
    }
    const index = Number(event.detail.imageIdIndex);
    if (!Number.isInteger(index) || index < 0) {
      return;
    }
    if (state.model.frameIndexByPlane[plane] === index) {
      return;
    }
    // Debounce a wheel burst into one SET_SLICE; keep the frozen view bound
    // to the exact plane/frame being displayed.
    const existing = sliceTimers.get(plane);
    if (existing) {
      clearTimeout(existing);
    }
    sliceTimers.set(
      plane,
      setTimeout(() => {
        sliceTimers.delete(plane);
        const records = registry.planeRecords(state.sessionId, plane);
        if (!Number.isInteger(index) || index >= records.length) {
          return;
        }
        void actionAndStatus(
          "SET_SLICE",
          { plane, frame_index: index },
          { view: { plane, frameIndex: index } }
        );
      }, 120)
    );
  }

  async function persistViewportMatrix() {
    const plane = state.dragPlane;
    state.dragPlane = null;
    if (!plane || !state.model || !elements[plane]) {
      return;
    }
    const raw = viewports.readMatrix(plane, elements[plane].host);
    const previous = state.model.viewportByPlane[plane];
    const changed =
      !previous ||
      Math.abs(previous.zoom - raw.zoom) > 1e-6 ||
      Math.abs(previous.pan_x - raw.pan_x) > 1e-6 ||
      Math.abs(previous.pan_y - raw.pan_y) > 1e-6;
    if (!changed) {
      return;
    }
    let params;
    try {
      // Validated BEFORE the queue: an out-of-range camera is surfaced and
      // never clamped into a persisted state different from the display.
      params = assertMatrix(raw);
    } catch (error) {
      setStatus(
        `${errorText(error)} — display not persisted (the server never clamps)`
      );
      return;
    }
    await actionAndStatus("SET_VIEWPORT_MATRIX", params);
  }

  async function applyPreset(presetId) {
    const view = currentView();
    const document = await actionAndStatus("SET_WL_PRESET", { preset_id: presetId });
    if (document) {
      await refreshRecords(PLANES.includes(view.plane) ? [view.plane] : [], false);
    }
  }

  async function clearWl() {
    const view = currentView();
    const document = await actionAndStatus("CLEAR_WL", {});
    if (document) {
      await refreshRecords(PLANES.includes(view.plane) ? [view.plane] : [], false);
    }
  }

  /**
   * Submit one v2 custom window/level for a plane (spec Part 4.2).
   *
   * The numerics are validated by the closed envelope BEFORE the network
   * (never clamped); the server re-renders and the fresh LAYER-R records are
   * reloaded, so the displayed bytes are always server-rendered.
   */
  async function applyCustomWl(plane, center, width) {
    const records = registry.planeRecords(state.sessionId, plane);
    const viewport = viewports.viewportFor(plane);
    const index = viewport ? viewport.getCurrentImageIdIndex() : 0;
    const frameIndex = pickFrameIndex({ [plane]: index }, plane, records.length);
    try {
      const document = await postAction(
        "SET_WL_CUSTOM",
        { center, width },
        { view: { plane, frameIndex } }
      );
      if (document) {
        await refreshRecords([plane], false);
      }
      return document;
    } catch (error) {
      setStatus(errorText(error));
      return null;
    }
  }

  async function applyCustomWlFromInputs() {
    const centerInput = $("wl-center-input");
    const widthInput = $("wl-width-input");
    if (!centerInput || !widthInput) {
      return;
    }
    const view = currentView();
    await applyCustomWl(
      view.plane,
      centerInput.valueAsNumber,
      widthInput.valueAsNumber
    );
  }

  function syncWlInputs() {
    const centerInput = $("wl-center-input");
    const widthInput = $("wl-width-input");
    if (!centerInput || !widthInput || !state.sessionId) {
      return;
    }
    const record = currentPlaneRecord(state.focusedPlane);
    if (!record || !record.wl_applied) {
      return;
    }
    const applied = record.wl_applied;
    if (doc.activeElement !== centerInput) {
      centerInput.value = String(applied.center);
    }
    if (doc.activeElement !== widthInput) {
      widthInput.value = String(applied.width);
    }
  }

  async function refreshRecords(planes, restore) {
    const binding = currentBinding();
    try {
      const records = await Promise.all(
        planes.map((plane) => loadPlaneRecords(state.sessionId, plane))
      );
      if (!bindingIsCurrent(binding) || !state.model) {
        return;
      }
      planes.forEach((plane, index) => {
        state.model.recordsByPlane[plane] = records[index];
      });
      await applyModel(state.model, { restore });
    } catch (error) {
      if (bindingIsCurrent(binding)) {
        setStatus(errorText(error));
      }
    }
  }

  async function refreshSession() {
    if (!state.sessionId) {
      return;
    }
    const binding = currentBinding();
    try {
      const stateDocument = await source.getState(state.sessionId);
      if (!bindingIsCurrent(binding)) {
        return;
      }
      const model = displayStateFrom(stateDocument);
      const records = await Promise.all(
        PLANES.map((plane) => loadPlaneRecords(state.sessionId, plane))
      );
      if (!bindingIsCurrent(binding)) {
        return;
      }
      PLANES.forEach((plane, index) => {
        model.recordsByPlane[plane] = records[index];
      });
      state.parentActionNonce = model.lastAcceptedNonce;
      await applyModel(model, { restore: true });
      if (model.qcPanelOpen) {
        await loadQcPanel(binding);
      }
      // Decision C: ACCEPT/EDIT/REJECT refresh all panels coherently.
      await loadOverlay(binding);
      await loadCpr(binding);
      await loadSchematic(binding);
      // The served centerline binding may have changed (e.g. a reseed
      // successor): re-render the accepted-seed provenance coherently.
      updateSeedControls();
      // MPR_SYNC_CURSOR: re-derive the ephemeral cursor against the fresh
      // served frames (no-op without a stored pick).
      await refreshSyncCursor();
      setStatus("");
    } catch (error) {
      if (bindingIsCurrent(binding)) {
        setStatus(errorText(error));
      }
    }
  }

  // -------------------------------------------------------------- QC seam

  function hideQcPanel() {
    const panel = $("qc-panel");
    if (panel) {
      panel.hidden = true;
    }
    state.qcDocument = null;
  }

  async function openQcPanel() {
    if (!state.sessionId) {
      return;
    }
    const binding = currentBinding();
    try {
      const result = await qcController.open();
      // The POST -> GET -> render chain spans awaits: if the physician
      // switched sessions (or the session was reopened) while it was in
      // flight, the document belongs to the ORIGINAL session/generation and
      // must be discarded before it can touch state or the DOM.
      if (!bindingIsCurrent(binding) || !result.state) {
        return;
      }
      state.qcDocument = result.document;
      renderQcPanel(result.document);
    } catch (error) {
      if (bindingIsCurrent(binding)) {
        setStatus(errorText(error));
      }
    }
  }

  async function loadQcPanel(binding) {
    try {
      const document = await source.getQc(state.sessionId);
      if (!bindingIsCurrent(binding)) {
        return;
      }
      state.qcDocument = document;
      renderQcPanel(document);
    } catch (error) {
      if (bindingIsCurrent(binding)) {
        setStatus(errorText(error));
      }
    }
  }

  function renderQcPanel(document) {
    const panel = $("qc-panel");
    if (!panel) {
      return;
    }
    panel.hidden = false;
    const outcomeList = $("qc-outcome");
    outcomeList.replaceChildren();
    const outcome = document.outcome || {};
    for (const [label, value] of [
      ["gate state", document.qc_gate_state_ref],
      ["decision", outcome.decision],
      ["reason", outcome.reason_code],
      ["gate id", outcome.gate_id],
      ["bound at", outcome.bound_at],
      ["superseded by", outcome.superseded_by],
      ["consumed refs", (outcome.consumed_refs || []).join(", ")],
      ["stale frames", (document.stale_frame_ids || []).length],
    ]) {
      const dt = doc.createElement("dt");
      dt.textContent = label;
      const dd = doc.createElement("dd");
      dd.textContent = value === null || value === undefined ? "—" : String(value);
      outcomeList.appendChild(dt);
      outcomeList.appendChild(dd);
    }
    const ledger = $("qc-ledger");
    ledger.replaceChildren();
    for (const entry of document.outcome_ledger || []) {
      const item = doc.createElement("li");
      item.textContent = `${entry.gate_id} · ${entry.decision}${
        entry.superseded_by ? ` → ${entry.superseded_by}` : ""
      }`;
      ledger.appendChild(item);
    }
    const capabilities = correctionCapabilities(document);
    const gradeSelect = $("correction-grade");
    gradeSelect.replaceChildren();
    for (const grade of QUALITY_GRADES) {
      const option = doc.createElement("option");
      option.value = grade;
      option.textContent = grade;
      option.disabled = !capabilities.actionable.includes(grade);
      gradeSelect.appendChild(option);
    }
    const firstActionable = capabilities.actionable.find((grade) =>
      QUALITY_GRADES.includes(grade)
    );
    gradeSelect.value = firstActionable || "";
    let studyRef = null;
    let studyRefError = null;
    try {
      studyRef = deriveStudyRef(document);
    } catch (error) {
      studyRefError = errorText(error);
    }
    const note = $("correction-note");
    const notes = [];
    if (studyRef) {
      notes.push(`bound subject: ${studyRef}`);
    } else {
      notes.push(studyRefError || "study reference unavailable — correction disabled");
    }
    notes.push(`actionable: ${capabilities.actionable.join(", ") || "none"}`);
    if (capabilities.deferred.length) {
      notes.push(
        `deferred: ${capabilities.deferred.join(", ")} (${capabilities.deferredReason || "no reason disclosed"})`
      );
    }
    note.textContent = notes.join(" · ");
    updateCorrectionControls();
  }

  function updateCorrectionControls() {
    const fieldset = $("qc-correction");
    const submit = $("submit-correction");
    const gradeSelect = $("correction-grade");
    if (!fieldset || !submit || !gradeSelect || !state.qcDocument) {
      return;
    }
    const capabilities = correctionCapabilities(state.qcDocument);
    let studyRefOk = false;
    try {
      deriveStudyRef(state.qcDocument);
      studyRefOk = true;
    } catch {
      studyRefOk = false;
    }
    const actionable = capabilities.actionable.includes(gradeSelect.value);
    fieldset.disabled = !(studyRefOk && capabilities.actionable.length);
    submit.disabled = !(studyRefOk && actionable);
  }

  async function submitCorrection() {
    if (!state.qcDocument) {
      return;
    }
    const grade = $("correction-grade").value;
    try {
      const document = await qcController.submitCorrection(state.qcDocument, grade);
      if (!document) {
        return;
      }
      setStatus(`Correction submitted: ${grade}`);
      await refreshRecords([...PLANES], false);
      await loadQcPanel(currentBinding());
    } catch (error) {
      setStatus(errorText(error));
    }
  }

  // ------------------------------------------------- coronary overlay

  /**
   * The one overlay-visibility toggle (closed enum-allowlist shape).
   *
   * The button is created dynamically so it can stay hidden while the session
   * carries no usable centerline; F8: an AMBIGUOUS/INVALID overlay read
   * refusal is surfaced through the sibling typed-failure element instead of
   * being silently swallowed.
   */
  function renderOverlayControls() {
    const toolbar = doc.querySelector(".toolbar");
    if (!toolbar || $(`overlay-toggle-${CORONARY_OVERLAY_ID}`)) {
      return;
    }
    const button = doc.createElement("button");
    button.type = "button";
    button.id = `overlay-toggle-${CORONARY_OVERLAY_ID}`;
    button.dataset.overlayId = CORONARY_OVERLAY_ID;
    button.textContent = "Path overlay";
    button.hidden = true;
    button.addEventListener("click", () => {
      void toggleOverlayVisibility();
    });
    const failure = doc.createElement("span");
    failure.id = `overlay-error-${CORONARY_OVERLAY_ID}`;
    failure.className = "overlay-error";
    failure.hidden = true;
    failure.setAttribute("role", "status");
    // Decision A: the explicitly labeled PROJECTION localizer toggle
    // (default OFF, visually distinct; never confusable with the normal
    // slice-local overlay).
    const modeButton = doc.createElement("button");
    modeButton.type = "button";
    modeButton.id = `overlay-mode-${CORONARY_OVERLAY_ID}`;
    modeButton.dataset.overlayId = CORONARY_OVERLAY_ID;
    modeButton.textContent = "Projection localizer: off";
    modeButton.hidden = true;
    modeButton.setAttribute("aria-pressed", "false");
    modeButton.addEventListener("click", () => {
      void toggleOverlayMode();
    });
    const caption = doc.createElement("span");
    caption.id = `overlay-caption-${CORONARY_OVERLAY_ID}`;
    caption.className = "overlay-caption";
    caption.hidden = true;
    caption.setAttribute("role", "status");
    // CORONARY_MULTI_VESSEL_NAV (Decision A): explicit vessel picker —
    // one button per frozen VESSEL:* token.  Selecting posts SELECT_VESSEL
    // and reloads both panels for the chosen path; the picker never infers
    // anatomy (labels come from the sealed assignment truth only).
    const picker = doc.createElement("span");
    picker.id = `vessel-picker-${CORONARY_OVERLAY_ID}`;
    picker.className = "vessel-picker";
    picker.hidden = true;
    picker.setAttribute("role", "group");
    picker.setAttribute("aria-label", "Vessel");
    for (const token of ["VESSEL:LAD", "VESSEL:LCX", "VESSEL:RCA"]) {
      const vesselButton = doc.createElement("button");
      vesselButton.type = "button";
      vesselButton.dataset.vessel = token;
      vesselButton.textContent = token.replace("VESSEL:", "");
      vesselButton.setAttribute("aria-pressed", "false");
      vesselButton.addEventListener("click", () => {
        void selectVessel(token);
      });
      picker.appendChild(vesselButton);
    }
    // CORONARY_MULTI_VESSEL_NAV (Decision B): assignment-source badge —
    // always rendered next to the picker, never bare anatomy.
    const badge = doc.createElement("span");
    badge.id = `vessel-source-${CORONARY_OVERLAY_ID}`;
    badge.className = "vessel-source";
    badge.hidden = true;
    badge.setAttribute("role", "status");
    toolbar.insertBefore(picker, $("open-qc") || null);
    toolbar.insertBefore(badge, $("open-qc") || null);
    toolbar.insertBefore(failure, $("open-qc") || null);
    toolbar.insertBefore(button, failure);
    toolbar.insertBefore(modeButton, failure);
    toolbar.insertBefore(caption, failure);
    renderCprPanelShell();
    renderSchematicPanelShell();
  }

  function updateOverlayToggle() {
    if (!$(`overlay-toggle-${CORONARY_OVERLAY_ID}`)) {
      renderOverlayControls();
    }
    const button = $(`overlay-toggle-${CORONARY_OVERLAY_ID}`);
    const failure = $(`overlay-error-${CORONARY_OVERLAY_ID}`);
    const modeButton = $(`overlay-mode-${CORONARY_OVERLAY_ID}`);
    const caption = $(`overlay-caption-${CORONARY_OVERLAY_ID}`);
    if (!button) {
      return;
    }
    const overlay = state.overlayDocument;
    const error = state.overlayError;
    if (failure) {
      failure.hidden = !error;
      failure.textContent = error ? `Path overlay: ${error.code}` : "";
      if (error) {
        failure.dataset.overlayError = error.code;
      } else {
        delete failure.dataset.overlayError;
      }
    }
    button.hidden = !overlay;
    if (!overlay) {
      button.setAttribute("aria-pressed", "false");
      button.textContent = "Path overlay";
      if (modeButton) {
        modeButton.hidden = true;
      }
      if (caption) {
        caption.hidden = true;
      }
      return;
    }
    button.setAttribute("aria-pressed", overlay.visible ? "true" : "false");
    button.textContent = `Path overlay: ${overlay.visible ? "on" : "off"}`;
    updateVesselPicker();
    // Decision A: PROJECTION is default-OFF and visually distinct.
    const isProjection = state.overlayMode === "PROJECTION";
    if (modeButton) {
      modeButton.hidden = false;
      modeButton.setAttribute("aria-pressed", isProjection ? "true" : "false");
      modeButton.textContent = `Projection localizer: ${isProjection ? "on" : "off"}`;
    }
    if (caption) {
      caption.hidden = !isProjection;
      caption.textContent = isProjection ? OVERLAY_PROJECTION_CAPTION : "";
      if (isProjection) {
        caption.dataset.projectionCaption = OVERLAY_PROJECTION_CAPTION;
      } else {
        delete caption.dataset.projectionCaption;
      }
    }
  }

  /** The LAYER-G frame index currently displayed by one plane's stack. */
  function overlayFrameIndex(plane) {
    const viewport = viewports.viewportFor(plane);
    const index = viewport ? viewport.getCurrentImageIdIndex() : 0;
    const records = registry.planeRecords(state.sessionId, plane);
    const record =
      records.find((item) => Number(item.frame_index) === index) || records[0] || null;
    return record ? Number(record.frame_index) : 0;
  }

  function redrawOverlay(plane) {
    const info = elements[plane];
    if (!info || !info.overlay) {
      return 0;
    }
    const drawn = drawCoronaryOverlay({
      canvas: info.overlay,
      viewport: viewports.viewportFor(plane),
      document: state.overlayDocument,
      plane,
      frameIndex: overlayFrameIndex(plane),
    });
    // The pending-seed marker shares the dedicated 2D layer and redraws in
    // the same pass so the two never desynchronize.
    drawSeedMarker(plane);
    // The ephemeral sync crosshair shares the same layer and pass (lines
    // only, served positions only); it redraws from the cached document on
    // every navigation event and re-derives on reselect/refresh.  FINDING 1:
    // the displayed frame binds the draw — a panel showing any frame other
    // than the served planes[plane].frame_index suppresses its lines (the
    // seed on-frame guard idiom) so the three crosshairs always represent
    // one world point; the visible reason lives in #sync-status (off-frame
    // note) alongside the per-canvas dataset.syncSuppressed flag.
    drawSyncCursor({
      canvas: info.overlay,
      viewport: viewports.viewportFor(plane),
      document: state.syncDocument,
      plane,
      frameIndex: overlayFrameIndex(plane),
    });
    updateSyncOffFrameNote();
    return drawn;
  }

  function redrawOverlays() {
    for (const plane of PLANES) {
      redrawOverlay(plane);
    }
  }

  /** Load (or clear) the overlay document for the bound session. */
  async function loadOverlay(binding) {
    const sessionId = state.sessionId;
    if (!sessionId) {
      return;
    }
    // F1-race (R3): a superseded navigation never fetches and never applies.
    // TREE_SCHEMATIC_NAV F2: nor does a load issued before the canonical
    // centerline set was replaced.
    if (!panelSetIsCurrent(binding)) {
      return;
    }
    // TSN_R5_F1 entry fence: an obsolete recovery/settlement epoch never claims an ordinal nor fetches.
    if ((binding.recoverySeq || 0) !== state.selectionRecoverySeq) {
      return;
    }
    // TREE_SCHEMATIC_NAV F2: this panel's own request ordinal, so a slower
    // earlier overlay read cannot overwrite a newer one.
    const request = (state.overlayReqSeq += 1);
    // TSN_R4_F1 (R10-F1 null-path): the REQUESTED vessel — the canonical
    // client selection this read is issued for — captured BEFORE the fetch.
    // An unfiltered read (null) may serve sole-path geometry for DISPLAY but
    // must never manufacture a selection from it.
    const requestedVessel =
      state.selectedVessel && typeof state.selectedVessel === "string"
        ? state.selectedVessel
        : null;
    let overlay = null;
    let failure = null;
    try {
      const needsOptions =
        (state.overlayMode && state.overlayMode !== OVERLAY_DEFAULT_MODE) ||
        requestedVessel !== null;
      const sourceWithMode = needsOptions
        ? {
            getCoronary: (id) =>
              source.getCoronary(id, {
                mode: state.overlayMode,
                vessel: requestedVessel,
              }),
          }
        : source;
      overlay = await loadCoronaryOverlay(sourceWithMode, sessionId);
    } catch (error) {
      // F8: a typed AMBIGUOUS/INVALID refusal is NOT the same as an ordinary
      // session without a centerline; it stays visible instead of silently
      // removing the overlay control.
      failure = {
        code: String((error && error.code) || "CENTERLINE_OVERLAY_ERROR"),
        message: errorText(error),
      };
    }
    // F1-race (R3): the response is applied only while the navigation that
    // started it is still the current one (the R3 stale-response variant).
    // TREE_SCHEMATIC_NAV F2: and only while it is still this panel's newest
    // request under the still-canonical ACTIVE centerline set.
    if (!panelSetIsCurrent(binding) || request !== state.overlayReqSeq || (binding.recoverySeq || 0) !== state.selectionRecoverySeq) {
      return;
    }
    state.overlayDocument = overlay;
    state.overlayError = failure;
    if (overlay && typeof overlay.vessel_label === "string") {
      // TSN_R4_F1 (R10-F1 null-path): adoption is provenance-gated — the
      // override-shaped document reports what was REQUESTED, so an
      // unfiltered (null-requested) geometry label is kept for display and
      // NEVER adopted as the canonical selection.  A filtered read adopts
      // its own label only when it still matches the requested token at
      // apply time (accepted-token and accepted/recovery reloads).
      if (
        requestedVessel !== null &&
        overlay.vessel_label === requestedVessel &&
        state.selectedVessel === requestedVessel &&
        (binding.recoverySeq || 0) === state.selectionRecoverySeq
      ) {
        state.selectedVessel = overlay.vessel_label;
      }
      /* else: keep document, never adopt its label */
    }
    if (failure) {
      setStatus(`${failure.code}: ${failure.message}`);
    }
    updateOverlayToggle();
    updateVesselPicker();
    redrawOverlays();
  }

  /** CORONARY_MULTI_VESSEL_NAV: post SELECT_VESSEL and reload all panels. */
  async function selectVessel(token) {
    // F1-race (R2): a vessel navigation invalidates any in-flight seed
    // submission's capture identity (``submissionIsCurrent``); bumped before
    // the optimistic selection and the queued SELECT_VESSEL action, so a
    // switch accepted while the /seed response is held is never invisible.
    state.vesselNavSeq += 1;
    // F1-race (R3): tag THIS navigation's panel loads with the sequence it
    // starts under, BEFORE the first await.  If a later navigation starts
    // while this one is in flight, the released response is older than the
    // current sequence and ``panelLoadIsCurrent`` drops it (and its follow-up
    // CPR/schematic loads) instead of mutating selection/documents.
    const navigation = currentBinding();
    // TREE_SCHEMATIC_NAV F1: the DISPLAYED selection only ever moves through
    // an ACCEPTED action.  ``token`` is applied locally so the click is
    // immediate and so the Q3 navigation tag above is still bumped at click
    // time, but the prior coherent selection stays the authority until the
    // server accepts.  On refusal the display is re-derived from the server.
    state.pendingSelection = { token, binding: navigation };
    updateVesselPicker();
    const accepted = await actionAndStatus("SELECT_VESSEL", { vessel_label: token });
    if (state.pendingSelection && state.pendingSelection.binding === navigation) {
      state.pendingSelection = null;
    }
    if (!accepted) {
      // Refused (F1): reconcile EVERY panel against the accepted selection.
      // Deliberately NO panel load is issued for the refused token - that is
      // exactly what made the panels disagree: the overlay and CPR requested
      // and rendered geometry for a selection the server had rejected while
      // the schematic showed the accepted vessel, and nothing reconciled them
      // afterwards.
      //
      // TREE_SCHEMATIC_NAV F1 (the concurrent form round 3 closed): a refusal
      // speaks for THIS navigation and for no other.  If a later navigation is
      // already the current one, this refusal must not touch the display at
      // all: an action accepted since this one was issued has superseded it,
      // so a recovery here would re-derive the panels for a navigation the
      // physician has already moved on from, and its authority read would
      // race that newer navigation's own recovery.  The later navigation's
      // own continuation owns the display; drop through.
      if (!panelLoadIsCurrent(navigation)) {
        return;
      }
      // TREE_SCHEMATIC_NAV F1 (R3-F1): the rollback authority is the server's
      // own STORED selection, and the ONLY served document that can carry it
      // is the tree schematic.  ``apps/viewer/server.py:584`` serves that
      // route as ``engine.tree_schematic_document()`` with NO argument, and
      // ``apps/viewer/engine.py:3976`` fills the served ``selected_vessel``
      // from ``state.selected_vessel`` unconditionally - so that key cannot be
      // anything but the stored selection.
      //
      // The overlay and CPR documents cannot serve that role.  Their routes
      // read an explicit ``vessel`` request parameter
      // (``apps/viewer/server.py:488``, ``:525``, ``:556``) and the engine
      // lets an explicit label OVERRIDE the stored one
      // (``apps/viewer/engine.py:3931`` and the CPR pair at ``:4041`` /
      // ``:4072``: ``state.selected_vessel if vessel_label is None else
      // vessel_label``).  This client hands the displayed selection to those
      // reads as exactly that parameter (``loadOverlay`` above, ``loadCpr``
      // below), so a served ``vessel_label`` reports what was REQUESTED - a
      // vessel that may have been refused and never accepted - and must never
      // be read back as the authority.  (The frozen state document's own
      // ``selected_vessel`` is equally authoritative, but the client holds no
      // cached copy of it, so using it would mean a new fetch and a new state
      // word; the schematic route already publishes the same value.)
      //
      // So the refused token leaves the display immediately, BEFORE any read
      // is issued: from here on this navigation cannot name it in a request,
      // and no later load can re-introduce it through an override-shaped
      // document.
      // TSN_R4_F1: refusal-recovery fence — bumped synchronously at
      // refusal-recovery entry before the first await, so pre-refusal
      // continuations fail their newest-ordinal checks at once and loads
      // whose binding.recoverySeq is older than state.selectionRecoverySeq
      // must never write state.selectedVessel nor apply documents.
      state.selectionRecoverySeq += 1;
      state.overlayReqSeq += 1; state.cprReqSeq += 1; state.schematicReqSeq += 1;
      state.selectedVessel = null;
      updateVesselPicker();
      const recovery = currentBinding();
      // The authority is fetched FIRST, because the two override-shaped reads
      // below take their request parameter out of ``state.selectedVessel``:
      // asked before the stored selection is known, they would ask for
      // whatever the client happens to hold.
      const authority = await loadSchematic(recovery);
      // Re-checked after the authority fetch: a newer navigation - or a
      // session switch, or a canonical-set replacement - owns the display now,
      // and this recovery must not take it back.
      if (!panelLoadIsCurrent(recovery)) {
        return;
      }
      // The displayed selection becomes the server's currently accepted
      // selection, or nothing at all when the authority could not be read: in
      // that case the two reads below are issued with no ``vessel`` parameter
      // and the server answers with the stored selection itself.
      state.selectedVessel =
        authority && typeof authority.selected_vessel === "string"
          ? authority.selected_vessel
          : null;
      updateVesselPicker();
      await loadOverlay(recovery);
      await loadCpr(recovery);
      // TSN_R4_F1: final reconcile — any same-navigation CPR pixels that
      // landed during recovery re-render under the authority selection.
      updateSchematicPanel();
      updateCprPanel();
      updateSeedControls();
      updateVesselPicker();
      return;
    }
    // TREE_SCHEMATIC_NAV F1: an ACCEPTED navigation loads ITS OWN token.  A
    // concurrent continuation may have left a different optimistic value
    // behind; without this the accepted navigation's overlay/CPR reads would
    // request whatever that continuation happened to leave, so the panels
    // would render one vessel while the server and the schematic showed the
    // accepted one.  Guarded by the same identity as the refusal branch: a
    // superseded navigation never writes the display either.
    if (!panelLoadIsCurrent(navigation)) {
      return;
    }
    const stored = await loadSchematic(navigation);
    if (!panelLoadIsCurrent(navigation)) {
      return;
    }
    // TSN_R5_F2 settlement epoch: acceptance settles under a fresh epoch so pre-acceptance refreshes (older recoverySeq) can never reclaim authority afterwards. Same synchronous-before-first-await idiom as refusal recovery; Q3/set guards untouched.
    state.selectionRecoverySeq += 1;
    state.overlayReqSeq += 1; state.cprReqSeq += 1; state.schematicReqSeq += 1;
    const settled = currentBinding();
    if (!panelLoadIsCurrent(settled)) {
      return;
    }
    const storedLabel = (stored && typeof stored.selected_vessel === 'string') ? stored.selected_vessel : null;
    if (storedLabel === null || storedLabel === token) {
      state.selectedVessel = token;
    } else {
      state.selectedVessel = storedLabel;
    }
    // settled carries the fresh epoch so the acceptance chain's own loads pass the entry fence while pre-settlement continuations bail.
    await loadOverlay(settled);
    await loadCpr(settled);
    await loadSchematic(settled);
    // The served centerline binding may have changed under an accepted seed:
    // re-render the provenance summary against the fresh documents.
    updateSeedControls();
    updateVesselPicker();
  }

  /** CORONARY_MULTI_VESSEL_NAV: picker pressed-state + Decision-B badge. */
  function updateVesselPicker() {
    const picker = $(`vessel-picker-${CORONARY_OVERLAY_ID}`);
    const badge = $(`vessel-source-${CORONARY_OVERLAY_ID}`);
    if (!picker) {
      return;
    }
    const overlay = state.overlayDocument;
    const label =
      (overlay && typeof overlay.vessel_label === "string" && overlay.vessel_label) ||
      (typeof state.selectedVessel === "string" && state.selectedVessel) ||
      "";
    picker.hidden = !overlay && !state.overlayError;
    for (const vesselButton of picker.querySelectorAll("button[data-vessel]")) {
      const active = vesselButton.dataset.vessel === label;
      vesselButton.setAttribute("aria-pressed", active ? "true" : "false");
      const pending = state.pendingSelection && state.pendingSelection.binding && panelLoadIsCurrent(state.pendingSelection.binding) && vesselButton.dataset.vessel === state.pendingSelection.token;
      if (pending) vesselButton.setAttribute('data-pending','true'); else vesselButton.removeAttribute('data-pending');
    }
    if (badge) {
      const source =
        typeof state.vesselAssignmentSource === "string" &&
        state.vesselAssignmentSource
          ? state.vesselAssignmentSource
          : "FIXTURE_DERIVED";
      badge.hidden = picker.hidden;
      badge.textContent = label ? `${label} · ${source}` : source;
      badge.dataset.vesselSource = source;
      if (label) {
        badge.dataset.vesselLabel = label;
      } else {
        delete badge.dataset.vesselLabel;
      }
    }
  }

  async function toggleOverlayVisibility() {
    const overlay = state.overlayDocument;
    if (!overlay) {
      return;
    }
    const desired = overlay.visible !== true;
    try {
      await postAction("SET_OVERLAY_VISIBILITY", {
        overlay_id: CORONARY_OVERLAY_ID,
        visible: desired,
        mode: state.overlayMode,
      });
    } catch (error) {
      setStatus(errorText(error));
      return;
    }
    await loadOverlay(currentBinding());
    await loadCpr(currentBinding());
    await loadSchematic(currentBinding());
    // A visibility toggle re-serves the overlay document: keep the
    // accepted-seed provenance bound to the fresh documents.
    updateSeedControls();
  }

  /** Decision A: flip the closed display mode (SLICE_LOCAL <-> PROJECTION). */
  async function toggleOverlayMode() {
    if (!state.overlayDocument) {
      return;
    }
    const next = state.overlayMode === "PROJECTION" ? "SLICE_LOCAL" : "PROJECTION";
    if (!OVERLAY_MODES.includes(next)) {
      return;
    }
    state.overlayMode = next;
    const visible = state.overlayDocument.visible === true;
    try {
      await postAction("SET_OVERLAY_VISIBILITY", {
        overlay_id: CORONARY_OVERLAY_ID,
        visible,
        mode: next,
      });
    } catch (error) {
      setStatus(errorText(error));
      state.overlayMode = next === "PROJECTION" ? "SLICE_LOCAL" : "PROJECTION";
      return;
    }
    await loadOverlay(currentBinding());
    await loadCpr(currentBinding());
    await loadSchematic(currentBinding());
    // A mode toggle re-serves the overlay document: keep the accepted-seed
    // provenance bound to the fresh documents.
    updateSeedControls();
  }

  // ---------------------------------------------------------- seed panel
  //
  // T2 physician/manual seed (Phase B): explicit voxel + direction entry with
  // canvas pick-assist, following the vessel-picker/canvas-dataset precedent
  // (dynamic toolbar group, aria-pressed toggles, dataset-carried state,
  // role=status typed failures).  Click sets the pending voxel, a drag
  // vector sets the pending direction (dx/dy; dz stays explicitly empty
  // until typed — a click alone never fabricates a direction), the pending
  // pick shows as a visible marker on the overlay canvas, reselect replaces
  // it, and submit posts the dedicated seed route.  Every value is visible
  // and editable; empty fields block submit (no hidden defaults); the server
  // re-validates bounds with typed refusals.
  //
  // Display-index note: the frozen client carries no patient geometry, so a
  // canvas pick maps through the viewport's own index-space transform (the
  // exact inverse of the overlay draw mapping) and pre-fills the EXPLICIT
  // fields — including the displayed frame index as pending k.  The physician
  // confirms the true volume indices before submit; the server's typed
  // bounds refusal is the authority, never the pick.

  /** The stroke of the display-only pending-seed marker. */
  const SEED_MARKER_STROKE = "#4ade80";

  // F2 truthful-wait (client-only observer, ADAPT of the primitives above):
  // the single ``state.actionQueue`` keeps EXACTLY today's semantics (the poll
  // below never enters/awaits/mutates/reorders it); the wait renders ONLY the
  // served ``seed_jobs`` word for THIS (session_id, client_nonce) identity,
  // verbatim in ``#seed-status`` text only.  No percentage, ratio, ETA,
  // duration, countdown or progress word is ever rendered; the words
  // COMPLETED, CANCELLED, STOPPED, FAILED, DONE and PARTIAL never appear in
  // any rendered line (COMPLETED is frozen-but-never-published).
  //
  // Closed wait vocabulary (verbatim, ``#seed-status`` text only):
  //   "Seed: QUEUED"                                  (served QUEUED)
  //   "Seed: RUNNING"                                 (served RUNNING)
  //   "Seed: REFUSED <CODE>" / "Seed: REFUSED"        (served REFUSED; code
  //     interpolated verbatim, never paraphrased/mapped; bare form only when
  //     the served refusal_code is absent/non-string — never invented)
  //   "Seed: waiting — state unreadable, job still server-owned"
  //     (poll-transport failure: getState rejected or a non-job document;
  //     transport error, never a job state; #seed-error stays hidden; the
  //     next tick proceeds unchanged)
  //   "Seed: wait stopped locally — job still server-owned"
  //     (waiter-detach: the client stopped waiting; the server still owns
  //     the job; #seed-error stays hidden, state.seedError is never set)
  // Silence (no state word written): before the first served word for this
  // identity (leave the pre-submit neutral text unchanged — claiming QUEUED
  // before the server observed it would be fabrication), after the execution
  // returned (stop the poll; the settled outcome surfaces speak), and on
  // supersession/session-switch (stop silently; the newer navigation owns
  // the display).
  //
  // Cadence: immediate first poll at seed-submit (t=0), then one
  // ``source.getState`` per 1000 ms as a serial ``setTimeout`` chain (each
  // iteration awaits the prior ``getState`` before scheduling the next;
  // never ``setInterval``, never overlapping/parallel polls for one
  // submission).  A failed poll renders the transport-error line and
  // schedules the next tick unchanged.  A bounded tick cap stops the chain
  // with the transport-error line shape, never with a fabricated state word.
  const SEED_WAIT_POLL_MS = 1000;
  const SEED_WAIT_MAX_TICKS = 600;
  const SEED_WAIT_TEXT_QUEUED = "Seed: QUEUED";
  const SEED_WAIT_TEXT_RUNNING = "Seed: RUNNING";
  const SEED_WAIT_TEXT_TRANSPORT_ERROR =
    "Seed: waiting — state unreadable, job still server-owned";
  const SEED_WAIT_TEXT_DETACHED =
    "Seed: wait stopped locally — job still server-owned";
  const seedWait = {
    controller: null,
    timer: null,
    active: null,
    ticks: 0,
  };

  /** The served ``seed_jobs`` entry for THIS (session_id, client_nonce) identity, or null. */
  function seedWaitEntryFor(document, sessionId, clientNonce) {
    const jobs =
      document && typeof document === "object" && Array.isArray(document.seed_jobs)
        ? document.seed_jobs
        : null;
    if (!jobs) {
      return null;
    }
    for (const entry of jobs) {
      if (
        entry &&
        typeof entry === "object" &&
        entry.session_id === sessionId &&
        entry.client_nonce === clientNonce
      ) {
        return entry;
      }
    }
    return null;
  }

  /** The verbatim ``#seed-status`` line for one served entry (closed set above). */
  function seedWaitLineFor(entry) {
    if (!entry || typeof entry !== "object") {
      return null;
    }
    if (entry.state === "QUEUED") {
      return SEED_WAIT_TEXT_QUEUED;
    }
    if (entry.state === "RUNNING") {
      return SEED_WAIT_TEXT_RUNNING;
    }
    if (entry.state === "REFUSED") {
      const code = entry.refusal_code;
      if (typeof code === "string" && code) {
        return `Seed: REFUSED ${code}`;
      }
      return "Seed: REFUSED";
    }
    return null;
  }

  /** Write one wait line to ``#seed-status`` text only (guarded; never a child node). */
  function writeSeedWaitLine(submission, binding, text) {
    if (!submissionIsCurrent(submission) || !bindingIsCurrent(binding)) {
      return false;
    }
    const status = $("seed-status");
    if (!status) {
      return false;
    }
    status.textContent = text;
    return true;
  }

  /** Stop the serial poll chain (clear the pending timeout; the loop checks identity).
   *
   * NOTE: stopping clears the poll identity + timer but NEVER the
   * per-submission AbortController: ``detachSeedWait`` must abort AFTER
   * stopping (else the held seed fetch never rejects), and the settle path
   * clears the controller explicitly once the fetch is no longer in flight.
   */
  function stopSeedWait() {
    if (seedWait.timer !== null) {
      clearTimeout(seedWait.timer);
      seedWait.timer = null;
    }
    seedWait.active = null;
    seedWait.ticks = 0;
    updateSeedWaitCancel();
  }

  /** Drop the per-submission AbortController once its fetch is no longer in flight. */
  function clearSeedWaitController() {
    seedWait.controller = null;
  }

  /** One serial poll tick: exactly one ``source.getState`` then (maybe) the next tick. */
  async function seedWaitTick(wait) {
    if (seedWait.active !== wait) {
      return;
    }
    if (!submissionIsCurrent(wait.submission) || !bindingIsCurrent(wait.binding)) {
      stopSeedWait();
      return;
    }
    wait.ticks += 1;
    seedWait.ticks = wait.ticks;
    let document = null;
    try {
      document = await source.getState(wait.sessionId);
    } catch {
      document = null;
    }
    if (seedWait.active !== wait) {
      return;
    }
    if (!submissionIsCurrent(wait.submission) || !bindingIsCurrent(wait.binding)) {
      stopSeedWait();
      return;
    }
    if (document === null) {
      writeSeedWaitLine(wait.submission, wait.binding, SEED_WAIT_TEXT_TRANSPORT_ERROR);
    } else {
      const entry = seedWaitEntryFor(document, wait.sessionId, wait.clientNonce);
      const line = seedWaitLineFor(entry);
      if (line !== null) {
        writeSeedWaitLine(wait.submission, wait.binding, line);
      } else if (typeof document !== "object" || document === null) {
        writeSeedWaitLine(wait.submission, wait.binding, SEED_WAIT_TEXT_TRANSPORT_ERROR);
      }
      // Silence otherwise: no entry for this identity yet (before the first
      // served word) or a non-job document — leave the display unchanged and
      // keep polling.  Silence is never a state word.
    }
    if (seedWait.active !== wait) {
      return;
    }
    if (wait.ticks >= SEED_WAIT_MAX_TICKS) {
      writeSeedWaitLine(wait.submission, wait.binding, SEED_WAIT_TEXT_TRANSPORT_ERROR);
      stopSeedWait();
      return;
    }
    seedWait.timer = setTimeout(() => {
      seedWait.timer = null;
      void seedWaitTick(wait);
    }, SEED_WAIT_POLL_MS);
  }

  /** Start the serial poll chain as a detached (non-awaited, non-queued) observer. */
  function startSeedWait({ submission, binding, sessionId, clientNonce }) {
    stopSeedWait();
    const wait = {
      submission,
      binding,
      sessionId,
      clientNonce,
      ticks: 0,
    };
    seedWait.active = wait;
    seedWait.ticks = 0;
    updateSeedWaitCancel();
    void seedWaitTick(wait);
  }

  /**
   * Cancel-as-waiter-detach (SEED POST fetch only): abort the in-flight seed
   * fetch, stop the poll chain, keep the queue alive, and render the detach
   * line.  Never ``showSeedError``, never ``state.seedError``.
   *
   * Ordering note: the per-submission controller is cleared by ``stopSeedWait``,
   * so capture it BEFORE stopping and abort AFTER — otherwise the abort never
   * fires and the held seed fetch never rejects (the T4 hang).
   */
  function detachSeedWait() {
    const wait = seedWait.active;
    if (!wait) {
      return false;
    }
    const submission = wait.submission;
    const binding = wait.binding;
    const controller = seedWait.controller;
    stopSeedWait();
    if (controller) {
      try {
        controller.abort();
      } catch {
        // Aborting is best-effort; the fetch rejection still classifies below.
      }
    }
    if (submissionIsCurrent(submission) && bindingIsCurrent(binding)) {
      const status = $("seed-status");
      if (status) {
        status.textContent = SEED_WAIT_TEXT_DETACHED;
      }
    }
    return true;
  }

  /**
   * One-shot re-check (never a resumed loop): exactly one ``getState``,
   * applying the same identity filter + verbatim table once, reporting the
   * still-live word or the settled ledger state without resuming polling.
   */
  async function recheckSeedWait() {
    const wait = seedWait.active;
    if (!wait) {
      return null;
    }
    if (!submissionIsCurrent(wait.submission) || !bindingIsCurrent(wait.binding)) {
      return null;
    }
    let document = null;
    try {
      document = await source.getState(wait.sessionId);
    } catch {
      document = null;
    }
    if (!submissionIsCurrent(wait.submission) || !bindingIsCurrent(wait.binding)) {
      return null;
    }
    if (document === null) {
      writeSeedWaitLine(wait.submission, wait.binding, SEED_WAIT_TEXT_TRANSPORT_ERROR);
      return "transport-error";
    }
    const entry = seedWaitEntryFor(document, wait.sessionId, wait.clientNonce);
    const line = seedWaitLineFor(entry);
    if (line !== null) {
      writeSeedWaitLine(wait.submission, wait.binding, line);
      return line;
    }
    const accepted =
      document &&
      typeof document === "object" &&
      Array.isArray(document.accepted_nonces)
        ? document.accepted_nonces
        : null;
    if (accepted && accepted.includes(wait.clientNonce)) {
      return "accepted";
    }
    return "no-live-word";
  }

  /** The Cancel control beside Submit: visible/enabled ONLY while waiting. */
  function updateSeedWaitCancel() {
    const cancel = $("seed-cancel");
    if (!cancel) {
      return;
    }
    const waiting = seedWait.active !== null;
    cancel.hidden = !waiting;
    cancel.disabled = !waiting;
  }

  /** True while the truthful-wait observer is active for the current submission. */
  function seedWaitIsWaiting() {
    return seedWait.active !== null;
  }

  function renderSeedControls() {
    const toolbar = doc.querySelector(".toolbar");
    if (!toolbar || $("seed-panel")) {
      return;
    }
    const panel = doc.createElement("span");
    panel.id = "seed-panel";
    panel.className = "seed-panel";
    panel.setAttribute("role", "group");
    panel.setAttribute("aria-label", "Physician seed");
    const arm = doc.createElement("button");
    arm.type = "button";
    arm.id = "seed-arm";
    arm.textContent = "Seed pick: off";
    arm.setAttribute("aria-pressed", "false");
    arm.addEventListener("click", () => {
      toggleSeedArm();
    });
    panel.appendChild(arm);
    for (const [id, label, step] of [
      ["seed-i", "i", "1"],
      ["seed-j", "j", "1"],
      ["seed-k", "k", "1"],
      ["seed-dx", "dx", "any"],
      ["seed-dy", "dy", "any"],
      ["seed-dz", "dz", "any"],
    ]) {
      const wrapper = doc.createElement("label");
      wrapper.textContent = `${label} `;
      const input = doc.createElement("input");
      input.id = id;
      input.type = "number";
      input.step = step;
      input.size = 5;
      input.setAttribute("aria-label", `seed ${label}`);
      if (step === "1") {
        input.min = "0";
      }
      input.addEventListener("input", () => {
        updateSeedControls();
      });
      wrapper.appendChild(input);
      panel.appendChild(wrapper);
    }
    const submit = doc.createElement("button");
    submit.type = "button";
    submit.id = "seed-submit";
    submit.textContent = "Submit seed";
    submit.disabled = true;
    submit.addEventListener("click", () => {
      void submitSeed();
    });
    panel.appendChild(submit);
    const cancel = doc.createElement("button");
    cancel.type = "button";
    cancel.id = "seed-cancel";
    cancel.textContent = "Cancel wait";
    cancel.hidden = true;
    cancel.disabled = true;
    cancel.setAttribute("aria-label", "Stop waiting for the seed (server work continues)");
    cancel.addEventListener("click", () => {
      detachSeedWait();
    });
    panel.appendChild(cancel);
    const clear = doc.createElement("button");
    clear.type = "button";
    clear.id = "seed-clear";
    clear.textContent = "Clear";
    clear.addEventListener("click", () => {
      clearSeed();
    });
    panel.appendChild(clear);
    const status = doc.createElement("span");
    status.id = "seed-status";
    status.className = "seed-status";
    status.setAttribute("role", "status");
    status.textContent = "no seed";
    panel.appendChild(status);
    const provenance = doc.createElement("span");
    provenance.id = "seed-prov";
    provenance.className = "seed-status";
    provenance.setAttribute("role", "status");
    provenance.hidden = true;
    provenance.textContent = "";
    panel.appendChild(provenance);
    const qcLink = doc.createElement("button");
    qcLink.type = "button";
    qcLink.id = "seed-qc-link";
    qcLink.textContent = "Seed QC";
    qcLink.hidden = true;
    qcLink.setAttribute("aria-label", "Open QC for the accepted seed");
    qcLink.addEventListener("click", () => {
      const control = $("open-qc");
      if (control) {
        control.click();
      }
    });
    panel.appendChild(qcLink);
    const failure = doc.createElement("span");
    failure.id = "seed-error";
    failure.className = "overlay-error";
    failure.hidden = true;
    failure.setAttribute("role", "status");
    panel.appendChild(failure);
    toolbar.insertBefore(panel, $("open-qc") || null);
  }

  function readSeedInputs() {
    const number = (id) => {
      const input = $(id);
      return input ? input.valueAsNumber : NaN;
    };
    return {
      voxel: [number("seed-i"), number("seed-j"), number("seed-k")],
      direction: [number("seed-dx"), number("seed-dy"), number("seed-dz")],
    };
  }

  function seedInputsValid() {
    const { voxel, direction } = readSeedInputs();
    try {
      assertSeedVoxel(voxel);
      assertSeedDirection(direction);
      return true;
    } catch {
      return false;
    }
  }

  function setSeedInputs(voxel, direction) {
    // Null means "leave unchanged"; use clearSeedInputs for a full reset.
    const ids = ["seed-i", "seed-j", "seed-k"];
    const dids = ["seed-dx", "seed-dy", "seed-dz"];
    if (voxel) {
      ids.forEach((id, index) => {
        const input = $(id);
        if (input) {
          input.value = Number.isFinite(voxel[index]) ? String(voxel[index]) : "";
        }
      });
    }
    if (direction) {
      dids.forEach((id, index) => {
        const input = $(id);
        if (input) {
          input.value =
            direction[index] === null || direction[index] === undefined
              ? ""
              : String(direction[index]);
        }
      });
    }
  }

  function clearSeedInputs() {
    for (const id of ["seed-i", "seed-j", "seed-k", "seed-dx", "seed-dy", "seed-dz"]) {
      const input = $(id);
      if (input) {
        input.value = "";
      }
    }
  }

  // Q3-SEED-PROV (physician-visible accepted-seed provenance, read/display
  // only): after a seed is accepted, the seed panel shows a compact summary
  // of what the seed path already records — the accepted voxel + direction
  // (``state.seedCurrent``), the bound session, and the served centerline
  // binding (``state.overlayDocument.centerline_ref`` + the bound generation
  // from ``state.cprDocument``, with the QC panel one click away via the
  // existing Open-QC control).  The pure text builder lives in
  // ``lib/seedProvenance.js`` (unit-covered); here only the DOM binding runs.
  // No new server state, no grammar change, no envelope keys, no anatomy
  // labels, no inference: every value is served or typed, and any malformed
  // binding renders a typed unavailable state (never invented text).  The
  // summary is a SIBLING of #seed-status (never a child), so the frozen
  // #seed-status text contract is byte-identical.
  //
  // F1 association: ``state.seedBinding`` is the served binding SNAPSHOTTED
  // when the seed was accepted (``submitSeed``, never on panel reloads).  A
  // render shows the binding only while the currently displayed documents
  // are exactly that snapshot; a vessel switch renders the typed unavailable
  // state instead of labelling another vessel's documents as this seed's
  // provenance.

  /**
   * The compact accepted-seed provenance summary, or null when no accepted
   * seed is displayed.  Malformed voxel/direction/binding values and an
   * unverifiable accepted-seed↔centerline association return the typed
   * unavailable sentinel instead of fabricated text.
   */
  function seedProvenanceSummary() {
    return seedProvenanceText({
      seedCurrent: state.seedCurrent,
      sessionId: state.sessionId,
      overlayDocument: state.overlayDocument,
      cprDocument: state.cprDocument,
      acceptedBinding: state.seedBinding,
    });
  }

  function updateSeedProvenance() {
    const provenance = $("seed-prov");
    const qcLink = $("seed-qc-link");
    if (!provenance) {
      return;
    }
    const text = seedProvenanceSummary();
    if (text === null) {
      provenance.hidden = true;
      provenance.textContent = "";
      delete provenance.dataset.seedProv;
      provenance.removeAttribute("title");
      if (qcLink) {
        qcLink.hidden = true;
      }
      return;
    }
    provenance.hidden = false;
    provenance.textContent = text;
    provenance.dataset.seedProv =
      text === SEED_PROVENANCE_UNAVAILABLE ? "unavailable" : "bound";
    provenance.title =
      text === SEED_PROVENANCE_UNAVAILABLE
        ? "the served seed binding is unreadable"
        : "accepted-seed provenance (served binding) · QC panel via Seed QC";
    if (qcLink) {
      // The link reuses the existing Open-QC control (no new seam): it stays
      // visible whenever an accepted seed is displayed, including the typed
      // unavailable state (the QC panel is still the audit path).
      qcLink.hidden = false;
    }
  }

  function seedStatusText() {
    if (state.seedCurrent) {
      const { voxel, direction } = state.seedCurrent;
      return `seed [${voxel.join(",")}] · dir [${direction.join(",")}]`;
    }
    const { voxel, direction } = readSeedInputs();
    const voxelText = voxel.every((value) => Number.isInteger(value) && value >= 0)
      ? `[${voxel.join(",")}]`
      : "—";
    const directionText = direction.every(
      (value) => typeof value === "number" && Number.isFinite(value)
    )
      ? `[${direction.join(",")}]`
      : "—";
    if (state.seedAnchor) {
      return (
        `pick ${state.seedAnchor.plane} f${state.seedAnchor.frameIndex}` +
        ` · voxel ${voxelText} · dir ${directionText}`
      );
    }
    return "no seed — arm pick or type voxel + direction";
  }

  function updateSeedControls() {
    if (!$("seed-panel")) {
      renderSeedControls();
    }
    const arm = $("seed-arm");
    const submit = $("seed-submit");
    const status = $("seed-status");
    const failure = $("seed-error");
    if (!arm || !submit) {
      return;
    }
    arm.setAttribute("aria-pressed", state.seedArmed ? "true" : "false");
    arm.textContent = `Seed pick: ${state.seedArmed ? "on" : "off"}`;
    submit.disabled = !(seedInputsValid() && state.sessionId);
    if (status && !seedWaitIsWaiting()) {
      status.textContent = seedStatusText();
    }
    updateSeedWaitCancel();
    updateSeedProvenance();
    if (failure) {
      failure.hidden = !state.seedError;
      failure.textContent = state.seedError
        ? `Seed: ${state.seedError.code}`
        : "";
      if (state.seedError) {
        failure.dataset.seedError = state.seedError.code;
        failure.title = state.seedError.message;
      } else {
        delete failure.dataset.seedError;
        failure.removeAttribute("title");
      }
    }
  }

  function toggleSeedArm() {
    state.seedArmed = !state.seedArmed;
    if (!state.seedArmed) {
      state.seedAnchor = null;
      state.seedDrag = null;
      redrawOverlays();
    }
    updateSeedControls();
  }

  function showSeedError(error) {
    const code =
      error && typeof error.code === "string" ? error.code : "SEED_ERROR";
    state.seedError = { code, message: errorText(error) };
    updateSeedControls();
    setStatus(`${code}: ${errorText(error)}`);
  }

  function clearSeed() {
    state.seedAnchor = null;
    state.seedDrag = null;
    state.seedError = null;
    clearSeedInputs();
    updateSeedControls();
    redrawOverlays();
  }

  /**
   * Map one canvas event to the displayed frame's index-space pixel.
   *
   * The inverse of the overlay draw mapping (``worldToCanvas`` over the
   * served ``(column, row)`` indices): for the uncalibrated display stack,
   * ``canvasToWorld`` returns the same index space.  Any mapping failure is
   * a typed client refusal — never a fabricated position.
   */
  function pickImagePixel(plane, event) {
    const info = elements[plane];
    const viewport = viewports.viewportFor(plane);
    if (!info || !viewport || typeof viewport.canvasToWorld !== "function") {
      throw new ViewerClientError(
        "SEED_PICK_INVALID",
        "seed picking needs the displayed viewport"
      );
    }
    const rect = info.host.getBoundingClientRect();
    const point = [event.clientX - rect.left, event.clientY - rect.top];
    let world = null;
    try {
      world = viewport.canvasToWorld(point);
    } catch {
      world = null;
    }
    if (
      !world ||
      !Number.isFinite(Number(world[0])) ||
      !Number.isFinite(Number(world[1]))
    ) {
      throw new ViewerClientError(
        "SEED_PICK_INVALID",
        "the click did not map to the displayed frame"
      );
    }
    return { column: Number(world[0]), row: Number(world[1]) };
  }

  function beginSeedPick(plane, event) {
    if (event.button !== 0) {
      return;
    }
    let pixel = null;
    try {
      pixel = pickImagePixel(plane, event);
    } catch (error) {
      showSeedError(error);
      return;
    }
    const frameIndex = overlayFrameIndex(plane);
    state.seedAnchor = {
      plane,
      column: pixel.column,
      row: pixel.row,
      frameIndex,
    };
    state.seedDrag = null;
    state.seedError = null;
    // The click pre-fills the EXPLICIT voxel fields (visible, editable) and
    // clears the direction until dragged or typed.
    setSeedInputs([Math.round(pixel.column), Math.round(pixel.row), frameIndex], [
      null,
      null,
      null,
    ]);
    updateSeedControls();
    redrawOverlay(plane);
  }

  function updateSeedDrag(plane, event) {
    const anchor = state.seedAnchor;
    if (!anchor || anchor.plane !== plane) {
      return;
    }
    if (typeof event.buttons === "number" && event.buttons !== 1) {
      return;
    }
    let pixel = null;
    try {
      pixel = pickImagePixel(plane, event);
    } catch {
      return;
    }
    const dx = pixel.column - anchor.column;
    const dy = pixel.row - anchor.row;
    if (!Number.isFinite(dx) || !Number.isFinite(dy)) {
      return;
    }
    state.seedDrag = { dx, dy };
    // The drag fills dx/dy only; dz stays explicitly as entered until typed.
    const rounded = (value) => Number(value.toFixed(3));
    const { voxel } = readSeedInputs();
    const dzInput = $("seed-dz");
    const dz = dzInput ? dzInput.valueAsNumber : NaN;
    setSeedInputs(voxel, [rounded(dx), rounded(dy), Number.isFinite(dz) ? dz : null]);
    updateSeedControls();
    redrawOverlay(plane);
  }

  /**
   * Draw the pending-seed marker on the plane's dedicated 2D layer.
   *
   * Display-only (the inputs carry the values): a crosshair at the picked
   * position plus a fixed-length direction arrow while dragging.  Drawn only
   * on the picked frame; runs after the overlay draw so the two stay in
   * sync.  ``canvas.dataset.seedDrawn`` records the marker state (0 when no
   * pending pick is shown).
   */
  function drawSeedMarker(plane) {
    const info = elements[plane];
    if (!info || !info.overlay) {
      return;
    }
    const canvas = info.overlay;
    const anchor = state.seedAnchor;
    const onFrame =
      anchor &&
      anchor.plane === plane &&
      anchor.frameIndex === overlayFrameIndex(plane);
    if (!onFrame) {
      canvas.dataset.seedDrawn = "0";
      delete canvas.dataset.seedVoxel;
      return;
    }
    const viewport = viewports.viewportFor(plane);
    if (!viewport || typeof viewport.worldToCanvas !== "function") {
      canvas.dataset.seedDrawn = "0";
      return;
    }
    let mapped = null;
    try {
      mapped = viewport.worldToCanvas([anchor.column, anchor.row, 0]);
    } catch {
      mapped = null;
    }
    if (
      !mapped ||
      !Number.isFinite(Number(mapped[0])) ||
      !Number.isFinite(Number(mapped[1]))
    ) {
      canvas.dataset.seedDrawn = "0";
      return;
    }
    const x = Number(mapped[0]);
    const y = Number(mapped[1]);
    const context = canvas.getContext("2d");
    const { voxel } = readSeedInputs();
    canvas.dataset.seedDrawn = "1";
    canvas.dataset.seedVoxel = voxel
      .map((value) => (Number.isFinite(value) ? String(value) : ""))
      .join(",");
    canvas.hidden = false;
    if (!context) {
      return;
    }
    context.strokeStyle = SEED_MARKER_STROKE;
    context.lineWidth = 1.5;
    context.setLineDash([]);
    context.beginPath();
    context.moveTo(x - 8, y);
    context.lineTo(x + 8, y);
    context.moveTo(x, y - 8);
    context.lineTo(x, y + 8);
    context.stroke();
    if (state.seedDrag) {
      const length = Math.hypot(state.seedDrag.dx, state.seedDrag.dy);
      if (length > 0) {
        const scale = 28 / length;
        const ex = x + state.seedDrag.dx * scale;
        const ey = y + state.seedDrag.dy * scale;
        context.beginPath();
        context.moveTo(x, y);
        context.lineTo(ex, ey);
        context.stroke();
        const angle = Math.atan2(ey - y, ex - x);
        context.beginPath();
        context.moveTo(ex, ey);
        context.lineTo(ex - 7 * Math.cos(angle - 0.4), ey - 7 * Math.sin(angle - 0.4));
        context.moveTo(ex, ey);
        context.lineTo(ex - 7 * Math.cos(angle + 0.4), ey - 7 * Math.sin(angle + 0.4));
        context.stroke();
      }
    }
  }

  function postSeedRequest(envelope, options = {}) {
    const binding = currentBinding();
    const run = state.actionQueue.then(async () => {
      if (!bindingIsCurrent(binding)) {
        return null;
      }
      let document;
      try {
        const seedOptions =
          options && options.signal !== undefined && options.signal !== null
            ? { signal: options.signal }
            : {};
        document = await source.postSeed(
          state.sessionId,
          envelope,
          canonicalJson(envelope),
          seedOptions
        );
      } catch (error) {
        // F2 truthful-wait (waiter-detach): an AbortError from the per-
        // submission AbortController is a DETACH, never a refusal — never
        // setStatus here (the submitSeed path renders the detach line),
        // never showSeedError, never state.seedError.  The queue stays
        // alive via the existing arm below; rethrow so the awaiting
        // submitSeed path observes the detach.
        if (
          error &&
          (error.name === "AbortError" ||
            (typeof DOMException !== "undefined" &&
              error instanceof DOMException &&
              error.name === "AbortError"))
        ) {
          throw error;
        }
        if (bindingIsCurrent(binding)) {
          setStatus(errorText(error));
        }
        throw error;
      }
      if (!bindingIsCurrent(binding)) {
        return null;
      }
      // TREE_SCHEMATIC_NAV F2: an ACCEPTED seed minted a NEW centerline
      // through the governed reseed seam, so the canonical ACTIVE set this
      // client mutates has ALREADY been replaced by the time this response
      // arrives.  Two things follow, and both must happen NOW - before the
      // ``applyModel`` await below.  That await can block for a whole frame
      // fetch (a new served frame record forces a new image), and advancing
      // the revision only after it returned left a window in which the
      // canonical set had changed while the client still treated the previous
      // revision as current - so a schematic/overlay/CPR response already in
      // flight under the superseded set could still pass its binding check and
      // repaint geometry the server had replaced:
      //   (1) advance the set revision, which invalidates every load already in
      //       flight under the superseded set - a response derived from the old
      //       paths can no longer pass its binding check; and
      //   (2) disarm the schematic derived from those now-superseded paths, so
      //       no current-looking geometry stays on screen while the replacement
      //       is pending (Rule 10: abstain rather than show something stale).
      // A REFUSED seed never reaches this point and advances neither.
      state.activeSetSeq += 1;
      disarmSchematicPanel();
      const model = displayStateFrom(document);
      state.parentActionNonce = model.lastAcceptedNonce;
      await applyModel(model, { restore: false });
      return document;
    });
    // Keep the queue alive on refusal; callers still observe the rejection.
    state.actionQueue = run.then(
      () => undefined,
      () => undefined
    );
    return run;
  }

  async function submitSeed() {
    if (!state.sessionId) {
      return;
    }
    // F1-race (R2): record the submission's navigation identity BEFORE the
    // first await.  ``selectVessel`` mutates the selection optimistically at
    // click time, so a switch accepted while the /seed response is in flight
    // is only visible here through ``vesselNavSeq``; the capture below fails
    // closed for that submission instead of snapshotting the switch's
    // documents as the accepted seed's provenance.
    const submission = {
      sessionId: state.sessionId,
      generation: state.generation,
      vesselNavSeq: state.vesselNavSeq,
    };
    const submitBinding = currentBinding();
    const submitSessionId = state.sessionId;
    const { voxel, direction } = readSeedInputs();
    let envelope;
    try {
      envelope = buildSeedEnvelope({
        voxel,
        direction,
        clientNonce: nextNonce("seed"),
        sessionId: state.sessionId,
        parentActionNonce: state.parentActionNonce,
      });
    } catch (error) {
      showSeedError(error);
      return;
    }
    // F2 truthful-wait: the per-submission waiter-detach controller lives
    // ONLY on the SEED POST fetch (passed through postSeedRequest into the
    // existing frameSource.postSeed signal pass-through).  getState/postAction
    // reads never receive it.  The poll observer starts ALONGSIDE (never
    // inside) the queued run, as a detached non-queued serial chain.
    const controller =
      typeof AbortController !== "undefined" ? new AbortController() : null;
    seedWait.controller = controller;
    startSeedWait({
      submission,
      binding: submitBinding,
      sessionId: submitSessionId,
      clientNonce: envelope.client_nonce,
    });
    const seedFailed = (error) => {
      const isAbort =
        error &&
        (error.name === "AbortError" ||
          (typeof DOMException !== "undefined" &&
            error instanceof DOMException &&
            error.name === "AbortError"));
      stopSeedWait();
    clearSeedWaitController();
    if (isAbort) {
      // Detach: never showSeedError, never state.seedError.  Render the
      // detach line only while THIS submission is still current; a
      // superseded/session-switched submission stops silently.
      if (submissionIsCurrent(submission) && bindingIsCurrent(submitBinding)) {
        const status = $("seed-status");
        if (status) {
          status.textContent = SEED_WAIT_TEXT_DETACHED;
        }
      }
      updateSeedWaitCancel();
      return;
    }
      showSeedError(error);
    };
    let document = null;
    try {
      document = await postSeedRequest(
        envelope,
        controller ? { signal: controller.signal } : {}
      );
    } catch (error) {
      seedFailed(error);
      return;
    }
    // The seed fetch settled: stop the observer and hand the display to the
    // settled surface (no trailing poll that could repaint a stale word).
    stopSeedWait();
    clearSeedWaitController();
    if (!document) {
      // A superseded/session-switched run settles to null without touching
      // the display (the newer navigation owns it); still refresh the
      // Cancel visibility from the stopped observer state.
      updateSeedWaitCancel();
      return;
    }
    state.seedError = null;
    state.seedCurrent = { voxel: [...voxel], direction: [...direction] };
    state.seedAnchor = null;
    state.seedDrag = null;
    // TREE_SCHEMATIC_NAV F2: an ACCEPTED seed replaced the canonical ACTIVE
    // set, so the set revision was ALREADY advanced and the superseded
    // schematic ALREADY disarmed at the acceptance itself (``postSeedRequest``,
    // before its ``applyModel`` await) - not here, after that await, which left
    // a window in which the canonical set had changed while the client still
    // considered the previous revision current.  A REFUSED seed returns above
    // and advanced neither.  The binding below therefore already carries the
    // NEW revision, and the replacement panels are fetched under it.
    const binding = currentBinding();
    // A seed changes the served generation: reload all panels coherently.
    await loadOverlay(binding);
    await loadCpr(binding);
    await loadSchematic(binding);
    // F1: snapshot the served binding observed right after ACCEPT — the
    // verifiable accepted-seed↔centerline association.  Only a render whose
    // displayed documents still match this snapshot shows the binding; a
    // vessel switch renders the typed unavailable state instead.
    // F1-race (R2): the snapshot is tied to THIS submission's navigation
    // identity — a vessel switch accepted while the /seed response was in
    // flight (or a session switch) fails the capture closed (typed
    // unavailable) instead of labelling the switch's documents as the
    // accepted seed's provenance.
    if (bindingIsCurrent(binding) && submissionIsCurrent(submission)) {
      state.seedBinding = captureAcceptedSeedBinding({
        sessionId: state.sessionId,
        overlayDocument: state.overlayDocument,
        cprDocument: state.cprDocument,
      });
    } else {
      state.seedBinding = null;
    }
    setStatus(`Seed accepted: voxel [${voxel.join(",")}]`);
    updateSeedControls();
    redrawOverlays();
  }

  // ------------------------------------------------- sync cursor
  //
  // MPR_SYNC_CURSOR (ephemeral view aid): double-click on any MPR viewport
  // places one shared crosshair pick on the displayed frame; the server
  // derives the world point plus the per-plane crosshair lines (the client
  // performs NO geometry and renders NO numbers — lines only).  Reselect
  // replaces the pick; Clear drops it; scroll/plane-change refresh re-derives
  // from the stored pick through the same binding-checked load.  A new
  // session starts with no cursor (nothing persists, Rule F).

  function renderSyncControls() {
    const toolbar = doc.querySelector(".toolbar");
    if (!toolbar || $("sync-panel")) {
      return;
    }
    const panel = doc.createElement("span");
    panel.id = "sync-panel";
    panel.className = "sync-panel";
    panel.setAttribute("role", "group");
    panel.setAttribute("aria-label", "Sync cursor");
    const clear = doc.createElement("button");
    clear.type = "button";
    clear.id = "sync-clear";
    clear.textContent = "Clear cursor";
    clear.disabled = true;
    clear.addEventListener("click", () => {
      clearSyncCursor();
    });
    panel.appendChild(clear);
    const status = doc.createElement("span");
    status.id = "sync-status";
    status.className = "sync-status";
    status.setAttribute("role", "status");
    status.textContent = "no cursor";
    panel.appendChild(status);
    const failure = doc.createElement("span");
    failure.id = "sync-error";
    failure.className = "overlay-error";
    failure.hidden = true;
    failure.setAttribute("role", "status");
    panel.appendChild(failure);
    toolbar.insertBefore(panel, $("open-qc") || null);
  }

  function updateSyncControls() {
    if (!$("sync-panel")) {
      renderSyncControls();
    }
    const clear = $("sync-clear");
    const status = $("sync-status");
    const failure = $("sync-error");
    if (!clear || !status) {
      return;
    }
    clear.disabled = !(state.syncSource && state.sessionId);
    status.textContent = state.syncDocument ? SYNC_CURSOR_CAPTION : "no cursor";
    if (status) {
      delete status.dataset.syncOffFrame;
    }
    if (failure) {
      failure.hidden = !state.syncError;
      failure.textContent = state.syncError
        ? `Sync cursor: ${state.syncError.code}`
        : "";
      if (state.syncError) {
        failure.dataset.syncError = state.syncError.code;
        failure.title = state.syncError.message;
      } else {
        delete failure.dataset.syncError;
        failure.removeAttribute("title");
      }
    }
    updateSyncOffFrameNote();
  }

  /**
   * The visible reason for FINDING 1 suppression (fixed prose, no numbers,
   * no measurement): when the cached document exists but any panel displays
   * a frame other than its served planes[plane].frame_index, that panel's
   * lines are suppressed and this note names it.  Runs after every overlay
   * redraw and every control update so scroll and reselect stay coherent.
   */
  function updateSyncOffFrameNote() {
    const status = $("sync-status");
    if (!status || !state.syncDocument) {
      return;
    }
    const planes = state.syncDocument.planes;
    if (!planes || typeof planes !== "object") {
      return;
    }
    let offFrame = false;
    for (const plane of PLANES) {
      const entry = planes[plane];
      const served =
        entry && typeof entry === "object" ? Number(entry.frame_index) : NaN;
      if (!Number.isFinite(served)) {
        continue;
      }
      if (Number(overlayFrameIndex(plane)) !== served) {
        offFrame = true;
        break;
      }
    }
    if (offFrame) {
      status.textContent = `${SYNC_CURSOR_CAPTION} · off-frame hidden`;
      status.dataset.syncOffFrame = "1";
    } else {
      status.textContent = SYNC_CURSOR_CAPTION;
      delete status.dataset.syncOffFrame;
    }
  }

  function showSyncError(error) {
    const code =
      error && typeof error.code === "string" ? error.code : "SYNC_CURSOR_ERROR";
    state.syncError = { code, message: errorText(error) };
    updateSyncControls();
    setStatus(`${code}: ${errorText(error)}`);
  }

  function clearSyncCursor() {
    // FINDING 2: invalidate any in-flight cursor GET so a cleared-while-
    // pending response cannot resurrect the cursor (bindingIsCurrent alone
    // only sees session+generation).
    state.syncSeq += 1;
    state.syncSource = null;
    state.syncDocument = null;
    state.syncError = null;
    updateSyncControls();
    redrawOverlays();
  }

  /**
   * Place (reselect) the crosshair pick on the displayed frame.
   *
   * Reuses the seed pick-assist inverse mapping (canvas event -> the
   * displayed frame's index-space pixel); the served document carries the
   * shared point and every plane's lines.  Mapping failures are typed client
   * refusals — never a fabricated position.
   */
  async function placeSyncCursor(plane, event) {
    if (!state.sessionId || state.seedArmed) {
      return;
    }
    let pixel = null;
    try {
      pixel = pickImagePixel(plane, event);
    } catch {
      showSyncError({ code: "SYNC_PICK_INVALID", message: "the click did not map" });
      return;
    }
    state.syncSource = {
      plane,
      frameIndex: overlayFrameIndex(plane),
      row: pixel.row,
      column: pixel.column,
    };
    state.syncError = null;
    // FINDING 2: reselect invalidates any in-flight older pick so rapid
    // picks resolving out of order cannot let the older overwrite the newer.
    state.syncSeq += 1;
    await refreshSyncCursor();
  }

  /** Re-derive the served document from the stored pick (refresh/reselect). */
  async function refreshSyncCursor() {
    if (!state.syncSource || !state.sessionId) {
      return;
    }
    const binding = currentBinding();
    // FINDING 2: the cursor-specific generation travels with this request;
    // BOTH the success and the error continuations below discard stale
    // generations (clear/reselect bumped syncSeq while this GET was in
    // flight), so a stale success cannot resurrect/overwrite and a stale
    // rejection cannot erase the latest cursor.
    const syncSeq = state.syncSeq;
    const pick = { ...state.syncSource };
    let document = null;
    try {
      document = await loadSyncPoint(source, state.sessionId, pick);
    } catch (error) {
      if (!bindingIsCurrent(binding) || syncSeq !== state.syncSeq) {
        return;
      }
      state.syncDocument = null;
      showSyncError(error);
      redrawOverlays();
      return;
    }
    if (!bindingIsCurrent(binding) || syncSeq !== state.syncSeq) {
      return;
    }
    // loadSyncPoint maps the ordinary empty-plane case to null (cursor
    // hidden, no error); any typed refusal above stays visible instead.
    state.syncDocument = document;
    state.syncError = null;
    updateSyncControls();
    redrawOverlays();
  }

  // ---------------------------------------------------------- CPR panel

  /**
   * The ONE dedicated straightened-view panel shell (Decision B).
   *
   * A plain 2D canvas reusing the frame/transport code path (no second
   * intensity pipeline, no client-side geometry): the server-rendered CPR
   * PNG bytes are drawn 1:1.  Created once alongside the overlay controls
   * so the panel layout is stable whether or not a CPR view exists.
   */
  function renderCprPanelShell() {
    const viewports = $("viewports");
    if (!viewports || $("cpr-panel")) {
      return;
    }
    const panel = doc.createElement("figure");
    panel.className = "viewport cpr-panel";
    panel.id = "cpr-panel";
    const host = doc.createElement("div");
    host.className = "viewport-canvas";
    host.id = "cpr-canvas-host";
    const canvas = doc.createElement("canvas");
    canvas.className = "cpr-canvas";
    canvas.id = "cpr-canvas";
    canvas.dataset.cprDrawn = "0";
    canvas.setAttribute("aria-label", "Straightened CPR view");
    canvas.hidden = true;
    const caption = doc.createElement("figcaption");
    caption.id = "cpr-caption";
    caption.textContent = "CPR: loading…";
    const failure = doc.createElement("span");
    failure.id = "cpr-error";
    failure.className = "overlay-error";
    failure.hidden = true;
    failure.setAttribute("role", "status");
    host.appendChild(canvas);
    panel.appendChild(host);
    panel.appendChild(caption);
    panel.appendChild(failure);
    viewports.appendChild(panel);
  }

  function updateCprPanel() {
    const panel = $("cpr-panel");
    const canvas = $("cpr-canvas");
    const caption = $("cpr-caption");
    const failure = $("cpr-error");
    if (!panel || !canvas) {
      return;
    }
    const view = state.cprDocument;
    const error = state.cprError;
    if (failure) {
      failure.hidden = !error;
      failure.textContent = error ? `CPR: ${error.code}` : "";
      if (error) {
        failure.dataset.cprError = error.code;
      } else {
        delete failure.dataset.cprError;
      }
    }
    if (!view) {
      canvas.hidden = true;
      if (caption) {
        caption.textContent = error ? `CPR: ${error.code}` : "CPR: unavailable";
      }
      return;
    }
    if (caption) {
      const matrix = Array.isArray(view.matrix_size) ? view.matrix_size : [];
      caption.textContent =
        `CPR · straightened · ${matrix.join("×") || "?"} ` +
        `· ${view.centerline_ref || "?"} · gen ${view.centerline_generation || "?"}`;
      caption.dataset.cprRef = String(view.cpr_ref || "");
      caption.dataset.centerlineRef = String(view.centerline_ref || "");
      caption.dataset.centerlineGeneration = String(view.centerline_generation || "");
    }
  }

  /** Load (or clear) the CPR view document + PNG for the bound session. */
  async function loadCpr(binding) {
    const sessionId = state.sessionId;
    if (!sessionId) {
      return;
    }
    // F1-race (R3): a superseded navigation never fetches and never applies.
    // TREE_SCHEMATIC_NAV F2: nor does a load issued before the canonical
    // centerline set was replaced.
    if (!panelSetIsCurrent(binding)) {
      return;
    }
    // TSN_R5_F1 entry fence: an obsolete recovery/settlement epoch never claims an ordinal nor fetches.
    if ((binding.recoverySeq || 0) !== state.selectionRecoverySeq) {
      return;
    }
    // TREE_SCHEMATIC_NAV F2: this panel's own request ordinal.
    const request = (state.cprReqSeq += 1);
    renderCprPanelShell();
    // TSN_R4_F1 (R10-F1 null-path): same requested-vessel capture as overlay.
    const requestedVessel =
      state.selectedVessel && typeof state.selectedVessel === "string"
        ? state.selectedVessel
        : null;
    let view = null;
    let failure = null;
    let pngBytes = null;
    try {
      const vesselOptions =
        requestedVessel !== null ? { vessel: requestedVessel } : undefined;
      view = await loadCprView(
        vesselOptions ? { getCpr: (id) => source.getCpr(id, vesselOptions) } : source,
        sessionId
      );
      if (view) {
        pngBytes = await source.getCprPngBytes(sessionId, vesselOptions);
      }
    } catch (error) {
      failure = {
        code: String((error && error.code) || "CPR_VIEW_ERROR"),
        message: errorText(error),
      };
      view = null;
      pngBytes = null;
    }
    // F1-race (R3): the response is applied only while the navigation that
    // started it is still the current one (the R3 stale-response variant).
    // TREE_SCHEMATIC_NAV F2: and only while it is still this panel's newest
    // request under the still-canonical ACTIVE centerline set.
    if (!panelSetIsCurrent(binding) || request !== state.cprReqSeq || (binding.recoverySeq || 0) !== state.selectionRecoverySeq) {
      return;
    }
    state.cprDocument = view;
    state.cprError = failure;
    if (view && typeof view.vessel_label === "string") {
      // TSN_R4_F1 (R10-F1 null-path): same provenance gate as overlay — an
      // unfiltered (null-requested) geometry label is kept for display and
      // NEVER adopted; a filtered read adopts only its own matching token.
      if (
        requestedVessel !== null &&
        view.vessel_label === requestedVessel &&
        state.selectedVessel === requestedVessel &&
        (binding.recoverySeq || 0) === state.selectionRecoverySeq
      ) {
        state.selectedVessel = view.vessel_label;
      }
      /* else: keep document, never adopt its label */
    }
    if (failure) {
      setStatus(`${failure.code}: ${failure.message}`);
    }
    updateCprPanel();
    updateVesselPicker();
    const canvas = $("cpr-canvas");
    if (canvas && view && pngBytes && pngBytes.length) {
      // TREE_SCHEMATIC_NAV F2 (the final paint): a guard and the effect it
      // authorises must not be separated by an await.  The bitmap decode is
      // that await - the shared CPR painter decoded between the check above and
      // the pixel write - and a newer CPR request (a second refresh, a later
      // vessel navigation), a session switch or a canonical-set replacement
      // can complete inside it.  So the PNG is decoded OFF-CANVAS here and the
      // pixels are committed only after re-checking, with nothing between the
      // re-check and the write.  The commit is the frozen 1:1 contract this
      // panel has always drawn under - no resampling, no windowing, no client
      // geometry, the same canvas state and the same dataset words - it simply
      // moved inside the guard that authorises it.
      //
      // The blank below is the one mutation that stays OUTSIDE the decode: it
      // runs synchronously, with no await between the guard above and it, so
      // it can never erase pixels a newer request has already committed.
      const blank = canvas.getContext("2d");
      if (blank) {
        blank.clearRect(0, 0, canvas.width, canvas.height);
      }
      canvas.dataset.cprDrawn = "0";
      let bitmap = null;
      try {
        bitmap = await createImageBitmap(new Blob([pngBytes], { type: "image/png" }));
      } catch (error) {
        setStatus(errorText(error));
      }
      if (bitmap) {
        if (panelSetIsCurrent(binding) && request === state.cprReqSeq && (binding.recoverySeq || 0) === state.selectionRecoverySeq) {
          const context = canvas.getContext("2d");
          if (context) {
            context.clearRect(0, 0, canvas.width, canvas.height);
          }
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
          if (context) {
            context.drawImage(bitmap, 0, 0);
          }
          canvas.dataset.cprDrawn = String(bitmap.width * bitmap.height);
          canvas.dataset.cprRows = String(bitmap.height);
          canvas.dataset.cprColumns = String(bitmap.width);
          canvas.hidden = false;
        }
        // A decoded bitmap this load is not allowed to paint is still closed,
        // exactly as the shared painter closed every bitmap it decoded.
        if (typeof bitmap.close === "function") {
          bitmap.close();
        }
      }
    } else if (canvas) {
      canvas.hidden = true;
      canvas.dataset.cprDrawn = "0";
    }
    updateCprPanel();
  }

  // ------------------------------------------------- tree schematic panel
  //
  // TREE_SCHEMATIC_NAV (read-only navigator): one simplified display-only
  // polyline per ACTIVE labeled centerline path, each labeled with its
  // frozen vessel token plus its assignment-source badge.  The panel follows
  // the CPR panel idioms (figure shell, typed-failure element with
  // role=status, binding-checked async loads).  The client performs NO
  // geometry and NO measurement: the SVG viewBox is a display-only fit of
  // the served coordinates computed at render time, and every per-vessel
  // click submits the EXISTING SELECT_VESSEL action (no grammar change).
  // Selection syncs both directions: the served selected_vessel updates
  // state.selectedVessel here, and every selectVessel reload re-renders
  // this panel alongside the overlay/CPR panels.

  /** The stroke of a schematic vessel path (unselected vs selected). */
  const SCHEMATIC_STROKE = "#a371f7";
  const SCHEMATIC_SELECTED_STROKE = "#ffd23f";

  /** Hide (or show) a schematic element with a REAL attribute.
   *
   * P1: SVGElement does not reflect the `hidden` IDL property into the
   * `hidden` content attribute, so `.tree-schematic-svg[hidden]` never
   * activates from `svg.hidden = true` alone and a later STALE/INVALID/
   * UNAVAILABLE response left the previous vessel paths visible AND
   * clickable.  Set the attribute (the CSS selector's trigger) plus the IDL
   * and an explicit display style, so a refusal always hides stale geometry
   * even if one layer is ignored.
   */
  function setSchematicHidden(element, hide) {
    try {
      element.hidden = hide;
    } catch {
      // SVG without a hidden IDL — the attribute below still applies.
    }
    if (hide) {
      element.setAttribute("hidden", "");
      if (element.style) {
        element.style.display = "none";
      }
    } else {
      element.removeAttribute("hidden");
      if (element.style) {
        element.style.display = "";
      }
    }
  }

  /** Drop stale schematic geometry and its listeners (removing nodes drops
   * their click/keydown listeners, so old paths can never stay clickable).
   */
  function clearSchematicGeometry(svg, vesselGroup) {
    while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
    }
    svg.removeAttribute("viewBox");
    delete svg.dataset.schematicDrawn;
    delete svg.dataset.schematicSelected;
    while (vesselGroup.firstChild) {
      vesselGroup.removeChild(vesselGroup.firstChild);
    }
  }

  /**
   * TREE_SCHEMATIC_NAV F3: leave no clickable schematic behind when the
   * session the geometry was derived for is gone.
   *
   * Drops the displayed document AND disarms the rendered panel in one
   * synchronous step, so the predecessor session's paths, buttons, caption,
   * badges and listeners cannot outlive it.  Removing the nodes is what
   * disarms them: a click listener dies with its element, so nothing that was
   * drawn for the previous session stays reachable.  A no-op before the panel
   * shell exists (nothing has been rendered yet, so nothing can be stale).
   */
  function disarmSchematicPanel() {
    state.schematicDocument = null;
    state.schematicError = null;
    const svg = $("tree-schematic-svg");
    const vesselGroup = $("tree-schematic-vessels");
    if (!svg || !vesselGroup) {
      return;
    }
    clearSchematicGeometry(svg, vesselGroup);
    setSchematicHidden(svg, true);
    setSchematicHidden(vesselGroup, true);
    const caption = $("tree-schematic-caption");
    if (caption) {
      delete caption.dataset.schematicCaption;
      caption.textContent = "Tree schematic: unavailable";
    }
    const failure = $("tree-schematic-error");
    if (failure) {
      failure.hidden = true;
      failure.textContent = "";
      delete failure.dataset.schematicError;
    }
  }

  function renderSchematicPanelShell() {
    const viewports = $("viewports");
    if (!viewports || $("tree-schematic-panel")) {
      return;
    }
    const panel = doc.createElement("figure");
    panel.className = "viewport tree-schematic-panel";
    panel.id = "tree-schematic-panel";
    const svg = doc.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.id = "tree-schematic-svg";
    svg.classList.add("tree-schematic-svg");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Coronary tree schematic");
    setSchematicHidden(svg, true);
    const vesselGroup = doc.createElement("span");
    vesselGroup.id = "tree-schematic-vessels";
    vesselGroup.className = "tree-schematic-vessels";
    setSchematicHidden(vesselGroup, true);
    vesselGroup.setAttribute("role", "group");
    vesselGroup.setAttribute("aria-label", "Tree schematic vessels");
    const caption = doc.createElement("figcaption");
    caption.id = "tree-schematic-caption";
    caption.textContent = "Tree schematic: loading…";
    const failure = doc.createElement("span");
    failure.id = "tree-schematic-error";
    failure.className = "overlay-error";
    failure.hidden = true;
    failure.setAttribute("role", "status");
    panel.appendChild(svg);
    panel.appendChild(vesselGroup);
    panel.appendChild(caption);
    panel.appendChild(failure);
    viewports.appendChild(panel);
  }

  function updateSchematicPanel() {
    const panel = $("tree-schematic-panel");
    const svg = $("tree-schematic-svg");
    const vesselGroup = $("tree-schematic-vessels");
    const caption = $("tree-schematic-caption");
    const failure = $("tree-schematic-error");
    if (!panel || !svg || !vesselGroup) {
      return;
    }
    const schematic = state.schematicDocument;
    const error = state.schematicError;
    if (failure) {
      failure.hidden = !error;
      failure.textContent = error ? `Tree schematic: ${error.code}` : "";
      if (error) {
        failure.dataset.schematicError = error.code;
      } else {
        delete failure.dataset.schematicError;
      }
    }
    if (!schematic) {
      // Refusal (STALE/INVALID/UNAVAILABLE): hide with a REAL attribute AND
      // drop the obsolete paths/buttons/listeners — never leave the previous
      // geometry visible or clickable behind the typed error.
      clearSchematicGeometry(svg, vesselGroup);
      setSchematicHidden(svg, true);
      setSchematicHidden(vesselGroup, true);
      if (caption) {
        caption.textContent = error
          ? `Tree schematic: ${error.code}`
          : "Tree schematic: unavailable";
      }
      return;
    }
    const labels = schematicVesselLabels(schematic);
    const selected =
      (typeof state.selectedVessel === "string" && state.selectedVessel) ||
      (typeof schematic.selected_vessel === "string" && schematic.selected_vessel) ||
      "";
    // TREE_SCHEMATIC_NAV F3: every control rendered below belongs to the
    // session this document was derived for.  Capture that identity once, at
    // render time, and re-check it at interaction time: a node that outlives
    // its session must never submit a SELECT_VESSEL into the session that
    // replaced it.  The check is on the DOCUMENT's own identity, not on the
    // selection, so it holds even for a retained node whose selection is
    // still displayed.
    const derivedSessionId = typeof schematic.session_id === "string" ? schematic.session_id : "";
    const submitFor = (token) => {
      if (derivedSessionId && derivedSessionId !== state.sessionId) {
        return;
      }
      selectVessel(token);
    };
    if (caption) {
      caption.textContent = TREE_SCHEMATIC_CAPTION;
      caption.dataset.schematicCaption = TREE_SCHEMATIC_CAPTION;
    }
    const viewBox = schematicViewBox(schematic);
    while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
    }
    if (viewBox) {
      svg.setAttribute("viewBox", viewBox.join(" "));
    } else {
      svg.removeAttribute("viewBox");
    }
    setSchematicHidden(svg, labels.length === 0);
    svg.dataset.schematicDrawn = String(labels.length);
    if (selected) {
      svg.dataset.schematicSelected = selected;
    } else {
      delete svg.dataset.schematicSelected;
    }
    for (const entry of Array.isArray(schematic.vessels) ? schematic.vessels : []) {
      const token = entry && typeof entry.vessel_label === "string" ? entry.vessel_label : "";
      if (!token) {
        continue;
      }
      const polyline = Array.isArray(entry.polyline) ? entry.polyline : [];
      const segments = [];
      for (const pair of polyline) {
        if (!Array.isArray(pair) || pair.length !== 2) {
          continue;
        }
        const u = Number(pair[0]);
        const v = Number(pair[1]);
        if (!Number.isFinite(u) || !Number.isFinite(v)) {
          continue;
        }
        segments.push(`${segments.length === 0 ? "M" : "L"}${u} ${v}`);
      }
      if (segments.length === 0) {
        continue;
      }
      const source =
        typeof entry.assignment_source === "string" && entry.assignment_source
          ? entry.assignment_source
          : "UNRECORDED";
      const path = doc.createElementNS("http://www.w3.org/2000/svg", "path");
      path.dataset.vessel = token;
      path.classList.add("tree-schematic-path");
      path.setAttribute("d", segments.join(" "));
      path.setAttribute("fill", "none");
      path.setAttribute(
        "stroke",
        token === selected ? SCHEMATIC_SELECTED_STROKE : SCHEMATIC_STROKE
      );
      path.setAttribute("stroke-width", token === selected ? "3" : "2");
      path.setAttribute("tabindex", "0");
      path.setAttribute("role", "button");
      path.setAttribute("aria-label", `${token} · ${source}`);
      path.setAttribute("aria-pressed", token === selected ? "true" : "false");
      const title = doc.createElementNS("http://www.w3.org/2000/svg", "title");
      title.textContent = `${token} · ${source}`;
      path.appendChild(title);
      path.addEventListener("click", () => {
        submitFor(token);
      });
      path.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          submitFor(token);
        }
      });
      svg.appendChild(path);
    }
    while (vesselGroup.firstChild) {
      vesselGroup.removeChild(vesselGroup.firstChild);
    }
    setSchematicHidden(vesselGroup, labels.length === 0);
    for (const token of labels) {
      const entry =
        (Array.isArray(schematic.vessels) ? schematic.vessels : []).find(
          (item) => item && item.vessel_label === token
        ) || {};
      const source =
        typeof entry.assignment_source === "string" && entry.assignment_source
          ? entry.assignment_source
          : "UNRECORDED";
      const vesselButton = doc.createElement("button");
      vesselButton.type = "button";
      vesselButton.dataset.vessel = token;
      vesselButton.textContent = `${token.replace("VESSEL:", "")} · ${source}`;
      vesselButton.setAttribute("aria-pressed", token === selected ? "true" : "false");
      vesselButton.addEventListener("click", () => {
        submitFor(token);
      });
      vesselGroup.appendChild(vesselButton);
    }
  }

  /** Load (or clear) the tree-schematic navigator document. */
  async function loadSchematic(binding) {
    const sessionId = state.sessionId;
    if (!sessionId) {
      return;
    }
    // F1-race (R3): a superseded navigation never fetches and never applies.
    // TREE_SCHEMATIC_NAV F2: nor does a load issued before the canonical
    // centerline set was replaced.
    if (!panelSetIsCurrent(binding)) {
      return;
    }
    // TSN_R5_F1 entry fence: an obsolete recovery/settlement epoch never claims an ordinal nor fetches.
    if ((binding.recoverySeq || 0) !== state.selectionRecoverySeq) {
      return;
    }
    // TREE_SCHEMATIC_NAV F2: claim this panel's next request ordinal.  Taken
    // BEFORE the fetch, it identifies the load even across the await below.
    // Only the newest schematic request of this client may paint, so a slower
    // earlier refresh can never overwrite a newer one - nor resurrect the
    // geometry a newer refresh has already replaced with a typed refusal.
    const request = (state.schematicReqSeq += 1);
    const requestIsCurrent = () => request === state.schematicReqSeq;
    renderSchematicPanelShell();
    let schematic = null;
    let failure = null;
    try {
      schematic = await loadTreeSchematic(source, sessionId);
    } catch (error) {
      // STALE/INVALID refusals surface a VISIBLE typed failure (never a
      // silent fallback); every refusal (STALE/INVALID/UNAVAILABLE) clears
      // stale geometry via updateSchematicPanel (P1: no visible/clickable
      // predecessor paths behind the error).
      failure = {
        code: String((error && error.code) || "TREE_SCHEMATIC_ERROR"),
        message: errorText(error),
      };
    }
    // F1-race (R3): the response is applied only while the navigation that
    // started it is still the current one (the R3 stale-response variant).
    // TREE_SCHEMATIC_NAV F2: and only while the ACTIVE centerline set it was
    // issued under is still the canonical one, and only while it is still the
    // newest schematic request - so a superseded document and a superseded
    // response both die here instead of repainting the panel.
    if (!panelSetIsCurrent(binding) || !requestIsCurrent() || (binding.recoverySeq || 0) !== state.selectionRecoverySeq) {
      return;
    }
    state.schematicDocument = schematic;
    state.schematicError = failure;
    if (schematic && typeof schematic.selected_vessel === "string") {
      // TSN_R4_F1: uniform fence — the schematic route takes no vessel param
      // so it is the authority source, and the recovery's own authority read
      // carries the fresh fence value so it passes; older continuations bail
      // at the guard already, and a same-fence stale stored selection is
      // impossible (stored selection only changes on accept).
      if ((binding.recoverySeq || 0) !== state.selectionRecoverySeq) {
        /* fenced: keep document, never adopt its label */
      } else {
        state.selectedVessel = schematic.selected_vessel;
      }
    } else if (
      // TSN_R4_F1 (R10-F1 null-path): the authority wins even for explicit
      // null.  A FRESH-EPOCH document that applied above (guard passed) and
      // carries an explicit null selected_vessel clears a geometry-derived
      // label (unfiltered overlay/CPR can no longer manufacture one).  A
      // fetch-failure document (schematic === null) leaves state untouched —
      // fail closed on display, never contradictory UI.
      schematic !== null &&
      schematic !== undefined &&
      Object.prototype.hasOwnProperty.call(schematic, "selected_vessel") &&
      schematic.selected_vessel !== undefined &&
      schematic.selected_vessel === null &&
      state.selectedVessel !== null
    ) {
      state.selectedVessel = null;
    }
    if (failure) {
      setStatus(`${failure.code}: ${failure.message}`);
    }
    updateSchematicPanel();
    updateVesselPicker();
    // TREE_SCHEMATIC_NAV F1 (R3-F1): hand back the document THIS load applied,
    // so a caller that needs the server's STORED selection can tell "this load
    // applied" from "this load bailed and left an older document in state"
    // (a bail returns nothing, above).  This route is the only one that cannot
    // be pointed at a requested vessel, so its served ``selected_vessel`` is
    // the only rollback authority.  A return value, not new state: every other
    // caller of this function ignores it.
    return schematic;
  }

  // -------------------------------------------------------------- lifecycle

  async function init() {
    renderViewportShells();
    renderPresetButtons();
    renderWlCustomControls();
    renderOverlayControls();
    renderSeedControls();
    updateSeedControls();
    renderSyncControls();
    updateSyncControls();
    updateOverlayToggle();
    updateFocusClasses();
    doc.body.dataset.clientState = "loading";
    const select = $("session-select");
    const refreshButton = $("refresh");
    const clearButton = $("clear-wl");
    const qcButton = $("open-qc");
    const submitButton = $("submit-correction");
    const gradeSelect = $("correction-grade");
    if (select) {
      select.addEventListener("change", (event) => {
        state.stickySessionId = event.target.value;
        void openSession(event.target.value);
      });
    }
    if (refreshButton) {
      refreshButton.addEventListener("click", () => {
        void refreshSession();
      });
    }
    if (clearButton) {
      clearButton.addEventListener("click", () => {
        void clearWl();
      });
    }
    if (qcButton) {
      qcButton.addEventListener("click", () => {
        void openQcPanel();
      });
    }
    if (submitButton) {
      submitButton.addEventListener("click", () => {
        void submitCorrection();
      });
    }
    if (gradeSelect) {
      gradeSelect.addEventListener("change", () => {
        updateCorrectionControls();
      });
    }
    let sessions = [];
    try {
      sessions = await source.getSessions();
    } catch (error) {
      doc.body.dataset.clientState = "error";
      setStatus(errorText(error));
      return;
    }
    if (select) {
      select.replaceChildren();
      for (const sessionId of sessions) {
        const option = doc.createElement("option");
        option.value = sessionId;
        option.textContent = sessionId;
        select.appendChild(option);
      }
    }
    const sessionId = pickPreferredSession(sessions, {
      stickySessionId: state.stickySessionId,
    });
    if (!sessionId) {
      doc.body.dataset.clientState = "empty";
      setStatus("No governed sessions are available.");
      return;
    }
    await openSession(sessionId);
  }

  return {
    init,
    openSession,
    refreshSession,
    openQcPanel,
    applyPreset,
    applyCustomWl,
    clearWl,
    submitCorrection,
    toggleOverlayVisibility,
    toggleOverlayMode,
    toggleSeedArm,
    submitSeed,
    clearSeed,
    detachSeedWait,
    recheckSeedWait,
    seedWaitIsWaiting,
    placeSyncCursor,
    clearSyncCursor,
    refreshSyncCursor,
    loadOverlay,
    loadCpr,
    registry,
    state,
  };
}

export default { createClientApp };
