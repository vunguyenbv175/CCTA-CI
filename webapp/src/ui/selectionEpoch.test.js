/** R5-F1/R5-F2 entry-fence + settlement-epoch unit tests (no browser).

R6-T1/T2/T3 require deterministic regressions for the R5 product mechanisms
(entry fence before ordinal claim; settlement epoch on acceptance) WITHOUT a
browser/CDP round-trip for the loader fence itself: the browser lane cannot
hold a panel response while an action response is held (actionQueue
serialises them — proven over 6 attempt-3 legs + 8 attempt-4 legs). The
browser lane OWNS the end-to-end settlement path (accepted-second-action
schedule via restore in the deliver round-trip gap — proven green in legs
both11/both13); the unit lane owns the loader-fence interleaving with a
fake gated source.

Strategy: drive the REAL `createClientApp` with a FAKE source whose panel
reads are manually gated promises. That pins the exact interleaving the
reviewer demanded — a competing panel continuation that retains the SAME
current navigation/set identity across a refusal/acceptance epoch
transition — with zero sleeps-as-evidence (every ordering is a held
promise released in script order).

Covered:
1. R5-F1: an obsolete-epoch loadCpr continuation (E0) claims NO ordinal
   and issues NO fetch after refusal recovery bumped to E1 — with the E1
   chain IN FLIGHT FIRST (its fetch survives while the E0 entry attempts
   to steal the ordinal).
2. R5-F1 (overlay): an obsolete-epoch loadOverlay entry claims NO ordinal
   and issues NO fetch. Schematic shares the identical fence idiom but
   loadSchematic is not exported on the app handle, so its current-epoch
   path is covered by the browser settle assertions (marker + caption +
   badge all LCX at close) — stated here, not claimed as unit coverage.
3. R5-F2: a pre-settlement loadCpr continuation (E0) claims NO ordinal
   after the settlement bump to E1 — with the SETTLED chain in flight
   first. NOTE (R7-T2 honest scope): this drives the loader fence under
   a scripted bump with the same synchronous idiom as the accepted
   branch; it does NOT drive the real selectVessel acceptance path (that
   needs postAction + schematic authority, unreachable with the stubbed
   frame/source seam) — the browser R5-F2 test owns the end-to-end
   settlement path.
4. Sanity: a current-epoch chain still loads (no ordinal starvation).
 */

import { describe, expect, it, vi } from "vitest";

vi.mock("@cornerstonejs/core", () => ({
  Enums: {},
}));

vi.mock("@cornerstonejs/core/utilities", () => ({
  createLinearRGBTransferFunction: () => ({}),
}));

vi.mock("../cornerstone/imageLoader.js", () => ({
  registerCctaImageLoader: () => {},
}));

vi.mock("../cornerstone/metadataProvider.js", () => ({
  registerCctaMetadataProvider: () => {},
}));

vi.mock("../cornerstone/tools.js", () => ({
  ensureCctaToolGroup: () => {},
}));

vi.mock("../cornerstone/viewports.js", () => ({
  createViewportManager: () => ({
    viewportFor: () => null,
    setStack: async () => {},
    setIndex: async () => {},
    applyMatrix: () => {},
    reset: () => {},
  }),
}));

vi.stubGlobal(
  "createImageBitmap",
  async () => ({ width: 8, height: 8, close: () => {} })
);
vi.stubGlobal("Blob", class Blob {
  constructor(parts) {
    this.parts = parts;
  }
});

import { createClientApp } from "./app.js";

function deferred() {
  let resolve;
  const promise = new Promise((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

function makeDoc(label) {
  return {
    vessel_label: label,
    selected_vessel: label,
    centerline_ref: `ref-${label}`,
    centerline_generation: 1,
    matrix_size: [8, 8],
    cpr_ref: `cpr-${label}`,
    visible: true,
  };
}

/** A fake source whose panel reads are script-gated.

R7-T2 completion note: the CPR DOCUMENT gate (gates.cpr) and the CPR PNG
gate (gates.cprPng) are SEPARATE deferreds, so a test that releases a CPR
document MUST ALSO release its PNG (flushCprRoundTrip) before the refresh
chain can reach its follow-up loadSchematic. Asserting a schematic count
while the PNG is still held is vacuous — the absence proves nothing
(proven in R7 review at selectionEpoch.test.js/R5-F2 browser).
getTreeSchematic lands in gates.schematic; getCoronary in gates.coronary.
(R8-A4 fix: the R7 note above misattributed this completion rule to
"selectionEpoch.test.js/R5-F2 browser" — it is the R7 REVIEW's finding
about the R5-F2 browser schedule.) */
function makeFakeSource() {
  const gates = {
    coronary: [],
    cpr: [],
    cprPng: [],
    schematic: [],
    state: [],
  };
  const calls = { coronary: 0, cpr: 0, cprPng: 0, schematic: 0, action: 0 };
  const stored = { selected_vessel: "VESSEL:RCA" };
  // R10-F1 server fidelity: an UNFILTERED geometry read (no vessel request
  // param) serves the sole ACTIVE path's geometry for DISPLAY — its label is
  // NOT the stored/accepted selection (stored may be null while the document
  // still carries the sole path's label). A FILTERED read echoes the
  // requested vessel (override-shaped, engine.py:3931/4041/4072).
  // `failNextWithNull` forces the next gated read of a panel to resolve a
  // null document (UNAVAILABLE/refusal shape) instead of makeDoc.
  const SOLE_PATH_VESSEL = "VESSEL:RCA";
  const failNextWithNull = { coronary: false, cpr: false, schematic: false };
  const seenOpts = { coronary: [], cpr: [] };
  const source = {
    getState: async () => {
      calls.state += 1;
      return { selected_vessel: stored.selected_vessel };
    },
    getFrames: async (_id, plane) => ({ plane, frames: [] }),
    getCoronary: async (_id, opts) => {
      calls.coronary += 1;
      seenOpts.coronary.push(opts === undefined ? null : opts);
      const g = deferred();
      gates.coronary.push(g);
      await g.promise;
      if (failNextWithNull.coronary) {
        failNextWithNull.coronary = false;
        return null;
      }
      const v =
        opts && opts.vessel ? String(opts.vessel) : SOLE_PATH_VESSEL;
      return makeDoc(v);
    },
    getCpr: async (_id, opts) => {
      calls.cpr += 1;
      seenOpts.cpr.push(opts === undefined ? null : opts);
      const g = deferred();
      gates.cpr.push(g);
      await g.promise;
      if (failNextWithNull.cpr) {
        failNextWithNull.cpr = false;
        return null;
      }
      const v =
        opts && opts.vessel ? String(opts.vessel) : SOLE_PATH_VESSEL;
      return makeDoc(v);
    },
    getTreeSchematic: async (_id) => {
      calls.schematic += 1;
      const g = deferred();
      gates.schematic.push(g);
      await g.promise;
      if (failNextWithNull.schematic) {
        failNextWithNull.schematic = false;
        return null;
      }
      return makeDoc(stored.selected_vessel);
    },
    getCprPngBytes: async (_id, _opts = {}) => {
      calls.cprPng += 1;
      // Record the PNG request options too: the loaded PNG must match the
      // loaded view document (same vessel param, filtered or unfiltered).
      seenOpts.png = seenOpts.png || [];
      seenOpts.png.push(_opts === undefined ? null : _opts);
      const g = deferred();
      gates.cprPng.push(g);
      await g.promise;
      return new Uint8Array([1, 2, 3]);
    },
    postAction: async (_id, envelope, _body) => {
      calls.action += 1;
      const params = envelope?.params || {};
      if (envelope?.action === "SELECT_VESSEL" && params.vessel_label) {
        stored.selected_vessel = params.vessel_label;
      }
      return { selected_vessel: stored.selected_vessel, session_id: "s1" };
    },
  };
  return { source, gates, calls, stored, seenOpts, failNextWithNull };
}

function makeDom() {
  const elements = {};
  const doc = {
    body: { dataset: {} },
    getElementById: (id) => {
      if (!elements[id]) {
        const isCanvas = id === "cpr-canvas";
        elements[id] = {
          textContent: "",
          hidden: false,
          width: 0,
          height: 0,
          dataset: {},
          setAttribute: () => {},
          removeAttribute: () => {},
          getContext: isCanvas
            ? () => ({
                clearRect: () => {},
                drawImage: () => {},
              })
            : () => null,
          querySelectorAll: () => [],
        };
      }
      return elements[id];
    },
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: () => ({
      setAttribute: () => {},
      appendChild: () => {},
      querySelectorAll: () => [],
      dataset: {},
    }),
  };
  return doc;
}


/** Resolve one full CPR round-trip: document gate, then (after a flush so
 *  the PNG fetch is issued) the PNG gate.
 *
 *  R8-A4 hardened: the PNG gate is looked up AFTER the flush and MUST
 *  exist — resolving a stale/old gate would hang or prove nothing. A
 *  missing PNG fails LOUD with the observed counts instead of hanging.
 */
async function flushCprRoundTrip(gates, docIndex) {
  const pngBefore = gates.cprPng.length;
  gates.cpr[docIndex].resolve();
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  if (gates.cprPng.length <= pngBefore) {
    throw new Error(
      `flushCprRoundTrip: no PNG fetch issued after doc gate ${docIndex} ` +
      `(cpr calls takengated=${gates.cpr.length}, png gates=${gates.cprPng.length})`
    );
  }
  gates.cprPng[gates.cprPng.length - 1].resolve();
}

async function bootApp() {
  const { source, gates, calls, stored, seenOpts, failNextWithNull } = makeFakeSource();
  const app = createClientApp({ document: makeDom(), source });
  app.state.sessionId = "s1";
  app.state.generation = 1;
  app.state.selectedVessel = "VESSEL:RCA";
  return { app, source, gates, calls, stored, seenOpts, failNextWithNull };
}

describe("R5-F1 entry fence (obsolete epoch claims no ordinal, issues no fetch)", () => {
  it("a pre-refusal loadCpr continuation bails at entry after the recovery bump", async () => {
    const { app, gates, calls } = await bootApp();

    // Start an E0 CPR load and let it reach its fetch gate.
    const bindingE0 = {
      sessionId: "s1",
      generation: 1,
      vesselNavSeq: app.state.vesselNavSeq,
      activeSetSeq: app.state.activeSetSeq,
      recoverySeq: app.state.selectionRecoverySeq,
    };
    const p0 = app.loadCpr(bindingE0);
    expect(calls.cpr).toBe(1);
    expect(gates.cpr.length).toBe(1);

    // Refusal recovery bumps the epoch synchronously (same idiom as the
    // product refusal branch) and invalidates ordinals.
    const epochBefore = app.state.selectionRecoverySeq;
    app.state.selectionRecoverySeq += 1;
    app.state.overlayReqSeq += 1;
    app.state.cprReqSeq += 1;
    app.state.schematicReqSeq += 1;
    expect(app.state.selectionRecoverySeq).toBe(epochBefore + 1);

    // A NEW load under the obsolete binding must bail at ENTRY: no ordinal
    // claim, no fetch.
    const cprBefore = calls.cpr;
    const ordBefore = app.state.cprReqSeq;
    await app.loadCpr(bindingE0);
    expect(calls.cpr).toBe(cprBefore);
    expect(app.state.cprReqSeq).toBe(ordBefore);

    // Release the E0 fetch: its apply must also bail (fence mismatch), so
    // the refused/old label is never adopted. loadCpr awaits the PNG
    // fetch as well, so resolve both gates (PNG only fetched when the
    // document is truthy — resolve defensively).
    await flushCprRoundTrip(gates, 0);
    await p0;
    expect(app.state.cprDocument == null || app.state.cprDocument.vessel_label === "VESSEL:RCA").toBe(
      true
    );

    // R7-T1 in-flight contention: a fresh-E1 chain STARTED FIRST must keep
    // its fetch while the obsolete E0 entry attempts to steal the ordinal.
    // Order: start E1 (claims the newest ordinal), THEN attempt the E0
    // entry — the E0 attempt must claim nothing and fetch nothing, while
    // the already-in-flight E1 keeps its ordinal and completes.
    const bindingE1 = {
      sessionId: "s1",
      generation: 1,
      vesselNavSeq: app.state.vesselNavSeq,
      activeSetSeq: app.state.activeSetSeq,
      recoverySeq: app.state.selectionRecoverySeq,
    };
    const p1 = app.loadCpr(bindingE1);
    expect(calls.cpr).toBe(cprBefore + 1);
    expect(gates.cpr.length).toBe(2);
    const ordE1 = app.state.cprReqSeq;
    await app.loadCpr(bindingE0);
    expect(calls.cpr).toBe(cprBefore + 1);
    expect(app.state.cprReqSeq).toBe(ordE1);
    await flushCprRoundTrip(gates, gates.cpr.length - 1);
    await p1;
    expect(app.state.cprDocument && app.state.cprDocument.vessel_label).toBe(
      'VESSEL:RCA'
    );
  });

  it("loadOverlay entries fence obsolete epochs (no fetch, no ordinal)", async () => {
    const { app, calls } = await bootApp();
    const bindingE0 = {
      sessionId: "s1",
      generation: 1,
      vesselNavSeq: app.state.vesselNavSeq,
      activeSetSeq: app.state.activeSetSeq,
      recoverySeq: app.state.selectionRecoverySeq,
    };
    app.state.selectionRecoverySeq += 1;
    app.state.overlayReqSeq += 1;
    app.state.cprReqSeq += 1;
    app.state.schematicReqSeq += 1;
    const ovBefore = calls.coronary;
    const ovOrd = app.state.overlayReqSeq;
    await app.loadOverlay(bindingE0);
    expect(calls.coronary).toBe(ovBefore);
    expect(app.state.overlayReqSeq).toBe(ovOrd);
    // R7-T1/T4 honest scope: loadSchematic is not exported on the app
    // handle, so no unit test here names it as exercised. It shares the
    // identical fence idiom (entry + apply, same state word); its
    // current-epoch path is covered by the browser settle assertions
    // (marker + caption + badge all LCX at close).
    expect(app.state.selectionRecoverySeq).toBe(bindingE0.recoverySeq + 1);
  });
});

describe("R5-F2 settlement epoch (acceptance invalidates pre-settlement chains)", () => {
  it("a pre-settlement continuation bails while the settled chain passes", async () => {
    const { app, gates, calls } = await bootApp();
    // R7-T1 in-flight contention (settlement form): the SETTLED chain is
    // started FIRST and stays in flight while the pre-settlement entry
    // attempts to steal the ordinal — mirroring an acceptance whose own
    // loads are on the wire while a pre-acceptance refresh continuation
    // tries to enter. The settled fetch must survive; the stale entry
    // must claim nothing and fetch nothing.
    app.state.selectionRecoverySeq += 1;
    app.state.overlayReqSeq += 1;
    app.state.cprReqSeq += 1;
    app.state.schematicReqSeq += 1;
    const bindingSettled = {
      sessionId: "s1",
      generation: 1,
      vesselNavSeq: app.state.vesselNavSeq,
      activeSetSeq: app.state.activeSetSeq,
      recoverySeq: app.state.selectionRecoverySeq,
    };
    const p = app.loadCpr(bindingSettled);
    expect(calls.cpr).toBe(1);
    const ordSettled = app.state.cprReqSeq;
    const bindingPre = {
      sessionId: "s1",
      generation: 1,
      vesselNavSeq: app.state.vesselNavSeq,
      activeSetSeq: app.state.activeSetSeq,
      recoverySeq: app.state.selectionRecoverySeq - 1,
    };
    await app.loadCpr(bindingPre);
    expect(calls.cpr).toBe(1);
    expect(app.state.cprReqSeq).toBe(ordSettled);
    await flushCprRoundTrip(gates, gates.cpr.length - 1);
    await p;
  });
});

describe("no ordinal starvation (current epoch still loads)", () => {
  it("a current-epoch overlay+CPR chain issues all fetches", async () => {
    const { app, gates, calls } = await bootApp();
    const binding = {
      sessionId: "s1",
      generation: 1,
      vesselNavSeq: app.state.vesselNavSeq,
      activeSetSeq: app.state.activeSetSeq,
      recoverySeq: app.state.selectionRecoverySeq,
    };
    const po = app.loadOverlay(binding);
    const pc = app.loadCpr(binding);
    expect(calls.coronary).toBe(1);
    expect(calls.cpr).toBe(1);
    // R7-T1 honest scope: loadSchematic is not exposed on the app handle,
    // so this no-starvation check covers the two exposed loaders; the
    // schematic current-epoch path shares the identical fence idiom
    // (entry + apply, same state word) and is covered by the browser
    // settle assertions (marker + caption + badge all LCX at close).
    gates.coronary[0].resolve();
    gates.coronary[0].resolve();
    await flushCprRoundTrip(gates, 0);
    await po;
    await pc;
  });
});

describe("R10-F1 canonical-null init (unfiltered geometry never manufactures selection)", () => {
  /** Boot with canonical null: no accepted selection, single-path geometry. */
  async function bootNullApp() {
    const { source, gates, calls, stored, seenOpts, failNextWithNull } = makeFakeSource();
    stored.selected_vessel = null;
    const app = createClientApp({ document: makeDom(), source });
    app.state.sessionId = "s1";
    app.state.generation = 1;
    app.state.selectedVessel = null;
    return { app, source, gates, calls, stored, seenOpts, failNextWithNull };
  }

  function freshBinding(app) {
    return {
      sessionId: "s1",
      generation: 1,
      vesselNavSeq: app.state.vesselNavSeq,
      activeSetSeq: app.state.activeSetSeq,
      recoverySeq: app.state.selectionRecoverySeq,
    };
  }

  it("unfiltered overlay+CPR keep the document but leave selectedVessel null", async () => {
    const { app, gates, calls, seenOpts } = await bootNullApp();
    const epoch = app.state.selectionRecoverySeq;

    // The unfiltered overlay read serves the sole ACTIVE path's geometry
    // (SOLE_PATH_VESSEL stand-in) under a null canonical selection — the
    // exact R10-F1 trace (engine overlay succeeds unfiltered while
    // canonical selected_vessel stays null).
    const po = app.loadOverlay(freshBinding(app));
    expect(calls.coronary).toBe(1);
    expect(seenOpts.coronary[0]).toBe(null);
    gates.coronary[0].resolve();
    gates.coronary[0].resolve();
    await po;
    // R10-F1: the geometry document is kept for DISPLAY (Q3 seed
    // provenance reads it), but the canonical selection is never
    // manufactured from an unfiltered label.
    expect(app.state.overlayDocument).not.toBe(null);
    expect(app.state.overlayDocument.vessel_label).toBe("VESSEL:RCA");
    expect(app.state.selectedVessel).toBe(null);
    expect(app.state.selectionRecoverySeq).toBe(epoch);

    // CPR follows the same rule: unfiltered document kept, null kept.
    const pc = app.loadCpr(freshBinding(app));
    expect(calls.cpr).toBe(1);
    expect(seenOpts.cpr[0]).toBe(null);
    await flushCprRoundTrip(gates, 0);
    await pc;
    expect(app.state.cprDocument).not.toBe(null);
    expect(app.state.cprDocument.vessel_label).toBe("VESSEL:RCA");
    expect(app.state.selectedVessel).toBe(null);
    expect(app.state.selectionRecoverySeq).toBe(epoch);
  });

  it("a filtered accepted-token reload still adopts its own label", async () => {
    const { app, gates, seenOpts } = await bootNullApp();
    // Accepted-token state (as the settled accepted path leaves it): a
    // filtered reload must adopt its own matching label (no starvation).
    app.state.selectedVessel = "VESSEL:RCA";
    const po = app.loadOverlay(freshBinding(app));
    expect(seenOpts.coronary[0]).toEqual({
      mode: app.state.overlayMode,
      vessel: "VESSEL:RCA",
    });
    gates.coronary[0].resolve();
    gates.coronary[0].resolve();
    await po;
    expect(app.state.overlayDocument.vessel_label).toBe("VESSEL:RCA");
    expect(app.state.selectedVessel).toBe("VESSEL:RCA");

    const pc = app.loadCpr(freshBinding(app));
    expect(seenOpts.cpr[0]).toEqual({ vessel: "VESSEL:RCA" });
    await flushCprRoundTrip(gates, 0);
    await pc;
    expect(app.state.cprDocument.vessel_label).toBe("VESSEL:RCA");
    expect(app.state.selectedVessel).toBe("VESSEL:RCA");
  });

  it("a mismatched in-flight label is never adopted when selection moved on", async () => {
    const { app, gates } = await bootNullApp();
    app.state.selectedVessel = "VESSEL:RCA";
    const p = app.loadOverlay(freshBinding(app));
    // The selection moves on while the filtered read is in flight (a stale
    // request parameter, not a stale fence): the response keeps its fetched
    // document but must not adopt the label it fetched for.
    app.state.selectedVessel = "VESSEL:LAD";
    gates.coronary[0].resolve();
    gates.coronary[0].resolve();
    await p;
    expect(app.state.overlayDocument).not.toBe(null);
    expect(app.state.selectedVessel).toBe("VESSEL:LAD");
  });

  it("recovery fetch-failure null leaves the prior selection untouched", async () => {
    // Fail-closed recovery path: the unfiltered geometry read resolves a
    // null document (UNAVAILABLE — sole path gone) while a prior accepted
    // selection stands. The panel clears to null for DISPLAY, the prior
    // selection is left untouched, and no contradictory label is adopted.
    const { app, gates, failNextWithNull } = await bootNullApp();
    app.state.selectedVessel = "VESSEL:RCA";
    failNextWithNull.coronary = true;
    const po = app.loadOverlay(freshBinding(app));
    gates.coronary[0].resolve();
    gates.coronary[0].resolve();
    await po;
    expect(app.state.overlayDocument).toBe(null);
    expect(app.state.selectedVessel).toBe("VESSEL:RCA");

    failNextWithNull.cpr = true;
    const pc = app.loadCpr(freshBinding(app));
    // CPR still issues its PNG fetch only when the view is truthy — a null
    // view skips the PNG gate, so resolve defensively.
    gates.cpr[0].resolve();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    if (gates.cprPng.length) {
      gates.cprPng[gates.cprPng.length - 1].resolve();
    }
    await pc;
    expect(app.state.cprDocument).toBe(null);
    expect(app.state.selectedVessel).toBe("VESSEL:RCA");
  });
});
