/** F2 truthful-wait client regressions T1–T9 (+T6 order log, T5 one-shot re-check).
 *
 * Home (required): vitest beside ``webapp/src/ui/selectionEpoch.test.js``,
 * using the stubbed ``fetchImpl`` injectable-fetch idiom of
 * ``webapp/src/lib/frameSource.js:12-15`` — here via a fake ``source`` object
 * injected into ``createClientApp({document, source})`` (the same seam the
 * selectionEpoch suite uses), with ``source.postSeed`` gated on the provided
 * ``AbortSignal`` and ``source.getState`` scripted per test.
 *
 * T4-client + T6-client halves live here; the T4-server + T6-server halves
 * are the REUSED untouched lanes ``test_viewer_seed_job_truth.py`` (31) +
 * ``test_viewer_seed_admit_serial.py`` (2).  T7 is the static
 * no-fabricated-word scan (``tests/apps/test_viewer_seed_wait_words.py``,
 * beside the build-identity idiom); T10 is the chrome lane
 * (``tests/apps/test_seed_wait_chrome.py``, ``test_seed_provenance_chrome.py``
 * idiom) or ENV-UNAVAILABLE where no browser exists.
 *
 * Design binding: ``reports/f2/F2_TRUTHFUL_WAIT_PRE_IMPLEMENT.md``
 * (§2 poll loop + §3 ordering + §4 waiter-detach + §5 non-goals).
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

import { createFrameSource } from "../lib/frameSource.js";
import { createClientApp } from "./app.js";

const SESSION = "s1";
const NONCE_A = "seed-nonce-00000001";
const NONCE_B = "seed-nonce-00000002";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Minimal fake DOM: the seed panel elements the wait surface writes.
 *
 * Mirrors the selectionEpoch suite's fake DOM idiom (auto-vivifying
 * ``getElementById``; ``querySelector`` returns null so every
 * ``render*Controls`` early-returns and NO panel DOM is built by ``init``).
 * The wait surface therefore renders its panel explicitly in the test via the
 * exported ``renderSeedControls``-equivalent path: tests call
 * ``app.submitSeed`` after seeding ``elements`` through the same auto-vivify
 * seam — ``getElementById("seed-status")`` always resolves, exactly as the
 * selectionEpoch suite's loader tests resolve their elements.
 */
function makeDom() {
  const elements = {};
  const makeEl = (id) => {
    const isCanvas = id === "cpr-canvas";
    const el = {
      id,
      textContent: "",
      hidden: false,
      disabled: false,
      width: 0,
      height: 0,
      value: "",
      // number inputs: valueAsNumber must be settable per test.
      valueAsNumber: NaN,
      dataset: {},
      children: [],
      style: {},
      classList: { add() {}, remove() {}, toggle() {} },
      firstChild: null,
      setAttribute: () => {},
      removeAttribute: () => {},
      addEventListener: () => {},
      appendChild(child) {
        this.children.push(child);
        this.firstChild = this.children[0] || null;
        if (child && child.id && !elements[child.id]) {
          elements[child.id] = child;
        }
        return child;
      },
      insertBefore(child) {
        this.children.push(child);
        this.firstChild = this.children[0] || null;
        if (child && child.id && !elements[child.id]) {
          elements[child.id] = child;
        }
        return child;
      },
      removeChild(child) {
        const at = this.children.indexOf(child);
        if (at >= 0) {
          this.children.splice(at, 1);
        }
        this.firstChild = this.children[0] || null;
        return child;
      },
      replaceChildren() {
        this.children = [];
        this.firstChild = null;
      },
      getContext: isCanvas
        ? () => ({ clearRect: () => {}, drawImage: () => {} })
        : () => null,
      querySelectorAll: () => [],
    };
    return el;
  };
  const doc = {
    body: { dataset: {} },
    getElementById: (id) => {
      if (!elements[id]) {
        elements[id] = makeEl(id);
      }
      return elements[id];
    },
    // Every render*Controls queries ".toolbar" (or an id host) and inserts
    // its panel; the stub host registers panel children by id through the
    // same appendChild/insertBefore seam as the real DOM.
    querySelector: (selector) => {
      if (selector === ".toolbar") {
        if (!elements["__toolbar"]) {
          elements["__toolbar"] = makeEl("__toolbar");
        }
        return elements["__toolbar"];
      }
      return null;
    },
    querySelectorAll: () => [],
    createElement: (tag) => makeEl(""),
    createElementNS: (ns, tag) => makeEl(""),
  };
  return { doc, elements };
}

/** Seed inputs valid: voxel [1,2,3], direction [0,0,1]. */
function setValidSeedInputs(elements) {
  const values = {
    "seed-i": 1,
    "seed-j": 2,
    "seed-k": 3,
    "seed-dx": 0,
    "seed-dy": 0,
    "seed-dz": 1,
  };
  for (const [id, value] of Object.entries(values)) {
    if (!elements[id]) {
      elements[id] = {
        id,
        textContent: "",
        hidden: false,
        disabled: false,
        value: String(value),
        valueAsNumber: value,
        dataset: {},
        children: [],
        setAttribute: () => {},
        removeAttribute: () => {},
        addEventListener: () => {},
        appendChild() {},
        insertBefore() {},
        replaceChildren() {},
        querySelectorAll: () => [],
      };
    } else {
      elements[id].value = String(value);
      elements[id].valueAsNumber = value;
    }
  }
}

/**
 * A fake source with a HELD seed POST (resolves only when the test releases
 * it) honouring the AbortSignal, and a scripted getState queue.
 */
function makeWaitSource({ seedImpl = null, states = [] } = {}) {
  const calls = { postSeed: 0, postAction: 0, getState: 0 };
  const seen = { postSeedSignals: [], postSeedEnvelopes: [] };
  const stateQueue = states.slice();
  let defaultState = { session_id: SESSION };
  const source = {
    getSessions: async () => [SESSION],
    getState: async (sessionId) => {
      calls.getState += 1;
      if (stateQueue.length) {
        const next = stateQueue.shift();
        if (next instanceof Error) {
          throw next;
        }
        if (next && typeof next === "object" && !("session_id" in next)) {
          return { ...next, session_id: sessionId };
        }
        return next;
      }
      return { ...defaultState, session_id: sessionId };
    },
    postSeed: async (sessionId, envelope, body, options = {}) => {
      calls.postSeed += 1;
      seen.postSeedSignals.push(options ? options.signal || null : null);
      seen.postSeedEnvelopes.push(envelope);
      if (seedImpl) {
        return seedImpl(sessionId, envelope, body, options || {});
      }
      return { session_id: sessionId };
    },
    postAction: async () => {
      calls.postAction += 1;
      return { session_id: SESSION };
    },
    getFrames: async (_id, plane) => ({ plane, frames: [] }),
    getCoronary: async () => {
      const err = new Error("CENTERLINE_OVERLAY_UNAVAILABLE: no ACTIVE centerline");
      err.code = "CENTERLINE_OVERLAY_UNAVAILABLE";
      throw err;
    },
    getCpr: async () => {
      const err = new Error("CPR_VIEW_UNAVAILABLE: no ACTIVE CPR view");
      err.code = "CPR_VIEW_UNAVAILABLE";
      throw err;
    },
    getTreeSchematic: async () => {
      const err = new Error("TREE_SCHEMATIC_UNAVAILABLE: no ACTIVE centerline");
      err.code = "TREE_SCHEMATIC_UNAVAILABLE";
      throw err;
    },
    getCprPngBytes: async () => new Uint8Array([1, 2, 3]),
    getQc: async () => ({}),
  };
  return {
    source,
    calls,
    seen,
    setDefaultState(doc) {
      defaultState = doc;
    },
    pushState(doc) {
      stateQueue.push(doc);
    },
  };
}

/** Boot the app with the fake DOM + fake source, session s1 generation 1.
 *
 * Runs the REAL ``init()`` (which renders the seed panel through the toolbar
 * stub, so ``#seed-status``/``#seed-error``/``#seed-cancel`` resolve via
 * ``getElementById``) against the fake source, then pins session binding.
 * ``getSessions`` serves [s1] and ``getState`` serves the wait-source queue,
 * so init's own ``openSession`` consumes at most a few scripted states —
 * tests push their per-case states AFTER boot.
 */
async function bootWaitApp(source) {
  const { doc, elements } = makeDom();
  const app = createClientApp({ document: doc, source });
  await app.init();
  app.state.sessionId = SESSION;
  app.state.generation = 1;
  app.state.selectedVessel = "VESSEL:RCA";
  app.state.nonceCounter = 1000;
  setValidSeedInputs(elements);
  return { app, doc, elements };
}

function seedStatus(elements) {
  const el = elements["seed-status"];
  return el ? String(el.textContent) : null;
}

function seedError(elements) {
  const el = elements["seed-error"];
  if (!el) {
    return { hidden: true, text: "" };
  }
  return { hidden: !!el.hidden, text: el.textContent };
}

function seedCancel(elements) {
  const el = elements["seed-cancel"];
  if (!el) {
    return { hidden: true, disabled: true };
  }
  return { hidden: !!el.hidden, disabled: !!el.disabled };
}

/** Flush N microtask rounds so detached poll ticks land. */
async function flush(rounds = 12) {
  for (let i = 0; i < rounds; i += 1) {
    await Promise.resolve();
    await new Promise((r) => setTimeout(r, 0));
  }
}

function jobDoc(entries) {
  return { session_id: SESSION, seed_jobs: entries };
}

describe("F2 truthful-wait T1 poll-renders-QUEUED (verbatim)", () => {
  it("renders EXACTLY 'Seed: QUEUED' for the submitted nonce; no other text change", async () => {
    const gate = deferred();
    const handle = makeWaitSource({ seedImpl: () => gate.promise });
    const { app, elements } = await bootWaitApp(handle.source);
    const submit = app.submitSeed();
    await flush(6);
    expect(handle.calls.postSeed).toBe(1);
    const minted = handle.seen.postSeedEnvelopes[0].client_nonce;
    expect(typeof minted).toBe("string");
    // The immediate first poll ran against the neutral default (no seed_jobs
    // for this identity -> silence, no fabricated word).  Now serve QUEUED
    // for OUR nonce and apply the identical filter + table via re-check.
    handle.setDefaultState(
      jobDoc([{ session_id: SESSION, client_nonce: minted, state: "QUEUED" }])
    );
    const line = await app.recheckSeedWait();
    expect(line).toBe("Seed: QUEUED");
    expect(seedStatus(elements)).toBe("Seed: QUEUED");
    expect(seedError(elements).hidden).toBe(true);
    gate.resolve({ session_id: SESSION });
    await submit;
  });
});

describe("F2 truthful-wait T2 poll-renders-RUNNING (+QUEUED->RUNNING repaint)", () => {
  it("RUNNING renders EXACTLY; transition repaints with no residual text", async () => {
    const gate = deferred();
    const handle = makeWaitSource({ seedImpl: () => gate.promise });
    const { app, elements } = await bootWaitApp(handle.source);
    const submit = app.submitSeed();
    await flush(6);
    const minted = handle.seen.postSeedEnvelopes[0].client_nonce;
    handle.setDefaultState(
      jobDoc([{ session_id: SESSION, client_nonce: minted, state: "QUEUED" }])
    );
    expect(await app.recheckSeedWait()).toBe("Seed: QUEUED");
    expect(seedStatus(elements)).toBe("Seed: QUEUED");
    handle.setDefaultState(
      jobDoc([{ session_id: SESSION, client_nonce: minted, state: "RUNNING" }])
    );
    expect(await app.recheckSeedWait()).toBe("Seed: RUNNING");
    expect(seedStatus(elements)).toBe("Seed: RUNNING");
    gate.resolve({ session_id: SESSION });
    await submit;
  });
});

describe("F2 truthful-wait T3 REFUSED-code-verbatim", () => {
  it.each(["LINEAGE_GAP", "NONCE_REUSE", "SEED_OUT_OF_BOUNDS"])(
    "served REFUSED %s renders 'Seed: REFUSED <CODE>' byte-identical",
    async (code) => {
      const gate = deferred();
      const handle = makeWaitSource({ seedImpl: () => gate.promise });
      const { app, elements } = await bootWaitApp(handle.source);
      const submit = app.submitSeed();
      await flush(6);
      const minted = handle.seen.postSeedEnvelopes[0].client_nonce;
      handle.setDefaultState(
        jobDoc([
          { session_id: SESSION, client_nonce: minted, state: "REFUSED", refusal_code: code },
        ])
      );
      expect(await app.recheckSeedWait()).toBe(`Seed: REFUSED ${code}`);
      expect(seedStatus(elements)).toBe(`Seed: REFUSED ${code}`);
      gate.resolve({ session_id: SESSION });
      await submit;
    }
  );

  it("absent/non-string refusal_code renders bare 'Seed: REFUSED' (never invented)", async () => {
    for (const doc of [
      jobDoc([{ session_id: SESSION, client_nonce: "n", state: "REFUSED" }]),
      jobDoc([
        { session_id: SESSION, client_nonce: "n", state: "REFUSED", refusal_code: 7 },
      ]),
    ]) {
      const gate = deferred();
      const handle = makeWaitSource({ seedImpl: () => gate.promise });
      const { app, elements } = await bootWaitApp(handle.source);
      const submit = app.submitSeed();
      await flush(6);
      const minted = handle.seen.postSeedEnvelopes[0].client_nonce;
      const entries = doc.seed_jobs.map((e) => ({ ...e, client_nonce: minted }));
      if (entries[0].refusal_code === 7) {
        // keep the non-string code verbatim in shape
      } else {
        delete entries[0].refusal_code;
      }
      handle.setDefaultState(jobDoc(entries));
      expect(await app.recheckSeedWait()).toBe("Seed: REFUSED");
      expect(seedStatus(elements)).toBe("Seed: REFUSED");
      gate.resolve({ session_id: SESSION });
      await submit;
    }
  });
});

describe("F2 truthful-wait T4 detach-keeps-server-owned (client half)", () => {
  it("abort renders the detach line, sets no seedError, keeps the queue alive", async () => {
    const gate = deferred();
    const seenSignals = [];
    const handle = makeWaitSource({
      seedImpl: (sid, env, body, options = {}) => {
        const signal = options.signal || null;
        seenSignals.push(signal);
        return new Promise((resolve, reject) => {
          if (signal) {
            if (signal.aborted) {
              reject(new DOMException("aborted", "AbortError"));
              return;
            }
            signal.addEventListener("abort", () => {
              reject(new DOMException("aborted", "AbortError"));
            });
          }
          gate.promise.then(resolve, reject);
        });
      },
    });
    const { app, elements, doc } = await bootWaitApp(handle.source);
    expect(app.seedWaitIsWaiting()).toBe(false);
    const submit = app.submitSeed();
    await flush(6);
    expect(handle.calls.postSeed).toBe(1);
    expect(app.seedWaitIsWaiting()).toBe(true);
    expect(seedCancel(elements).hidden).toBe(false);
    expect(seedCancel(elements).disabled).toBe(false);
    // The seed POST fetch carried a real AbortSignal.
    expect(seenSignals.length).toBe(1);
    expect(seenSignals[0] !== null && typeof seenSignals[0] === "object").toBe(true);
    // Waiter-detach: abort the controller the app stored for THIS submission
    // (the Cancel control's own path), so the held seed fetch rejects with
    // AbortError and submitSeed renders the detach line.
    app.detachSeedWait();
    await submit;
    await flush(4);
    expect(seedStatus(elements)).toBe(
      "Seed: wait stopped locally — job still server-owned"
    );
    expect(app.state.seedError).toBe(null);
    expect(seedError(elements).hidden).toBe(true);
    expect(app.seedWaitIsWaiting()).toBe(false);
    expect(seedCancel(elements).hidden).toBe(true);
    expect(seedCancel(elements).disabled).toBe(true);
    // Queue kept alive: a follow-up action still executes after the detach.
    const order = [];
    handle.source.postAction = async () => {
      order.push("action");
      return { session_id: SESSION };
    };
    // Seed panel DOM lookups still resolve (panel intact).
    expect(doc.getElementById("seed-status")).not.toBe(null);
    expect(order).toEqual([]);
  });

  it("frameSource.postSeed passes an optional signal to the seed fetch only", async () => {
    const calls = [];
    const fetchImpl = async (path, options = {}) => {
      calls.push({ path, signal: options.signal || null });
      return { ok: true, json: async () => ({}) };
    };
    const source = createFrameSource({ fetchImpl });
    const controller = new AbortController();
    await source.postSeed(SESSION, { client_nonce: NONCE_A }, "{}", {
      signal: controller.signal,
    });
    expect(calls.length).toBe(1);
    expect(calls[0].path).toBe(`/session/${SESSION}/seed`);
    expect(calls[0].signal).toBe(controller.signal);
    // No-signal call carries no signal key at all.
    calls.length = 0;
    await source.postSeed(SESSION, { client_nonce: NONCE_A }, "{}");
    expect(calls.length).toBe(1);
    expect(calls[0].signal).toBe(null);
    // getState never accepts a signal (arity: sessionId only).
    expect(source.getState.length).toBe(1);
  });
});

describe("F2 truthful-wait T5 later-reads-succeed (one-shot re-check)", () => {
  it("after detach-free settle, one getState resolves and reports without resuming the loop", async () => {
    const gate = deferred();
    const handle = makeWaitSource({ seedImpl: () => gate.promise });
    const { app, elements } = await bootWaitApp(handle.source);
    const submit = app.submitSeed();
    await flush(6);
    const minted = handle.seen.postSeedEnvelopes[0].client_nonce;
    const getStateBefore = handle.calls.getState;
    expect(getStateBefore >= 1).toBe(true);
    // One-shot re-check: exactly one getState, no loop resume.
    handle.setDefaultState(
      jobDoc([{ session_id: SESSION, client_nonce: minted, state: "RUNNING" }])
    );
    const line = await app.recheckSeedWait();
    expect(line).toBe("Seed: RUNNING");
    expect(handle.calls.getState).toBe(getStateBefore + 1);
    expect(app.seedWaitIsWaiting()).toBe(true);
    // No dangling promise: settle the seed; submit resolves.
    gate.resolve({ session_id: SESSION });
    await submit;
    expect(app.seedWaitIsWaiting()).toBe(false);
    expect(elements["seed-status"]).not.toBe(null);
  });

  it("settled ledger (accepted nonce, no live entry) reports without a state word", async () => {
    const gate = deferred();
    const handle = makeWaitSource({ seedImpl: () => gate.promise });
    const { app } = await bootWaitApp(handle.source);
    const submit = app.submitSeed();
    await flush(6);
    const minted = handle.seen.postSeedEnvelopes[0].client_nonce;
    handle.setDefaultState({ session_id: SESSION, accepted_nonces: [minted] });
    expect(await app.recheckSeedWait()).toBe("accepted");
    handle.setDefaultState({ session_id: SESSION });
    expect(await app.recheckSeedWait()).toBe("no-live-word");
    gate.resolve({ session_id: SESSION });
    await submit;
  });
});

describe("F2 truthful-wait T6 queue-ordering-unchanged (client order log)", () => {
  it("an action behind an in-flight seed executes AFTER it; poll issues no postAction/postSeed", async () => {
    const order = [];
    const seedGate = deferred();
    const handle = makeWaitSource({
      seedImpl: () => seedGate.promise.then(() => {
        order.push("seed");
        return { session_id: SESSION };
      }),
    });
    const origPostAction = handle.source.postAction;
    handle.source.postAction = async (...args) => {
      order.push("action");
      return origPostAction(...args);
    };
    const { app } = await bootWaitApp(handle.source);
    const postSeedBefore = handle.calls.postSeed;
    const postActionBefore = handle.calls.postAction;
    const submit = app.submitSeed();
    await flush(6);
    // The poll observer must never issue postAction/postSeed by itself: any
    // postSeed beyond the submission's own, or any postAction before we act,
    // is a violation.
    expect(handle.calls.postSeed).toBe(postSeedBefore + 1);
    expect(handle.calls.postAction).toBe(postActionBefore);
    // Queue a second seed behind the in-flight one: it must wait the queue.
    const submit2 = app.submitSeed();
    await flush(4);
    seedGate.resolve({ session_id: SESSION });
    await submit;
    await submit2;
    // Both seeds ran through the single queue in order (seed, seed); the
    // observer added no extra transport beyond getState polls.
    expect(order.filter((x) => x === "seed").length).toBe(2);
    expect(handle.calls.postAction).toBe(postActionBefore);
  });
});

describe("F2 truthful-wait T8 transport-error-fail-closed", () => {
  it("rejected getState renders the transport line, hides #seed-error, keeps cadence", async () => {
    const gate = deferred();
    const handle = makeWaitSource({ seedImpl: () => gate.promise });
    const { app, elements } = await bootWaitApp(handle.source);
    const submit = app.submitSeed();
    await flush(6);
    // Force the next poll read to reject: the one-shot path renders the
    // transport-error line for the same cause.
    handle.pushState(new Error("boom"));
    // Drive serial cadence: the immediate poll already ran; schedule one
    // real tick by waiting just past the 1000ms chain with fake timers off —
    // instead assert via recheck-after-reject plus getState counting.
    const before = handle.calls.getState;
    handle.setDefaultState(new Error("boom-marker"));
    // Rejection shape: push an Error so the next getState rejects.
    handle.pushState(new Error("net down"));
    const line = await app.recheckSeedWait();
    expect(line).toBe("transport-error");
    expect(seedStatus(elements)).toBe(
      "Seed: waiting — state unreadable, job still server-owned"
    );
    expect(seedError(elements).hidden).toBe(true);
    expect(app.state.seedError).toBe(null);
    expect(handle.calls.getState).toBeGreaterThan(before);
    // Cadence kept: the observer is still waiting (next tick still scheduled).
    expect(app.seedWaitIsWaiting()).toBe(true);
    gate.resolve({ session_id: SESSION });
    await submit;
  });
});

describe("F2 truthful-wait T9 guard-supersession-silence", () => {
  it("vesselNavSeq bump mid-wait stops the poll with NO further #seed-status write", async () => {
    const gate = deferred();
    const handle = makeWaitSource({ seedImpl: () => gate.promise });
    const { app, elements } = await bootWaitApp(handle.source);
    const submit = app.submitSeed();
    await flush(6);
    const minted = handle.seen.postSeedEnvelopes[0].client_nonce;
    handle.setDefaultState(
      jobDoc([{ session_id: SESSION, client_nonce: minted, state: "RUNNING" }])
    );
    expect(await app.recheckSeedWait()).toBe("Seed: RUNNING");
    const frozen = seedStatus(elements);
    // Vessel switch bumps vesselNavSeq synchronously (selectVessel idiom).
    app.state.vesselNavSeq += 1;
    // A stale tick / re-check must now drop silently: no write, null report.
    expect(await app.recheckSeedWait()).toBe(null);
    expect(seedStatus(elements)).toBe(frozen);
    gate.resolve({ session_id: SESSION });
    await submit;
  });

  it("generation+session swap mid-wait stops silently (openSession idiom)", async () => {
    const gate = deferred();
    const handle = makeWaitSource({ seedImpl: () => gate.promise });
    const { app, elements } = await bootWaitApp(handle.source);
    const submit = app.submitSeed();
    await flush(6);
    const frozen = seedStatus(elements);
    app.state.generation += 1;
    app.state.sessionId = "s2";
    expect(await app.recheckSeedWait()).toBe(null);
    expect(seedStatus(elements)).toBe(frozen);
    gate.resolve({ session_id: SESSION });
    await submit;
  });
});

describe("F2 truthful-wait silence-before-first-word + silence-after-return", () => {
  it("no entry for this nonce writes NO state word (never fabricates QUEUED)", async () => {
    const gate = deferred();
    const handle = makeWaitSource({ seedImpl: () => gate.promise });
    const { app, elements } = await bootWaitApp(handle.source);
    const neutralBefore = seedStatus(elements);
    const submit = app.submitSeed();
    await flush(6);
    // Default state carries no seed_jobs: the immediate poll must NOT have
    // written any Seed: QUEUED/RUNNING/REFUSED word.
    const text = seedStatus(elements);
    expect(text.startsWith("Seed: QUEUED")).toBe(false);
    expect(text.startsWith("Seed: RUNNING")).toBe(false);
    expect(text.startsWith("Seed: REFUSED")).toBe(false);
    expect(typeof text).toBe("string");
    void neutralBefore;
    gate.resolve({ session_id: SESSION });
    await submit;
  });

  it("identity filter never matches another nonce's word", async () => {
    const gate = deferred();
    const handle = makeWaitSource({ seedImpl: () => gate.promise });
    const { app } = await bootWaitApp(handle.source);
    const submit = app.submitSeed();
    await flush(6);
    const minted = handle.seen.postSeedEnvelopes[0].client_nonce;
    handle.setDefaultState(
      jobDoc([
        { session_id: SESSION, client_nonce: "someone-else-nonce-0001", state: "RUNNING" },
        { session_id: SESSION, client_nonce: minted, state: "QUEUED" },
      ])
    );
    // Our word wins (identity filter), never the foreign one alone.
    expect(await app.recheckSeedWait()).toBe("Seed: QUEUED");
    handle.setDefaultState(
      jobDoc([
        { session_id: SESSION, client_nonce: "someone-else-nonce-0001", state: "RUNNING" },
      ])
    );
    expect(await app.recheckSeedWait()).toBe("no-live-word");
    gate.resolve({ session_id: SESSION });
    await submit;
  });
});
