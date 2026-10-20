/**
 * Accepted-seed provenance summary (Q3-SEED-PROV, read/display only).
 *
 * After a physician seed is accepted, the seed panel shows a compact summary
 * of what the seed path already records: the accepted voxel + direction
 * (``state.seedCurrent``), the bound session, and the served centerline
 * binding (the overlay document's ``centerline_ref`` plus the bound
 * generation from the CPR view document), with a pointer to the existing QC
 * panel (the ``open-qc`` control).  No new server state, no grammar change,
 * no envelope keys, no anatomy labels, no inference: every value is served
 * or typed, and any malformed binding renders a typed unavailable state
 * (never invented text).
 *
 * Association + validation (R1 findings F1/F2/F3):
 * * F1: the served binding observed when the seed was ACCEPTED is captured
 *   once (``captureAcceptedSeedBinding``) and every render re-verifies that
 *   the currently displayed overlay/CPR documents are that same binding.  A
 *   vessel switch (or any other document change) therefore renders the typed
 *   unavailable state: arbitrary current-vessel documents are never labelled
 *   as the accepted seed's provenance.
 * * F2: a binding is bound only when the overlay ref matches the frozen
 *   grammar form, the CPR document cross-references the SAME centerline, and
 *   the generation is the served ``<centerline_id>:v<n>`` form.  The
 *   server's ``UNKNOWN`` sentinel (``apps/viewer/cpr_view.py``: no matching
 *   geometry generation) and arbitrary generation text are refused.
 * * F3: voxel/direction validation REUSES the canonical client validators
 *   (``seedEnvelope.assertSeedVoxel`` / ``assertSeedDirection``), including
 *   the non-zero-vector rule; no weaker duplicate rule lives here.
 */

import { CENTERLINE_REF_RE } from "./actionEnvelope.js";
import { ViewerClientError } from "./errors.js";
import { assertSeedDirection, assertSeedVoxel } from "./seedEnvelope.js";

/** The typed unavailable sentinel (never invented text). */
export const SEED_PROVENANCE_UNAVAILABLE = "seed provenance unavailable";

/**
 * The served sentinel for an unbound geometry generation: the CPR view
 * document carries ``centerline_generation = "UNKNOWN"`` when no matching
 * geometry generation exists (``apps/viewer/cpr_view.py``).  It is never a
 * bound generation.
 */
export const SEED_GENERATION_UNKNOWN = "UNKNOWN";

/**
 * Extract the verifiable served binding of ONE overlay/CPR document pair.
 *
 * The pair is accepted only when it is internally consistent: the overlay
 * ref matches the frozen grammar centerline form, the CPR document
 * cross-references the SAME centerline, and the generation is the served
 * ``<centerline_id>:v<n>`` form (never the server's ``UNKNOWN`` sentinel).
 * Anything else returns null so the caller renders the typed unavailable
 * state instead of declaring the binding bound.
 */
function bindingFrom({ sessionId, overlayDocument, cprDocument }) {
  if (typeof sessionId !== "string" || !sessionId) {
    return null;
  }
  const centerlineRef =
    overlayDocument && typeof overlayDocument === "object"
      ? overlayDocument.centerline_ref
      : null;
  if (typeof centerlineRef !== "string" || !CENTERLINE_REF_RE.test(centerlineRef)) {
    return null;
  }
  const cprRef =
    cprDocument && typeof cprDocument === "object"
      ? cprDocument.centerline_ref
      : null;
  if (cprRef !== centerlineRef) {
    return null;
  }
  const generation =
    cprDocument && typeof cprDocument === "object"
      ? cprDocument.centerline_generation
      : null;
  if (typeof generation !== "string" || generation === SEED_GENERATION_UNKNOWN) {
    return null;
  }
  const prefix = `${centerlineRef.split(":", 2)[1]}:v`;
  if (
    !generation.startsWith(prefix) ||
    !/^\d+$/.test(generation.slice(prefix.length))
  ) {
    return null;
  }
  return { sessionId, centerlineRef, generation };
}

/**
 * Capture the served binding of an ACCEPTED seed (the association snapshot).
 *
 * Called once, right after a seed is accepted and the panels were re-served:
 * the returned triple is the verifiable accepted-seed↔centerline association
 * the summary later re-checks against the displayed documents.  Null when
 * the served documents cannot establish one (the summary then renders the
 * typed unavailable state).
 *
 * @param {{
 *   sessionId: unknown,
 *   overlayDocument: unknown,
 *   cprDocument: unknown,
 * }} input the already-served values (no fetching, no defaults)
 * @returns {{sessionId: string, centerlineRef: string, generation: string} | null}
 */
export function captureAcceptedSeedBinding(input) {
  if (!input || typeof input !== "object") {
    return null;
  }
  return bindingFrom(input);
}

/**
 * Build the compact accepted-seed provenance summary.
 *
 * @param {{
 *   seedCurrent: {voxel: unknown, direction: unknown} | null,
 *   sessionId: unknown,
 *   overlayDocument: {centerline_ref?: unknown} | null,
 *   cprDocument: {centerline_ref?: unknown, centerline_generation?: unknown} | null,
 *   acceptedBinding: {sessionId?: unknown, centerlineRef?: unknown, generation?: unknown} | null,
 * }} input the already-served/typed values (no fetching, no defaults)
 * @returns {string | null} the summary text, the unavailable sentinel on
 * malformed data or an unverifiable association, or null when no accepted
 * seed is displayed
 */
export function seedProvenanceText(input) {
  if (!input || typeof input !== "object") {
    throw new ViewerClientError(
      "SEED_PROVENANCE_INVALID",
      "seed provenance input must be an object"
    );
  }
  const { seedCurrent, sessionId, overlayDocument, cprDocument, acceptedBinding } =
    input;
  if (seedCurrent === null || seedCurrent === undefined) {
    return null;
  }
  if (typeof seedCurrent !== "object") {
    return SEED_PROVENANCE_UNAVAILABLE;
  }
  try {
    // F3: reuse the canonical validators (voxel + finite non-zero direction);
    // a malformed accepted seed renders the typed sentinel, never text.
    assertSeedVoxel(seedCurrent.voxel);
    assertSeedDirection(seedCurrent.direction);
  } catch {
    return SEED_PROVENANCE_UNAVAILABLE;
  }
  const displayed = bindingFrom({ sessionId, overlayDocument, cprDocument });
  if (displayed === null) {
    return SEED_PROVENANCE_UNAVAILABLE;
  }
  // F1: only the binding captured at ACCEPT time may be displayed; the
  // current documents must still be exactly that binding.
  if (
    !acceptedBinding ||
    typeof acceptedBinding !== "object" ||
    acceptedBinding.sessionId !== displayed.sessionId ||
    acceptedBinding.centerlineRef !== displayed.centerlineRef ||
    acceptedBinding.generation !== displayed.generation
  ) {
    return SEED_PROVENANCE_UNAVAILABLE;
  }
  return `session ${displayed.sessionId} · ${displayed.centerlineRef} · gen ${displayed.generation}`;
}

export default {
  SEED_GENERATION_UNKNOWN,
  SEED_PROVENANCE_UNAVAILABLE,
  captureAcceptedSeedBinding,
  seedProvenanceText,
};
