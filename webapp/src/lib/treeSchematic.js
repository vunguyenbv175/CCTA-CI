/**
 * Read-only coronary tree schematic navigator (display-only).
 *
 * The panel renders the accepted ``ccta.viewer.tree_schematic.v1`` read
 * document: one simplified display-only polyline per ACTIVE labeled
 * centerline path, each labeled with its frozen vessel token plus its
 * assignment-source badge.  The client performs NO geometry, NO measurement
 * and NO anatomy inference: polylines are served 2D coordinates, the SVG
 * viewBox is a display-only fit computed at render time, and every click
 * submits the EXISTING SELECT_VESSEL action through the standard action
 * seam (the closed grammar is untouched).
 */
import { ViewerClientError } from "./errors.js";

/** The closed tree-schematic document schema. */
export const TREE_SCHEMATIC_SCHEMA = "ccta.viewer.tree_schematic.v1";

/** The mandatory document kind marker (every payload + caption is SCHEMATIC). */
export const TREE_SCHEMATIC_KIND = "SCHEMATIC";

/** The mandatory visible caption (fixed so a schematic is never anatomy). */
export const TREE_SCHEMATIC_CAPTION = "SCHEMATIC — not anatomy";

/** The frozen fixed-orientation projection rule of the served document. */
export const TREE_SCHEMATIC_PROJECTION_RULE = "ORTHOGRAPHIC_WORLD_XY_V1";

/** The frozen simplification rule of the served document. */
export const TREE_SCHEMATIC_SIMPLIFICATION_RULE = "STRIDE_DECIMATION_MAX_32_V1";

/** The simplification cap (display-only point budget per vessel polyline). */
export const TREE_SCHEMATIC_MAX_POINTS = 32;

/** The explicit badge when no bound assignment record is served. */
export const TREE_SCHEMATIC_BADGE_UNRECORDED = "UNRECORDED";

/** The typed schematic refusal codes (closed vocabulary). */
export const TREE_SCHEMATIC_REFUSALS = Object.freeze([
  "TREE_SCHEMATIC_UNAVAILABLE",
  "TREE_SCHEMATIC_INVALID",
  "TREE_SCHEMATIC_STALE",
]);

/**
 * Load the tree-schematic navigator document.
 *
 * Only ``TREE_SCHEMATIC_UNAVAILABLE`` (no ACTIVE labeled centerline) is the
 * ordinary "no panel" case and returns ``null``; stale/malformed states
 * (``TREE_SCHEMATIC_STALE`` / ``TREE_SCHEMATIC_INVALID``) are rethrown so the
 * caller keeps a VISIBLE typed failure.  Any other failure is rethrown as
 * well (never swallowed).
 */
export async function loadTreeSchematic(source, sessionId) {
  if (!source || typeof source.getTreeSchematic !== "function" || !sessionId) {
    return null;
  }
  try {
    return await source.getTreeSchematic(sessionId);
  } catch (error) {
    if (
      error instanceof ViewerClientError &&
      String(error.code || "") === "TREE_SCHEMATIC_UNAVAILABLE"
    ) {
      return null;
    }
    throw error;
  }
}

/**
 * The ordered vessel labels of one schematic document (click-to-select map).
 *
 * Each label is a frozen ``VESSEL:*`` token that submits verbatim as the
 * ``vessel_label`` param of the EXISTING SELECT_VESSEL action.  Returns []
 * for a missing/foreign document (the caller hides the panel, never a
 * fabricated selection).
 */
export function schematicVesselLabels(document) {
  const vessels =
    document && typeof document === "object" && Array.isArray(document.vessels)
      ? document.vessels
      : [];
  const labels = [];
  for (const entry of vessels) {
    const label = entry && typeof entry.vessel_label === "string" ? entry.vessel_label : "";
    if (label && !labels.includes(label)) {
      labels.push(label);
    }
  }
  return labels;
}

/**
 * The schematic entry for one vessel label, or null.
 */
export function schematicEntryForVessel(document, vesselLabel) {
  const vessels =
    document && typeof document === "object" && Array.isArray(document.vessels)
      ? document.vessels
      : [];
  for (const entry of vessels) {
    if (entry && entry.vessel_label === vesselLabel) {
      return entry;
    }
  }
  return null;
}

/**
 * The display-only SVG viewBox fitting every served polyline.
 *
 * Purely presentational (computed at render from served coordinates, never
 * posted anywhere): ``[minU - pad, minV - pad, width + 2 * pad,
 * height + 2 * pad]`` with a relative pad so a degenerate (single-point
 * span) polyline still frames.  Returns null when no plottable point exists.
 */
export function schematicViewBox(document) {
  const vessels =
    document && typeof document === "object" && Array.isArray(document.vessels)
      ? document.vessels
      : [];
  let minU = Infinity;
  let minV = Infinity;
  let maxU = -Infinity;
  let maxV = -Infinity;
  for (const entry of vessels) {
    const polyline = entry && Array.isArray(entry.polyline) ? entry.polyline : [];
    for (const pair of polyline) {
      if (!Array.isArray(pair) || pair.length !== 2) {
        continue;
      }
      const u = Number(pair[0]);
      const v = Number(pair[1]);
      if (!Number.isFinite(u) || !Number.isFinite(v)) {
        continue;
      }
      if (u < minU) minU = u;
      if (v < minV) minV = v;
      if (u > maxU) maxU = u;
      if (v > maxV) maxV = v;
    }
  }
  if (!Number.isFinite(minU) || !Number.isFinite(minV)) {
    return null;
  }
  const spanU = Math.max(maxU - minU, 1e-6);
  const spanV = Math.max(maxV - minV, 1e-6);
  const pad = 0.08 * Math.max(spanU, spanV);
  return [minU - pad, minV - pad, spanU + 2 * pad, spanV + 2 * pad];
}

export default {
  TREE_SCHEMATIC_BADGE_UNRECORDED,
  TREE_SCHEMATIC_CAPTION,
  TREE_SCHEMATIC_KIND,
  TREE_SCHEMATIC_MAX_POINTS,
  TREE_SCHEMATIC_PROJECTION_RULE,
  TREE_SCHEMATIC_REFUSALS,
  TREE_SCHEMATIC_SCHEMA,
  TREE_SCHEMATIC_SIMPLIFICATION_RULE,
  loadTreeSchematic,
  schematicEntryForVessel,
  schematicVesselLabels,
  schematicViewBox,
};
