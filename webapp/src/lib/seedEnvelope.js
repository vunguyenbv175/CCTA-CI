/**
 * Physician-seed envelope construction + closed-envelope validation (client side).
 *
 * The dedicated ``POST /session/:id/seed`` route (T2 Phase A, server side) is
 * NOT a grammar action: its envelope is exactly
 * ``{voxel, direction, client_nonce, lineage}`` with
 * ``lineage = {session_id, parent_action_nonce}`` (the SELECT_VESSEL
 * nonce/lineage pattern).  This module is the strict client-side mirror of
 * ``apps/viewer/engine.py`` (``_SEED_ENVELOPE_KEYS`` / ``_SEED_LINEAGE_KEYS``
 * / ``_validate_seed_request``): a malformed envelope is refused BEFORE the
 * network, never clamped, never defaulted — omitted voxel/direction values
 * throw instead of filling anything in.
 */
import { NONCE_RE } from "./constants.js";
import { ViewerClientError } from "./errors.js";

/** The exact closed key set of the seed route envelope (server order). */
export const SEED_ENVELOPE_KEYS = Object.freeze([
  "client_nonce",
  "direction",
  "lineage",
  "voxel",
]);

/** The exact closed key set of the seed lineage object. */
export const SEED_LINEAGE_KEYS = Object.freeze([
  "parent_action_nonce",
  "session_id",
]);

/**
 * Validate one physician-picked voxel WITHOUT defaulting.
 *
 * @param {unknown} voxel candidate ``[i, j, k]`` index triple
 * @returns {[number, number, number]} the validated triple
 */
export function assertSeedVoxel(voxel) {
  if (
    !Array.isArray(voxel) ||
    voxel.length !== 3 ||
    !voxel.every((value) => Number.isInteger(value) && value >= 0)
  ) {
    throw new ViewerClientError(
      "UNKNOWN_PARAM",
      "voxel must be three non-negative integers [i, j, k] (no hidden defaults)"
    );
  }
  return [voxel[0], voxel[1], voxel[2]];
}

/**
 * Validate one physician-set direction hint WITHOUT defaulting.
 *
 * @param {unknown} direction candidate finite non-zero 3-vector
 * @returns {[number, number, number]} the validated triple
 */
export function assertSeedDirection(direction) {
  if (
    !Array.isArray(direction) ||
    direction.length !== 3 ||
    !direction.every((value) => typeof value === "number" && Number.isFinite(value)) ||
    direction.every((value) => value === 0)
  ) {
    throw new ViewerClientError(
      "UNKNOWN_PARAM",
      "direction must be a finite non-zero 3-vector (no hidden defaults)"
    );
  }
  return [direction[0], direction[1], direction[2]];
}

/**
 * Validate one voxel/direction pair against the closed seed domain.
 *
 * @param {unknown} voxel
 * @param {unknown} direction
 */
export function validateSeedPayload(voxel, direction) {
  assertSeedVoxel(voxel);
  assertSeedDirection(direction);
}

/**
 * Build one canonical seed envelope (no serialization here; see canonicalJson).
 *
 * @param {{
 *   voxel: unknown, direction: unknown, clientNonce: string, sessionId: string,
 *   parentActionNonce: string | null,
 * }} input
 * @returns {{voxel: number[], direction: number[], client_nonce: string, lineage: object}}
 */
export function buildSeedEnvelope(input) {
  const { voxel, direction, clientNonce, sessionId, parentActionNonce } = input;
  const validVoxel = assertSeedVoxel(voxel);
  const validDirection = assertSeedDirection(direction);
  if (typeof sessionId !== "string" || !sessionId) {
    throw new ViewerClientError("SESSION_INVALID", "session id is required");
  }
  if (typeof clientNonce !== "string" || !NONCE_RE.test(clientNonce)) {
    throw new ViewerClientError(
      "NONCE_INVALID",
      "client_nonce must match the frozen grammar"
    );
  }
  if (parentActionNonce !== null && parentActionNonce !== undefined) {
    if (typeof parentActionNonce !== "string" || !NONCE_RE.test(parentActionNonce)) {
      throw new ViewerClientError(
        "NONCE_INVALID",
        "parent_action_nonce must match the grammar"
      );
    }
  }
  return {
    voxel: validVoxel,
    direction: validDirection,
    client_nonce: clientNonce,
    lineage: {
      session_id: sessionId,
      parent_action_nonce:
        parentActionNonce === undefined ? null : parentActionNonce,
    },
  };
}

export default {
  SEED_ENVELOPE_KEYS,
  SEED_LINEAGE_KEYS,
  assertSeedDirection,
  assertSeedVoxel,
  buildSeedEnvelope,
  validateSeedPayload,
};
