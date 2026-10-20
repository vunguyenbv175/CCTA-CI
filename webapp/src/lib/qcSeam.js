/**
 * QC seam orchestration (F3): the QC presentation is only ever rendered AFTER
 * the open traverses the frozen action seam.
 *
 *   open():  POST OPEN_QC_PANEL (closed empty param set)  ->  GET /qc
 *   correction params: subject_ref + proposed_grade + justification_code only,
 *   with subject_ref DERIVED from the server-issued gate outcome
 *   (``consumed_refs`` carries study_id:<ULID>), never from free-form input.
 *
 * The frozen grammar carries no evidence-flag vocabulary, so the gate's own
 * ``correction_capabilities`` disclosure decides which grades are actionable;
 * ACCEPTABLE/LIMITED stay disabled with the disclosed reason (never invented).
 */
import { JUSTIFICATION_HUMAN_EVIDENCE, QUALITY_GRADES, STUDY_REF_RE } from "./constants.js";
import { ViewerClientError } from "./errors.js";

/**
 * Derive the bound study reference from a served QC document.
 *
 * Scans the append-only outcome ledger (newest last) for exactly one unique
 * ``study_id:<ULID>`` consumed ref.  Missing or ambiguous refs fail closed.
 *
 * @param {object} qcDocument the GET /session/:id/qc document
 * @returns {string} exactly ``study_id:<ULID>``
 */
export function deriveStudyRef(qcDocument) {
  if (qcDocument === null || typeof qcDocument !== "object") {
    throw new ViewerClientError("QC_DOCUMENT_INVALID", "QC document must be an object");
  }
  const candidates = new Set();
  const collect = (outcome) => {
    if (!outcome || typeof outcome !== "object") {
      return;
    }
    const refs = Array.isArray(outcome.consumed_refs) ? outcome.consumed_refs : [];
    for (const ref of refs) {
      if (typeof ref === "string" && STUDY_REF_RE.test(ref)) {
        candidates.add(ref);
      }
    }
  };
  collect(qcDocument.outcome);
  if (Array.isArray(qcDocument.outcome_ledger)) {
    for (const outcome of qcDocument.outcome_ledger) {
      collect(outcome);
    }
  }
  if (candidates.size !== 1) {
    throw new ViewerClientError(
      "STUDY_REF_UNAVAILABLE",
      `QC document carries ${candidates.size} distinct study refs; correction is disabled`
    );
  }
  return [...candidates][0];
}

/**
 * The actionable/deferred correction grades disclosed by the server.
 *
 * @param {object} qcDocument
 * @returns {{actionable: string[], deferred: string[], deferredReason: string | null}}
 */
export function correctionCapabilities(qcDocument) {
  const capabilities =
    qcDocument && typeof qcDocument === "object" ? qcDocument.correction_capabilities : null;
  const actionable =
    capabilities && Array.isArray(capabilities.actionable_grades)
      ? capabilities.actionable_grades.filter((grade) => QUALITY_GRADES.includes(grade))
      : [];
  const deferred =
    capabilities && Array.isArray(capabilities.deferred_grades)
      ? capabilities.deferred_grades.filter((grade) => QUALITY_GRADES.includes(grade))
      : [];
  return {
    actionable,
    deferred,
    deferredReason:
      capabilities && typeof capabilities.deferred_reason === "string"
        ? capabilities.deferred_reason
        : null,
  };
}

/**
 * Build the closed REQUEST_CORRECTION params (exactly three keys).
 *
 * @param {{studyRef: string, grade: string, actionableGrades: readonly string[]}} input
 */
export function buildCorrectionParams(input) {
  const { studyRef, grade, actionableGrades } = input;
  if (typeof studyRef !== "string" || !STUDY_REF_RE.test(studyRef)) {
    throw new ViewerClientError("STUDY_REF_UNAVAILABLE", "no server-derived study reference");
  }
  if (!QUALITY_GRADES.includes(grade)) {
    throw new ViewerClientError("GRADE_NOT_IN_ENUM", `unknown grade ${grade}`);
  }
  if (!actionableGrades.includes(grade)) {
    throw new ViewerClientError(
      "GRADE_DEFERRED",
      `${grade} is deferred by the disclosed correction capabilities`
    );
  }
  return {
    subject_ref: studyRef,
    proposed_grade: grade,
    justification_code: JUSTIFICATION_HUMAN_EVIDENCE,
  };
}

/**
 * The QC controller: the ONLY path that renders the QC presentation.
 *
 * @param {{
 *   postAction: (action: string, params: object) => Promise<object|null>,
 *   getQc: () => Promise<object>,
 * }} dependencies
 */
export function createQcController(dependencies) {
  const { postAction, getQc } = dependencies;
  const calls = [];
  return {
    /** Ordered open: action seam FIRST, then the QC document. */
    async open() {
      calls.length = 0;
      calls.push("POST OPEN_QC_PANEL");
      const state = await postAction("OPEN_QC_PANEL", {});
      calls.push("GET /qc");
      const document = await getQc();
      return { state, document, calls: calls.slice() };
    },
    /** Submit one closed correction (params validated above). */
    async submitCorrection(qcDocument, grade) {
      const studyRef = deriveStudyRef(qcDocument);
      const { actionable } = correctionCapabilities(qcDocument);
      const params = buildCorrectionParams({
        studyRef,
        grade,
        actionableGrades: actionable,
      });
      return postAction("REQUEST_CORRECTION", params);
    },
    calls() {
      return calls.slice();
    },
  };
}

export default {
  buildCorrectionParams,
  correctionCapabilities,
  createQcController,
  deriveStudyRef,
};
