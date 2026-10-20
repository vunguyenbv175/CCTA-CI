/**
 * Typed client-side refusal.
 *
 * Every refusal raised by the physician-viewer client carries a closed code so
 * the UI never degrades to an anonymous error string.  These codes are
 * CLIENT-side presentation codes only: they are not server rejection codes and
 * never enter an action envelope.
 */
export class ViewerClientError extends Error {
  /**
   * @param {string} code closed client-side refusal code
   * @param {string} message human-readable detail
   */
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "ViewerClientError";
    this.code = code;
  }
}

export default ViewerClientError;
