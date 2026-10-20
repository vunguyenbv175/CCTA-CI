/**
 * Canonical JSON serialization mirrored from core.digest.canonical_json.
 *
 * The frozen grammar only accepts request bytes equal to the canonical form
 * (sorted keys, tight separators, UTF-8).  JavaScript already emits strings,
 * booleans, null, arrays and objects byte-identically; the ONLY divergent
 * surface is number formatting: ``JSON.stringify`` and Python's
 * ``repr``/``json.dumps`` spell the same IEEE-754 double differently at the
 * exponent thresholds (``0.000001`` vs Python's ``1e-06``, ``0.00001`` vs
 * ``1e-05``), and the frozen server rejects such bytes with
 * ``NON_CANONICAL_ENCODING`` (grammar.parse_action_body re-serializes the
 * parsed body with core.digest.canonical_json and requires byte equality).
 *
 * Every number is therefore formatted by :func:`formatCanonicalNumber`,
 * which reproduces Python's float repr rules (shortest round-trip digits,
 * scientific notation iff ``decpt <= -4 || decpt > 16``, signed two-digit
 * minimum exponents, trailing ``.0`` only where Python would emit it).  A
 * number that is integer-valued serializes in integer form because it MUST
 * round-trip through the frozen grammar's ``_require_int`` gates (e.g.
 * ``view.frame_index``); Python parses it back as ``int`` and its canonical
 * re-serialization is byte-identical.  This serializer never quantizes by
 * itself (see actionEnvelope.js for the matrix precision transform).
 */
export function canonicalJson(value) {
  const serialized = serializeCanonical(value);
  if (serialized === undefined) {
    throw new TypeError("canonicalJson: value is not canonically serializable");
  }
  return serialized;
}

function serializeCanonical(value) {
  if (value === null) {
    return "null";
  }
  const type = typeof value;
  if (type === "number") {
    return formatCanonicalNumber(value);
  }
  if (type === "string") {
    return JSON.stringify(value);
  }
  if (type === "boolean") {
    return value ? "true" : "false";
  }
  if (Array.isArray(value)) {
    const items = [];
    for (const item of value) {
      const text = serializeCanonical(item);
      items.push(text === undefined ? "null" : text);
    }
    return `[${items.join(",")}]`;
  }
  if (type === "object") {
    // Python's json.dumps(sort_keys=True) sorts by code point; the closed
    // envelope keys are ASCII, where the default JS sort is identical.
    const parts = [];
    for (const key of Object.keys(value).sort()) {
      const text = serializeCanonical(value[key]);
      if (text !== undefined) {
        parts.push(`${JSON.stringify(key)}:${text}`);
      }
    }
    return `{${parts.join(",")}}`;
  }
  if (type === "bigint") {
    throw new TypeError("canonicalJson: BigInt values are not serializable");
  }
  // undefined / function / symbol: same omission semantics as JSON.stringify.
  return undefined;
}

/**
 * Format one finite number exactly as Python's ``json.dumps`` would.
 *
 * @param {number} value
 * @returns {string}
 */
function formatCanonicalNumber(value) {
  if (!Number.isFinite(value)) {
    throw new TypeError(`canonicalJson: non-finite number ${value} is not canonical`);
  }
  if (Number.isInteger(value)) {
    // Integer form: "2" (never "2.0") so the frozen grammar's integer gates
    // keep working; Python re-serializes the parsed int byte-identically.
    return Object.is(value, -0) ? "0" : String(value);
  }
  return pythonFloatRepr(value);
}

/**
 * Reproduce CPython's ``repr(float)`` spelling (format_float_short, 'r').
 *
 * The shortest round-trip digits are shared with ``Number.prototype.toString``
 * (both are shortest-representation algorithms); only the placement of the
 * decimal point, the scientific-notation threshold and the exponent spelling
 * differ, and those are re-derived here from the decimal exponent.
 *
 * @param {number} value a finite, non-integer number
 * @returns {string}
 */
function pythonFloatRepr(value) {
  const shortest = Math.abs(value).toString();
  const exponentIndex = shortest.indexOf("e");
  const mantissa = exponentIndex === -1 ? shortest : shortest.slice(0, exponentIndex);
  const exponent = exponentIndex === -1 ? 0 : Number(shortest.slice(exponentIndex + 1));
  const dotIndex = mantissa.indexOf(".");
  const integerDigits = dotIndex === -1 ? mantissa : mantissa.slice(0, dotIndex);
  const fractionDigits = dotIndex === -1 ? "" : mantissa.slice(dotIndex + 1);
  let digits = integerDigits + fractionDigits;
  // decpt is the count of digits that precede the decimal point, i.e.
  // value === 0.<digits> * 10**decpt (CPython format_float_short).
  let decpt = integerDigits.length + exponent;
  let leading = 0;
  while (leading < digits.length - 1 && digits[leading] === "0") {
    leading += 1;
    decpt -= 1;
  }
  digits = digits.slice(leading).replace(/0+$/, "");
  if (digits === "") {
    digits = "0";
  }
  let text;
  if (decpt <= -4 || decpt > 16) {
    const head = digits[0];
    const tail = digits.slice(1);
    text = tail ? `${head}.${tail}` : head;
    const scientificExponent = decpt - 1;
    const sign = scientificExponent < 0 ? "-" : "+";
    text += `e${sign}${String(Math.abs(scientificExponent)).padStart(2, "0")}`;
  } else if (decpt <= 0) {
    text = `0.${"0".repeat(-decpt)}${digits}`;
  } else if (decpt >= digits.length) {
    text = `${digits}${"0".repeat(decpt - digits.length)}.0`;
  } else {
    text = `${digits.slice(0, decpt)}.${digits.slice(decpt)}`;
  }
  return value < 0 ? `-${text}` : text;
}

export default canonicalJson;
