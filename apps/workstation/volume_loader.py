"""Canonical volume materialization + deterministic measurements (§B2).

Voxel arrays are stored as raw ``.bin`` blobs under content-addressed paths
(``core.volume.content_blob_path``) with a JSON envelope beside them — the
WS-00 storage pattern, no numpy in Phase 0. This loader interface IS the
future numpy adapter seam (Part C1): swapping in an array engine later must
not change any caller.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any, Final, cast

from core.coordinates import FrameSpec
from core.dicom.instance import stored_padding_bits_are_zero
from core.digest import canonical_digest, require_sha256_hex
from core.enums import ReleaseLane, VolumeKind
from core.errors import CanonicalValidationError, DigestMismatchError
from core.volume import Volume, VolumeBlobRef, content_blob_path, create_volume

try:  # numpy is an admitted runtime dependency; keep the legacy shell importable without it.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only by the pre-001 runtime shell
    _np = cast(Any, None)

np: Any = _np

VOXEL_CODEC: Final[str] = "uint16-le"
CANONICAL_INT16_CODEC: Final[str] = "int16-le"
CANONICAL_FLOAT32_CODEC: Final[str] = "float32-le"
CANONICAL_VOLUME_CODECS: Final[frozenset[str]] = frozenset(
    {CANONICAL_INT16_CODEC, CANONICAL_FLOAT32_CODEC}
)
LEGACY_VOLUME_CODECS: Final[frozenset[str]] = frozenset({VOXEL_CODEC})
SCALAR_DOMAINS: Final[frozenset[str]] = frozenset(
    {"STORED", "MODALITY_TRANSFORMED", "HU"}
)
ENVELOPE_SCHEMA: Final[str] = "ccta.volume.envelope.v1"
#: AMENDMENT V1.1 (IMAGING_CORE_002_ADMISSION): the envelope now carries the
#: source bit-depth tuple (Part 4.1 provenance).  Additive field set only; the
#: scalar/codec contract is unchanged.
SOURCE_BIT_DEPTH_FIELDS: Final[tuple[str, ...]] = (
    "source_bits_allocated",
    "source_bits_stored",
    "source_high_bit",
    "source_pixel_representation",
)
#: FOLLOW-UP 01 (IC002-R3-R1, cross-version envelope preservation): the FULL
#: scalar binding a canonical content address carries once its envelope is
#: written.  The collision guard compares EXACTLY these nine fields — the
#: source bit-depth tuple PLUS the scalar metadata that decides what the
#: samples MEAN (STORED vs HU, applied rescale, slope/intercept) — because
#: equal pixel bytes and an equal tuple do not imply an equal scalar binding:
#: an identity-rescale re-import after an upgrade produces byte-identical
#: pixels under a DIFFERENT domain (R2 STORED -> target HU), and overwriting
#: the envelope rewrites the earlier volume's MPR schema and breaks its saved
#: sessions.  Enumerated explicitly here and never inferred.
SCALAR_BINDING_FIELDS: Final[tuple[str, ...]] = (
    *SOURCE_BIT_DEPTH_FIELDS,
    "scalar_domain",
    "intensity_unit",
    "rescale_applied",
    "rescale_slope",
    "rescale_intercept",
)
#: FOLLOW-UP 01 (F19, P1): the canonical envelope's voxel SHAPE is part of the
#: structural binding of a content address, alongside the nine scalar fields
#: above.  A blob digest is a content address over raw BYTES, and bytes alone do
#: not determine shape: 4 int16 samples are the same byte string whether they are
#: laid out as (1,2,2) or (2,1,2) (or (4,1,1)).  Equal bytes plus an equal
#: nine-field scalar binding therefore does NOT establish equal geometry, and an
#: incoming write of a different shape over an occupied address would silently
#: re-label the earlier volume's MPR schema and break its saved sessions — the
#: identical provenance hazard the scalar guard exists to prevent, in structural
#: space.  Shape is compared STRUCTURALLY (validated/normalized into a 3-tuple of
#: positive ints), never by raw ``==`` on the stored JSON list, so an equal shape
#: spelled differently still compares equal.  It is deliberately NOT appended to
#: SCALAR_BINDING_FIELDS, which stays the nine SCALAR fields the R1/R2/R3 record
#: enumerates; shape is the structural dimension of the same comparison.
VOLUME_SHAPE_FIELD: Final[str] = "shape"
MEASUREMENT_SCHEMA: Final[str] = "ccta.payload.measurement_set.v1"
MEASUREMENT_ALGORITHM: Final[str] = "central-box-hu-stats"
MEASUREMENT_PARAMS_VERSION: Final[str] = "v1"


@dataclass(frozen=True)
class StoredBlob:
    """Content-addressed write result."""

    sha256: str
    byte_size: int


class VolumeAbstentionError(CanonicalValidationError):
    """Typed, series-level refusal for unsafe canonical scalar materialization."""

    def __init__(self, reason_code: str, detail: str | None = None) -> None:
        self.reason_code = reason_code
        self.detail_code = detail
        codes = [reason_code] if detail is None else [reason_code, detail]
        super().__init__(codes)


@dataclass(frozen=True)
class MaterializedVolumeData:
    """Immutable result of canonical scalar materialization.

    ``array`` is a read-only NumPy view of the exact bytes written to the
    content-addressed blob.  ``envelope`` is the complete scalar-domain
    contract persisted beside that blob.
    """

    array: Any
    blob: StoredBlob
    envelope: dict[str, object]
    frame: FrameSpec | None = None

    @property
    def volume_blob_ref(self) -> VolumeBlobRef:
        return VolumeBlobRef(
            sha256=self.blob.sha256,
            codec=str(self.envelope["codec"]),
            byte_size=self.blob.byte_size,
        )


class CanonicalVolumeLoader:
    """Read/write access to the workspace blob store (§B2)."""

    def __init__(self, blobs_root: Path) -> None:
        self._root = Path(blobs_root)

    # -- blob store ------------------------------------------------------------

    def write_blob(self, data: bytes) -> StoredBlob:
        digest = hashlib.sha256(data).hexdigest()
        path = self.blob_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return StoredBlob(sha256=digest, byte_size=len(data))

    def blob_path(self, digest: str) -> Path:
        require_sha256_hex(digest, "digest")
        return self._root / (content_blob_path(digest) + ".bin")

    def envelope_path(self, digest: str) -> Path:
        require_sha256_hex(digest, "digest")
        return self._root / (content_blob_path(digest) + ".json")

    def read_blob(self, digest: str) -> bytes:
        """Return blob bytes after verifying their content address."""
        path = self.blob_path(digest)
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise DigestMismatchError(
                f"voxel blob {digest[:12]}… fails its content address (found {actual[:12]}…)"
            )
        return data

    def write_envelope(self, digest: str, envelope: dict[str, object]) -> None:
        require_sha256_hex(digest, "digest")
        if not isinstance(envelope, dict):
            raise VolumeAbstentionError("DICOM_SYNTAX_INVALID", "ENVELOPE_NOT_OBJECT")
        path = self.envelope_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**envelope, "schema": ENVELOPE_SCHEMA, "blob_sha256": digest}
        self._validate_envelope(payload, digest=digest)
        path.write_text(
            json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8", newline="\n"
        )

    def read_envelope(self, digest: str) -> dict[str, object]:
        try:
            decoded = json.loads(self.envelope_path(digest).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VolumeAbstentionError(
                "DICOM_SYNTAX_INVALID", "ENVELOPE_UNREADABLE"
            ) from exc
        if not isinstance(decoded, dict):
            raise VolumeAbstentionError("DICOM_SYNTAX_INVALID", "ENVELOPE_NOT_OBJECT")
        document: dict[str, object] = decoded
        self._validate_envelope(document, digest=digest)
        return document

    @staticmethod
    def _validate_envelope(document: dict[str, object], *, digest: str) -> None:
        """Validate versioned scalar metadata without widening legacy codecs."""
        if document.get("schema") != ENVELOPE_SCHEMA:
            raise VolumeAbstentionError("DICOM_SYNTAX_INVALID", "ENVELOPE_SCHEMA_INVALID")
        if document.get("blob_sha256") != digest:
            raise VolumeAbstentionError("DIGEST_MISMATCH", "ENVELOPE_BLOB_REF_MISMATCH")
        codec = document.get("codec")
        if not isinstance(codec, str):
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "CODEC_UNSUPPORTED")
        if codec in LEGACY_VOLUME_CODECS:
            shape = document.get("shape")
            if not isinstance(shape, list) or len(shape) != 3:
                raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "VOLUME_SHAPE_INVALID")
            CanonicalVolumeLoader._validate_shape(shape)
            return
        if codec not in CANONICAL_VOLUME_CODECS:
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "CODEC_UNSUPPORTED")
        shape = document.get("shape")
        if not isinstance(shape, list) or len(shape) != 3:
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "VOLUME_SHAPE_INVALID")
        dimensions = CanonicalVolumeLoader._validate_shape(shape)
        raw_byte_size = document.get("byte_size")
        item_size = 2 if codec == CANONICAL_INT16_CODEC else 4
        expected = dimensions[0] * dimensions[1] * dimensions[2] * item_size
        if (
            isinstance(raw_byte_size, bool)
            or not isinstance(raw_byte_size, int)
            or raw_byte_size != expected
        ):
            raise VolumeAbstentionError("DICOM_SYNTAX_INVALID", "PIXEL_LENGTH_MISMATCH")
        if document.get("ordering") != "(slice,row,column) = (k,j,i)":
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "ORDERING_INVALID")
        scalar_domain = document.get("scalar_domain")
        intensity_unit = document.get("intensity_unit")
        if (
            not isinstance(scalar_domain, str)
            or scalar_domain not in SCALAR_DOMAINS
            or intensity_unit != scalar_domain
        ):
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "INTENSITY_UNIT_MISMATCH")
        if document.get("rounding_rule") != "SOURCE_ROUND_HALF_AWAY_FLOAT32_CAST_RNE_V1":
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "ROUNDING_RULE_INVALID")
        rescale_slope = document.get("rescale_slope")
        rescale_intercept = document.get("rescale_intercept")
        try:
            finite_rescale = (
                not isinstance(rescale_slope, bool)
                and isinstance(rescale_slope, (int, float))
                and isfinite(float(rescale_slope))
                and not isinstance(rescale_intercept, bool)
                and isinstance(rescale_intercept, (int, float))
                and isfinite(float(rescale_intercept))
            )
        except (TypeError, ValueError, OverflowError):
            finite_rescale = False
        if not finite_rescale:
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "RESCALE_INVALID")
        rescale_type = document.get("rescale_type")
        if not isinstance(rescale_type, str) or rescale_type not in {"HU", "NONE"}:
            raise VolumeAbstentionError("RESCALE_TYPE_UNSUPPORTED")
        if not isinstance(document.get("rescale_applied"), bool):
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "RESCALE_FLAG_INVALID")
        if document["rescale_applied"] != (scalar_domain != "STORED"):
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "RESCALE_FLAG_INVALID")
        source_integral = document.get("source_values_exactly_integral")
        if not isinstance(source_integral, bool):
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "INTEGRAL_FLAG_INVALID")
        if codec == CANONICAL_INT16_CODEC and not source_integral:
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "INTEGRAL_FLAG_INVALID")
        # AMENDMENT V1.1: source bit-depth provenance.  Present envelopes must
        # carry all four fields as non-negative integers; legacy envelopes
        # written before this amendment are accepted unchanged (the fields did
        # not exist and no caller may infer bit depth from their absence).
        present = [name for name in SOURCE_BIT_DEPTH_FIELDS if name in document]
        if present and len(present) != len(SOURCE_BIT_DEPTH_FIELDS):
            raise VolumeAbstentionError(
                "SCALAR_DOMAIN_INVALID", "SOURCE_BIT_DEPTH_INCOMPLETE"
            )
        for name in present:
            value = document[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VolumeAbstentionError(
                    "SCALAR_DOMAIN_INVALID", "SOURCE_BIT_DEPTH_INVALID"
                )

    # -- NumPy canonical scalar path -----------------------------------------

    @staticmethod
    def _require_numpy() -> Any:
        if np is None:
            raise RuntimeError(
                "numpy is required for canonical volume materialization; "
                "use the admitted runtime environment"
            )
        return np

    @staticmethod
    def _validate_shape(shape: Sequence[int]) -> tuple[int, int, int]:
        if (
            isinstance(shape, (str, bytes, bytearray))
            or not isinstance(shape, Sequence)
            or len(shape) != 3
            or any(
                isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in shape
            )
        ):
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "VOLUME_SHAPE_INVALID")
        return (int(shape[0]), int(shape[1]), int(shape[2]))

    @staticmethod
    def _normalize_scalar_contract(
        *,
        scalar_domain: object,
        intensity_unit: object,
        rescale_slope: object,
        rescale_intercept: object,
        rescale_type: object,
        modality_lut_present: object,
        pixel_padding_present: object,
    ) -> tuple[str, str, float, float, str]:
        if not isinstance(scalar_domain, str) or scalar_domain not in SCALAR_DOMAINS:
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID")
        if intensity_unit is None:
            intensity_unit = scalar_domain
        if (
            not isinstance(intensity_unit, str)
            or intensity_unit not in SCALAR_DOMAINS
            or intensity_unit != scalar_domain
        ):
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "INTENSITY_UNIT_MISMATCH")
        if not isinstance(modality_lut_present, bool) or not isinstance(
            pixel_padding_present, bool
        ):
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "PRESENCE_FLAG_INVALID")
        if modality_lut_present:
            raise VolumeAbstentionError("MODALITY_LUT_UNSUPPORTED")
        if pixel_padding_present:
            raise VolumeAbstentionError("PIXEL_PADDING_UNSUPPORTED")
        if isinstance(rescale_slope, bool) or isinstance(rescale_intercept, bool):
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "RESCALE_INVALID")
        try:
            slope = float(cast(Any, rescale_slope))
            intercept = float(cast(Any, rescale_intercept))
        except (TypeError, ValueError, OverflowError) as exc:
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "RESCALE_INVALID") from exc
        if not (np is not None and np.isfinite(slope) and np.isfinite(intercept)):
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "RESCALE_NONFINITE")
        if rescale_type is not None and not isinstance(rescale_type, str):
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "RESCALE_TYPE_INVALID")
        normalized_type = (
            "NONE"
            if rescale_type is None or not rescale_type.strip()
            else rescale_type.strip().upper()
        )
        if normalized_type not in {"HU", "NONE"}:
            raise VolumeAbstentionError("RESCALE_TYPE_UNSUPPORTED")
        if scalar_domain == "STORED" and (slope != 1.0 or intercept != 0.0):
            # STORED is deliberately raw.  The metadata is retained, but a
            # caller must not accidentally claim transformed values are raw.
            pass
        return scalar_domain, intensity_unit, slope, intercept, normalized_type

    @staticmethod
    def round_half_away_from_zero(value: float) -> int:
        """Frozen source-domain integralization rule (no banker rounding)."""
        import math

        if not math.isfinite(value):
            raise VolumeAbstentionError("SCALAR_NONFINITE")
        return int(math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5))

    @classmethod
    def _decode_stored_values(
        cls,
        stored_values: Any,
        *,
        shape: tuple[int, int, int] | None,
        pixel_representation: int | None,
    ) -> Any:
        module = cls._require_numpy()
        if isinstance(stored_values, (bytes, bytearray, memoryview)):
            if shape is None:
                raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "VOLUME_SHAPE_REQUIRED")
            raw_shape = cls._validate_shape(shape)
            expected = raw_shape[0] * raw_shape[1] * raw_shape[2] * 2
            data = bytes(stored_values)
            if len(data) != expected:
                raise VolumeAbstentionError("DICOM_SYNTAX_INVALID", "PIXEL_LENGTH_MISMATCH")
            if pixel_representation is None:
                representation = 0
            elif isinstance(pixel_representation, bool) or not isinstance(
                pixel_representation, int
            ):
                raise VolumeAbstentionError(
                    "SCALAR_DOMAIN_INVALID", "PIXEL_REPRESENTATION_INVALID"
                )
            else:
                representation = int(pixel_representation)
            if representation not in {0, 1}:
                raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "PIXEL_REPRESENTATION_INVALID")
            dtype = module.dtype("<i2" if representation == 1 else "<u2")
            values = module.frombuffer(data, dtype=dtype).copy().reshape(raw_shape)
        else:
            try:
                values = module.asarray(stored_values)
            except (TypeError, ValueError) as exc:
                raise VolumeAbstentionError(
                    "SCALAR_DOMAIN_INVALID", "STORED_VALUES_INVALID"
                ) from exc
            if values.ndim != 3:
                raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "VOLUME_SHAPE_INVALID")
            raw_shape = cls._validate_shape(tuple(int(d) for d in values.shape))
            if shape is not None and raw_shape != cls._validate_shape(shape):
                raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "VOLUME_SHAPE_MISMATCH")
            if values.dtype.kind not in {"i", "u"}:
                # Stored DICOM values are integer samples.  An explicitly
                # integral float array is accepted only after an exact check.
                if values.dtype.kind != "f":
                    raise VolumeAbstentionError(
                        "SCALAR_DOMAIN_INVALID", "STORED_VALUES_NOT_INTEGER"
                    )
                if not bool(module.all(module.isfinite(values))):
                    raise VolumeAbstentionError("SCALAR_NONFINITE")
                if not bool(module.all(values == module.trunc(values))):
                    raise VolumeAbstentionError(
                        "SCALAR_DOMAIN_INVALID", "STORED_VALUES_NOT_INTEGER"
                    )
            values = values.copy()
            if pixel_representation is not None and (
                isinstance(pixel_representation, bool)
                or not isinstance(pixel_representation, int)
                or int(pixel_representation) not in {0, 1}
            ):
                raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "PIXEL_REPRESENTATION_INVALID")
        return values

    @classmethod
    def materialize_array(
        cls,
        stored_values: Any,
        *,
        shape: tuple[int, int, int] | None = None,
        pixel_representation: int | None = None,
        scalar_domain: str = "STORED",
        intensity_unit: str | None = None,
        rescale_slope: float = 1.0,
        rescale_intercept: float = 0.0,
        rescale_type: str | None = None,
        modality_lut_present: bool = False,
        modality_lut_sequence: object | None = None,
        pixel_padding_present: bool = False,
        pixel_padding_value: object | None = None,
        pixel_padding_range_limit: object | None = None,
        source_bits_allocated: int | None = None,
        source_bits_stored: int | None = None,
        source_high_bit: int | None = None,
        source_pixel_representation: int | None = None,
    ) -> tuple[Any, str, dict[str, object]]:
        """Convert stored samples to the frozen canonical scalar contract.

        The return value is ``(array, codec, envelope_fields)``.  Integer
        storage is selected iff *all* transformed source-domain values are
        exactly integral and lie in signed int16 range.  Every other safe
        value is explicitly cast to little-endian float32 with NumPy's
        IEEE-754 round-to-nearest-even cast.  No integer conversion can wrap.

        The optional ``source_*`` bit-depth arguments are provenance only:
        when supplied (all four together) they are recorded verbatim in the
        envelope; they never alter the scalar transform, the codec decision,
        or the rounding rule.
        """
        module = cls._require_numpy()
        if modality_lut_sequence is not None:
            modality_lut_present = True
        if pixel_padding_value is not None or pixel_padding_range_limit is not None:
            pixel_padding_present = True
        domain, unit, slope, intercept, normalized_type = cls._normalize_scalar_contract(
            scalar_domain=scalar_domain,
            intensity_unit=intensity_unit,
            rescale_slope=rescale_slope,
            rescale_intercept=rescale_intercept,
            rescale_type=rescale_type,
            modality_lut_present=modality_lut_present,
            pixel_padding_present=pixel_padding_present,
        )
        raw = cls._decode_stored_values(
            stored_values, shape=shape, pixel_representation=pixel_representation
        )
        raw64 = module.asarray(raw, dtype=module.dtype("<f8"))
        if not bool(module.all(module.isfinite(raw64))):
            raise VolumeAbstentionError("SCALAR_NONFINITE")
        if domain == "STORED":
            transformed = raw64
            applied = False
        else:
            # Keep the transform in binary64 until the final storage decision;
            # the storage cast itself is the frozen RNE operation.
            transformed = module.add(module.multiply(raw64, slope), intercept)
            applied = True
        if not bool(module.all(module.isfinite(transformed))):
            raise VolumeAbstentionError("SCALAR_NONFINITE")
        exactly_integral = bool(module.all(transformed == module.trunc(transformed)))
        in_int16_range = bool(
            module.all((transformed >= -32768.0) & (transformed <= 32767.0))
        )
        if exactly_integral and in_int16_range:
            codec = CANONICAL_INT16_CODEC
            output = module.asarray(transformed, dtype=module.dtype("<i2"))
        else:
            finfo = module.finfo(module.float32)
            if bool(module.any(transformed > finfo.max)) or bool(
                module.any(transformed < -finfo.max)
            ):
                raise VolumeAbstentionError("SCALAR_OVERFLOW", "FLOAT32_RANGE")
            output = module.asarray(transformed, dtype=module.dtype("<f4"))
            if not bool(module.all(module.isfinite(output))):
                raise VolumeAbstentionError("SCALAR_OVERFLOW", "FLOAT32_CAST_NONFINITE")
            codec = CANONICAL_FLOAT32_CODEC
        output = module.ascontiguousarray(output)
        output.setflags(write=False)
        envelope_fields: dict[str, object] = {
            "codec": codec,
            "shape": list(int(d) for d in output.shape),
            "byte_size": int(output.nbytes),
            "ordering": "(slice,row,column) = (k,j,i)",
            "scalar_domain": domain,
            "intensity_unit": unit,
            "rescale_slope": slope,
            "rescale_intercept": intercept,
            "rescale_type": normalized_type,
            "rescale_applied": applied,
            "source_values_exactly_integral": exactly_integral,
            "rounding_rule": "SOURCE_ROUND_HALF_AWAY_FLOAT32_CAST_RNE_V1",
        }
        bit_depth = {
            "source_bits_allocated": source_bits_allocated,
            "source_bits_stored": source_bits_stored,
            "source_high_bit": source_high_bit,
            "source_pixel_representation": source_pixel_representation,
        }
        supplied = [name for name, value in bit_depth.items() if value is not None]
        if supplied:
            if len(supplied) != len(SOURCE_BIT_DEPTH_FIELDS):
                raise VolumeAbstentionError(
                    "SCALAR_DOMAIN_INVALID", "SOURCE_BIT_DEPTH_INCOMPLETE"
                )
            for name, value in bit_depth.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise VolumeAbstentionError(
                        "SCALAR_DOMAIN_INVALID", "SOURCE_BIT_DEPTH_INVALID"
                    )
            envelope_fields.update(bit_depth)
        return output, codec, envelope_fields

    def materialize_canonical(
        self,
        stored_values: Any,
        **kwargs: object,
    ) -> MaterializedVolumeData:
        """Materialize and persist one canonical scalar volume.

        The source-bit-depth record (AMENDMENT V1.1) is bound to the pixel
        blob's content address: a re-materialization of EQUAL pixel bytes may
        never overwrite a DIFFERENT recorded source tuple (R1 F3), and a
        record-bearing materialization may never overwrite a record-free
        (pre-amendment) one (R2 F3-residual) — equal pixels do not establish
        the source representation of an earlier volume.  The guard is
        fail-closed and typed, and v1 (record-free) envelopes stay readable.
        """
        output, _codec, fields = self.materialize_array(
            stored_values, **cast(dict[str, Any], kwargs)
        )
        blob = self.write_blob(output.tobytes(order="C"))
        self._guard_source_bit_depth_binding(blob.sha256, fields)
        self.write_envelope(blob.sha256, fields)
        return MaterializedVolumeData(array=output, blob=blob, envelope={
            "schema": ENVELOPE_SCHEMA,
            "blob_sha256": blob.sha256,
            **fields,
        })

    @staticmethod
    def _normalize_rescale_pair(
        slope: object, intercept: object
    ) -> tuple[float, float] | None:
        """Normalize one recorded/incoming rescale pair for EQUALITY only.

        The SPEC records an ABSENT rescale as the identity pair (slope 1.0,
        intercept 0.0) — the reader already materializes it that way
        (``core.dicom.instance``: ``rescale_slope=1.0 if slope is None else
        slope``).  R2-era and post-fix envelopes can therefore describe the
        same physical binding in two spellings, and a naive raw comparison
        would refuse a LEGITIMATE idempotent same-domain re-materialization.
        This normalization makes absent-identical-to-1/0 EXACT:

        * ``None`` (absent) for BOTH fields == the identity pair (1.0, 0.0);
        * a bool is never accepted as an int (``True`` is not 1.0 here), and
          only a real number (int/float, not a string) is a usable value;
        * numeric values are compared as binary64, so ``1`` == ``1.0``.

        Returns ``None`` when a value is present but not a finite real number
        — the caller then treats the pair as NOT equal rather than guessing
        (fail-closed: no coercion, no fabricated rescale).
        """
        if slope is None and intercept is None:
            return (1.0, 0.0)
        if slope is None or intercept is None:
            return None
        for value in (slope, intercept):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
        try:
            normalized = (float(cast(Any, slope)), float(cast(Any, intercept)))
        except (TypeError, ValueError, OverflowError):  # pragma: no cover - typed above
            return None
        if not (isfinite(normalized[0]) and isfinite(normalized[1])):
            return None
        return normalized

    @classmethod
    def _shape_binding(cls, document: dict[str, object]) -> object:
        """The comparable structural shape of one envelope/field mapping.

        FOLLOW-UP 01 (F19): returns the shape as a NORMALIZED 3-tuple of
        positive ints, so it is compared STRUCTURALLY rather than as a raw JSON
        list.  An absent/invalid/non-3D shape is returned as a distinct,
        unequal sentinel instead of being normalized into a guess, and an
        unusable entry (bool, non-int, <= 0, wrong length) is likewise its own
        state — fail-closed, never coerced, never fabricated.
        """
        shape = document.get(VOLUME_SHAPE_FIELD)
        if shape is None:
            return ("shape-absent", None)
        if not isinstance(shape, Sequence) or isinstance(
            shape, (str, bytes, bytearray)
        ):
            return ("shape-invalid", repr(shape))
        try:
            return cls._validate_shape(cast(Sequence[int], shape))
        except VolumeAbstentionError:
            return ("shape-invalid", repr(shape))

    @classmethod
    def _scalar_binding(cls, document: dict[str, object]) -> tuple[object, ...]:
        """The comparable full scalar binding of one envelope/field mapping.

        Compares EXACTLY :data:`SCALAR_BINDING_FIELDS` — the source bit-depth
        tuple plus ``scalar_domain``, ``intensity_unit``, ``rescale_applied``
        and the normalized (slope, intercept) pair.  Every field is enumerated
        explicitly; no field is inferred from another, and ``rescale_applied``
        (a bool) is never conflated with the numeric fields.

        FOLLOW-UP 01 (F19): the voxel SHAPE is appended as the structural
        dimension of the same comparison (see :meth:`_shape_binding`).  The
        nine scalar fields remain exactly as enumerated above.
        """
        binding: list[object] = []
        for name in SCALAR_BINDING_FIELDS:
            if name in ("rescale_slope", "rescale_intercept"):
                continue
            value = document.get(name)
            if name == "rescale_applied" and not isinstance(value, bool):
                # Absent/invalid is a distinct, unequal state — never
                # defaulted to False, which would silently match a STORED
                # envelope against an incoming HU one.
                binding.append(("rescale_applied-absent", value))
                continue
            binding.append(value)
        binding.append(
            cls._normalize_rescale_pair(
                document.get("rescale_slope"), document.get("rescale_intercept")
            )
        )
        binding.append(cls._shape_binding(document))
        return tuple(binding)

    def _guard_source_bit_depth_binding(
        self, digest: str, fields: dict[str, object]
    ) -> None:
        """Refuse a re-materialization that would change a recorded binding.

        The volume envelope is keyed by the pixel blob digest alone, so two
        different sources CAN produce byte-identical canonical blobs.  The
        FIRST materialization to occupy a content address owns it: whatever
        source-bit-depth state it established (a recorded tuple, or the
        deliberate absence of one) is immutable for that address.  An incoming
        state that disagrees with the existing one is a typed refusal, not a
        silent provenance overwrite.

        First-writer-wins is the established guard semantic — the
        recorded-tuple case already refused this way — and R2 extends it to
        the record-free case: a record-free (pre-amendment, v1) envelope is
        NOT upgraded in place, because a later source's tuple says nothing
        about the earlier volume whose pixels happen to be equal, and
        overwriting it silently rewrites that volume's MPR schema and breaks
        its saved sessions.  Refusing the collision is the smallest
        fail-closed change; preserving both sources under tuple-namespaced
        envelope keys is a governed task of its own (tracked F17) and is out
        of scope here.

        FOLLOW-UP 01 (IC002-R3-R1): the comparison is the FULL scalar
        binding, not the 4-tuple alone.  Equal pixels and an equal tuple do
        NOT imply an equal binding: an identity-rescale re-import after the
        HU-selection upgrade produces byte-identical pixels under a different
        domain (R2 STORED, rescale_applied false -> HU, true), and the old
        :573 equal-tuple accept let :537 overwrite it — the earlier session
        then failed reopen with a size mismatch.  Any incoming write that
        would change ANY field of :data:`SCALAR_BINDING_FIELDS` — with the
        absent-identity normalization of :meth:`_normalize_rescale_pair` —
        raises the SAME closed typed refusal.  Legitimate same-domain
        idempotent re-materialization (HU+HU, STORED+STORED, same tuple and
        metadata) stays ALLOWED.

         FOLLOW-UP 01 (F19): the comparison ALSO includes the voxel SHAPE
         as a STRUCTURAL binding of the content address.  A blob digest
         addresses raw bytes, and equal bytes do not determine geometry --
         4 int16 samples are one byte string whether laid out as (1,2,2),
         (2,1,2) or (4,1,1).  Equal bytes plus an equal nine-field scalar
         binding but a DIFFERENT shape is therefore a structural
         re-labelling of the earlier volume, and it raises the SAME closed
         typed refusal (no new codes).  Shape is compared as a normalized
         3-tuple (:meth:`_shape_binding`), so an idempotent SAME-shape
         re-materialization stays ALLOWED, exactly as before.
        """
        if not self.envelope_path(digest).exists():
            return
        existing = self.read_envelope(digest)
        recorded = self._scalar_binding(existing)
        incoming = self._scalar_binding(fields)
        if recorded == incoming:
            return
        if any(value is not None for value in recorded[: len(SOURCE_BIT_DEPTH_FIELDS)]):
            # An immutable record already exists at this content address: a
            # different (or absent) incoming tuple would corrupt provenance.
            raise VolumeAbstentionError(
                "SCALAR_DOMAIN_INVALID", "SOURCE_BIT_DEPTH_MISMATCH"
            )
        # A record-free (pre-amendment, v1) envelope is FIRST-WRITER-WINS too:
        # a record-bearing materialization of equal bytes cannot claim an
        # earlier volume's source representation.  Refuse before overwriting
        # (R2 F3-residual); no migration, no fabrication.
        if any(value is not None for value in incoming[: len(SOURCE_BIT_DEPTH_FIELDS)]):
            raise VolumeAbstentionError(
                "SCALAR_DOMAIN_INVALID", "SOURCE_BIT_DEPTH_MISMATCH"
            )
        # Both sides are record-free: an equal-byte envelope whose scalar
        # BINDING would change (e.g. an R2 STORED v1 envelope met by an HU
        # write) is the same provenance hazard in scalar space — refuse with
        # the identical closed code rather than re-labelling the earlier
        # volume's samples.
        raise VolumeAbstentionError(
            "SCALAR_DOMAIN_INVALID", "SOURCE_BIT_DEPTH_MISMATCH"
        )

    # Names used by adapters/tests are intentionally aliases to the one
    # implementation above; they cannot drift into separate scalar rules.
    materialize_volume_data = materialize_canonical
    materialize = materialize_canonical
    write_canonical_volume = materialize_canonical

    def read_array(self, digest: str, *, envelope: dict[str, object] | None = None) -> Any:
        """Decode a canonical int16/float32 blob after validating its envelope."""
        module = self._require_numpy()
        if envelope is None:
            document = self.read_envelope(digest)
        else:
            if not isinstance(envelope, dict):
                raise VolumeAbstentionError("DICOM_SYNTAX_INVALID", "ENVELOPE_NOT_OBJECT")
            self._validate_envelope(envelope, digest=digest)
            document = envelope
        codec = document.get("codec")
        if codec not in CANONICAL_VOLUME_CODECS and codec not in LEGACY_VOLUME_CODECS:
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "CODEC_UNSUPPORTED")
        raw_shape = document.get("shape")
        if not isinstance(raw_shape, list) or len(raw_shape) != 3:
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "VOLUME_SHAPE_INVALID")
        shape = self._validate_shape(raw_shape)
        data = self.read_blob(digest)
        dtype = module.dtype(
            "<i2"
            if codec == CANONICAL_INT16_CODEC
            else "<f4"
            if codec == CANONICAL_FLOAT32_CODEC
            else "<u2"
        )
        expected = shape[0] * shape[1] * shape[2] * dtype.itemsize
        raw_byte_size = document.get("byte_size")
        if (
            len(data) != expected
            or (
                codec != VOXEL_CODEC
                and (
                    isinstance(raw_byte_size, bool)
                    or not isinstance(raw_byte_size, int)
                    or raw_byte_size != expected
                )
            )
            or (codec == VOXEL_CODEC and raw_byte_size is not None and raw_byte_size != expected)
        ):
            raise VolumeAbstentionError("DICOM_SYNTAX_INVALID", "PIXEL_LENGTH_MISMATCH")
        values = module.frombuffer(data, dtype=dtype).copy().reshape(shape)
        if not bool(module.all(module.isfinite(values))):
            raise VolumeAbstentionError("SCALAR_NONFINITE")
        # A true flag is a promise about the pre-storage transformed values.
        # A fractional decoded float cannot be paired with
        # source_values_exactly_integral=true.  A false flag is intentionally
        # not inferred from float32 bytes: a fractional source may have
        # rounded to an integer-valued binary32 representation, and that must
        # not license an int16 MPR.
        if (
            codec == CANONICAL_FLOAT32_CODEC
            and document.get("source_values_exactly_integral") is True
            and not bool(module.all(values == module.trunc(values)))
        ):
            raise VolumeAbstentionError("SCALAR_DOMAIN_INVALID", "INTEGRAL_FLAG_INVALID")
        values.setflags(write=False)
        return values

    load_array = read_array

    def materialize_dicom_stack(
        self,
        instances: Sequence[Any],
        *,
        scalar_domain: str = "STORED",
        intensity_unit: str | None = None,
        modality_lut_present: bool = False,
        modality_lut_sequence: object | None = None,
        pixel_padding_present: bool = False,
        pixel_padding_value: object | None = None,
        pixel_padding_range_limit: object | None = None,
        frame: FrameSpec | None = None,
    ) -> MaterializedVolumeData:
        """Materialize an already-admitted sequence of DICOM DTO slices.

        The DICOM reader remains the authority for syntax and stack admission;
        this adapter only verifies that scalar metadata is uniform before
        concatenating ``(k,j,i)`` bytes.
        """
        try:
            stack = tuple(instances)
        except (TypeError, ValueError) as exc:
            raise VolumeAbstentionError("SERIES_STACK_INVALID", "SCALAR_METADATA_INVALID") from exc
        if not stack:
            raise VolumeAbstentionError("SERIES_INCOMPLETE")
        first = stack[0]
        try:
            rows = int(first.rows)
            columns = int(first.columns)
            representation = int(first.pixel_representation)
            slope = float(getattr(first, "rescale_slope", 1.0))
            intercept = float(getattr(first, "rescale_intercept", 0.0))
            rescale_type = getattr(first, "rescale_type", None)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise VolumeAbstentionError(
                "DICOM_SYNTAX_INVALID", "SCALAR_METADATA_INVALID"
            ) from exc
        # AMENDMENT V1.1 (R1 F7): the legacy fallback must distinguish three
        # closed cases — (a) EVERY member omits ALL depth fields (a duck-typed
        # legacy adapter; the envelope stays record-free, no inference), (b) a
        # member carries a PARTIAL set (any field present with the rest
        # absent, or a recorded member beside a record-free member) ->
        # SOURCE_BIT_DEPTH_INCOMPLETE, and (c) a member carries a malformed
        # value (non-int / bool where an integer belongs) ->
        # SOURCE_BIT_DEPTH_INVALID.  A present-but-bad record is never
        # silently dropped into a legacy envelope.
        depth_names = ("bits_allocated", "bits_stored", "high_bit")
        raw_depths = [
            tuple(getattr(instance, name, None) for name in depth_names)
            for instance in stack
        ]
        if all(all(value is None for value in member) for member in raw_depths):
            raw_depths = []
        else:
            for member in raw_depths:
                if any(value is None for value in member):
                    raise VolumeAbstentionError(
                        "SCALAR_DOMAIN_INVALID", "SOURCE_BIT_DEPTH_INCOMPLETE"
                    )
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in member
                ):
                    raise VolumeAbstentionError(
                        "SCALAR_DOMAIN_INVALID", "SOURCE_BIT_DEPTH_INVALID"
                    )
        # AMENDMENT V1.1: a stack that would concatenate mixed bit depths is a
        # series-level refusal; only a fully present, well-typed UNIFORM tuple
        # is recorded into the envelope (no inference, no default).
        bit_depth_kwargs: dict[str, object] = {}
        if raw_depths:
            if any(member != raw_depths[0] for member in raw_depths[1:]):
                raise VolumeAbstentionError(
                    "SERIES_STACK_INVALID", "SOURCE_BIT_DEPTH_MISMATCH"
                )
            # The classification above guarantees three plain non-bool ints.
            head = cast(tuple[int, int, int], raw_depths[0])
            bit_depth_kwargs = {
                "source_bits_allocated": int(head[0]),
                "source_bits_stored": int(head[1]),
                "source_high_bit": int(head[2]),
                "source_pixel_representation": representation,
            }
        chunks: list[bytes] = []
        for index, instance in enumerate(stack):
            try:
                instance_rescale_type = getattr(instance, "rescale_type", None)
                metadata_mismatch = (
                    int(instance.rows) != rows
                    or int(instance.columns) != columns
                    or int(instance.pixel_representation) != representation
                    or float(getattr(instance, "rescale_slope", 1.0)) != slope
                    or float(getattr(instance, "rescale_intercept", 0.0)) != intercept
                    or (instance_rescale_type or None) != (rescale_type or None)
                )
                data = bytes(instance.pixel_data)
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise VolumeAbstentionError(
                    "DICOM_SYNTAX_INVALID", "SCALAR_METADATA_INVALID"
                ) from exc
            if metadata_mismatch:
                raise VolumeAbstentionError("SERIES_STACK_INVALID", "SCALAR_METADATA_MISMATCH")
            if len(data) != rows * columns * 2:
                raise VolumeAbstentionError("DICOM_SYNTAX_INVALID", "PIXEL_LENGTH_MISMATCH")
            # 12-bit arm conformance: nonconformant padding abstains here too,
            # so no silently masked word can ever reach the canonical blob.
            # (A record-free legacy member carries no depth metadata, so the
            # padding predicate is trivially satisfied for it.)
            if not stored_padding_bits_are_zero(
                data,
                bits_stored=raw_depths[index][1] if raw_depths else None,
                high_bit=raw_depths[index][2] if raw_depths else None,
            ):
                raise VolumeAbstentionError(
                    "DICOM_SYNTAX_INVALID", "SOURCE_PADDING_BITS_NONCONFORMANT"
                )
            # DTOs from a future adapter may expose these optional fields.  A
            # present padding/LUT is always a series-level abstention, with a
            # stable reason for each closed unsupported feature.
            if any(
                getattr(instance, name, None) is not None
                for name in ("modality_lut_sequence", "modality_lut")
            ) or any(
                getattr(instance, name, False) is True
                for name in ("modality_lut_present",)
            ):
                raise VolumeAbstentionError("MODALITY_LUT_UNSUPPORTED")
            if any(
                getattr(instance, name, None) is not None
                for name in ("pixel_padding_value", "pixel_padding_range_limit")
            ) or any(
                getattr(instance, name, False) is True
                for name in ("pixel_padding_present",)
            ):
                raise VolumeAbstentionError("PIXEL_PADDING_UNSUPPORTED")
            chunks.append(data)
        result = self.materialize_canonical(
            b"".join(chunks),
            shape=(len(stack), rows, columns),
            pixel_representation=representation,
            scalar_domain=scalar_domain,
            intensity_unit=intensity_unit,
            rescale_slope=slope,
            rescale_intercept=intercept,
            rescale_type=rescale_type,
            modality_lut_present=modality_lut_present,
            modality_lut_sequence=modality_lut_sequence,
            pixel_padding_present=pixel_padding_present,
            pixel_padding_value=pixel_padding_value,
            pixel_padding_range_limit=pixel_padding_range_limit,
            **bit_depth_kwargs,
        )
        if frame is not None:
            frame = replace(
                frame,
                rescale_slope=slope,
                rescale_intercept=intercept,
                rescale_applied=scalar_domain != "STORED",
            )
        return MaterializedVolumeData(
            array=result.array, blob=result.blob, envelope=result.envelope, frame=frame
        )

    def create_canonical_volume(
        self,
        *,
        stored_values: Any,
        study_ref: str,
        source_series_refs: tuple[str, ...],
        coordinate_frame: FrameSpec,
        provenance_ref: str,
        release_lane: ReleaseLane,
        kind: VolumeKind = VolumeKind.CANONICAL,
        transform_ref: str | None = None,
        registration_ref: str | None = None,
        volume_id: str | None = None,
        **kwargs: object,
    ) -> tuple[Volume, MaterializedVolumeData]:
        """Persist canonical samples and construct the existing Volume record."""
        if not isinstance(coordinate_frame, FrameSpec):
            raise VolumeAbstentionError("GEOMETRY_INVALID", "FRAME_SPEC_INVALID")
        result = self.materialize_canonical(
            stored_values, **cast(dict[str, Any], kwargs)
        )
        if tuple(int(value) for value in coordinate_frame.dimensions) != tuple(
            int(value) for value in result.array.shape
        ):
            raise VolumeAbstentionError("GEOMETRY_INVALID", "VOLUME_SHAPE_MISMATCH")
        coordinate_frame = replace(
            coordinate_frame,
            rescale_slope=float(cast(Any, result.envelope["rescale_slope"])),
            rescale_intercept=float(cast(Any, result.envelope["rescale_intercept"])),
            rescale_applied=bool(result.envelope["rescale_applied"]),
        )
        result = replace(result, frame=coordinate_frame)
        volume = create_volume(
            study_ref=study_ref,
            source_series_refs=source_series_refs,
            kind=kind,
            coordinate_frame=coordinate_frame,
            blob_ref=result.volume_blob_ref,
            provenance_ref=provenance_ref,
            release_lane=release_lane,
            transform_ref=transform_ref,
            registration_ref=registration_ref,
            volume_id=volume_id,
        )
        return volume, result


# ---------------------------------------------------------------------------
# Deterministic measurement engine (Rule C: no AI here, ever)
# ---------------------------------------------------------------------------


def measurement_parameters_digest() -> str:
    """Frozen parameters of the Phase-0 measurement engine."""
    params = {
        "algorithm": MEASUREMENT_ALGORITHM,
        "params_version": MEASUREMENT_PARAMS_VERSION,
        "roi_rule": "central box, lo = dim // 3, hi = dim - lo per axis",
        "input": "uint16-le voxels, shape (slices, rows, columns)",
    }
    return canonical_digest(params)


def decode_voxels(data: bytes, shape: tuple[int, int, int]) -> list[list[list[int]]]:
    """Decode uint16-LE voxel bytes into nested ints (little volumes only).

    Phase-0 fixtures are tiny; the numpy adapter owns large arrays later.
    """
    expected = shape[0] * shape[1] * shape[2] * 2
    if len(data) != expected:
        raise ValueError(f"voxel payload is {len(data)} bytes, expected {expected}")
    values = [
        int.from_bytes(data[i : i + 2], "little", signed=False)
        for i in range(0, len(data), 2)
    ]
    return [
        [
            [
                values[(k * shape[1] + j) * shape[2] + i]
                for i in range(shape[2])
            ]
            for j in range(shape[1])
        ]
        for k in range(shape[0])
    ]


def compute_measurements(
    voxels: bytes, shape: tuple[int, int, int], *, params_version: str = MEASUREMENT_PARAMS_VERSION
) -> dict[str, object]:
    """Deterministic central-box HU statistics over a decoded volume.

    The ROI rule is frozen in :func:`measurement_parameters_digest`; identical
    inputs always produce bit-identical output floats (integer sums, one final
    division), which is what round-trip check (c) re-verifies.
    """
    if params_version != MEASUREMENT_PARAMS_VERSION:
        raise ValueError(f"unsupported measurement params version {params_version!r}")
    cube = decode_voxels(voxels, shape)
    lo = tuple(dim // 3 for dim in shape)
    hi = tuple(dim - low for dim, low in zip(shape, lo))
    total = 0
    count = 0
    minimum: int | None = None
    maximum: int | None = None
    for k in range(lo[0], hi[0]):
        for j in range(lo[1], hi[1]):
            for i in range(lo[2], hi[2]):
                value = cube[k][j][i]
                total += value
                count += 1
                if minimum is None or value < minimum:
                    minimum = value
                if maximum is None or value > maximum:
                    maximum = value
    assert minimum is not None and maximum is not None, "ROI is never empty (dim >= 1)"
    return {
        "schema": MEASUREMENT_SCHEMA,
        "algorithm": MEASUREMENT_ALGORITHM,
        "params_version": params_version,
        "params_sha256": measurement_parameters_digest(),
        "shape": list(shape),
        "roi_lo": list(lo),
        "roi_hi": list(hi),
        "mean_hu": total / count,
        "min_hu": minimum,
        "max_hu": maximum,
        "voxel_count": count,
    }


def build_frame_spec(
    *,
    frame_of_reference_uid: str,
    image_position_patient: tuple[float, float, float],
    image_orientation_patient: tuple[float, float, float, float, float, float],
    pixel_spacing_mm: tuple[float, float],
    slice_spacing_mm: float,
    dimensions: tuple[int, int, int],
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
    rescale_applied: bool = False,
) -> FrameSpec:
    """FrameSpec from parsed DICOM geometry (only via this helper; Rule B).

    The scalar fields are retained on the existing FrameSpec for geometry
    consumers, while the complete scalar-domain contract lives in the volume
    envelope (the canonical Volume dataclass shape is unchanged).
    """
    return FrameSpec(
        frame_of_reference_uid=frame_of_reference_uid,
        image_position_patient=image_position_patient,
        image_orientation_patient=image_orientation_patient,
        spacing_mm=(pixel_spacing_mm[1], pixel_spacing_mm[0], slice_spacing_mm),
        dimensions=dimensions,
        rescale_slope=float(rescale_slope),
        rescale_intercept=float(rescale_intercept),
        rescale_applied=bool(rescale_applied),
    )


__all__ = [
    "CANONICAL_FLOAT32_CODEC",
    "CANONICAL_INT16_CODEC",
    "CANONICAL_VOLUME_CODECS",
    "ENVELOPE_SCHEMA",
    "LEGACY_VOLUME_CODECS",
    "MEASUREMENT_ALGORITHM",
    "MEASUREMENT_PARAMS_VERSION",
    "MEASUREMENT_SCHEMA",
    "CanonicalVolumeLoader",
    "MaterializedVolumeData",
    "materialize_canonical_volume",
    "materialize_volume",
    "StoredBlob",
    "SCALAR_DOMAINS",
    "VolumeAbstentionError",
    "build_frame_spec",
    "compute_measurements",
    "decode_voxels",
    "measurement_parameters_digest",
]


def materialize_canonical_volume(
    loader: CanonicalVolumeLoader, stored_values: Any, **kwargs: object
) -> MaterializedVolumeData:
    """Functional adapter spelling for callers that do not retain a loader."""
    return loader.materialize_canonical(
        stored_values, **cast(dict[str, Any], kwargs)
    )


def materialize_volume(
    loader: CanonicalVolumeLoader, stored_values: Any, **kwargs: object
) -> MaterializedVolumeData:
    """Alias for :func:`materialize_canonical_volume`."""
    return materialize_canonical_volume(loader, stored_values, **kwargs)
