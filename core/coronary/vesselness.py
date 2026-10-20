"""Frangi-class vesselness over the canonical HU volume (pure NumPy).

The derivation is fully deterministic: the same volume, spacing and
:class:`~core.coronary.types.VesselnessParameters` produce a bit-identical
response (no RNG, no hidden state, no BLAS-dependent reduction).  Only
element-wise arithmetic, ``np.pad``, and fixed-order ``np.take_along_axis`` /
``np.argsort(kind="stable")`` are used, so the result does not depend on
threading or library heuristics.

Frozen arithmetic (recorded in ``parameters_ref`` via the parameter document):

1. **Smoothing** - for every scale ``sigma_mm`` a separable normalized
   Gaussian kernel with ``sigma_points = sigma_mm / spacing`` is applied along
   i, j, k in that fixed order; the radius is ``ceil(3 * sigma_points)`` and
   the boundary rule is edge replication (``SMOOTHING_RULE``).
2. **Hessian** - second derivatives by central differences of the smoothed
   volume, divided by the squared physical spacing, with edge replication at
   the volume boundary (``HESSIAN_RULE``).
3. **Scale normalization** - the Hessian is multiplied by ``sigma_mm ** 2``
   (the standard gamma-normalization that makes the response comparable
   across scales; ``VESSELNESS_RULE``).
4. **Eigenvalues** - the real eigenvalues of every symmetric 3x3 Hessian are
   computed with the closed-form trigonometric solution, then ordered by
   increasing absolute value with a STABLE sort.
5. **Frangi response** - for bright structures (CTA contrast lumen,
   ``polarity=BRIGHT``): the response is zero unless ``l2 < 0`` and
   ``l3 < 0``; otherwise
   ``(1 - exp(-Ra^2 / (2 alpha^2))) * exp(-Rb^2 / (2 beta^2)) *
   (1 - exp(-S^2 / (2 c^2)))`` with ``Ra = |l2| / |l3|``,
   ``Rb = |l1| / sqrt(|l2 l3|)``, ``S = sqrt(l1^2 + l2^2 + l3^2)`` and
   ``c = gamma * max(S)`` over the scale (a deterministic global scaling that
   keeps the response in [0, 1]).
6. **Multi-scale combination** - element-wise MAXIMUM over the scales, in the
   parameter order.

Memory-bounded full-volume evaluation (F30-OPT-C, additive; the frozen
arithmetic above is untouched):

* The full volume is processed in ``k``-slabs spanning full ``(j, i)`` planes.
  A slab carries a ``k``-halo of ``R_k + 1`` rows on each side of its valid
  centre, where ``R_k`` is the exact ``k``-axis support radius of
  :func:`_gaussian_kernel` (``max(1, ceil(3 * sigma_mm / spacing_k))``) and the
  extra row covers the Hessian central difference.  Only the TRUE volume rows
  of that window are copied (:func:`_slab_window`): rows outside the volume are
  never materialized, because the UNMODIFIED :func:`_smooth` (dispatcher:
  blocked above its own floor) edge-replicates the window boundary exactly as
  the full-volume path replicates a true volume face, while interior seams
  receive true neighbour rows.  The smoothing of every row needed by the valid
  centre is therefore bit-identical to the full-volume smoothing.
* The smoothed window is reduced to the ``width + 2`` rows the Hessian of the
  valid centre reads (the valid rows plus their outside ``k`` neighbours); at a
  true volume face the outside neighbour is set to the adjacent smoothed row,
  exactly the reference ``mode="edge"`` value.  The result then runs through
  the UNMODIFIED :func:`_second_derivatives` and - in halo-free sub-blocks -
  :func:`_symmetric_eigenvalues`, :func:`_ordered_by_magnitude` and the Frangi
  formula, so every per-voxel operation is literally the same code in the same
  order on the same values; only the valid centre rows are kept.
* The scale constant ``c = gamma * max(S)`` is global to the scale: pass 1
  reduces the exact per-voxel ``S`` norm maximum over the valid rows of every
  slab, propagating NaN through both reduction levels exactly like the
  reference ``np.max`` (``np.maximum`` over the sub-block and slab maxima),
  and pass 2 evaluates the response with that global ``c`` through
  :func:`_frangi_response_with_global_c` (a line mirror of the frozen
  :func:`_frangi_response`).
* The cross-scale combination is the same element-wise MAXIMUM, accumulated
  slab by slab into the single full-volume result in parameter order.
* Volumes below ``_BLOCKED_MIN_VOXELS`` (32^3) keep the verbatim historical
  full-volume path (:func:`_frangi_vesselness_unblocked`); fixture volumes and
  every real volume take the slab path.  The returned bytes are identical on
  both paths (see ``tests/core/coronary/test_vesselness_blocked_parity.py``).

The mask is the closed predicate ``response > threshold``.  An all-zero mask
is an abstention (``NOT_ANALYZABLE``), never a fabricated path - the caller
(``core.coronary.centerline``) raises the typed refusal.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from core.coronary.types import (
    CoronaryRefusalCode,
    VesselnessParameters,
    _refuse,
)

#: Public dtype of the vesselness response (frozen, digest-bound evidence).
RESPONSE_DTYPE: Final[np.dtype[np.float64]] = np.dtype("<f8")
#: The radius multiplier of the Gaussian kernel support (3 sigma).
_GAUSSIAN_RADIUS_SIGMAS: Final[float] = 3.0
#: Physical-spacing component of each NumPy array axis.  The canonical array
#: is ``(k, j, i)`` (``VOLUME_ORDERING``) while ``spacing_mm`` is ``(i, j, k)``
#: (DICOM row/column/slice, see ``build_frame_spec``): array axis 0 (k) maps to
#: ``spacing_mm[2]``, axis 1 (j) to ``spacing_mm[1]``, axis 2 (i) to
#: ``spacing_mm[0]``.  Every physical-spacing consumer in this module uses this
#: explicit mapping - never the raw axis index.
_SPACING_INDEX_BY_ARRAY_AXIS: Final[tuple[int, int, int]] = (2, 1, 0)


def _require_volume(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume)
    if array.ndim != 3:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "volume must be 3-D (k,j,i)")
    if any(int(dimension) <= 0 for dimension in array.shape):
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "volume dimensions must be positive")
    if array.dtype.kind not in {"i", "u", "f"}:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            f"volume dtype {array.dtype} is not a numeric scalar type",
        )
    if array.dtype.kind == "f":
        for value in array.flat:
            if not math.isfinite(float(value)):
                raise _refuse(
                    CoronaryRefusalCode.VOLUME_INVALID, "volume contains non-finite samples"
                )
    return array


def _require_spacing(spacing_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(spacing_mm, tuple) or len(spacing_mm) != 3:
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "spacing_mm must be a 3-tuple (i, j, k) of millimetres per index step",
        )
    result: list[float] = []
    for axis, value in enumerate(spacing_mm):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise _refuse(
                CoronaryRefusalCode.VOLUME_INVALID,
                f"spacing_mm[{axis}] must be positive and finite",
            )
        result.append(float(value))
    return (result[0], result[1], result[2])


def _gaussian_kernel(sigma_points: float) -> np.ndarray:
    """One normalized 1-D Gaussian kernel with edge-truncated support."""

    radius = max(1, int(math.ceil(_GAUSSIAN_RADIUS_SIGMAS * sigma_points)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets * offsets) / (2.0 * sigma_points * sigma_points))
    return np.asarray(kernel / kernel.sum(), dtype=np.float64)


def _convolve_axis(volume: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Separable 1-D convolution with edge replication (fixed order)."""

    radius = (kernel.shape[0] - 1) // 2
    paddings: list[tuple[int, int]] = [(0, 0), (0, 0), (0, 0)]
    paddings[axis] = (radius, radius)
    padded = np.pad(volume, paddings, mode="edge")
    result = np.zeros_like(volume, dtype=np.float64)
    shape = volume.shape
    for tap, weight in enumerate(kernel):
        if weight == 0.0:
            continue
        slices: list[slice] = [slice(None), slice(None), slice(None)]
        slices[axis] = slice(tap, tap + shape[axis])
        result += float(weight) * padded[tuple(slices)]
    return result


#: Floor for halo-blocked smoothing: inputs with fewer voxels take the
#: verbatim unblocked path (identical code path as before this change).
#: 32^3 keeps the small analytic probes (e.g. the 21^3 F3 impulse volume)
#: on the unblocked path while routing fixture volumes (>= 48^3) and every
#: real crop through the blocked path.
_SMOOTH_BLOCKED_MIN_VOXELS: Final[int] = 32 * 32 * 32
#: Target slab size for the adaptive block count: one slab holds roughly
#: this many voxels (~8 MiB in float64), bounding the padded-slab transient
#: independently of the full-volume size.
_SMOOTH_BLOCKED_TARGET_SLAB_VOXELS: Final[int] = 2**20
#: Hard cap on the per-axis block count (bounds halo bookkeeping; the exact
#: result does not depend on the count - see the parity test).
_SMOOTH_BLOCKED_MAX_BLOCKS: Final[int] = 8


def _chunk_bounds(count: int, nblocks: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` index ranges partitioning ``range(count)``."""

    base, rem = divmod(count, nblocks)
    bounds: list[tuple[int, int]] = []
    start = 0
    for block in range(nblocks):
        size = base + (1 if block < rem else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def _blocked_convolve_axis(
    volume: np.ndarray, kernel: np.ndarray, axis: int, nblocks: int
) -> np.ndarray:
    """Halo-blocked form of :func:`_convolve_axis` (bit-exact).

    Each output chunk ``[s, e)`` is produced by running the repo
    :func:`_convolve_axis` verbatim on the halo-extended slab
    ``[max(0, s - radius), min(n, e + radius))`` (halo = kernel radius) and
    stitching the interior ``[s - es, e - es)``.  Interior voxels see the
    identical tap neighbourhood as the full-volume convolution; boundary
    voxels of the full volume keep the same edge-replication semantics
    because a slab touching the volume face pads with the same edge rule.
    """

    radius = (int(kernel.shape[0]) - 1) // 2
    n = int(volume.shape[axis])
    blocks = max(1, min(int(nblocks), n))
    out = np.empty_like(volume, dtype=np.float64)
    for start, end in _chunk_bounds(n, blocks):
        ext_start = max(0, start - radius)
        ext_end = min(n, end + radius)
        get: list[slice] = [slice(None), slice(None), slice(None)]
        get[axis] = slice(ext_start, ext_end)
        slab = np.ascontiguousarray(volume[tuple(get)], dtype=np.float64)
        conv = _convolve_axis(slab, kernel, axis)
        put_src: list[slice] = [slice(None), slice(None), slice(None)]
        put_src[axis] = slice(start - ext_start, end - ext_start)
        put_dst: list[slice] = [slice(None), slice(None), slice(None)]
        put_dst[axis] = slice(start, end)
        out[tuple(put_dst)] = conv[tuple(put_src)]
    return out


def _smooth_nblocks(shape: tuple[int, int, int]) -> int:
    """Adaptive per-axis block count for :func:`_smooth` (1 = unblocked).

    Returns 1 below ``_SMOOTH_BLOCKED_MIN_VOXELS`` (the unblocked path);
    otherwise ``clamp(ceil(voxels / target), 2, 8)`` so the slab transient
    stays bounded as the volume grows while small inputs keep the exact
    historical code path.
    """

    voxels = int(shape[0]) * int(shape[1]) * int(shape[2])
    if voxels < _SMOOTH_BLOCKED_MIN_VOXELS:
        return 1
    nblocks = (voxels + _SMOOTH_BLOCKED_TARGET_SLAB_VOXELS - 1) // (
        _SMOOTH_BLOCKED_TARGET_SLAB_VOXELS
    )
    return max(2, min(_SMOOTH_BLOCKED_MAX_BLOCKS, nblocks))


def _smooth_unblocked(
    volume: np.ndarray, spacing_mm: tuple[float, float, float], sigma_mm: float
) -> np.ndarray:
    """The historical full-volume separable smoothing (frozen arithmetic)."""

    smoothed = np.asarray(volume, dtype=np.float64)
    for axis in range(3):
        # F3: the array is (k, j, i); the spacing tuple is (i, j, k).  Map the
        # physical spacing to the array axis explicitly, so an anisotropic
        # volume smooths by ``sigma_mm`` along every PHYSICAL axis.
        sigma_points = sigma_mm / spacing_mm[_SPACING_INDEX_BY_ARRAY_AXIS[axis]]
        kernel = _gaussian_kernel(sigma_points)
        smoothed = _convolve_axis(smoothed, kernel, axis)
    return smoothed


def _smooth_blocked(
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    sigma_mm: float,
    nblocks: int,
) -> np.ndarray:
    """Halo-blocked separable smoothing (same kernels, axis order, radius)."""

    smoothed = np.asarray(volume, dtype=np.float64)
    for axis in range(3):
        sigma_points = sigma_mm / spacing_mm[_SPACING_INDEX_BY_ARRAY_AXIS[axis]]
        kernel = _gaussian_kernel(sigma_points)
        smoothed = _blocked_convolve_axis(smoothed, kernel, axis, nblocks)
    return smoothed


def _smooth(
    volume: np.ndarray, spacing_mm: tuple[float, float, float], sigma_mm: float
) -> np.ndarray:
    array = np.asarray(volume)
    nblocks = _smooth_nblocks((int(array.shape[0]), int(array.shape[1]), int(array.shape[2])))
    if nblocks <= 1:
        return _smooth_unblocked(volume, spacing_mm, sigma_mm)
    return _smooth_blocked(volume, spacing_mm, sigma_mm, nblocks)


def _second_derivatives(
    smoothed: np.ndarray, spacing_mm: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The six unique second-derivative volumes of the smoothed volume."""

    padded = np.pad(smoothed, 1, mode="edge")
    center = padded[1:-1, 1:-1, 1:-1]
    inv = [1.0 / (spacing_mm[axis] * spacing_mm[axis]) for axis in range(3)]
    d_ii = (padded[1:-1, 1:-1, 2:] - 2.0 * center + padded[1:-1, 1:-1, :-2]) * inv[0]
    d_jj = (padded[1:-1, 2:, 1:-1] - 2.0 * center + padded[1:-1, :-2, 1:-1]) * inv[1]
    d_kk = (padded[2:, 1:-1, 1:-1] - 2.0 * center + padded[:-2, 1:-1, 1:-1]) * inv[2]
    d_ij = (
        padded[1:-1, 2:, 2:]
        - padded[1:-1, 2:, :-2]
        - padded[1:-1, :-2, 2:]
        + padded[1:-1, :-2, :-2]
    ) / (4.0 * spacing_mm[0] * spacing_mm[1])
    d_ik = (
        padded[2:, 1:-1, 2:]
        - padded[2:, 1:-1, :-2]
        - padded[:-2, 1:-1, 2:]
        + padded[:-2, 1:-1, :-2]
    ) / (4.0 * spacing_mm[0] * spacing_mm[2])
    d_jk = (
        padded[2:, 2:, 1:-1]
        - padded[2:, :-2, 1:-1]
        - padded[:-2, 2:, 1:-1]
        + padded[:-2, :-2, 1:-1]
    ) / (4.0 * spacing_mm[1] * spacing_mm[2])
    return (
        np.ascontiguousarray(d_ii),
        np.ascontiguousarray(d_jj),
        np.ascontiguousarray(d_kk),
        np.ascontiguousarray(d_ij),
        np.ascontiguousarray(d_ik),
        np.ascontiguousarray(d_jk),
    )


def _symmetric_eigenvalues(
    a_ii: np.ndarray,
    a_jj: np.ndarray,
    a_kk: np.ndarray,
    a_ij: np.ndarray,
    a_ik: np.ndarray,
    a_jk: np.ndarray,
) -> np.ndarray:
    """Real eigenvalues of many symmetric 3x3 matrices (closed form).

    The trigonometric solution for symmetric tridiagonal-free 3x3 matrices:
    ``q = trace / 3``, ``p = sqrt(sum((a_kk - q)^2) / 6)`` and
    ``r = det((A - q I) / p) / 2`` clamped into [-1, 1]; the eigenvalues are
    ``q + 2 p cos(phi)``, ``q + 2 p cos(phi + 2 pi / 3)`` and
    ``3 q - e1 - e3`` with ``phi = acos(r) / 3``.  The returned array has the
    eigenvalues along the last axis in the order they are computed (NOT yet
    ordered by magnitude).
    """

    one = np.float64(1.0)
    three = np.float64(3.0)
    two = np.float64(2.0)
    q = (a_ii + a_jj + a_kk) / three
    p_squared = (
        (a_ii - q) * (a_ii - q) + (a_jj - q) * (a_jj - q) + (a_kk - q) * (a_kk - q)
        + two * (a_ij * a_ij + a_ik * a_ik + a_jk * a_jk)
    ) / (three * two)
    p = np.sqrt(np.maximum(p_squared, 0.0))
    safe_p = np.where(p > 0.0, p, one)
    b_ii = (a_ii - q) / safe_p
    b_jj = (a_jj - q) / safe_p
    b_kk = (a_kk - q) / safe_p
    b_ij = a_ij / safe_p
    b_ik = a_ik / safe_p
    b_jk = a_jk / safe_p
    det = (
        b_ii * (b_jj * b_kk - b_jk * b_jk)
        - b_ij * (b_ij * b_kk - b_jk * b_ik)
        + b_ik * (b_ij * b_jk - b_jj * b_ik)
    )
    r = np.clip(det / two, -1.0, 1.0)
    phi = np.arccos(r) / three
    pi = np.float64(math.pi)
    e1 = q + two * p * np.cos(phi)
    e3 = q + two * p * np.cos(phi + two * pi / three)
    e2 = three * q - e1 - e3
    return np.stack([e1, e2, e3], axis=-1)


def _ordered_by_magnitude(eigenvalues: np.ndarray) -> np.ndarray:
    """Order the three eigenvalues by increasing |value| (stable, fixed)."""

    order = np.argsort(np.abs(eigenvalues), axis=-1, kind="stable")
    return np.take_along_axis(eigenvalues, order, axis=-1)


def _nan_propagating_max(existing: float, candidate: float) -> float:
    """Two-value maximum with the reference ``np.max`` NaN semantics.

    ``max(a, b)`` would discard a NaN candidate whenever the accumulator is a
    finite value; the reference :func:`_frangi_response` reduces the whole ``S``
    array at once, so a NaN anywhere must poison the scale maximum.  This
    helper uses ``np.maximum``, which returns NaN when either operand is NaN,
    matching ``np.max`` of the concatenation for both the all-finite and the
    NaN cases; ``max`` over floats is exact and order-independent, so reducing
    per-sub-block / per-slab maxima with it yields the identical global value
    in any partition.
    """

    return float(np.maximum(existing, candidate))


def _frangi_response(
    eigenvalues: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """The bright-structure Frangi response from ordered eigenvalues."""

    l1 = eigenvalues[..., 0]
    l2 = eigenvalues[..., 1]
    l3 = eigenvalues[..., 2]
    s = np.sqrt(np.maximum(l1 * l1 + l2 * l2 + l3 * l3, 0.0))
    max_s = float(np.max(s)) if s.size else 0.0
    c = gamma * max_s
    eps = np.finfo(np.float64).tiny
    ab_l2 = np.abs(l2)
    ab_l3 = np.maximum(np.abs(l3), eps)
    ra = ab_l2 / ab_l3
    rb = np.abs(l1) / np.sqrt(np.maximum(ab_l2 * ab_l3, eps))
    first = 1.0 - np.exp(-(ra * ra) / (2.0 * alpha * alpha))
    second = np.exp(-(rb * rb) / (2.0 * beta * beta))
    if c > 0.0:
        third = 1.0 - np.exp(-(s * s) / (2.0 * c * c))
    else:
        third = np.zeros_like(s)
    response = first * second * third
    # Bright structures: a bright tube has two large negative eigenvalues.
    response = np.where((l2 < 0.0) & (l3 < 0.0), response, 0.0)
    return response


#: Floor for the memory-bounded full-volume pipeline: inputs with fewer
#: voxels take the verbatim historical path (identical code path as before
#: this change).  32^3 keeps the small analytic probes (e.g. the 21^3 F3
#: impulse volume) on the unblocked path while routing fixture volumes
#: (>= 48^3) and every real volume through the slab path.
_BLOCKED_MIN_VOXELS: Final[int] = 32 * 32 * 32
#: Target row budget of one ``k``-extended slab (valid rows + both ``k``
#: halos).  The per-scale valid width is ``budget - 2 * halo`` (with a floor),
#: so the working set of one slab is bounded independently of the volume size.
_BLOCKED_MAX_EXTENDED_SLICES: Final[int] = 64
#: Floor for the valid slab width so a large ``k``-halo cannot degenerate into
#: one-row slabs (the exact result does not depend on the width).
_BLOCKED_MIN_VALID_SLICES: Final[int] = 16
#: Row width of the pointwise sub-blocks.  The eigensolve, the stable ordering
#: and the Frangi formula are pointwise, so they need no halo; sub-blocking
#: bounds their temporaries to a fraction of one slab.
_BLOCKED_EIG_SUB_SLICES: Final[int] = 16
#: Row of the valid centre inside the trimmed derivative arrays returned by
#: :func:`_slab_scaled_derivatives`: row 0 is the lower outside ``k`` neighbour
#: and row ``width + 1`` the upper one.
_TRIMMED_VALID_ROW: Final[int] = 1


def _k_smoothing_radius(sigma_mm: float, spacing_mm: tuple[float, float, float]) -> int:
    """The ``k`` array-axis support radius of the smoothing kernel at one scale.

    Mirrors the radius line of :func:`_gaussian_kernel` for array axis 0 (k),
    whose physical spacing is ``spacing_mm[2]`` per
    :data:`_SPACING_INDEX_BY_ARRAY_AXIS`; kept as a function so the frozen
    kernel is never re-expressed.
    """

    sigma_points = float(sigma_mm) / float(spacing_mm[_SPACING_INDEX_BY_ARRAY_AXIS[0]])
    return (int(_gaussian_kernel(sigma_points).shape[0]) - 1) // 2


def _slab_bounds(count: int, valid: int) -> list[tuple[int, int]]:
    """Partition ``range(count)`` into contiguous valid slabs of width <= valid."""

    bounds: list[tuple[int, int]] = []
    start = 0
    while start < count:
        stop = min(count, start + valid)
        bounds.append((start, stop))
        start = stop
    return bounds


def _slab_window(
    array: np.ndarray, start: int, stop: int, halo: int
) -> tuple[np.ndarray, int]:
    """Copy the TRUE rows ``[start - halo, stop + halo)`` along ``k``.

    Rows outside ``[0, count)`` are NOT materialized: the window is clipped to
    the volume (``np.take`` with the raw ``k`` indices), so a thin volume with a
    large ``k``-halo copies at most ``count`` planes instead of
    ``width + 2 * halo`` edge repetitions.  The UNMODIFIED :func:`_smooth`
    applied to this window edge-replicates its first/last row exactly as
    ``mode="edge"`` replicates a true volume face, while interior slab seams
    receive true neighbour rows - the same argument as the previous clamped
    extraction, with the repetitions produced on demand instead of stored.

    The valid rows ``[start, stop)`` sit at ``[start - begin, stop - begin)``
    of the copy, where ``begin = max(0, start - halo)`` is the first window row;
    the valid offset is returned (rather than recomputed by callers) and cannot
    drift from the extraction site.
    """

    count = int(array.shape[0])
    begin = max(0, start - halo)
    end = min(count, stop + halo)
    indices = np.arange(begin, end, dtype=np.intp)
    return np.ascontiguousarray(np.take(array, indices, axis=0)), start - begin


def _slab_scaled_derivatives(
    window: np.ndarray,
    spacing_mm: tuple[float, float, float],
    sigma_mm: float,
    *,
    offset: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed, Hessian, gamma-normalized derivatives of one slab window.

    Runs the unmodified :func:`_smooth` on the true-row window, then reduces the
    smoothed window to the ``width + 2`` rows the Hessian of the valid centre
    reads: the valid rows plus exactly one outside ``k`` neighbour on each side,
    which is the same dependency the reference ``np.pad(..., mode="edge")``
    feeds to :func:`_second_derivatives` - and only along ``k``.  That keeps the
    six derivative arrays at ``(width + 2, j, i)``, never at the full smoothing
    halo, which is the F2 memory bound for thin volumes with large ``k``-halos.

    At a TRUE volume face the outside neighbour does not exist in the window;
    it is set to the adjacent valid smoothed row, exactly the reference
    ``np.pad(..., mode="edge")`` value.  Interior seams keep their true
    neighbour rows untouched.

    The valid centre always starts at row 1 of the returned derivatives (the
    first row is the lower outside ``k`` neighbour), so callers index it with
    :data:`_TRIMMED_VALID_ROW`.

    The ``sigma_mm ** 2`` normalization is applied in place (``*=`` performs
    the same single correctly-rounded multiply per element as ``d * scale``
    with identical bits, while avoiding six extra temporaries).
    """

    smoothed = _smooth(window, spacing_mm, float(sigma_mm))
    # Needed window rows: [offset - 1, offset + width] (width + 2 rows).
    lower = offset - 1
    row_count = int(smoothed.shape[0])
    rows = np.empty((width + 2, *smoothed.shape[1:]), dtype=np.float64)
    # Destination row r reads smoothed row r + lower; fill exactly the rows
    # that exist, then synthesize the missing outside neighbours below.
    first = max(0, -lower)
    last = min(width + 1, row_count - 1 - lower)
    rows[first : last + 1] = smoothed[first + lower : last + lower + 1]
    del smoothed
    if lower < 0:
        # True lower volume face: replicate the smoothed first valid row.
        rows[0] = rows[1]
    if offset + width >= row_count:
        # True upper volume face: replicate the smoothed last valid row.
        rows[width + 1] = rows[width]
    derivatives = _second_derivatives(rows, spacing_mm)
    del rows
    scale = np.float64(sigma_mm) * np.float64(sigma_mm)
    for derivative in derivatives:
        derivative *= scale
    return derivatives


def _slab_max_s(
    scaled: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    offset: int,
    width: int,
) -> float:
    """Maximum of the Frangi structure norm ``S`` over one slab's VALID rows.

    Only rows ``[offset, offset + width)`` (the valid centre) participate: halo
    rows near interior slab seams see edge-replicated padding instead of true
    neighbours, so their ``S`` values are NOT part of the scale-global
    ``max(S)``.  The ``s`` expression is the exact line mirror of
    :func:`_frangi_response` (same operations in the same order per voxel).

    The per-sub-block reduction is ``float(np.max(s))`` and the accumulation is
    :func:`_nan_propagating_max`: together they reproduce the reference
    ``np.max`` semantics through BOTH reduction levels, including the
    propagation of a NaN sub-block maximum (``max`` would discard it whenever
    the accumulator is finite).  ``max`` over floats is exact, so reducing per
    sub-block / per slab / per scale in any order yields the identical global
    ``max(S)``.
    """

    best = 0.0
    step = int(_BLOCKED_EIG_SUB_SLICES)
    stop = offset + width
    edge = offset
    while edge < stop:
        sub = min(stop, edge + step)
        eigenvalues = _symmetric_eigenvalues(
            scaled[0][edge:sub],
            scaled[1][edge:sub],
            scaled[2][edge:sub],
            scaled[3][edge:sub],
            scaled[4][edge:sub],
            scaled[5][edge:sub],
        )
        ordered = _ordered_by_magnitude(eigenvalues)
        del eigenvalues
        l1 = ordered[..., 0]
        l2 = ordered[..., 1]
        l3 = ordered[..., 2]
        s = np.sqrt(np.maximum(l1 * l1 + l2 * l2 + l3 * l3, 0.0))
        del ordered
        best = _nan_propagating_max(best, float(np.max(s)))
        del s
        edge = sub
    return best


def _frangi_response_with_global_c(
    eigenvalues: np.ndarray,
    alpha: float,
    beta: float,
    c: float,
) -> np.ndarray:
    """Bright-structure Frangi response for a caller-supplied scale ``c``.

    A line mirror of the frozen :func:`_frangi_response`: every per-voxel
    operation below appears in the same order with the same operands, except
    that ``c`` - the deterministic global ``gamma * max(S)`` of the whole
    scale - is passed in instead of being reduced from the local input.  Feeding
    this function the ordered eigenvalues of any partition with the scale-global
    ``c`` therefore reproduces :func:`_frangi_response` bit-identically on that
    partition.
    """

    l1 = eigenvalues[..., 0]
    l2 = eigenvalues[..., 1]
    l3 = eigenvalues[..., 2]
    s = np.sqrt(np.maximum(l1 * l1 + l2 * l2 + l3 * l3, 0.0))
    eps = np.finfo(np.float64).tiny
    ab_l2 = np.abs(l2)
    ab_l3 = np.maximum(np.abs(l3), eps)
    ra = ab_l2 / ab_l3
    rb = np.abs(l1) / np.sqrt(np.maximum(ab_l2 * ab_l3, eps))
    first = 1.0 - np.exp(-(ra * ra) / (2.0 * alpha * alpha))
    second = np.exp(-(rb * rb) / (2.0 * beta * beta))
    if c > 0.0:
        third = 1.0 - np.exp(-(s * s) / (2.0 * c * c))
    else:
        third = np.zeros_like(s)
    response = first * second * third
    # Bright structures: a bright tube has two large negative eigenvalues.
    response = np.where((l2 < 0.0) & (l3 < 0.0), response, 0.0)
    return response


def _slab_response_into(
    scaled: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    alpha: float,
    beta: float,
    c: float,
    offset: int,
    out: np.ndarray,
) -> None:
    """Evaluate the scale response of one slab into the preallocated ``out``.

    The eigensolve, stable ordering and response are pointwise, so the valid
    rows are evaluated in halo-free sub-blocks; ``out`` spans exactly the valid
    rows ``(width, j, i)`` while ``scaled`` spans the trimmed slab starting at
    row ``offset``.  Every voxel sees the same operations in the same order as
    the full-volume path.
    """

    step = int(_BLOCKED_EIG_SUB_SLICES)
    stop = offset + int(out.shape[0])
    edge = offset
    while edge < stop:
        sub = min(stop, edge + step)
        eigenvalues = _symmetric_eigenvalues(
            scaled[0][edge:sub],
            scaled[1][edge:sub],
            scaled[2][edge:sub],
            scaled[3][edge:sub],
            scaled[4][edge:sub],
            scaled[5][edge:sub],
        )
        ordered = _ordered_by_magnitude(eigenvalues)
        del eigenvalues
        out[edge - offset : sub - offset] = _frangi_response_with_global_c(
            ordered, alpha, beta, c
        )
        del ordered
        edge = sub


def _frangi_vesselness_blocked(
    array: np.ndarray,
    spacing: tuple[float, float, float],
    parameters: VesselnessParameters,
) -> np.ndarray:
    """Memory-bounded form of :func:`_frangi_vesselness_unblocked` (bit-exact).

    See the "Memory-bounded full-volume evaluation" section of the module
    docstring for the halo-exactness argument.  The returned array is
    bit-identical to the historical full-volume path.

    The scale-global ``max(S)`` is reduced in two levels -
    :func:`_slab_max_s` per slab, :func:`_nan_propagating_max` across slabs and
    scales - so the NaN semantics of the reference ``np.max`` (which never
    skips NaN) survive the partition: a NaN ``S`` value makes ``c`` NaN exactly
    as in :func:`_frangi_response`, which then takes the same zero third factor
    and yields the same (possibly all-zero) response bytes.  A genuinely
    non-finite response is refused by the same final finiteness check on both
    paths.
    """

    shape = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
    result: np.ndarray | None = None
    for sigma_mm in parameters.scales_mm:
        sigma = float(sigma_mm)
        # Halo-exact slabbing: the halo is the smoothing support plus the
        # Hessian stencil row; the valid width is the extended-slab budget
        # minus the two halos (floored so large halos cannot degenerate).
        halo = _k_smoothing_radius(sigma, spacing) + 1
        valid = max(int(_BLOCKED_MIN_VALID_SLICES), int(_BLOCKED_MAX_EXTENDED_SLICES) - 2 * halo)
        bounds = _slab_bounds(shape[0], valid)
        # Pass 1: the scale-global max(S) for c = gamma * max(S).
        scale_max_s = 0.0
        for start, stop in bounds:
            window, offset = _slab_window(array, start, stop, halo)
            scaled = _slab_scaled_derivatives(
                window,
                spacing,
                sigma,
                offset=offset,
                width=stop - start,
            )
            del window
            slab_max_s = _slab_max_s(scaled, _TRIMMED_VALID_ROW, stop - start)
            del scaled
            scale_max_s = _nan_propagating_max(scale_max_s, slab_max_s)
        c = float(parameters.gamma) * float(scale_max_s)
        # Pass 2: the scale response, max-accumulated slab by slab into the
        # single full-volume result (element-wise MAXIMUM in parameter order,
        # identical to the frozen per-scale combination).
        for start, stop in bounds:
            window, offset = _slab_window(array, start, stop, halo)
            scaled = _slab_scaled_derivatives(
                window,
                spacing,
                sigma,
                offset=offset,
                width=stop - start,
            )
            del window
            valid_response = np.empty(
                (stop - start, shape[1], shape[2]), dtype=np.float64, order="C"
            )
            _slab_response_into(
                scaled,
                parameters.alpha,
                parameters.beta,
                c,
                _TRIMMED_VALID_ROW,
                valid_response,
            )
            del scaled
            if result is None:
                result = np.zeros(shape, dtype=np.float64, order="C")
                result[start:stop] = valid_response
            else:
                result[start:stop] = np.maximum(result[start:stop], valid_response)
            del valid_response
    assert result is not None, "scales_mm is validated non-empty"
    if not bool(np.all(np.isfinite(result))):
        raise _refuse(
            CoronaryRefusalCode.NONFINITE_RESPONSE,
            "the vesselness response contains non-finite values",
        )
    output = np.asarray(result, dtype=RESPONSE_DTYPE, order="C")
    output.setflags(write=False)
    return output


def _frangi_vesselness_unblocked(
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    parameters: VesselnessParameters,
) -> np.ndarray:
    """The historical full-volume pipeline (frozen reference, verbatim).

    Retained as the independent Tier-A reference for the memory-bounded path:
    the parity battery compares the dispatcher output against this function
    bit-for-bit.
    """

    if not isinstance(parameters, VesselnessParameters):
        raise _refuse(
            CoronaryRefusalCode.PARAMETERS_INVALID,
            "parameters must be a VesselnessParameters instance",
        )
    array = _require_volume(volume)
    spacing = _require_spacing(spacing_mm)

    result: np.ndarray | None = None
    for sigma_mm in parameters.scales_mm:
        smoothed = _smooth(array, spacing, float(sigma_mm))
        d_ii, d_jj, d_kk, d_ij, d_ik, d_jk = _second_derivatives(smoothed, spacing)
        scale = np.float64(sigma_mm) * np.float64(sigma_mm)
        eigenvalues = _symmetric_eigenvalues(
            d_ii * scale, d_jj * scale, d_kk * scale, d_ij * scale, d_ik * scale, d_jk * scale
        )
        ordered = _ordered_by_magnitude(eigenvalues)
        response = _frangi_response(ordered, parameters.alpha, parameters.beta, parameters.gamma)
        result = response if result is None else np.maximum(result, response)
    assert result is not None, "scales_mm is validated non-empty"
    if not bool(np.all(np.isfinite(result))):
        raise _refuse(
            CoronaryRefusalCode.NONFINITE_RESPONSE,
            "the vesselness response contains non-finite values",
        )
    output = np.asarray(result, dtype=RESPONSE_DTYPE, order="C")
    output.setflags(write=False)
    return output


def frangi_vesselness(
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    parameters: VesselnessParameters,
) -> np.ndarray:
    """The multi-scale Frangi-class vesselness of one canonical volume.

    ``volume`` is the canonical ``(k, j, i)`` array (int16 HU samples or a
    float array); ``spacing_mm`` is ``(i, j, k)`` millimetres per index step.
    The returned read-only ``<f8`` array is bit-identical for identical
    inputs (no RNG anywhere).

    Inputs at or above :data:`_BLOCKED_MIN_VOXELS` take the memory-bounded
    slab path; smaller inputs take the verbatim historical full-volume path.
    Both paths are bit-identical by construction and by the parity battery.
    """

    if not isinstance(parameters, VesselnessParameters):
        raise _refuse(
            CoronaryRefusalCode.PARAMETERS_INVALID,
            "parameters must be a VesselnessParameters instance",
        )
    array = _require_volume(volume)
    spacing = _require_spacing(spacing_mm)
    if int(array.size) < int(_BLOCKED_MIN_VOXELS):
        return _frangi_vesselness_unblocked(array, spacing, parameters)
    return _frangi_vesselness_blocked(array, spacing, parameters)


def vessel_mask(response: np.ndarray, parameters: VesselnessParameters) -> np.ndarray:
    """The closed mask predicate ``response > threshold`` (read-only bool)."""

    if not isinstance(parameters, VesselnessParameters):
        raise _refuse(
            CoronaryRefusalCode.PARAMETERS_INVALID,
            "parameters must be a VesselnessParameters instance",
        )
    array = np.asarray(response, dtype=np.float64)
    if array.ndim != 3 or not bool(np.all(np.isfinite(array))):
        raise _refuse(
            CoronaryRefusalCode.VOLUME_INVALID,
            "response must be a finite 3-D array",
        )
    mask = np.asarray(array > float(parameters.threshold), dtype=bool, order="C")
    mask.setflags(write=False)
    return mask


__all__ = [
    "RESPONSE_DTYPE",
    "frangi_vesselness",
    "vessel_mask",
]
