"""Pure-NumPy distance-guided thinning and skeleton trace.

Skeletonization ("thinning") of the binary derivation mask:

1. **Exact Euclidean distance transform** (Felzenszwalb's separable squared
   transform; pure NumPy): every mask voxel receives its exact distance to
   the nearest background voxel, in millimetres.
2. **Geodesic spine**: two deterministic breadth-first searches over the
   26-connected mask resolve the geodesic diameter of the seed's connected
   component (stable neighbor order; the first-discovered voxel breaks a
   distance tie), and the parent chain yields a connected one-voxel-wide
   path through the mask.
3. **Centering**: every spine voxel is relaxed towards the distance ridge by
   moving to the candidate in its 26-neighborhood that maximizes the
   distance transform, subject to the candidate staying 26-adjacent to BOTH
   neighbours of the spine (so the chain remains connected and consecutive
   points stay within one voxel), iterated until stable.

The result is a connected one-voxel-wide medial spine of the mask, centered
on the distance ridge.  The whole procedure is deterministic - fixed orders,
index tie-breaks, no RNG - and is a pure function of the mask.

The trace then orients the ordered spine with the human direction hint (the
hint points INTO the vessel: the polyline starts at the end that lies BEHIND
the seed along the hint and ends ahead of it).

Rule C boundary: this module produces structure only - an ordered voxel
polyline; no diameter, area or stenosis is computed anywhere.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Final

import numpy as np

from core.coronary.types import CoronaryRefusalCode, _refuse

#: The fixed 26-neighborhood offset order (deterministic BFS/relaxation).
_OFFSETS26: Final[tuple[tuple[int, int, int], ...]] = tuple(
    (di, dj, dk)
    for dk in (-1, 0, 1)
    for dj in (-1, 0, 1)
    for di in (-1, 0, 1)
    if not (di == 0 and dj == 0 and dk == 0)
)

#: Centering rounds of the spine relaxation (fixpoint bounded by the mask).
_CENTERING_ROUNDS: Final[int] = 24


def exact_edt(mask: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance transform of a boolean mask (millimetres).

    Felzenszwalb & Huttenlocher's separable squared transform: the 1-D
    squared distance to the nearest background element is computed exactly
    along each axis (in ``i``, ``j``, ``k`` order), then square-rooted once.
    Voxels outside the mask are 0 by construction; the caller only reads the
    transform at mask voxels.  The result of this function is a pure
    deterministic function of the mask (no RNG, no reduction order hazards).
    """

    def transform_1d(line: np.ndarray) -> np.ndarray:
        n = line.shape[0]
        result = np.empty(n, dtype=np.float64)
        vertex = np.zeros(n, dtype=np.int64)
        boundary = np.zeros(n + 1, dtype=np.float64)
        vertex[0] = 0
        boundary[0] = -np.inf
        boundary[1] = np.inf
        k = 0
        for q in range(1, n):
            s = ((line[q] + q * q) - (line[vertex[k]] + vertex[k] * vertex[k])) / (
                2.0 * q - 2.0 * vertex[k]
            )
            while s <= boundary[k]:
                k -= 1
                s = ((line[q] + q * q) - (line[vertex[k]] + vertex[k] * vertex[k])) / (
                    2.0 * q - 2.0 * vertex[k]
                )
            k += 1
            vertex[k] = q
            boundary[k] = s
            boundary[k + 1] = np.inf
        k = 0
        for q in range(n):
            while boundary[k + 1] < q:
                k += 1
            result[q] = float((q - vertex[k]) ** 2) + float(line[vertex[k]])
        return result

    array = np.asarray(mask)
    if array.ndim != 3 or array.dtype != bool:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "mask must be a 3-D bool array")
    outside = 1e12
    transform = np.where(array, outside, 0.0)
    for axis in range(3):
        transform = np.apply_along_axis(transform_1d, axis, transform)
    return np.sqrt(transform)


def _bfs(
    mask: np.ndarray, start: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = mask.shape
    distance = np.full(shape, -1, dtype=np.int64)
    order = np.full(shape, -1, dtype=np.int64)
    parent = np.full(shape + (3,), -1, dtype=np.int64)
    distance[start] = 0
    order[start] = 0
    discovery = 1
    queue: deque[tuple[int, int, int]] = deque([start])
    while queue:
        ck, cj, ci = queue.popleft()
        for di, dj, dk in _OFFSETS26:
            nk, nj, ni = ck + dk, cj + dj, ci + di
            if 0 <= nk < shape[0] and 0 <= nj < shape[1] and 0 <= ni < shape[2]:
                if mask[nk, nj, ni] and distance[nk, nj, ni] == -1:
                    distance[nk, nj, ni] = distance[ck, cj, ci] + 1
                    parent[nk, nj, ni] = (ck, cj, ci)
                    order[nk, nj, ni] = discovery
                    discovery += 1
                    queue.append((nk, nj, ni))
    return distance, parent, order


def _extreme(distance: np.ndarray, order: np.ndarray) -> tuple[int, int, int]:
    best_distance = int(distance.max())
    candidates = np.argwhere(distance == best_distance)
    keys = order[candidates[:, 0], candidates[:, 1], candidates[:, 2]]
    chosen = candidates[int(np.argmin(keys))]
    return (int(chosen[0]), int(chosen[1]), int(chosen[2]))


def _chain(
    parent: np.ndarray, start: tuple[int, int, int], end: tuple[int, int, int]
) -> list[tuple[int, int, int]]:
    chain: list[tuple[int, int, int]] = []
    cursor = end
    while cursor != start:
        chain.append(cursor)
        previous = parent[cursor[0], cursor[1], cursor[2]]
        cursor = (int(previous[0]), int(previous[1]), int(previous[2]))
    chain.append(start)
    chain.reverse()
    return chain


def _neighborhood(
    voxel: tuple[int, int, int], shape: tuple[int, ...]
) -> list[tuple[int, int, int]]:
    ck, cj, ci = voxel
    result: list[tuple[int, int, int]] = []
    for di, dj, dk in _OFFSETS26:
        nk, nj, ni = ck + dk, cj + dj, ci + di
        if 0 <= nk < shape[0] and 0 <= nj < shape[1] and 0 <= ni < shape[2]:
            result.append((nk, nj, ni))
    return result


def _adjacent(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> bool:
    return max(
        abs(left[0] - right[0]), abs(left[1] - right[1]), abs(left[2] - right[2])
    ) <= 1


def _geodesic_segment(
    mask: np.ndarray, start: tuple[int, int, int], end: tuple[int, int, int]
) -> list[tuple[int, int, int]]:
    """Shortest 26-connected voxel path from *start* to *end* through *mask*.

    Deterministic: the fixed neighbor order decides the BFS discovery order
    and therefore every parent link; the segment excludes *start*.
    """

    if _adjacent(start, end):
        return [end]
    parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    queue: deque[tuple[int, int, int]] = deque([start])
    seen = {start}
    while queue:
        current = queue.popleft()
        if current == end:
            break
        for neighbor in _neighborhood(current, mask.shape):
            if neighbor in seen or not bool(mask[neighbor]):
                continue
            seen.add(neighbor)
            parent[neighbor] = current
            queue.append(neighbor)
    if end not in parent:
        return [end]
    segment: list[tuple[int, int, int]] = []
    cursor = end
    while cursor != start:
        segment.append(cursor)
        cursor = parent[cursor]
    segment.reverse()
    return segment


def _densify(
    mask: np.ndarray, path: list[tuple[int, int, int]]
) -> list[tuple[int, int, int]]:
    """Reconnect every consecutive pair of spine voxels with a geodesic run."""

    result = [path[0]]
    for point in path[1:]:
        if _adjacent(result[-1], point):
            result.append(point)
            continue
        result.extend(_geodesic_segment(mask, result[-1], point))
    return result


def _ridge_path(
    mask: np.ndarray,
    field: np.ndarray,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    """Ridge-weighted shortest voxel path from *start* to *end*.

    The step cost is ``length / (distance + 0.5)``: a long diagonal step is
    more expensive than an axis step, and a voxel deep inside the vessel is
    cheaper than a voxel at the wall, so the path hugs the distance ridge
    instead of cutting a corner across a junction blob.  Ties are broken by
    insertion order (a monotonic counter), so the result is deterministic.
    """

    step_cost = {1: 1.0, 2: 1.4142135623730951, 3: 1.7320508075688772}
    best: dict[tuple[int, int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    counter = 0
    heap: list[tuple[float, int, tuple[int, int, int]]] = [(0.0, counter, start)]
    settled = 0
    while heap:
        cost, _, current = heapq.heappop(heap)
        if cost > best.get(current, float("inf")) + 1e-12:
            continue
        settled += 1
        if current == end:
            break
        for neighbor in _neighborhood(current, mask.shape):
            if not bool(mask[neighbor]):
                continue
            step = max(
                abs(neighbor[0] - current[0]),
                abs(neighbor[1] - current[1]),
                abs(neighbor[2] - current[2]),
            )
            if step > 1:
                continue
            length = step_cost[
                abs(neighbor[0] - current[0])
                + abs(neighbor[1] - current[1])
                + abs(neighbor[2] - current[2])
            ]
            candidate = cost + length * (1.0 / (float(field[neighbor]) + 0.5))
            if candidate < best.get(neighbor, float("inf")) - 1e-12:
                best[neighbor] = candidate
                parent[neighbor] = current
                counter += 1
                heapq.heappush(heap, (candidate, counter, neighbor))
    if end not in parent:
        return _geodesic_segment(mask, start, end)
    segment: list[tuple[int, int, int]] = []
    cursor = end
    while cursor != start:
        segment.append(cursor)
        cursor = parent[cursor]
    segment.reverse()
    return segment


def thinning_skeleton(
    mask: np.ndarray, seed_voxel: tuple[int, int, int]
) -> np.ndarray:
    """One connected one-voxel-wide medial spine of the seed's component.

    Returns an ``(N, 3)`` int array of voxels in canonical ``(k, j, i)``
    order, walking the component's geodesic diameter after the centering
    relaxation.  The seed must be a foreground mask voxel (the caller refuses
    ``NOT_ANALYZABLE`` otherwise - this function never invents voxels).

    F4: the geodesic diameter of the component can bypass the seed's branch
    entirely (e.g. a T-shaped mask seeded at its short-branch tip).  The
    selection therefore verifies that the chosen spine passes through the
    seed's local lumen; when it does not, ONE seed-anchored path is selected
    (seed branch -> attachment junction -> the farther diameter end).  If even
    that path cannot carry the seed's branch, the derivation abstains with
    ``NOT_ANALYZABLE`` - it never discards the human seed silently.
    """

    array = np.asarray(mask)
    if array.ndim != 3 or array.dtype != bool:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "mask must be a 3-D bool array")
    seed = (int(seed_voxel[0]), int(seed_voxel[1]), int(seed_voxel[2]))
    shape = array.shape
    if not (
        0 <= seed[0] < shape[0] and 0 <= seed[1] < shape[1] and 0 <= seed[2] < shape[2]
    ):
        raise _refuse(
            CoronaryRefusalCode.SEED_OUT_OF_BOUNDS,
            f"seed voxel {seed} is outside the volume {shape}",
        )
    if not bool(array[seed]):
        raise _refuse(
            CoronaryRefusalCode.NOT_ANALYZABLE,
            "the seed voxel carries no vessel evidence in the mask",
        )
    distance, parent, order = _bfs(array, seed)
    end_a = _extreme(distance, order)
    distance_b, _parent_b, order_b = _bfs(array, end_a)
    end_b = _extreme(distance_b, order_b)
    field = exact_edt(array)
    spine = _ridge_path(array, field, end_a, end_b)
    if len(spine) < 2:
        raise _refuse(
            CoronaryRefusalCode.NOT_ANALYZABLE,
            "the seed's mask component is a single voxel",
        )
    seed_anchored = False
    if not _seed_inside_spine_tube(spine, seed, field):
        # F4: the diameter path discarded the seed's branch.  Select the ONE
        # path that carries the seed instead of snapping the seed onto an
        # unrelated branch.
        spine = _seed_branch_spine(
            array, seed, distance, parent, end_a, end_b, spine
        )
        seed_anchored = True
    current = _center_spine(array, field, spine)
    refined = list(current)
    if seed_anchored:
        # The seed end is a human-placed lumen point (the branch tip); only
        # the far end is walked to the distance ridge, so the branch is never
        # swallowed by end refinement.
        _refine_path_end(array, field, refined, len(refined) - 1)
    else:
        _refine_path_end(array, field, refined, 0)
        _refine_path_end(array, field, refined, len(refined) - 1)
    dense = _densify(array, refined)
    result = np.asarray(dense, dtype=np.int64)
    # F4 final guard: the finished spine must still pass through the seed's
    # local lumen; otherwise the single path cannot be resolved and the
    # derivation abstains (Rule 10 - no fabricated seed-directed trace).
    if not _seed_inside_spine_tube(result, seed, field):
        raise _refuse(
            CoronaryRefusalCode.NOT_ANALYZABLE,
            "the selected path cannot resolve the seed's branch",
        )
    return result


def _seed_inside_spine_tube(
    spine: np.ndarray | list[tuple[int, int, int]],
    seed: tuple[int, int, int],
    field: np.ndarray,
) -> bool:
    """Does the spine pass through the seed's local cross-section?

    A spine voxel ``p`` covers the seed when the seed lies inside ``p``'s
    distance-to-background tube: ``|seed - p| <= EDT(p) + 0.5`` (index units).
    A seed on a discarded side branch (T-mask probe: 9 voxels away from the
    diameter path, local radius ~2) fails the test; a seed inside the lumen of
    the selected path passes it regardless of which exact voxel it clicked.
    """

    seed_array = np.asarray(seed, dtype=np.float64)
    for voxel in spine:
        delta = (
            float(voxel[0]) - seed_array[0],
            float(voxel[1]) - seed_array[1],
            float(voxel[2]) - seed_array[2],
        )
        euclidean = math.sqrt(
            delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
        )
        if euclidean <= float(field[int(voxel[0]), int(voxel[1]), int(voxel[2])]) + 0.5:
            return True
    return False


def _seed_branch_spine(
    array: np.ndarray,
    seed: tuple[int, int, int],
    seed_distance: np.ndarray,
    seed_parent: np.ndarray,
    end_a: tuple[int, int, int],
    end_b: tuple[int, int, int],
    diameter_path: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """ONE path through the seed's branch to the farther diameter end.

    * ``junction`` - the diameter-path voxel nearest to the seed (its BFS
      distance from the seed is minimal; ties break on the path index, so the
      selection is deterministic);
    * ``seed arm`` - the BFS chain seed -> junction (contains the seed);
    * ``far arm`` - the BFS chain junction -> the diameter end that is farther
      from the junction.

    The concatenation is a simple 26-connected path (the two arms only share
    the junction); a connectivity failure is a typed abstention, never a
    silently densified jump.
    """

    junction_index = min(
        range(len(diameter_path)),
        key=lambda index: float(
            seed_distance[
                diameter_path[index][0], diameter_path[index][1], diameter_path[index][2]
            ]
        ),
    )
    junction = diameter_path[junction_index]
    seed_arm = _chain(seed_parent, seed, junction)
    junction_distance, junction_parent, _order = _bfs(array, junction)
    far = (
        end_a
        if float(junction_distance[end_a]) >= float(junction_distance[end_b])
        else end_b
    )
    far_arm = _chain(junction_parent, junction, far)
    candidate = seed_arm + far_arm[1:]
    for left, right in zip(candidate, candidate[1:]):
        if not _adjacent(left, right):
            raise _refuse(
                CoronaryRefusalCode.NOT_ANALYZABLE,
                "the seed-anchored path does not stay 26-connected",
            )
    return candidate


def _center_spine(
    array: np.ndarray,
    field: np.ndarray,
    spine: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Relax every spine voxel towards the distance ridge (bounded fixpoint)."""

    shape = array.shape
    current = spine
    for _round in range(_CENTERING_ROUNDS):
        improved = False
        moved: list[tuple[int, int, int]] = []
        for index, voxel in enumerate(current):
            previous = moved[-1] if moved else None
            following = current[index + 1] if index + 1 < len(current) else None
            best = voxel
            best_value = float(field[voxel])
            for candidate in _neighborhood(voxel, shape):
                if not bool(array[candidate]):
                    continue
                if previous is not None and not _adjacent(previous, candidate):
                    continue
                if following is not None and not _adjacent(following, candidate):
                    continue
                value = float(field[candidate])
                if value > best_value:
                    best_value = value
                    best = candidate
            if best != voxel:
                improved = True
            moved.append(best)
        deduped: list[tuple[int, int, int]] = []
        for point in moved:
            if not deduped or deduped[-1] != point:
                deduped.append(point)
        current = deduped
        if not improved:
            break
    if len(current) < 2:
        raise _refuse(
            CoronaryRefusalCode.NOT_ANALYZABLE,
            "the centered spine degenerated below two voxels",
        )
    return current


def _refine_path_end(
    array: np.ndarray,
    field: np.ndarray,
    spine: list[tuple[int, int, int]],
    index: int,
) -> None:
    """Walk one spine end to the local distance-ridge maximum (bounded).

    The geodesic extremes of a capped tube sit on the CAP RIM (up to one tube
    radius off the axis); walking the end to its local distance maximum (up to
    a bounded number of steps) puts it back on the vessel axis.  Deterministic:
    the first improving neighbour in the fixed offset order wins.
    """

    shape = array.shape
    cursor = spine[index]
    for _step in range(8):
        best = cursor
        best_value = float(field[cursor])
        for candidate in _neighborhood(cursor, shape):
            if not bool(array[candidate]):
                continue
            value = float(field[candidate])
            if value > best_value:
                best_value = value
                best = candidate
        if best == cursor:
            break
        cursor = best
    spine[index] = cursor


def orient_skeleton_path(
    spine: np.ndarray,
    seed_voxel: tuple[int, int, int],
    direction_voxel: tuple[float, float, float],
) -> np.ndarray:
    """Orient one ordered spine with the human direction hint.

    The hint is the walking direction INTO the vessel: the returned polyline
    starts at the spine end BEHIND the seed (the smaller projection onto the
    hint) and ends AHEAD of it (the larger projection).  The comparison is
    strict-inclusive so an exact tie keeps the input order (deterministic,
    documented).  ``seed_voxel`` and ``direction_voxel`` are both in the
    canonical ``(k, j, i)`` voxel order used by the spine.
    """

    array = np.asarray(spine)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 2:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "spine must be an (N,3) voxel path")
    hint = np.asarray(direction_voxel, dtype=np.float64)
    if (
        hint.shape != (3,)
        or not bool(np.all(np.isfinite(hint)))
        or float(np.linalg.norm(hint)) == 0.0
    ):
        raise _refuse(
            CoronaryRefusalCode.SEED_DIRECTION_INVALID,
            "the traced direction must be a finite non-zero 3-vector",
        )
    seed = np.asarray(seed_voxel, dtype=np.float64)
    start = array[0].astype(np.float64)
    end = array[-1].astype(np.float64)
    start_dot = float(np.dot(start - seed, hint))
    end_dot = float(np.dot(end - seed, hint))
    # F2: start BEHIND the seed (the smaller projection onto the hint) and end
    # AHEAD of it.  The previous form kept the path whose start was ahead,
    # reversing the human direction hint.
    if start_dot <= end_dot:
        return np.ascontiguousarray(array)
    return np.ascontiguousarray(array[::-1])


def _component_size(skeleton: np.ndarray, seed: tuple[int, int, int]) -> int:
    """Diagnostic count of the seed's 26-connected component (voxel set)."""

    if not bool(skeleton[seed[0], seed[1], seed[2]]):
        return 0
    distance, _parent, _order = _bfs(skeleton, seed)
    return int(np.count_nonzero(distance >= 0))


def trace_skeleton(
    skeleton: np.ndarray,
    seed_voxel: tuple[int, int, int],
    direction_voxel: tuple[float, float, float],
) -> np.ndarray:
    """Backwards-compatible seam: orient a *voxel-set* skeleton.

    The slice's degenerate voxel-set case is the one-voxel component; the
    governed pipeline uses :func:`thinning_skeleton` (ordered spine) plus
    :func:`orient_skeleton_path`.  This helper exists for callers that hold a
    skeleton SET and resolves its geodesic diameter exactly as the thinning
    does, then orients it.
    """

    array = np.asarray(skeleton)
    if array.ndim != 3 or array.dtype != bool:
        raise _refuse(CoronaryRefusalCode.VOLUME_INVALID, "skeleton must be a 3-D bool array")
    spine = thinning_skeleton(array, seed_voxel)
    return orient_skeleton_path(spine, seed_voxel, direction_voxel)


__all__ = [
    "exact_edt",
    "orient_skeleton_path",
    "thinning_skeleton",
    "trace_skeleton",
]
