"""Frozen CPR PNG transport (Decision B/C): server-authoritative WL rendering.

The straightened CPR panel reuses the server-WL discipline (#25 authority):
the CPR pixel array (decoded via the frozen ``decode_cpr_pixel_blob``) is
mapped through the EXISTING frozen binary32 window/level transfer
(``apps.viewer.transfer.window_bytes``) and encoded with the EXISTING frozen
PNG encoder (``apps.viewer.png_encoder.encode_png``).  There is NO second
intensity pipeline and NO client-side LUT: the WL inputs are the session's
current effective window (preset / custom / raw-identity), resolved by the
engine exactly like the MPR LAYER-R path.

The CPR view has no LAYER-G geometry frames of its own, so there are no
``ViewerFrameRecord``\\ s here: the PNG identity is
``(cpr_artifact_id, wl_preset_id, wl_applied)`` and the bytes are
re-derivable from the sealed payload + the stored window.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np

from apps.viewer.png_encoder import encode_png, pixel_digest
from apps.viewer.transfer import (
    TRANSFER_FUNCTION_ID,
    exact_extrema,
    identity_window_bounds,
    window_bytes,
)

#: The transfer applied to CPR pixels (identical frozen function as MPR).
CPR_TRANSFER_FUNCTION_ID: Final[str] = TRANSFER_FUNCTION_ID


def cpr_window_for_pixels(
    pixels: np.ndarray,
    *,
    wl_preset_id: str | None,
    preset_window: tuple[float, float] | None = None,
    custom_window: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Resolve the effective CPR window under server WL authority.

    Exactly one source wins: an explicit ``custom_window`` (the validated
    binary32 SET_WL_CUSTOM values), a preset (``wl_preset_id`` + its stored
    ``preset_window``), or the raw identity window over the decoded CPR pixel
    extrema when both are absent.  Supplying both is a caller error (never a
    silent pick).
    """

    if custom_window is not None and wl_preset_id is not None:
        raise ValueError("custom_window and wl_preset_id are mutually exclusive")
    if custom_window is not None:
        center, width = custom_window
        return (float(center), float(width))
    if wl_preset_id is not None:
        if preset_window is None:
            raise ValueError(f"preset {wl_preset_id!r} carries no stored window")
        center, width = preset_window
        return (float(center), float(width))
    minimum, maximum = exact_extrema(pixels)
    return identity_window_bounds(minimum, maximum)


def render_cpr_png(
    pixels: np.ndarray, center: float, width: float
) -> tuple[bytes, str]:
    """Render CPR pixels through the frozen WL transfer + PNG encoder.

    Returns ``(png_bytes, canvas_digest)`` where the digest is the semantic
    pixel digest (SHA-256 over the decoded RGB24 bytes, before PNG encoding).
    Inputs are never mutated.
    """

    array = np.asarray(pixels)
    if array.ndim != 2:
        raise ValueError("render_cpr_png requires a 2-D CPR pixel array")
    canvas = window_bytes(array, float(center), float(width))
    return (encode_png(canvas), pixel_digest(canvas))


def cpr_png_identity(
    cpr_artifact_id: str,
    wl_preset_id: str | None,
    wl_applied: Mapping[str, float],
) -> tuple[str, str | None, float, float]:
    """The stable PNG identity tuple of one CPR render."""

    return (
        str(cpr_artifact_id),
        wl_preset_id,
        float(wl_applied["center"]),
        float(wl_applied["width"]),
    )


__all__ = [
    "CPR_TRANSFER_FUNCTION_ID",
    "cpr_png_identity",
    "cpr_window_for_pixels",
    "render_cpr_png",
]
