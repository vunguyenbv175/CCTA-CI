"""Frozen window/level preset table (VIEWER_001_SPEC Part 4).

Presets are the ONLY window/level values in v1 (no free numeric entry) so the
display vocabulary stays closed and deterministic.  Numeric/interactive WL is
a documented v2 candidate (spec Part 4.2) and must not be smuggled in here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class WLPreset:
    """One frozen window/level preset (center/width in HU)."""

    preset_id: str
    center: float
    width: float


#: The frozen v1 preset table, exactly as pinned in spec Part 4.  The tuple
#: order IS the frozen preset order (the session's ``wl_presets_active``
#: default subset follows this order).
WL_PRESETS: Final[tuple[WLPreset, ...]] = (
    WLPreset(preset_id="CT_MEDIASTINUM", center=40.0, width=400.0),
    WLPreset(preset_id="CT_VESSEL_CONTRAST", center=300.0, width=800.0),
    WLPreset(preset_id="CT_LUNG", center=-600.0, width=1500.0),
    WLPreset(preset_id="CT_BONE", center=300.0, width=1500.0),
    WLPreset(preset_id="CT_SOFT_TISSUE", center=50.0, width=350.0),
)

WL_PRESET_IDS: Final[tuple[str, ...]] = tuple(
    preset.preset_id for preset in WL_PRESETS
)

_PRESET_BY_ID: Final[dict[str, WLPreset]] = {
    preset.preset_id: preset for preset in WL_PRESETS
}


def preset_by_id(preset_id: str) -> WLPreset:
    """Return the frozen preset for *preset_id*; unknown ids are a KeyError."""

    return _PRESET_BY_ID[preset_id]


def preset_exists(preset_id: object) -> bool:
    """True iff *preset_id* is exactly one frozen preset id (closed set)."""

    return isinstance(preset_id, str) and preset_id in _PRESET_BY_ID


__all__ = [
    "WL_PRESETS",
    "WL_PRESET_IDS",
    "WLPreset",
    "preset_by_id",
    "preset_exists",
]
