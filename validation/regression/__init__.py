"""Golden-digest regression harness (WS02_DESIGN.md §B9)."""

from validation.regression.runner import (
    GOLDEN_SCHEMA,
    IDENTITY_IGNORE_KEYS,
    RegressionDifference,
    RegressionResult,
    RegressionRunner,
)

__all__ = [
    "GOLDEN_SCHEMA",
    "IDENTITY_IGNORE_KEYS",
    "RegressionDifference",
    "RegressionResult",
    "RegressionRunner",
]
