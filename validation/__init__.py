"""Validation framework baseline (WS02_DESIGN.md §B9).

Feeds acceptance EVIDENCE only — promotion authority stays in
``models.registry`` (Rule H). Subpackages:

- ``regression``: golden-digest regression harness over the exit-gate report;
- ``metrics``: §48 workflow metric vocabulary computed from EventLog replays.
"""

from validation.metrics.engine import (
    STRATA_FIELDS,
    StratumFields,
    WorkflowMetrics,
    WorkflowMetricsEngine,
)
from validation.regression.runner import (
    GOLDEN_SCHEMA,
    RegressionDifference,
    RegressionResult,
    RegressionRunner,
)

__all__ = [
    "GOLDEN_SCHEMA",
    "STRATA_FIELDS",
    "RegressionDifference",
    "RegressionResult",
    "RegressionRunner",
    "StratumFields",
    "WorkflowMetrics",
    "WorkflowMetricsEngine",
]
