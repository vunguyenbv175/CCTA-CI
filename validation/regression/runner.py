"""Golden-digest regression harness over the exit-gate report (§B9).

Runs a saved session through :class:`RoundTripVerificationEngine` and diffs
the fresh machine-readable report against a committed golden document.
Identity fields minted per-run (ULIDs) are pruned before comparison; every
content digest, measurement value and check outcome must match exactly —
golden-file discipline with locked digests (Part D).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from apps.workstation.round_trip import (
    REPORT_SCHEMA,
    RoundTripVerificationEngine,
    VerificationReport,
)

GOLDEN_SCHEMA: Final[str] = "ccta.validation.golden_report.v1"

#: Identity minted at run time; excluded from golden comparison. Everything
#: else — digests, matrices, measurements, verdicts — is locked.
IDENTITY_IGNORE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact_id",
        "provenance_id",
        "series_id",
        "session_digest",
        "study_id",
        "volume_id",
    }
)


def _prune(value: object, path: str) -> tuple[object, list[tuple[str, str]]]:
    """Return value with ignored identity keys removed (recursively)."""
    if isinstance(value, dict):
        out: dict[str, object] = {}
        removals: list[tuple[str, str]] = []
        for key in sorted(value, key=str):
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if key_text in IDENTITY_IGNORE_KEYS:
                removals.append((child_path, "identity"))
                continue
            pruned, nested = _prune(value[key], child_path)
            out[key_text] = pruned
            removals.extend(nested)
        return out, removals
    if isinstance(value, list):
        items: list[object] = []
        nested_removals: list[tuple[str, str]] = []
        for index, item in enumerate(value):
            pruned, nested = _prune(item, f"{path}[{index}]")
            items.append(pruned)
            nested_removals.extend(nested)
        return items, nested_removals
    return value, []


def _flatten(value: object, path: str) -> dict[str, object]:
    flat: dict[str, object] = {}
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            flat.update(_flatten(value[key], f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flat.update(_flatten(item, f"{path}[{index}]"))
    else:
        flat[path or "<root>"] = value
    return flat


@dataclass(frozen=True)
class RegressionDifference:
    pointer: str
    golden: object
    fresh: object

    def to_dict(self) -> dict[str, object]:
        return {"golden": self.golden, "fresh": self.fresh, "pointer": self.pointer}


@dataclass(frozen=True)
class RegressionResult:
    matched: bool
    fresh_verdict: str
    golden_verdict: str
    differences: tuple[RegressionDifference, ...]

    @property
    def verdict(self) -> str:
        return "PASS" if self.matched else "FAIL"

    def to_dict(self) -> dict[str, object]:
        return {
            "fresh_verdict": self.fresh_verdict,
            "golden_verdict": self.golden_verdict,
            "differences": [diff.to_dict() for diff in self.differences],
            "matched": self.matched,
            "verdict": self.verdict,
        }


class RegressionRunner:
    """Fresh exit-gate run vs committed golden document."""

    def __init__(self, engine: RoundTripVerificationEngine) -> None:
        self._engine = engine

    def run_fresh(self, workspace_root: Path) -> VerificationReport:
        from apps.workstation.session_store import DerivedStateStoreManager, WorkstationPaths

        paths = WorkstationPaths.create(Path(workspace_root))
        session = DerivedStateStoreManager(paths).reopen()
        return self._engine.verify(session)

    def write_golden(self, report: VerificationReport, path: Path) -> None:
        document = {
            "schema": GOLDEN_SCHEMA,
            "report_schema": REPORT_SCHEMA,
            "report": report.to_dict(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(document, sort_keys=True, indent=2), encoding="utf-8", newline="\n"
        )

    def load_golden(self, path: Path) -> dict[str, object]:
        loaded: object = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or loaded.get("schema") != GOLDEN_SCHEMA:
            raise ValueError(f"{path} is not a {GOLDEN_SCHEMA} document")
        return loaded

    def compare(
        self,
        fresh: VerificationReport,
        golden_document: dict[str, object],
    ) -> RegressionResult:
        golden_report = golden_document.get("report")
        if not isinstance(golden_report, dict):
            raise ValueError("golden document lacks a report object")
        golden_pruned, _ = _prune(golden_report, "")
        fresh_dict = fresh.to_dict()
        fresh_pruned, _ = _prune(fresh_dict, "")

        golden_flat = _flatten(golden_pruned, "")
        fresh_flat = _flatten(fresh_pruned, "")
        differences: list[RegressionDifference] = []
        for pointer in sorted(set(golden_flat) | set(fresh_flat)):
            if pointer not in golden_flat:
                differences.append(
                    RegressionDifference(pointer=pointer, golden=None, fresh=fresh_flat[pointer])
                )
            elif pointer not in fresh_flat:
                differences.append(
                    RegressionDifference(pointer=pointer, golden=golden_flat[pointer], fresh=None)
                )
            elif golden_flat[pointer] != fresh_flat[pointer]:
                differences.append(
                    RegressionDifference(
                        pointer=pointer,
                        golden=golden_flat[pointer],
                        fresh=fresh_flat[pointer],
                    )
                )
        return RegressionResult(
            matched=not differences,
            fresh_verdict=fresh.verdict,
            golden_verdict=str(golden_report.get("verdict", "")),
            differences=tuple(differences),
        )


__all__ = [
    "GOLDEN_SCHEMA",
    "IDENTITY_IGNORE_KEYS",
    "RegressionDifference",
    "RegressionResult",
    "RegressionRunner",
]
