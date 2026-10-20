"""Deterministic builders for models.registry tests (Part C)."""

from __future__ import annotations

import hashlib
from typing import Any

from core.enums import ReleaseLane
from models.registry.record import (
    DataPolicyBlock,
    DatasetBinding,
    EnsembleBlock,
    InputContract,
    IntendedUseBlock,
    LicenseBlock,
    LineageBlock,
    OutputContract,
    RegistryRecord,
    RiskBlock,
    RuntimeBlock,
    SourceBlock,
    ValidationBlock,
    WeightsBlock,
)

AT = "2026-02-01T08:00:00Z"
COMMIT = "abcdef0123456789abcdef0123456789abcdef01"

WEIGHTS_BYTES = b"weights-bytes-v1"


def sha(text: object) -> str:
    if isinstance(text, bytes):
        return hashlib.sha256(text).hexdigest()
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def weights_sha() -> str:
    return sha(WEIGHTS_BYTES)


def make_training_ref(**overrides: Any) -> DatasetBinding:
    kwargs: dict[str, Any] = dict(
        dataset_id="ds-train",
        snapshot_version="2026-01",
        split_role="train",
        locked=True,
        checksum_sha256=sha("ds-train"),
        source_class="public",
        patient_disjoint_asserted=True,
    )
    kwargs.update(overrides)
    return DatasetBinding(**kwargs)


def make_benchmark_ref(**overrides: Any) -> DatasetBinding:
    kwargs: dict[str, Any] = dict(
        dataset_id="ds-locked-test",
        snapshot_version="2026-01",
        split_role="locked_test",
        locked=True,
        checksum_sha256=sha("ds-locked-test"),
        source_class="local_real_world",
        patient_disjoint_asserted=True,
    )
    kwargs.update(overrides)
    return DatasetBinding(**kwargs)


def make_snapshot_lineage(**overrides: Any) -> LineageBlock:
    snap_kwargs: dict[str, Any] = dict(
        dataset_id="ds-train",
        snapshot_version="2026-01",
        split_role="train",
        locked=True,
        checksum_sha256=sha("ds-train"),
        source_class="public",
        patient_disjoint_asserted=True,
    )
    snap_kwargs.update(overrides.pop("training_snapshot_overrides", {}))
    base = overrides.pop("base_model", None)
    from models.registry.record import TrainingSnapshot

    return LineageBlock(
        base_model=base,
        training_snapshot=TrainingSnapshot(**snap_kwargs),
    )


def make_record(model_version: str = "1.2.0", **overrides: Any) -> RegistryRecord:
    kwargs: dict[str, Any] = dict(
        model_id="coronary-seg",
        model_version=model_version,
        task="coronary_segmentation",
        kind="model",
        source=SourceBlock(repository="https://git.example/coronary-seg", commit=COMMIT),
        weights=WeightsBlock(
            location="models/weights/coronary-seg.onnx",
            sha256=weights_sha(),
            format="onnx",
            adapter="models/adapters/segmentation.py@abc123",
        ),
        license=LicenseBlock(
            code="MIT",
            weights="CC-BY-4.0",
            commercial_allowed=False,
            review_ref="docs/license-review/lr-001",
        ),
        data_policy=DataPolicyBlock(
            logs_sensitive_data=False,
            remote_processing=False,
            network_required=False,
            training_data_refs=(make_training_ref(),),
        ),
        input_contract=InputContract(
            modality="CCTA",
            protocol="acq-proto@1",
            preprocessing="preproc-pipeline@2.0",
        ),
        output_contract=OutputContract(schema="ccta.payload.segmentation.v1"),
        intended_use=IntendedUseBlock(
            release_lane=ReleaseLane.PERSONAL_RESEARCH,
            target_user="radiologist-owner",
            limitations=("personal research only",),
        ),
        runtime=RuntimeBlock(local_supported=True, network_required=False, hardware="cpu-min"),
        validation=ValidationBlock(
            claim_ids=("CLM-SEG-001",),
            claim_version_at_registration="v3",
            benchmark_ref=make_benchmark_ref(),
            performance_path="reports/bench/coronary-seg.json",
            performance_sha256=sha("perf-report"),
        ),
        risk=RiskBlock(
            known_failure_modes=("severe motion",),
            required_qc=("qc-gate-motion",),
        ),
        registered_by="owner@opencode",
        registered_at=AT,
        lineage=make_snapshot_lineage(),
    )
    kwargs.update(overrides)
    return RegistryRecord(**kwargs)


def make_ensemble_record(members: tuple[str, ...], **overrides: Any) -> RegistryRecord:
    kwargs: dict[str, Any] = dict(
        model_id="seg-ensemble",
        model_version="1.0.0",
        task="coronary_segmentation",
        kind="ensemble",
        ensemble=EnsembleBlock(members=members, combination_strategy="weighted-vote"),
    )
    kwargs.update(overrides)
    return make_record(**kwargs)
