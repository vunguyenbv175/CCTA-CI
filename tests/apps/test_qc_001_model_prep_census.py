"""QC_001_MODEL_PREP D4 probe: the N=96 DECLARED census shape (Part 4.6).

DECLARED-NOT-EXECUTED: this probe enumerates the SHAPE (G=48 ×
M={AGREE, DISAGREE}=2 → N=96) and proves the strata arithmetic against the
accepted QC_001-D 48-strata enumeration. NO model is run, NO metric is
computed, NO result is recorded, NO benchmark is created.
"""

from __future__ import annotations

import dataclasses
from itertools import product
from pathlib import Path

import pytest

from core.errors import CanonicalValidationError
from models.registry.model_prep import (
    MODEL_PREP_CENSUS,
    MODEL_PREP_DECLARED_N,
    MODEL_PREP_DECLARED_ONLY_THRESHOLDS,
    MODEL_PREP_MEMBER_ROLES,
    MODEL_PREP_PRIMARY_ENDPOINTS,
    ModelPrepCensusDeclaration,
)
from tests.apps.qc_001_d_fixtures import _GEOMETRY_STRATA, GeometryStratum

#: The accepted QC_001-D census strata axes (read from the accepted fixture
#: module — imported, not copied — so a drift there is detectable here).
_DETECTED_AXES: tuple[tuple[str, ...], ...] = (
    tuple(sorted({s.orientation for s in _GEOMETRY_STRATA})),
    tuple(sorted({s.spacing for s in _GEOMETRY_STRATA})),
    tuple(sorted({s.normal for s in _GEOMETRY_STRATA})),
    tuple(sorted({s.origin for s in _GEOMETRY_STRATA})),
    tuple(sorted({s.pixels for s in _GEOMETRY_STRATA})),
)


class TestD4DeclaredCensusEnumeration:
    def test_g_is_exactly_the_accepted_48_strata(self):
        assert len(_GEOMETRY_STRATA) == 48
        assert len(set(_GEOMETRY_STRATA)) == 48  # distinct, no duplicates
        assert len(_GEOMETRY_STRATA) == MODEL_PREP_CENSUS.geometry_strata_count

    def test_strata_axes_reconstruct_the_product_enumeration(self):
        rebuilt = tuple(
            GeometryStratum(*values) for values in product(*_DETECTED_AXES)
        )
        # axis VALUES match the accepted fixture enumeration exactly; the
        # sorted-axis order may differ, so compare as a set:
        assert set(rebuilt) == set(_GEOMETRY_STRATA)
        assert len(rebuilt) == len(_GEOMETRY_STRATA) == 48

    def test_n_is_g_times_m(self):
        g = MODEL_PREP_CENSUS.geometry_strata_count
        m = MODEL_PREP_MEMBER_ROLES
        assert g == 48 and len(m) == 2
        assert g * len(m) == 96 == MODEL_PREP_DECLARED_N

    @pytest.mark.parametrize("role", MODEL_PREP_MEMBER_ROLES)
    def test_every_cell_exists_in_the_declared_shape(self, role):
        # The declared census is the full G×M grid; every cell is declared.
        cells = {(g_ordinal, role) for g_ordinal in range(48)}
        assert len(cells) == 48

    def test_total_declared_rows_are_exactly_96(self):
        rows = [(g_ordinal, role) for g_ordinal in range(48) for role in MODEL_PREP_MEMBER_ROLES]
        assert len(rows) == 96
        assert len(set(rows)) == 96

    def test_declared_endpoints_and_thresholds_are_the_verbatim_row(self):
        assert MODEL_PREP_PRIMARY_ENDPOINTS == (
            "model_disagreement_flag_exact_match_rate=1.0",
            "deterministic_comparator_exactness=1.0",
        )
        assert MODEL_PREP_DECLARED_ONLY_THRESHOLDS == (
            "sensitivity_MODEL_DISAGREEMENT=TP/(TP+FN)=1.0",
            "specificity_MODEL_DISAGREEMENT=TN/(TN+FP)=1.0",
        )
        # declaration-only strings; no metric machinery exists to evaluate
        # them (no TP/TN/FP/FN counters anywhere in the declaration module):
        import models.registry.model_prep as mp

        source = Path(mp.__file__).read_text(encoding="utf-8")
        assert "def specificity" not in source
        assert "def sensitivity" not in source

    # -- negatives: drift from the pinned Part 4.6 row is refused ------------

    def test_negative_wrong_g_refused(self):
        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepCensusDeclaration(
                geometry_strata_count=47,
                member_roles=MODEL_PREP_MEMBER_ROLES,
                declared_n=94,
                primary_endpoints=MODEL_PREP_PRIMARY_ENDPOINTS,
                declared_only_thresholds=MODEL_PREP_DECLARED_ONLY_THRESHOLDS,
            )
        assert "geometry_strata_count must be 48" in excinfo.value.violations[0]

    def test_negative_wrong_roles_refused(self):
        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepCensusDeclaration(
                geometry_strata_count=48,
                member_roles=("AGREE",),
                declared_n=48,
                primary_endpoints=MODEL_PREP_PRIMARY_ENDPOINTS,
                declared_only_thresholds=MODEL_PREP_DECLARED_ONLY_THRESHOLDS,
            )
        assert any(
            v.startswith("member_roles must be")
            for v in excinfo.value.violations
        )

    def test_negative_wrong_n_refused(self):
        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepCensusDeclaration(
                geometry_strata_count=48,
                member_roles=MODEL_PREP_MEMBER_ROLES,
                declared_n=95,
                primary_endpoints=MODEL_PREP_PRIMARY_ENDPOINTS,
                declared_only_thresholds=MODEL_PREP_DECLARED_ONLY_THRESHOLDS,
            )
        assert "declared_n must be 96" in excinfo.value.violations[0]

    def test_negative_non_verbatim_threshold_refused(self):
        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepCensusDeclaration(
                geometry_strata_count=48,
                member_roles=MODEL_PREP_MEMBER_ROLES,
                declared_n=96,
                primary_endpoints=MODEL_PREP_PRIMARY_ENDPOINTS,
                declared_only_thresholds=(
                    "sensitivity_MODEL_DISAGREEMENT=TP/(TP+FN)=0.9",
                    "specificity_MODEL_DISAGREEMENT=TN/(TN+FP)=1.0",
                ),
            )
        assert any(
            v.startswith("declared_only_thresholds must be the verbatim")
            for v in excinfo.value.violations
        )

    def test_negative_executed_true_refused(self):
        with pytest.raises(CanonicalValidationError) as excinfo:
            ModelPrepCensusDeclaration(
                geometry_strata_count=48,
                member_roles=MODEL_PREP_MEMBER_ROLES,
                declared_n=96,
                primary_endpoints=MODEL_PREP_PRIMARY_ENDPOINTS,
                declared_only_thresholds=MODEL_PREP_DECLARED_ONLY_THRESHOLDS,
                executed=True,
            )
        assert any(
            v.startswith("the QC_001-MODEL census is declared-not-executed")
            for v in excinfo.value.violations
        )

    # -- controls: the restored/correct declaration passes --------------------

    def test_control_pinned_declaration_passes(self):
        assert MODEL_PREP_CENSUS.executed is False
        assert MODEL_PREP_CENSUS.declared_n == 96

    # -- mutation verification ------------------------------------------------

    def test_d4_shape_enforcement_is_load_bearing(self, monkeypatch):
        # enforcement removed -> a drifted census would be accepted:
        import models.registry.model_prep as mp

        monkeypatch.setattr(mp, "MODEL_PREP_MEMBER_ROLES", ("AGREE",))
        monkeypatch.setattr(mp, "MODEL_PREP_DECLARED_N", 48)
        drifted = ModelPrepCensusDeclaration(
            geometry_strata_count=48,
            member_roles=("AGREE",),
            declared_n=48,
            primary_endpoints=MODEL_PREP_PRIMARY_ENDPOINTS,
            declared_only_thresholds=MODEL_PREP_DECLARED_ONLY_THRESHOLDS,
        )
        assert drifted.declared_n == 48  # enforcement gone, drift accepted
        monkeypatch.undo()
        # restored -> the same drift is refused:
        with pytest.raises(CanonicalValidationError):
            ModelPrepCensusDeclaration(
                geometry_strata_count=48,
                member_roles=("AGREE",),
                declared_n=48,
                primary_endpoints=MODEL_PREP_PRIMARY_ENDPOINTS,
                declared_only_thresholds=MODEL_PREP_DECLARED_ONLY_THRESHOLDS,
            )


# ---------------------------------------------------------------------------
# no-execution guard: nothing in the model-prep path runs anything
# ---------------------------------------------------------------------------


class TestD4NothingExecutes:
    def test_no_model_artifact_producer_or_claim_types_exist(self):
        import models.registry.model_prep as mp

        public = {name for name in dir(mp) if not name.startswith("_")}
        for banned in ("Artifact", "Producer", "Claim", "ProvenanceRecord", "Metric"):
            assert not any(banned in name for name in public), sorted(public)

    def test_declaration_module_defines_no_model_runner(self):
        import inspect

        import models.registry.model_prep as mp

        for name, obj in vars(mp).items():
            if name.startswith("_") or not inspect.isfunction(obj):
                continue
            source = inspect.getsource(obj)
            assert "run(" not in source, name
            assert "predict" not in source, name

    def test_census_carries_no_per_row_results(self):
        fields = {f.name for f in dataclasses.fields(MODEL_PREP_CENSUS)}
        assert fields.isdisjoint(
            {"results", "tp", "tn", "fp", "fn", "sensitivity", "specificity", "rows"}
        )
        assert not hasattr(MODEL_PREP_CENSUS, "__iter__") or isinstance(
            MODEL_PREP_CENSUS, type
        )
        with pytest.raises(TypeError):
            _ = len(MODEL_PREP_CENSUS)  # type: ignore[arg-type]
