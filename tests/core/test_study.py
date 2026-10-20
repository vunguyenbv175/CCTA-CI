"""Study/Series creation: fail-closed validations + manifest freezing (§3.1-3.2)."""

from __future__ import annotations

import pytest

from core.enums import PrivacyClass
from core.errors import CanonicalValidationError, DigestMismatchError
from core.study import AcquisitionPhase, compute_composite_series_sha256
from tests.core.factories import AT, make_instances, make_series_manifest, make_study


class TestStudy:
    def test_happy_path_freezes_at_import(self):
        study = make_study()
        assert study.created_at == AT == study.import_info.imported_at
        assert study.privacy_class == PrivacyClass.SYNTHETIC

    @pytest.mark.parametrize(
        "overrides",
        [
            {"dicom_study_instance_uid": ""},
            {"privacy_class": None},
            {"import_job_ref": ""},
            {"source_root_path": ""},
            {"classifier": ""},
            {"created_by": ""},
            {"imported_at": "not-a-timestamp"},
        ],
    )
    def test_fail_closed_validations(self, overrides: dict):
        with pytest.raises(CanonicalValidationError) as err:
            make_study(**overrides)
        assert len(err.value.violations) >= 1

    def test_annotation_is_append_only_copy(self):
        from core.study import AnnotationEvent

        study = make_study()
        annotated = study.with_annotation(
            AnnotationEvent(annotation_id="ann-1", text="motion artifact", at=AT, by="owner")
        )
        assert len(study.notes) == 0  # original untouched
        assert len(annotated.notes) == 1


class TestSeriesManifest:
    def test_manifest_seals_composite_digest(self):
        series = make_series_manifest()
        expected = compute_composite_series_sha256(
            series.manifest.ordered_sop_uids,
            tuple(inst.sha256 for inst in series.manifest.instances),
        )
        assert series.manifest.composite_series_sha256 == expected
        assert series.manifest.instance_count == 3
        assert series.manifest.acquisition_phase is None

    def test_composite_digest_ignores_slice_order_disagreement(self):
        instances = make_instances(3)
        uids = [inst.sop_instance_uid for inst in instances]
        shas = [inst.sha256 for inst in instances]
        assert compute_composite_series_sha256(tuple(uids), tuple(shas)) == (
            compute_composite_series_sha256(tuple(reversed(uids)), tuple(reversed(shas)))
        )

    def test_multiphase_study_requires_acquisition_phase(self):
        with pytest.raises(CanonicalValidationError) as err:
            make_series_manifest(study_multiphase=True)
        assert any("acquisition_phase" in v for v in err.value.violations)

    def test_acquisition_phase_satisfies_multiphase_rule(self):
        series = make_series_manifest(
            study_multiphase=True,
            acquisition_phase=AcquisitionPhase(label="75% R-R", temporal_position=0.75),
        )
        assert series.manifest.acquisition_phase is not None
        assert series.classification.phase_label == "75% R-R"

    def test_instances_must_match_ordered_uids(self):
        instances = make_instances(3)
        reordered = (instances[2], instances[0], instances[1])
        with pytest.raises(CanonicalValidationError):
            make_series_manifest(
                ordered_sop_uids=tuple(i.sop_instance_uid for i in instances),
                instances=reordered,
            )

    def test_duplicate_sop_uids_rejected(self):
        instances = make_instances(3)
        duplicated = instances + (instances[-1],)
        with pytest.raises(CanonicalValidationError):
            make_series_manifest(
                instances=duplicated,
                ordered_sop_uids=tuple(i.sop_instance_uid for i in duplicated),
            )

    def test_original_dicom_ref_requires_digest(self):
        with pytest.raises(CanonicalValidationError) as err:
            from core.study import OriginalDicomRef

            make_series_manifest(
                original_dicom_ref=OriginalDicomRef(root_path="/data/raw", manifest_sha256="nope")
            )
        assert any("manifest_sha256" in v for v in err.value.violations)

    def test_classification_provenance_required(self):
        with pytest.raises(CanonicalValidationError):
            make_series_manifest(classification_provenance_ref="")


class TestReadOnlySourceStore:
    def test_open_bytes_verifies_against_manifest(self, tmp_path):
        import hashlib

        from core.dicom import ReadOnlySourceStore, SourceManifest

        payload = b"DICM-bytes"
        digest = hashlib.sha256(payload).hexdigest()
        slice_dir = tmp_path / "slices"
        slice_dir.mkdir()
        target = slice_dir / "0.dcm"
        target.write_bytes(payload)

        manifest = SourceManifest(
            root_path=str(tmp_path), entries=(("1.2.3", digest),)
        )
        store = ReadOnlySourceStore(manifest=manifest)
        assert store.open_bytes("1.2.3", "slices/0.dcm") == payload
        assert len(store.manifest.manifest_sha256) == 64

    def test_tampered_bytes_rejected(self, tmp_path):
        import hashlib

        from core.dicom import ReadOnlySourceStore, SourceManifest

        good = b"DICM-bytes"
        bad = b"TAMPERED!"
        digest = hashlib.sha256(good).hexdigest()
        target = tmp_path / "0.dcm"
        target.write_bytes(bad)

        store = ReadOnlySourceStore(
            manifest=SourceManifest(root_path=str(tmp_path), entries=(("1.2.3", digest),))
        )
        with pytest.raises(DigestMismatchError):
            store.open_bytes("1.2.3", "0.dcm")

    def test_store_handle_frozen(self, tmp_path):
        from core.dicom import ReadOnlySourceStore, SourceManifest

        store = ReadOnlySourceStore(
            manifest=SourceManifest(root_path=str(tmp_path), entries=(("1.2.3", "0" * 64),))
        )
        with pytest.raises(AttributeError):  # frozen dataclass has no setters
            setattr(store, "manifest", None)

    def test_verify_file_digest(self, tmp_path):
        import hashlib

        from core.dicom import verify_file_digest

        target = tmp_path / "f.dcm"
        target.write_bytes(b"data")
        verify_file_digest(target, hashlib.sha256(b"data").hexdigest())
        with pytest.raises(DigestMismatchError):
            verify_file_digest(target, "f" * 64)

    def test_unknown_sop_uid(self, tmp_path):
        from core.dicom import ReadOnlySourceStore, SourceManifest

        store = ReadOnlySourceStore(
            manifest=SourceManifest(root_path=str(tmp_path), entries=(("1.2.3", "0" * 64),))
        )
        with pytest.raises(KeyError):
            store.open_bytes("9.9.9", "x.dcm")


def test_source_manifest_rejects_empty_entries():
    from core.dicom import SourceManifest

    with pytest.raises(ValueError):
        SourceManifest(root_path="/x", entries=())
