"""Plugin mount validation tests (WS02 §B6, Rule D/E)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.events import EventLog
from plugins import PluginRegistryMount, PluginRejectedError

FIXTURES = Path(__file__).parent / "fixtures"


def test_conforming_engine_mounts_and_registers(tmp_path: Path) -> None:
    log = EventLog()
    mount = PluginRegistryMount(log=log)
    mounted = mount.mount(FIXTURES / "good_engine")
    assert mounted.plugin_id == "phase0-measure-engine"
    assert mounted.kind.value == "DETERMINISTIC_ENGINE"
    assert [cap.value for cap in mounted.capabilities] == ["MEASUREMENT_SET"]
    assert mounted.claims == ("CLM-MEAS-001",)
    assert len(mounted.manifest_sha256) == 64
    projection = log.project_state("phase0-measure-engine", entity="plugin")
    assert projection.get("registry_status") == "REGISTERED"


def test_llm_report_draft_plugin_is_allowed() -> None:
    mount = PluginRegistryMount()
    mounted = mount.mount(FIXTURES / "good_llm_report")
    assert mounted.plugin_id == "report-drafter"
    assert [cap.value for cap in mounted.capabilities] == ["REPORT_DRAFT"]


def test_unknown_manifest_key_is_rejected() -> None:
    mount = PluginRegistryMount()
    with pytest.raises(PluginRejectedError) as excinfo:
        mount.mount(FIXTURES / "bad_unknown_key")
    assert any("unknown manifest key" in reason for reason in excinfo.value.violations)
    assert not mount.mounted


def test_network_required_true_is_rejected() -> None:
    mount = PluginRegistryMount()
    with pytest.raises(PluginRejectedError) as excinfo:
        mount.mount(FIXTURES / "bad_network_required")
    assert any("network_required" in reason for reason in excinfo.value.violations)


def test_llm_outside_report_draft_is_rejected_rule_d() -> None:
    mount = PluginRegistryMount()
    with pytest.raises(PluginRejectedError) as excinfo:
        mount.mount(FIXTURES / "bad_llm_scope")
    reasons = " | ".join(excinfo.value.violations)
    assert "PRODUCER_MATRIX" in reasons
    assert "Rule D" in reasons
    assert not mount.mounted


def test_unknown_claim_is_rejected() -> None:
    mount = PluginRegistryMount()
    with pytest.raises(PluginRejectedError) as excinfo:
        mount.mount(FIXTURES / "bad_claim")
    assert any("CLAIM_REGISTRY" in reason for reason in excinfo.value.violations)


def test_missing_attr_entrypoint_is_rejected_not_loaded(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "broken-entry"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.toml").write_text(
        'id = "broken-entry"\nversion = "0.1.0"\nkind = "DETERMINISTIC_ENGINE"\n'
        'entrypoint = "mod_missing_attr:run"\ncapabilities = ["MEASUREMENT_SET"]\n'
        'claims = ["CLM-MEAS-001"]\nnetwork_required = false\n',
        encoding="utf-8",
    )
    (plugin_dir / "mod_missing_attr.py").write_text("other = 1\n", encoding="utf-8")
    mount = PluginRegistryMount()
    with pytest.raises(PluginRejectedError) as excinfo:
        mount.mount(plugin_dir)
    assert any("failed to load" in reason for reason in excinfo.value.violations)


def test_mount_all_records_rejections_without_raising(tmp_path: Path) -> None:
    plugins_root = tmp_path / "plugins"
    plugins_root.mkdir()
    import shutil

    shutil.copytree(FIXTURES / "good_engine", plugins_root / "good_engine")
    shutil.copytree(FIXTURES / "bad_claim", plugins_root / "bad_claim")
    shutil.copytree(FIXTURES / "bad_network_required", plugins_root / "bad_net")
    mount = PluginRegistryMount()
    mounted = mount.mount_all(plugins_root)
    assert [m.plugin_id for m in mounted] == ["phase0-measure-engine"]
    assert len(mount.rejections) == 2
