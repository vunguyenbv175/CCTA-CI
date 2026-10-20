"""F2_SEED_ADMIT: concurrent first-build serialization regression.

F2_SEED_ADMIT serializes first-build seed admission with a per-ViewerEngine
``threading.Lock`` (stdlib only): validate + nonce-consume admission is
atomic under the lock; the long compute (``_execute_seed_first_build``
derivation + ``store.add``) runs OUTSIDE the lock.  A second concurrent
same-session first-build is refused with the EXISTING closed vocabulary
(``LINEAGE_GAP`` — its parent lineage no longer names the current session
tip; ``NONCE_REUSE`` for a true duplicate nonce).  No new rejection code,
no retune/ROI/latency change, reseed untouched, single-threaded behavior
unchanged.

Proves (synthetic phantom session/synthetic compute only — no patient
anatomy, no clinical claim; ThreadingHTTPServer + sleep-patched
``_execute_seed_first_build`` over the coronary analytic-phantom
volume-only session, whose first-build derivation carries real vessel
evidence):

* two concurrent same-session first-build seeds -> exactly one HTTP 200 +
  one typed refusal (existing code: 400 LINEAGE_GAP for the second
  distinct-nonce first-build);
* replay of the admitted nonce -> 400 NONCE_REUSE (fail-closed duplicate).
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from apps.viewer.build_identity import resolve_build_identity
from apps.viewer.engine import ViewerEngine
from apps.viewer.grammar import RejectionCode as Code
from apps.viewer.server import ViewerServer
from core.digest import canonical_json
from core.enums import ArtifactKind
from tests.core.coronary.support import helix_points
from tests.core.coronary.viewer_support import build_coronary_viewer_fixture


def _offset_ulids(start: int):
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

    class _Factory:
        def __init__(self) -> None:
            self._n = start

        def __call__(self, at_ms: int | None = None) -> str:
            del at_ms
            self._n += 1
            value = (0x01A9 << 80) | self._n
            return "".join(
                alphabet[(value >> (shift * 5)) & 0x1F] for shift in range(25, -1, -1)
            )

    return _Factory()


@pytest.fixture(autouse=True)
def _run_inside_bound_checkout(clean_checkout: Path, monkeypatch: pytest.MonkeyPatch):
    """F11 pattern: run engine operations from inside the bound checkout."""

    monkeypatch.chdir(clean_checkout)


@pytest.fixture(scope="module")
def coronary_volume_only(tmp_path_factory: pytest.TempPathFactory, clean_checkout: Path):
    """One centerline-free phantom session (the real T2 first-build case)."""

    checkout = clean_checkout
    code_commit_sha = resolve_build_identity(checkout)
    coronary = build_coronary_viewer_fixture(
        tmp_path_factory.mktemp("f2-seed-admit"),
        repo_root=checkout,
        code_commit_sha=code_commit_sha,
    )
    drop: set[str] = set()
    for artifact in coronary.session.artifacts:
        if artifact.kind is ArtifactKind.CENTERLINE or artifact.kind is ArtifactKind.CORRECTION:
            drop.add(artifact.artifact_id)
        elif artifact.kind is ArtifactKind.MPR_VIEW:
            document = coronary.session.payload_documents.get(artifact.artifact_id)
            if isinstance(document, dict) and document.get("schema_version") == (
                "ccta.cpr.view.v1"
            ):
                drop.add(artifact.artifact_id)
    session = replace(
        coronary.session,
        artifacts=tuple(
            artifact for artifact in coronary.session.artifacts if artifact.artifact_id not in drop
        ),
        payload_documents={
            key: value
            for key, value in coronary.session.payload_documents.items()
            if key not in drop
        },
    )
    return checkout, session


def _phantom_seed(index: int, ahead: int, behind: int) -> tuple[list[int], list[float]]:
    helix = helix_points()
    probe = helix[index]
    voxel = [
        int(round(float(probe[0]))),
        int(round(float(probe[1]))),
        int(round(float(probe[2]))),
    ]
    direction = [
        float(helix[ahead][0] - helix[behind][0]),
        float(helix[ahead][1] - helix[behind][1]),
        float(helix[ahead][2] - helix[behind][2]),
    ]
    return voxel, direction


def _seed_body(
    *, session_id: str, voxel: list[int], direction: list[float], nonce: str, parent: str | None
) -> bytes:
    return canonical_json(
        {
            "client_nonce": nonce,
            "direction": direction,
            "lineage": {"parent_action_nonce": parent, "session_id": session_id},
            "voxel": voxel,
        }
    ).encode("utf-8")


def _post_seed(port: int, session_id: str, body: bytes) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/session/{session_id}/seed",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Origin": f"http://127.0.0.1:{port}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _open_volume_only_engine(
    coronary_volume_only, *, ulid_start: int, monkeypatch: pytest.MonkeyPatch | None = None
) -> ViewerEngine:
    checkout, session = coronary_volume_only
    engine = ViewerEngine(
        session,
        id_factory=_offset_ulids(ulid_start),
        now=lambda: "2026-09-17T00:00:00Z",
        repo_root=checkout,
    )
    engine.open()
    return engine


def test_concurrent_first_build_serializes_second_is_lineage_gap(
    coronary_volume_only, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two concurrent same-session first-builds -> one 200 + one typed refusal."""

    engine = _open_volume_only_engine(coronary_volume_only, ulid_start=900000)
    session_id = engine.session_id()

    # Synthetic compute only (no patient anatomy, no clinical claim): hold
    # the first build inside the derivation so the second admission races it.
    entered = threading.Event()
    release = threading.Event()
    real_first_build = ViewerEngine._execute_seed_first_build

    def _slow_first_build(self, request, store):  # noqa: ANN001, ANN002
        if request.client_nonce == "f2admit-seed-00000001":
            entered.set()
            assert release.wait(timeout=60), "slow first-build was never released"
        return real_first_build(self, request, store)

    monkeypatch.setattr(ViewerEngine, "_execute_seed_first_build", _slow_first_build)
    srv = ViewerServer(audit_sink=None)
    srv.add_engine(engine)
    _, port = srv.start_background()
    try:
        voxel_a, direction_a = _phantom_seed(40, 60, 20)
        voxel_b, direction_b = _phantom_seed(200, 220, 180)
        body1 = _seed_body(
            session_id=session_id,
            voxel=voxel_a,
            direction=direction_a,
            nonce="f2admit-seed-00000001",
            parent=None,
        )
        body2 = _seed_body(
            session_id=session_id,
            voxel=voxel_b,
            direction=direction_b,
            nonce="f2admit-seed-00000002",
            parent=None,
        )
        outcomes: dict[str, tuple[int, dict[str, Any]]] = {}

        def _fire(name: str, body: bytes) -> None:
            outcomes[name] = _post_seed(port, session_id, body)

        first = threading.Thread(target=_fire, args=("first", body1))
        first.start()
        assert entered.wait(timeout=60), "first build never entered compute"
        # The second admission races WHILE the first compute is in flight.
        second = threading.Thread(target=_fire, args=("second", body2))
        second.start()
        time.sleep(0.5)
        release.set()
        first.join(timeout=120)
        second.join(timeout=120)
        assert set(outcomes) == {"first", "second"}
        first_status, _ = outcomes["first"]
        second_status, second_payload = outcomes["second"]
        # Exactly one HTTP 200 (the admitted first build) ...
        assert first_status == 200
        # ... and one typed refusal with the EXISTING closed vocabulary
        # (LINEAGE_GAP: the second concurrent first-build's parent lineage
        # no longer names the current session tip).  No new rejection code.
        assert second_status == 400
        assert isinstance(second_payload.get("error"), dict)
        assert cast(dict[str, Any], second_payload["error"])["code"] == Code.LINEAGE_GAP.value
        # Fail-closed: the admitted nonce is accepted exactly once, the
        # refused second nonce is consumed (seen -> NONCE_REUSE on replay).
        document = engine.state_document()
        assert document["accepted_nonces"] == ["f2admit-seed-00000001"]
        assert sorted(cast(list[str], document["used_nonces"])) == [
            "f2admit-seed-00000001",
            "f2admit-seed-00000002",
        ]
        assert document["action_count"] == 1
        # Replay of the admitted nonce -> 400 NONCE_REUSE (fixture only).
        replay_status, replay_payload = _post_seed(port, session_id, body1)
        assert replay_status == 400
        assert isinstance(replay_payload.get("error"), dict)
        assert cast(dict[str, Any], replay_payload["error"])["code"] == Code.NONCE_REUSE.value
    finally:
        srv.stop()


def test_sequential_first_build_still_succeeds_single_threaded(
    coronary_volume_only, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-threaded behavior unchanged: one first-build admits cleanly."""

    engine = _open_volume_only_engine(coronary_volume_only, ulid_start=910000)
    session_id = engine.session_id()
    real_first_build = ViewerEngine._execute_seed_first_build

    def _fast_first_build(self, request, store):  # noqa: ANN001, ANN002
        assert self._seed_first_build_inflight == request.client_nonce
        return real_first_build(self, request, store)

    monkeypatch.setattr(ViewerEngine, "_execute_seed_first_build", _fast_first_build)
    srv = ViewerServer(audit_sink=None)
    srv.add_engine(engine)
    _, port = srv.start_background()
    try:
        voxel, direction = _phantom_seed(40, 60, 20)
        body = _seed_body(
            session_id=session_id,
            voxel=voxel,
            direction=direction,
            nonce="f2admit-seed-00000001",
            parent=None,
        )
        status, _ = _post_seed(port, session_id, body)
        assert status == 200
        assert engine._seed_first_build_inflight is None
        document = engine.state_document()
        assert document["accepted_nonces"] == ["f2admit-seed-00000001"]
        assert document["action_count"] == 1
    finally:
        srv.stop()
