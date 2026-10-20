"""F2_JOB_HANDLE_TRUTH: a running seed's own state is served; nothing outlives it.

The measurement in ``reports/q1/F2_PROGRESS_CANCEL_MEASURE_20260923.md``
found the served defect this suite closes: a ``GET /session/:id/state`` issued
while a seed is in flight answers HTTP 200 in ~1 ms and carries NO jobish key
(``poll_exposes_inflight_status = false``), so a polling client cannot tell a
running from a queued build.  The lifecycle now rides that existing document,
keyed to the EXISTING ``(session_id, client_nonce)`` identity pair.

THE RETAINED GUARANTEE (manager decisions 2026-09-27 and 2026-09-28, after the
contract-specified reviewer's CHANGES_REQUESTED verdicts).  The handle reports
ONE running execution's own observed state — ``QUEUED``, ``RUNNING`` or
``REFUSED`` — and that entry is REMOVED when the execution returns, on success,
refusal, cancellation or exception alike.  Nothing outlives its execution, and
that is the whole of the claim:

* while an execution is in flight, its own state is served, and it is one of
  those three words;
* when it returns, its identity serves NO job entry at all;
* a refusal observed by an execution is visible only while that execution is
  in flight — never afterwards, whatever a later attempt of the same nonce
  really does;
* a really successful seed's entry disappears when its execution returns, while
  the frozen document's own ``accepted_nonces`` and ``action_count`` still
  record the acceptance in full;
* nothing in the served state expresses a percentage, ratio, stage, estimate,
  remaining time, duration, elapsed time or timestamp.

The handle therefore says NOTHING about any final or completed outcome of a
``(session_id, client_nonce)`` identity, and no test in this file may claim one.
Seven review rounds established why none can be earned here: as long as a
record can outlive the execution that produced it, every "is this still true?"
question has to be answered by reasoning about OTHER executions — concurrent
requests, nonce consumption by unrelated callers, the order between a snapshot
and a read — and that is exactly where every P1 was found.  The first narrowing
(2026-09-27) withdrew the identity-level ``COMPLETED`` but kept publishing an
identity-level ``REFUSED``, which still required an irreversibility predicate;
that predicate tested current consumption of the nonce rather than whether the
REFUSING execution had consumed it, so a retryable refusal was still published
after a successful same-nonce retry, and three amended tests had come to
require that behaviour.  The manager accepted the reviewer's criticism of those
amendments.  The second narrowing (2026-09-28) removes the class instead of
arguing it: with no record outliving its execution there is no record left to
publish after a later retry, no cross-execution question to answer, and
therefore no snapshot/read ordering, no live-registry withholding rule and no
irreversibility predicate that could be wrong.  The tests below were RESTORED to
the product's truth, not adjusted to accommodate it.

Proves (synthetic phantom session over the coronary analytic-phantom
volume-only session, real ThreadingHTTPServer, the real F2 admission rule and
the real first-build derivation with real vessel evidence — no patient anatomy,
no clinical claim, and no mocked refusal anywhere):

* the state document is served while work is in flight and carries only the
  frozen lifecycle words, in order, for the execution that owns the seed;
* a served ``REFUSED`` is derived ONLY from a real refusal, and its
  ``refusal_code`` is the EXACT code the server answered the client with, and
  it is served only for as long as the refusing execution is in flight;
* every observation and every removal is bound to the request execution that
  OWNS the record, so a second request reusing the same ``client_nonce`` — even
  when it is really refused — can neither overwrite, downgrade, re-attribute
  nor delete the running job (F1 of the 2026-09-27 codex review);
* the public handle is AWARDED to the execution that really wins the governed
  first-build admission, not to the execution that happened to create the
  record, in BOTH orders of arrival, so the admitted build is never invisible
  and a loser's real refusal never becomes the handle's state
  (R2-F1 of the 2026-09-27 codex review round 2);
* and the award carries NO WORD with it: whatever state and ``refusal_code`` the
  former owner left on the record, the new owner is handed the provisional
  ``QUEUED`` — the one served word that makes no claim about a particular
  execution — with the former owner's refusal code cleared, so its own
  ``RUNNING`` observation is accepted on top of it and no word that a returned
  execution earned is ever published for a live one (F1, P1, of the
  2026-09-28 codex review, fixed under manager decision
  ``F2_JOB_HANDLE_OWNERSHIP_TRANSFER_FIX``);
* an execution that has returned contributes NOTHING to the served document:
  its own success and its own refusal alike are gone, so no later attempt of
  the same nonce — successful or not — can be masked, or masked in turn, by a
  record that no longer exists;
* the handle is bound to the execution that holds it, so the first-build
  ordering survives the change, and the second-narrowing change is visible as a
  PRODUCT difference rather than an expectation adjustment: every test that once
  read a served word AFTER its execution had returned now reads it WHILE the
  execution is in flight (parked at the real route-exit hook) and reads
  NOTHING after it returns;
* a detached client (the seed POST connection closed before the response) still
  learns the job is owned by the server, never that it stopped, and the later
  state read is uncorrupted;
* a crash BEFORE the first observation and a crash AFTER the record was really
  inserted both leave nothing behind, and the second is proven to have had a
  record to lose;
* no percentage, ratio, stage, estimate, ETA, remaining time, duration, elapsed
  time or timestamp is emitted anywhere in the served document, and no job is
  persisted (no job store);
* ``COMPLETED`` is never emitted on any path, for any identity, at any time.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from apps.viewer.build_identity import resolve_build_identity
from apps.viewer.engine import SeedRequest, ViewerEngine
from apps.viewer.schemas import STATE_SCHEMA
from apps.viewer.server import ViewerServer
from core.enums import ArtifactKind
from tests.apps.test_viewer_seed_admit_serial import (
    _offset_ulids,
    _open_volume_only_engine,
    _phantom_seed,
    _post_seed,
    _seed_body,
)
from tests.core.coronary.viewer_support import build_coronary_viewer_fixture

_JOBS_KEY = "seed_jobs"
_QUEUED = "QUEUED"
_RUNNING = "RUNNING"
_COMPLETED = "COMPLETED"
_REFUSED = "REFUSED"
_FROZEN_STATES = (_QUEUED, _RUNNING, _COMPLETED, _REFUSED)
#: Every estimate-shaped key the server must NEVER emit: it measures no such
#: quantity, so it may not imply one (Rules C/D/G).
_FORBIDDEN_ESTIMATE_TOKENS = (
    "percent",
    "progress",
    "ratio",
    "eta",
    "remaining",
    "estimate",
    "duration",
    "elapsed",
    "timeout",
    "deadline",
    "fraction",
    "forecast",
    "left",
    "speed",
    "throughput",
)
_NONCE = "f2job-seed-00000001"
_OTHER_NONCE = "f2job-seed-00000002"
#: A THIRD distinct identity, for the interleaving in which a different request
#: makes the initial centerline exist while a same-nonce execution is parked.
_THIRD_NONCE = "f2job-seed-00000003"


@pytest.fixture(autouse=True)
def _run_inside_bound_checkout(clean_checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F11 pattern: run engine operations from inside the bound checkout."""

    monkeypatch.chdir(clean_checkout)


@pytest.fixture(scope="module")
def volume_only_session(
    tmp_path_factory: pytest.TempPathFactory, clean_checkout: Path
) -> tuple[Path, Any]:
    """One centerline-free phantom session (the real T2 first-build case).

    The SAME construction as the accepted F2_SEED_ADMIT battery fixture
    (``tests/apps/test_viewer_seed_admit_serial.py``), rebuilt here so this
    module binds no other test module's fixtures.  The centerline, correction
    and CPR-view artifacts are dropped from the built fixture, so the session
    has no ACTIVE centerline and the first-build route is the real one.
    """

    checkout = clean_checkout
    code_commit_sha = resolve_build_identity(checkout)
    coronary = build_coronary_viewer_fixture(
        tmp_path_factory.mktemp("f2-job-truth"),
        repo_root=checkout,
        code_commit_sha=code_commit_sha,
    )
    drop: set[str] = set()
    for artifact in coronary.session.artifacts:
        if artifact.kind in (ArtifactKind.CENTERLINE, ArtifactKind.CORRECTION):
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
            artifact
            for artifact in coronary.session.artifacts
            if artifact.artifact_id not in drop
        ),
        payload_documents={
            key: value
            for key, value in coronary.session.payload_documents.items()
            if key not in drop
        },
    )
    return checkout, session


def _get_state(port: int, session_id: str) -> dict[str, Any]:
    """One ``GET /session/:id/state`` read of the existing endpoint."""

    request = urllib.request.Request(f"http://127.0.0.1:{port}/session/{session_id}/state")
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 200
        return cast("dict[str, Any]", json.loads(response.read()))


def _jobs(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every served job record of the document (served truth only, no internals)."""

    jobs = document.get(_JOBS_KEY)
    assert isinstance(jobs, list), f"served state carries no {_JOBS_KEY}: {sorted(document)}"
    return cast("list[dict[str, Any]]", jobs)


def _job(document: dict[str, Any], nonce: str) -> dict[str, Any]:
    """The single served job record for ``nonce`` (identity-checked)."""

    jobs = _jobs(document)
    matching = [job for job in jobs if job["client_nonce"] == nonce]
    assert len(matching) == 1, f"expected exactly one job for {nonce}, got {jobs}"
    job = matching[0]
    assert job["session_id"] == document["session_id"], (
        "a job is keyed to the session id of the document that serves it"
    )
    assert job["state"] in _FROZEN_STATES, f"non-frozen lifecycle word: {job['state']!r}"
    return job


def _all_keys(value: object) -> set[str]:
    """Every mapping key anywhere in a served document."""

    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for item in cast("dict[str, object]", value).values():
            keys |= _all_keys(item)
        return keys
    if isinstance(value, list):
        keys = set()
        for item in cast("list[object]", value):
            keys |= _all_keys(item)
        return keys
    return set()


def _gate_first_build(
    monkeypatch: pytest.MonkeyPatch, entered: threading.Event, release: threading.Event
) -> None:
    """Hold the governed first-build compute open (no fabricated result)."""

    real_first_build = ViewerEngine._execute_seed_first_build

    def _gated(self, request, store):  # noqa: ANN001, ANN202
        entered.set()
        assert release.wait(timeout=120), "the compute gate was never released"
        return real_first_build(self, request, store)

    monkeypatch.setattr(ViewerEngine, "_execute_seed_first_build", _gated)


def _park_before_admission(monkeypatch: pytest.MonkeyPatch, turnstile: _Turnstile) -> None:
    """Park each validated request after its job EXISTS, before the F2 rule.

    Validation does not consume the nonce, so two same-session requests
    carrying the SAME nonce can both be validated (and both observe the
    record) before either reaches the governed admission.  The wrapper sits on
    the serialized entry point and therefore OUTSIDE ``_seed_admit_lock``:
    parking never holds the admission lock, and ``_admit_seed_first_build`` is
    then called unchanged and decides exactly what it decided before.  The
    turnstile only decides WHEN a validated request may reach the rule, which
    turns the reviewer's decisive interleaving into a deterministic test
    instead of a timing race.
    """

    real_serialized = ViewerEngine._execute_seed_first_build_serialized

    def _gated(self, request, store):  # noqa: ANN001, ANN202
        turnstile.park()
        return real_serialized(self, request, store)

    monkeypatch.setattr(ViewerEngine, "_execute_seed_first_build_serialized", _gated)


def _seed_request(engine: ViewerEngine, nonce: str) -> SeedRequest:
    """One validated seed request; its OBJECT identity is the ownership token."""

    return SeedRequest(
        voxel=(40, 60, 20),
        direction=(0.0, 0.0, 1.0),
        client_nonce=nonce,
        session_id=engine.session_id(),
        parent_action_nonce=None,
    )


class _Turnstile:
    """Hands out admission turns deterministically, in arrival order.

    The governed F2 admission rule is untouched and still runs for real (see
    :func:`_park_before_admission`); this only decides which validated request
    enters it next, so a same-nonce interleaving is reproducible instead of
    timing-dependent.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._parked: list[threading.Event] = []
        self._permits: list[threading.Event] = []

    def park(self) -> None:
        """Block the calling request until its turn is granted."""

        with self._lock:
            parked = threading.Event()
            permit = threading.Event()
            self._parked.append(parked)
            self._permits.append(permit)
            parked.set()
        assert permit.wait(timeout=120), "the admission turnstile granted no turn"

    def await_parked(self, index: int) -> None:
        """Wait until the ``index``-th validated request is parked at the rule."""

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            with self._lock:
                if index < len(self._parked):
                    return
            time.sleep(0.01)
        raise AssertionError(f"request {index} never parked before the admission rule")

    def grant(self, index: int) -> None:
        """Let the ``index``-th parked request enter the real F2 rule."""

        with self._lock:
            assert index < len(self._permits), "no parked request to grant"
            self._permits[index].set()

    def grant_all(self) -> None:
        """Release every parked request (never wedge a server thread)."""

        with self._lock:
            for permit in self._permits:
                permit.set()


def _post_seed_detached(port: int, session_id: str, body: bytes) -> socket.socket:
    """Start one seed POST and hand back the still-open connection.

    The caller closes it while the governed build runs: the client stops
    waiting, keeps no response and no promise, and is left with the state
    route alone.  No real work is faked — the server keeps building.
    """

    connection = socket.create_connection(("127.0.0.1", port), timeout=30)
    connection.sendall(
        (
            f"POST /session/{session_id}/seed HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            "Content-Type: application/json\r\n"
            f"Origin: http://127.0.0.1:{port}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("ascii")
        + body
    )
    return connection


def test_seed_job_states_are_observed_in_order_while_the_build_is_in_flight(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QUEUED -> RUNNING while the build is in flight; nothing after it lands.

    AMENDED IN THE F2_JOB_HANDLE_TRUTH NARROWING (2026-09-27) — the ORDER is
    preserved exactly and one element is removed.  The original expectation was
    ``QUEUED -> RUNNING -> COMPLETED``, i.e. that the handle would publish the
    identity's terminal outcome once the route returned.  That is the claim the
    manager withdrew after the contract-specified reviewer's CHANGES_REQUESTED
    verdict: no identity-level terminal publication can be established with the
    current coordination between nonce consumption, the live-execution registry
    and the serve path.  The narrowed guarantee is per execution, so the
    in-flight observations are unchanged and the post-landing read is replaced
    by the STRICTEST possible assertion: no ``COMPLETED`` is ever published and
    the accepted identity is left unreported (abstention, not a guess), while
    the real acceptance is still asserted on the server's own ledger and on the
    unchanged HTTP 200.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=920000)
    session_id = engine.session_id()
    admitted = threading.Event()
    admit_release = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    real_admit = ViewerEngine._admit_seed_first_build

    def _gate_admission(self, request):  # noqa: ANN001, ANN202
        # The REAL F2 admission runs; the gate holds only the window between
        # the admission and the compute, which is where the job is QUEUED.
        real_admit(self, request)
        admitted.set()
        assert admit_release.wait(timeout=120), "the admission gate was never released"

    _gate_first_build(monkeypatch, entered, release)
    monkeypatch.setattr(ViewerEngine, "_admit_seed_first_build", _gate_admission)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    observed: list[str] = []
    try:
        voxel, direction = _phantom_seed(40, 60, 20)
        body = _seed_body(
            session_id=session_id,
            voxel=voxel,
            direction=direction,
            nonce=_NONCE,
            parent=None,
        )
        status: dict[str, int] = {}

        def _fire() -> None:
            status["value"] = _post_seed(port, session_id, body)[0]

        worker = threading.Thread(target=_fire)
        worker.start()
        try:
            assert admitted.wait(timeout=120), "the seed never reached admission"
            # The build has NOT started; the server is serving QUEUED.
            observed.append(_job(_get_state(port, session_id), _NONCE)["state"])
            admit_release.set()
            assert entered.wait(timeout=120), "the governed build was never entered"
            # The build is in flight; the server is serving RUNNING.
            observed.append(_job(_get_state(port, session_id), _NONCE)["state"])
            release.set()
            worker.join(timeout=300)
            assert not worker.is_alive(), "the seed request never returned"
            assert status["value"] == 200
            # AMENDED IN THE NARROWING: with the execution returned, the handle
            # reports NOTHING for that identity.  It must not restate the
            # acceptance as a completion, and it must not leave a stale
            # QUEUED/RUNNING behind either.
            landed = _get_state(port, session_id)
            served_after = _served_states(landed, _NONCE)
            assert served_after == (), (
                f"a returned execution published a terminal identity state: "
                f"{_reported_jobs(landed)}"
            )
            assert _COMPLETED not in served_after, served_after
        finally:
            admit_release.set()
            release.set()
            worker.join(timeout=300)
    finally:
        release.set()
        server.stop()
    # AMENDED IN THE NARROWING: the observation order while the work is really
    # in flight is unchanged and is the retained guarantee.
    assert observed == [_QUEUED, _RUNNING]
    assert _COMPLETED not in observed
    # Observed facts, not words: the ACCEPTANCE really happened, exactly as the
    # seed route has always done it — the handle simply no longer restates it.
    document = engine.state_document()
    assert document["accepted_nonces"] == [_NONCE]
    assert document["action_count"] == 1
    assert document["used_nonces"] == [_NONCE]


def test_concurrent_second_first_build_is_refused_with_the_real_admission_code(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REFUSED comes from the F2 admission rule and names its EXACT code.

    RESTORED TO THE PRODUCT'S TRUTH BY THE F2_JOB_HANDLE_TRUTH SECOND NARROWING
    (manager decision 2026-09-28).  ORIGINAL INTENT, preserved: a second
    concurrent first-build really races the admitted, in-flight job and the F2
    rule really refuses it with its own real code, while the job that owns the
    build keeps its own live state.

    WHY THE ENDING MOVED, AND WHERE IT MOVED TO: the original test read the
    served ``REFUSED`` from a state poll taken AFTER the refusing request had
    already answered its HTTP response, i.e. after that execution had returned.
    That is no longer a moment at which anything can be served for it, because
    the route-exit hook now removes the record of every execution that returns.
    So the served word is read at the only instant it can be read: while the
    refusing execution is still in flight, parked at the REAL route-exit hook
    (nothing acquired, the real removal one line away), with the owning build
    still running.  The expectation is the same real admission refusal with the
    same exact code — it is the TIMING of the read that the product change
    forces, not what the server says.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=921000)
    session_id = engine.session_id()
    entered = threading.Event()
    release = threading.Event()
    refused_parked = threading.Event()
    refused_release = threading.Event()
    other_voxel, other_direction = _phantom_seed(200, 220, 180)
    _gate_first_build(monkeypatch, entered, release)
    # Only the LOSER parks: the owning first build must keep running untouched.
    # Keyed on the LOSER's own real voxel, never on the out-of-bounds literal.
    _park_at_route_exit(monkeypatch, other_voxel, refused_parked, refused_release)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    first_status: dict[str, int] = {}
    second_outcome: dict[str, tuple[int, dict[str, Any]]] = {}
    second: threading.Thread | None = None
    try:
        first_voxel, first_direction = _phantom_seed(40, 60, 20)
        first_body = _seed_body(
            session_id=session_id,
            voxel=first_voxel,
            direction=first_direction,
            nonce=_NONCE,
            parent=None,
        )
        other_body = _seed_body(
            session_id=session_id,
            voxel=other_voxel,
            direction=other_direction,
            nonce=_OTHER_NONCE,
            parent=None,
        )

        def _fire() -> None:
            first_status["value"] = _post_seed(port, session_id, first_body)[0]

        worker = threading.Thread(target=_fire)
        worker.start()
        try:
            assert entered.wait(timeout=120), "the first build never entered compute"
            # A second concurrent first-build races the admitted, in-flight
            # job: the F2 rule refuses it (real HTTP answer, real code), and
            # its execution is then held open INSIDE its own route.
            second = threading.Thread(
                target=lambda: second_outcome.update(
                    value=_post_seed(port, session_id, other_body)
                ),
                name="f2job-concurrent-second",
            )
            second.start()
            assert refused_parked.wait(timeout=120), (
                "the refused second first-build never parked inside its route"
            )
            served = _get_state(port, session_id)
            refused = _job(served, _OTHER_NONCE)
            assert refused["state"] == _REFUSED
            assert refused["refusal_code"] == "LINEAGE_GAP"
            # ... and the refusal did not disturb the job that owns the build.
            assert _job(served, _NONCE)["state"] == _RUNNING
            assert set(refused) == {"session_id", "client_nonce", "state", "refusal_code"}
            # The exact code the server answered the refusing client with is the
            # exact code the handle served, and that answer is real.
            refused_release.set()
            assert second is not None
            second.join(timeout=300)
            assert not second.is_alive(), "the refused second first-build never returned"
            second_status, second_payload = second_outcome["value"]
            assert second_status == 400
            answered = cast("dict[str, Any]", second_payload["error"])["code"]
            assert answered == "LINEAGE_GAP", (
                f"the served code is not the code the server answered: {answered}"
            )
        finally:
            refused_release.set()
            if second is not None:
                second.join(timeout=300)
        release.set()
        assert worker is not None
        worker.join(timeout=300)
        assert not worker.is_alive(), "the owning first build never returned"
        assert first_status["value"] == 200
        # THE RETAINED GUARANTEE, END TO END: BOTH executions of the session
        # have now returned, so NEITHER identity is served a word.  The refusal
        # was real and is still on the wire; the acceptance is real and is
        # still in the frozen document's own ledgers.  Nothing is restated as an
        # identity outcome, and no refusal is retained to be read after the fact.
        settled = _get_state(port, session_id)
        assert _served_states(settled, _NONCE) == ()
        assert _served_states(settled, _OTHER_NONCE) == ()
        assert _served_refusal_codes(settled, _OTHER_NONCE) == ()
        # F2 (2026-09-28 review, P3): the assertion that used to sit here read
        # ``assert _COMPLETED not in _reported_jobs(settled)``, which compares a
        # STRING against the served job DICTIONARIES and therefore cannot fail.
        # It is REMOVED, not rewritten, and this is why: the two absence
        # assertions immediately above it already cover every identity this test
        # ever creates (``_NONCE`` and ``_OTHER_NONCE``), so the served job list
        # is empty and no served word of any kind — a completion included — can
        # be present.  No completion coverage is claimed from this site.
        assert engine.state_document()["accepted_nonces"] == [_NONCE]
        assert engine.state_document()["action_count"] == 1
        assert _nonce_ledger(engine.state_document(), _OTHER_NONCE) == (1, 0)
    finally:
        refused_release.set()
        release.set()
        server.stop()


def test_out_of_bounds_seed_is_refused_with_the_real_seam_code(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real derivation refusal is REFUSED while it runs, never COMPLETED.

    RESTORED TO THE PRODUCT'S TRUTH BY THE F2_JOB_HANDLE_TRUTH SECOND NARROWING
    (manager decision 2026-09-28).  ORIGINAL INTENT, preserved: a real
    derivation refusal is ``REFUSED`` with the exact code the server raised —
    never a heuristic, never a timeout, and never dressed up as a success.

    WHY THE REFUSAL IS NOW READ WHILE IT IS IN FLIGHT: the original test read
    the served word from a poll taken after the only execution of that identity
    had returned, which was exactly the moment the old retention rule was built
    to keep truthful.  The route-exit hook now removes the record of every
    execution that returns, so that poll legitimately serves nothing.  The same
    real refusal is therefore read at the only instant it can be read — its own
    execution still inside the route, parked at the real hook — and the
    identity is then asserted unreported for good.  Both halves are stronger
    together than either was alone: the server really does say ``REFUSED`` when
    it can, and it really does stop saying it the instant it cannot.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=922000)
    session_id = engine.session_id()
    parked = threading.Event()
    release = threading.Event()
    _park_at_route_exit(monkeypatch, [1000000, 1000000, 1000000], parked, release)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    outcome: dict[str, tuple[int, dict[str, Any]]] = {}
    worker: threading.Thread | None = None
    try:
        # Far outside every session volume: the governed first-build route
        # raises its own typed refusal (no clamp, no default, no retry).
        body = _seed_body(
            session_id=session_id,
            voxel=[1000000, 1000000, 1000000],
            direction=[0.0, 0.0, 1.0],
            nonce=_NONCE,
            parent=None,
        )
        worker = threading.Thread(
            target=lambda: outcome.update(value=_post_seed(port, session_id, body)),
            name="f2job-out-of-bounds",
        )
        worker.start()
        try:
            assert parked.wait(timeout=120), (
                "the refused seed never parked inside its own route"
            )
            in_flight = _get_state(port, session_id)
            refused = _job(in_flight, _NONCE)
            assert refused["state"] == _REFUSED
            # The refusal_code is served while the refusal's own execution runs
            # and is the very code the server is answering the client with.
            _assert_a_code_belongs_to_its_own_state(in_flight)
            _assert_document_never_contradicts_itself(in_flight)
            release.set()
            worker.join(timeout=300)
            assert not worker.is_alive(), "the refused seed never returned"
            status, payload = outcome["value"]
            assert status == 409
            answered = cast("dict[str, Any]", payload["error"])["code"]
            assert refused["refusal_code"] == answered
            assert set(refused) == {
                "session_id",
                "client_nonce",
                "state",
                "refusal_code",
            }
            # And once that execution has returned, the identity serves NO word
            # at all: nothing outlives its execution, in either direction.
            returned = _get_state(port, session_id)
            assert _served_states(returned, _NONCE) == (), (
                f"a returned execution is still served: {_reported_jobs(returned)}"
            )
            assert _COMPLETED not in _served_states(returned, _NONCE)
            # Nothing was accepted and no accepted bookkeeping was fabricated.
            assert engine.state_document()["accepted_nonces"] == []
            assert engine.state_document()["action_count"] == 0
        finally:
            release.set()
            parked.set()
            if worker is not None:
                worker.join(timeout=300)
    finally:
        release.set()
        server.stop()


def test_detached_client_sees_the_job_still_owned_by_the_server(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel-as-detach: the client stops waiting; the server keeps the work.

    AMENDED IN THE F2_JOB_HANDLE_TRUTH NARROWING (2026-09-27) — the detach
    property is untouched and the post-landing read is re-encoded.  The original
    expectation was that the detached job "reached its own terminal state" and
    was served ``COMPLETED``; the identity-level terminal claim is withdrawn, so
    the ending is now the strictest available one: the job is never reported as
    completed, and once its execution returns the identity is not reported at
    all — while the accepted chain says the action really landed, and the
    detached client still learned nothing that implies the work stopped.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=923000)
    session_id = engine.session_id()
    entered = threading.Event()
    release = threading.Event()
    _gate_first_build(monkeypatch, entered, release)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    try:
        voxel, direction = _phantom_seed(40, 60, 20)
        body = _seed_body(
            session_id=session_id,
            voxel=voxel,
            direction=direction,
            nonce=_NONCE,
            parent=None,
        )
        connection = _post_seed_detached(port, session_id, body)
        try:
            assert entered.wait(timeout=120), "the detached seed never entered compute"
            # DETACH: the client stops waiting and closes the connection. It
            # holds no response, no promise and no progress — only this route.
            connection.close()
            served = _get_state(port, session_id)
            job = _job(served, _NONCE)
            # The detach is NOT a stop: the server still owns the work, and no
            # served word claims the work stopped or was cancelled.
            assert job["state"] == _RUNNING
            assert all(
                word not in str(job).upper()
                for word in ("STOP", "CANCEL", "ABORT", "PAUSE", "KILL")
            )
            assert set(job) == {"session_id", "client_nonce", "state"}
            # The later read is uncorrupted: the frozen document is intact and
            # the additive job key is the ONLY difference from it.
            assert served["schema_version"] == STATE_SCHEMA
            assert set(served["renders"]) == {"AXIAL", "CORONAL", "SAGITTAL"}
            assert {key: value for key, value in served.items() if key != _JOBS_KEY} == (
                engine.state_document()
            )
            release.set()
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if _served_states(_get_state(port, session_id), _NONCE) == ():
                    break
                time.sleep(0.05)
            # The detached job was NOT orphaned: its execution really returned
            # and the accepted chain says the action landed.  AMENDED IN THE
            # NARROWING: the handle publishes no terminal success for it — the
            # identity is simply not reported once the execution is done, which
            # is abstention rather than a fabricated word, and it is never
            # reported as COMPLETED at any point.
            settled = _get_state(port, session_id)
            assert _served_states(settled, _NONCE) == ()
            assert _COMPLETED not in _served_states(settled, _NONCE)
            assert engine.state_document()["accepted_nonces"] == [_NONCE]
            assert engine.state_document()["action_count"] == 1
        finally:
            release.set()
            connection.close()
    finally:
        release.set()
        server.stop()


def test_served_state_document_estimates_nothing_and_persists_no_job(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No fabricated percentage/ETA/timing, and no job anywhere but the read.

    RESTORED TO THE PRODUCT'S TRUTH BY THE F2_JOB_HANDLE_TRUTH SECOND NARROWING
    (manager decision 2026-09-28).  ORIGINAL INTENT, preserved in full: no
    estimate-shaped key may appear anywhere in the served document, the per-job
    key set stays closed, the key is purely additive, and nothing is persisted.

    WHY THE SWEEP NOW RUNS ON A REFUSAL THAT IS STILL IN FLIGHT: the review of
    2026-09-27 rightly insisted that this sweep must not degenerate into an
    empty-list exercise, so it is run over a document that really carries a
    served job — the richest served key set, the one that also carries a
    ``refusal_code``.  The first amendment obtained that by refusing a seed and
    polling afterwards, which the second narrowing makes impossible.  The same
    coverage is obtained from the real refusal while its OWN execution is still
    inside the route (parked at the real route-exit hook), which is the only
    instant such a document exists at all.  The sweep, the closed key set, the
    additive check and the no-persistence checks are all unchanged.
    """

    checkout, session = volume_only_session
    engine = _open_volume_only_engine(volume_only_session, ulid_start=924000)
    session_id = engine.session_id()
    parked = threading.Event()
    release = threading.Event()
    _park_at_route_exit(monkeypatch, [1000000, 1000000, 1000000], parked, release)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    worker: threading.Thread | None = None
    try:
        voxel, direction = _phantom_seed(40, 60, 20)
        body = _seed_body(
            session_id=session_id,
            voxel=voxel,
            direction=direction,
            nonce=_NONCE,
            parent=None,
        )
        assert _post_seed(port, session_id, body)[0] == 200
        served = _get_state(port, session_id)
        # The accepted identity is NOT reported: its execution has returned.
        assert _served_states(served, _NONCE) == ()
        assert _COMPLETED not in _served_states(served, _NONCE)
        # The acceptance really happened, in the frozen document's own ledger.
        assert engine.state_document()["accepted_nonces"] == [_NONCE]
        # A REAL refusal on the now-ACTIVE centerline, held open inside its own
        # route, so the sweep below runs against a document that really carries
        # a served job.
        parent = cast("list[str]", engine.state_document()["accepted_nonces"])[-1]
        refused_body = _seed_body(
            session_id=session_id,
            voxel=[1000000, 1000000, 1000000],
            direction=[0.0, 0.0, 1.0],
            nonce=_OTHER_NONCE,
            parent=parent,
        )
        worker = threading.Thread(
            target=lambda: _post_seed(port, session_id, refused_body),
            name="f2job-sweep-refusal",
        )
        worker.start()
        try:
            assert parked.wait(timeout=120), (
                "the refused re-seed never parked inside its own route"
            )
            served = _get_state(port, session_id)
            job = _job(served, _OTHER_NONCE)
            assert job["state"] == _REFUSED
            # A served refusal carries its code and nothing else: the closed key
            # set.
            assert set(job) == {"session_id", "client_nonce", "state", "refusal_code"}
            # Nothing estimate-shaped anywhere in the served document.
            for key in _all_keys(served):
                lowered = key.lower()
                for token in _FORBIDDEN_ESTIMATE_TOKENS:
                    assert token not in lowered, f"estimate-shaped key served: {key!r}"
            # No served VALUE may be a measured quantity either, not just a
            # timing-shaped KEY: every served job value is one of the session
            # id, the client nonce, a frozen state word or a refusal code, so
            # the server cannot be leaking a duration or a ratio under any name.
            assert all(
                isinstance(value, str)
                for job in cast("list[dict[str, Any]]", served[_JOBS_KEY])
                for value in job.values()
            ), f"a served job carries a non-string (measured?) value: {served[_JOBS_KEY]}"
            # The job is ADDITIVE: the served document is the frozen document
            # plus the job key and nothing else.
            assert {key: value for key, value in served.items() if key != _JOBS_KEY} == (
                engine.state_document()
            )
        finally:
            release.set()
            parked.set()
            if worker is not None:
                worker.join(timeout=300)
        # Once the refusing execution has returned the identity serves no job
        # entry at all.
        assert _served_states(_get_state(port, session_id), _OTHER_NONCE) == ()
        # The persisted state document never carries a job (no job store), and
        # an engine opened over the same restored session reports no job.
        assert _JOBS_KEY not in engine.state_document()
        reopened = ViewerEngine(
            session,
            id_factory=_offset_ulids(926000),
            now=lambda: "2026-09-17T00:00:00Z",
            repo_root=checkout,
        )
        reopened.open()
        assert _JOBS_KEY not in reopened.planes_state()
    finally:
        release.set()
        server.stop()


def test_a_refusal_is_reported_only_while_its_own_execution_is_in_flight(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal repeats byte-identically while it runs, then it is GONE.

    RENAMED BY THE F2_JOB_HANDLE_TRUTH SECOND NARROWING (manager decision
    2026-09-28) from ``test_terminal_job_keeps_reporting_its_terminal_state``.
    The old name promised the property that was withdrawn: a finished job
    keeping its terminal word for ever ("terminal is FINAL, so a late poller
    still learns the build finished").  No record outlives its execution any
    more, so that promise cannot be kept and must not be tested.  The original
    intent — a real refusal is reported, and reporting it does not contradict
    the frozen document — is kept in full, and re-anchored to the only window in
    which the server can still support the claim: the refusing execution's own
    route.  A repeated poll inside that window must return the byte-identical
    document (a client that polls twice is not told two different things), and
    the instant that execution returns the identity must serve nothing at all.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=925000)
    session_id = engine.session_id()
    parked = threading.Event()
    release = threading.Event()
    _park_at_route_exit(monkeypatch, [1000000, 1000000, 1000000], parked, release)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    worker: threading.Thread | None = None
    try:
        voxel, direction = _phantom_seed(40, 60, 20)
        body = _seed_body(
            session_id=session_id,
            voxel=voxel,
            direction=direction,
            nonce=_NONCE,
            parent=None,
        )
        assert _post_seed(port, session_id, body)[0] == 200
        first = _served_states(_get_state(port, session_id), _NONCE)
        second = _served_states(_get_state(port, session_id), _NONCE)
        # The accepted identity is NEVER reported as a success, and never
        # reverts to a live word: the server makes no claim about it at all.
        assert first == second == ()
        assert _COMPLETED not in first + second
        # ... while the real acceptance is reported where it always was.
        assert engine.state_document()["accepted_nonces"] == [_NONCE]
        assert engine.state_document()["action_count"] == 1

        # A second seed against the now-ACTIVE centerline runs the reseed job
        # and is refused by the governed seam with its own closed code.  Its
        # execution is held open inside its own route, which is the whole
        # window in which the server may still speak for it.
        parent = cast("list[str]", engine.state_document()["accepted_nonces"])[-1]
        refused_body = _seed_body(
            session_id=session_id,
            voxel=[1000000, 1000000, 1000000],
            direction=[0.0, 0.0, 1.0],
            nonce=_OTHER_NONCE,
            parent=parent,
        )
        worker = threading.Thread(
            target=lambda: _post_seed(port, session_id, refused_body),
            name="f2job-refused-window",
        )
        worker.start()
        try:
            assert parked.wait(timeout=120), (
                "the refused re-seed never parked inside its own route"
            )
            first_read = _job(_get_state(port, session_id), _OTHER_NONCE)
            # Repeated polls while the refusal's own execution is in flight are
            # byte-identical: the word never reverts to a live one, and a late
            # poller inside that window really does learn this build was refused.
            assert _job(_get_state(port, session_id), _OTHER_NONCE) == first_read
            assert first_read["state"] == _REFUSED
            assert first_read["refusal_code"] == "SEED_OUT_OF_BOUNDS"
            assert first_read["client_nonce"] == _OTHER_NONCE
        finally:
            release.set()
            parked.set()
            if worker is not None:
                worker.join(timeout=300)
        # And when that execution has returned, the identity serves no job
        # entry at all — not the refusal, not a live word, not a completion.
        after = _get_state(port, session_id)
        assert _served_states(after, _OTHER_NONCE) == (), (
            f"a returned execution is still served: {_reported_jobs(after)}"
        )
        # F2 (2026-09-28 review, P3): the ``_COMPLETED not in _reported_jobs``
        # line that used to sit here compared a string against served
        # DICTIONARIES and could never fail, so it is REMOVED rather than
        # presented as completion coverage.  The effective check is the document
        # invariant on the very next line, which reads every served job's
        # ``state`` and fails on a served ``COMPLETED``; the absence assertion
        # above independently guarantees this document is empty.
        _assert_document_never_contradicts_itself(after)
    finally:
        release.set()
        server.stop()


def test_duplicate_same_nonce_refusal_never_overwrites_the_running_job(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review's interleaving, served: RUNNING, never REFUSED, never COMPLETED.

    AMENDED IN THE F2_JOB_HANDLE_TRUTH NARROWING (2026-09-27) — the whole point
    of this test is preserved and the ending is re-encoded.  The original
    sequence was ``[_RUNNING, _RUNNING, _COMPLETED]``: while the duplicate's real
    ``NONCE_REUSE`` refusal was withheld, the running job stayed ``RUNNING`` and
    the handle then published the owner's completion.  The withheld-refusal
    property — the retained truth — is unchanged and still asserted twice; only
    the terminal success is replaced by the narrowed expectation that the
    accepted identity is never reported at all, so no ``COMPLETED`` can ever
    appear in this sequence.

    Two same-session requests carry the SAME ``client_nonce``.  Validation does
    not consume the nonce, so both are validated and both observe the record
    before either reaches the governed F2 admission.  The first is admitted and
    holds the real compute open; the second is then refused by the REAL rule
    with ``NONCE_REUSE`` because the running first build consumed that nonce.
    That refusal is a fact about the SECOND execution only: while the first is
    still building, the served job must stay ``RUNNING`` — never ``REFUSED``, and
    never a refusal code attached to a running build.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=927000)
    session_id = engine.session_id()
    entered = threading.Event()
    release = threading.Event()
    turnstile = _Turnstile()
    _gate_first_build(monkeypatch, entered, release)
    _park_before_admission(monkeypatch, turnstile)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    owner_status: dict[str, int] = {}
    duplicate_outcome: dict[str, tuple[int, dict[str, Any]]] = {}
    served_states: list[str] = []
    started: list[threading.Thread] = []
    try:
        voxel, direction = _phantom_seed(40, 60, 20)
        # The SAME nonce in both bodies: this is a reused nonce, not a new job.
        body = _seed_body(
            session_id=session_id,
            voxel=voxel,
            direction=direction,
            nonce=_NONCE,
            parent=None,
        )

        def _own_the_seed() -> None:
            owner_status["value"] = _post_seed(port, session_id, body)[0]

        def _duplicate_the_seed() -> None:
            duplicate_outcome["value"] = _post_seed(port, session_id, body)

        owner = threading.Thread(target=_own_the_seed, name="f2job-owner")
        owner.start()
        started.append(owner)
        try:
            turnstile.await_parked(0)
            duplicate = threading.Thread(target=_duplicate_the_seed, name="f2job-duplicate")
            duplicate.start()
            started.append(duplicate)
            turnstile.await_parked(1)
            # Both requests are validated and both observed the ONE record the
            # first created: no second record, no forged duplicate identity.
            parked = _get_state(port, session_id)
            assert len(_jobs(parked)) == 1
            assert _job(parked, _NONCE)["state"] == _QUEUED
            # The first enters the REAL F2 admission and the governed build.
            turnstile.grant(0)
            assert entered.wait(timeout=120), "the owning build never entered compute"
            served_states.append(_job(_get_state(port, session_id), _NONCE)["state"])
            # Now the duplicate reaches the same rule and is REALLY refused.
            turnstile.grant(1)
            duplicate.join(timeout=300)
            assert not duplicate.is_alive(), "the duplicate seed never returned"
            status, payload = duplicate_outcome["value"]
            assert status == 400
            assert cast("dict[str, Any]", payload["error"])["code"] == "NONCE_REUSE"
            # The duplicate's route has returned by now, so BOTH of its
            # lifecycle effects have already run: its refusal observation and
            # its release.
            served = _get_state(port, session_id)
            served_states.append(_job(served, _NONCE)["state"])
            # A real refusal of the duplicate is never attributed to the job
            # that is still running.
            assert all(job["state"] != _REFUSED for job in _jobs(served)), (
                f"the duplicate's refusal was attributed to the running job: {_jobs(served)}"
            )
            # No served job may be terminal while the request that owns it is
            # still executing.  (A ``REFUSED`` is served only while ITS OWN
            # execution is in flight, so a record still owned by a building
            # execution can never be one.)
            assert owner.is_alive(), "the owning request already returned"
            assert all(
                job["state"] not in (_COMPLETED, _REFUSED) for job in _jobs(served)
            )
            # The owner really finishes, and its identity is then not reported.
            release.set()
            owner.join(timeout=300)
            assert not owner.is_alive(), "the owning seed never returned"
            assert owner_status["value"] == 200
            landed = _get_state(port, session_id)
            # AMENDED IN THE NARROWING: the owner's acceptance is real and the
            # handle publishes no terminal success for it, so this identity is
            # absent from the served document rather than served COMPLETED.
            assert _served_states(landed, _NONCE) == ()
            assert _COMPLETED not in _served_states(landed, _NONCE)
        finally:
            turnstile.grant_all()
            release.set()
            for thread in started:
                thread.join(timeout=300)
    finally:
        turnstile.grant_all()
        release.set()
        server.stop()
    assert served_states == [_RUNNING, _RUNNING], served_states
    assert _REFUSED not in served_states
    assert _COMPLETED not in served_states
    # The owner's acceptance bookkeeping is the one the seed route always did.
    assert engine.state_document()["accepted_nonces"] == [_NONCE]
    assert engine.state_document()["action_count"] == 1


def test_refused_duplicate_never_deletes_or_downgrades_the_owning_record(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror case: the owned record survives a duplicate's whole route.

    The same interleaving as the review's, asserted on the OWNED RECORD itself:
    a refused duplicate must not delete it, must not downgrade it, must not
    re-attribute it to itself, and must not re-open it afterwards.  The served
    record is byte-identical before and after the duplicate's request.

    AMENDED IN THE F2_JOB_HANDLE_TRUTH NARROWING (2026-09-27) — the "replay
    leaves the record untouched" half is re-encoded, nothing weakened.  The
    original finished by serving the owner's ``COMPLETED`` terminal record and
    re-reading it after a replay; the identity-level terminal claim is
    withdrawn, so the ending is now: the owner's acceptance publishes nothing,
    a later REPLAY of the same nonce is still refused by the real grammar gate
    and still touches nothing (ownership is not transferable through a reused
    nonce), and a late poller still cannot resurrect any word for that
    identity — the real accepted chain is the only place the success appears.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=928000)
    session_id = engine.session_id()
    entered = threading.Event()
    release = threading.Event()
    turnstile = _Turnstile()
    _gate_first_build(monkeypatch, entered, release)
    _park_before_admission(monkeypatch, turnstile)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    started: list[threading.Thread] = []
    try:
        voxel, direction = _phantom_seed(40, 60, 20)
        body = _seed_body(
            session_id=session_id,
            voxel=voxel,
            direction=direction,
            nonce=_NONCE,
            parent=None,
        )
        owner_status: dict[str, int] = {}
        duplicate_outcome: dict[str, tuple[int, dict[str, Any]]] = {}

        def _own_the_seed() -> None:
            owner_status["value"] = _post_seed(port, session_id, body)[0]

        def _duplicate_the_seed() -> None:
            duplicate_outcome["value"] = _post_seed(port, session_id, body)

        owner = threading.Thread(target=_own_the_seed, name="f2job-owner")
        owner.start()
        started.append(owner)
        try:
            turnstile.await_parked(0)
            duplicate = threading.Thread(target=_duplicate_the_seed, name="f2job-duplicate")
            duplicate.start()
            started.append(duplicate)
            turnstile.await_parked(1)
            turnstile.grant(0)
            assert entered.wait(timeout=120), "the owning build never entered compute"
            # The owned record, exactly as the state route serves it.
            owned = _jobs(_get_state(port, session_id))
            assert len(owned) == 1
            assert owned[0]["state"] == _RUNNING
            # The duplicate is really refused, and its whole route has returned.
            turnstile.grant(1)
            duplicate.join(timeout=300)
            assert not duplicate.is_alive(), "the duplicate seed never returned"
            assert duplicate_outcome["value"][0] == 400
            assert (
                cast("dict[str, Any]", duplicate_outcome["value"][1]["error"])["code"]
                == "NONCE_REUSE"
            )
            # NOT DELETED, NOT DOWNGRADED, NOT RE-ATTRIBUTED: the served record
            # is byte-identical, so the duplicate neither removed the owner's
            # job nor replaced its state (no refusal code can appear on a
            # running job, and the closed key set is unchanged).
            after = _jobs(_get_state(port, session_id))
            assert after == owned
            assert set(after[0]) == {"session_id", "client_nonce", "state"}
            # The owner still lands its own acceptance.
            release.set()
            owner.join(timeout=300)
            assert not owner.is_alive(), "the owning seed never returned"
            assert owner_status["value"] == 200
            # AMENDED IN THE NARROWING: no completion is published, so nothing
            # of the landed execution survives in the served document.  Read
            # through the abstention-tolerant helper: ``planes_state`` omits the
            # key entirely when the server has no claim to make.
            assert _reported_jobs(_get_state(port, session_id)) == []
            # A later REPLAY of the same nonce is refused by the real grammar
            # gate and touches nothing: ownership is not transferable through a
            # reused nonce.
            replay_status, replay_payload = _post_seed(port, session_id, body)
            assert replay_status == 400
            assert (
                cast("dict[str, Any]", replay_payload["error"])["code"] == "NONCE_REUSE"
            )
            assert _reported_jobs(_get_state(port, session_id)) == []
            assert _COMPLETED not in [
                str(job["state"])
                for job in _reported_jobs(_get_state(port, session_id))
            ]
        finally:
            turnstile.grant_all()
            release.set()
            for thread in started:
                thread.join(timeout=300)
    finally:
        turnstile.grant_all()
        release.set()
        server.stop()
    assert engine.state_document()["accepted_nonces"] == [_NONCE]
    assert engine.state_document()["action_count"] == 1


def test_every_non_owner_observation_and_release_is_a_no_op(
    volume_only_session,
) -> None:
    """Every non-owner OBSERVATION and RELEASE is a NO-OP, on the served document.

    Driven straight at the observation API with two DISTINCT request objects
    that carry the same ``client_nonce`` (the reused-nonce shape), so the whole
    ORDINARY ownership surface is pinned in-process: a non-owner can neither
    transition, mutate, refuse nor remove the owner's record, while the owner
    itself still transitions and still removes its own record when its route
    returns.  Only the served document (``planes_state``) is read — ownership is
    enforced, never exposed.

    F3 (2026-09-28 review, P3) — THE NAME AND THE SCOPE ARE CORRECTED HERE.
    This test used to be called ``..._and_never_transferred`` and its docstring
    claimed it pinned "the whole ownership surface".  Both were false: it
    exercises the observation and release APIs only, and ownership IS
    transferred — once, by the F2 admission award
    (:meth:`ViewerEngine._seed_job_admit_owner`), which is the very operation
    the P1 finding F1 was about.  The award is covered by its own regressions
    (``test_handle_ownership_is_awarded_by_admission_and_never_moves_again`` and
    the two F1 transfer schedules added with the F1 fix), not by this one.  What
    this test does pin is the complementary half, and it is load-bearing: every
    ORDINARY effect by a request that does not own the record is dropped.

    RESTORED TO THE PRODUCT'S TRUTH BY THE F2_JOB_HANDLE_TRUTH SECOND NARROWING
    (manager decision 2026-09-28).  Two things changed, and both follow from the
    product: (i) the owner no longer has to have a nonce consumed for its own
    refusal to be served — the old version appended to ``used_nonces`` by hand
    purely to satisfy the irreversibility predicate, which no longer exists, and
    that hand-written bookkeeping is removed here rather than kept as dead
    scaffolding; (ii) the owner's OWN release now removes its record whatever
    state it holds, so the identity serves nothing at all once the owner has
    returned, instead of keeping a retained terminal word.  Every non-owner
    effect in between is unchanged and still asserted.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=929000)
    owner = _seed_request(engine, _NONCE)
    intruder = _seed_request(engine, _NONCE)

    def served() -> dict[str, Any]:
        """The SERVED document (the only thing these assertions read)."""

        return cast("dict[str, Any]", engine.planes_state())

    # Only the first request may create the record; it OWNS it.
    engine._seed_job_queued(owner)
    assert _job(served(), _NONCE)["state"] == _QUEUED
    # A different execution with the same nonce finds the record already there.
    engine._seed_job_queued(intruder)
    assert len(_jobs(served())) == 1
    assert _job(served(), _NONCE)["state"] == _QUEUED
    # Every other observation the intruder could make is dropped: RUNNING, a
    # REAL refusal with a real closed code, and the removal.
    engine._seed_job_running(intruder)
    engine._seed_job_refused(intruder, "NONCE_REUSE")
    engine._seed_job_release(intruder)
    assert _job(served(), _NONCE)["state"] == _QUEUED
    # The refusal could not even add a key: the closed key set is unchanged and
    # the ownership binding itself is never served.
    assert set(_jobs(served())[0]) == {"session_id", "client_nonce", "state"}
    # The owner advances its own job, and the intruder still cannot touch it.
    engine._seed_job_running(owner)
    engine._seed_job_running(intruder)
    engine._seed_job_release(intruder)
    assert _job(served(), _NONCE)["state"] == _RUNNING
    # The owner's own REAL refusal is served while the owner's execution is
    # still in flight, with its exact code and the closed key set.  No consumed
    # nonce is required or consulted any more.
    engine._seed_job_refused(owner, "SEED_OUT_OF_BOUNDS")
    terminal = _jobs(served())
    assert terminal[0]["state"] == _REFUSED
    assert terminal[0]["refusal_code"] == "SEED_OUT_OF_BOUNDS"
    # The owner's own record survives every non-owner effect, and no non-owner
    # can replace it or re-attribute it.
    engine._seed_job_release(intruder)
    engine._seed_job_refused(intruder, "NONCE_REUSE")
    assert _jobs(served()) == terminal
    # A fresh job on a fresh nonce is its own owner: a non-owner still cannot
    # drop it, while the owner's own route exit removes it — the retained
    # guarantee, which is that an execution's observations are served while it
    # is running and gone once it returns.
    later = _seed_request(engine, _OTHER_NONCE)
    engine._seed_job_queued(later)
    engine._seed_job_running(later)
    engine._seed_job_release(_seed_request(engine, _OTHER_NONCE))
    assert _job(served(), _OTHER_NONCE)["state"] == _RUNNING
    engine._seed_job_release(later)
    assert _served_states(served(), _OTHER_NONCE) == (), (
        f"a returned execution is still served: {_jobs(served())}"
    )
    document = served()
    assert [job["client_nonce"] for job in _jobs(document)] == [_NONCE]
    assert _job(document, _NONCE)["state"] == _REFUSED
    # The owner itself returns: its identity is then served nothing at all.
    engine._seed_job_release(owner)
    assert _served_states(served(), _NONCE) == (), (
        f"a returned execution is still served: {_reported_jobs(served())}"
    )
    # The job is still only served: the persisted document never carries it.
    assert _JOBS_KEY not in cast("dict[str, Any]", engine.state_document())


def test_the_admitted_execution_owns_the_handle_it_did_not_create(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review R2-F1, the reviewer's exact order: served RUNNING, never REFUSED.

    AMENDED IN THE F2_JOB_HANDLE_TRUTH NARROWING (2026-09-27) — the award
    property is untouched; only the identity's terminal success is replaced by
    the narrowed expectation that the accepted identity is not reported at all.
    The original sequence ended ``[_RUNNING, _RUNNING, _COMPLETED]``; the two
    ``RUNNING`` reads and every withholding assertion are unchanged, and the
    third element is now the absence of any served word for that identity — so
    the review's decisive interleaving still ends with the handle reporting the
    execution that really ran, and never reporting a terminal identity outcome.

    Two same-session requests carry the SAME ``client_nonce``, and validation
    does not consume the nonce, so both enter the route and both observe the
    ONE record the first one created.  The FIRST is then parked before the
    governed F2 admission and the SECOND is granted the admission first — the
    review's decisive interleaving, in which record creation and admission
    select DIFFERENT executions.  The unchanged real rule admits the second,
    it enters the real governed build, and the public handle must then be
    ITS: ``RUNNING`` while it builds.

    Only afterwards does the record creator resume, where the unchanged real
    rule really refuses it with ``NONCE_REUSE`` (asserted against the creator's
    own HTTP 400 body).  That refusal is a fact about the creator alone, so it
    must not touch the handle the admitted execution was really awarded: the
    served record stays ``RUNNING``, byte-identical across the creator's whole
    route, and the handle is never ``REFUSED``.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=930000)
    session_id = engine.session_id()
    entered = threading.Event()
    release = threading.Event()
    turnstile = _Turnstile()
    _gate_first_build(monkeypatch, entered, release)
    _park_before_admission(monkeypatch, turnstile)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    creator_outcome: dict[str, tuple[int, dict[str, Any]]] = {}
    admitted_status: dict[str, int] = {}
    served_states: list[str] = []
    started: list[threading.Thread] = []
    try:
        voxel, direction = _phantom_seed(40, 60, 20)
        # The SAME nonce in both bodies: one public handle, two executions.
        body = _seed_body(
            session_id=session_id,
            voxel=voxel,
            direction=direction,
            nonce=_NONCE,
            parent=None,
        )

        def _create_the_record() -> None:
            creator_outcome["value"] = _post_seed(port, session_id, body)

        def _win_the_admission() -> None:
            admitted_status["value"] = _post_seed(port, session_id, body)[0]

        creator = threading.Thread(target=_create_the_record, name="f2job-creator")
        creator.start()
        started.append(creator)
        try:
            turnstile.await_parked(0)
            admitted = threading.Thread(target=_win_the_admission, name="f2job-admitted")
            admitted.start()
            started.append(admitted)
            turnstile.await_parked(1)
            # Both executions are validated and both saw the ONE record the
            # creator made: no second record, no forged duplicate identity.
            parked = _get_state(port, session_id)
            assert len(_jobs(parked)) == 1
            assert _job(parked, _NONCE)["state"] == _QUEUED
            # THE REVIEW'S ORDER: the SECOND request is admitted FIRST and is
            # really building.  The handle must already be its own.
            turnstile.grant(1)
            assert entered.wait(timeout=120), "the admitted build never entered compute"
            admitted_running = _jobs(_get_state(port, session_id))
            assert len(admitted_running) == 1
            assert admitted_running[0]["state"] == _RUNNING
            served_states.append(admitted_running[0]["state"])
            # NOW the record creator resumes and the REAL rule refuses it.
            turnstile.grant(0)
            creator.join(timeout=300)
            assert not creator.is_alive(), "the record creator never returned"
            status, payload = creator_outcome["value"]
            assert status == 400
            assert cast("dict[str, Any]", payload["error"])["code"] == "NONCE_REUSE"
            # The creator's whole route has run (its refusal AND its release):
            # the handle is neither refused, nor downgraded, nor re-attributed.
            after = _jobs(_get_state(port, session_id))
            assert after == admitted_running
            assert set(after[0]) == {"session_id", "client_nonce", "state"}
            served_states.append(after[0]["state"])
            # No served job is terminal while its own request thread is running.
            # (A ``REFUSED`` is served only while ITS OWN execution is in flight,
            # so the still-building awarded record can never be one.)
            assert admitted.is_alive(), "the admitted request already returned"
            assert all(job["state"] not in (_COMPLETED, _REFUSED) for job in after)
            assert _REFUSED not in served_states
            # The admitted execution really completes; AMENDED IN THE NARROWING:
            # the handle publishes no terminal identity outcome for it, so the
            # identity is simply not reported once the execution returns.
            release.set()
            admitted.join(timeout=300)
            assert not admitted.is_alive(), "the admitted seed never returned"
            assert admitted_status["value"] == 200
            landed = _get_state(port, session_id)
            assert _served_states(landed, _NONCE) == ()
            assert _COMPLETED not in _served_states(landed, _NONCE)
        finally:
            turnstile.grant_all()
            release.set()
            for thread in started:
                thread.join(timeout=300)
    finally:
        turnstile.grant_all()
        release.set()
        server.stop()
    assert served_states == [_RUNNING, _RUNNING], served_states
    assert _REFUSED not in served_states
    assert _COMPLETED not in served_states
    # The work that really ran is the work the accepted chain reports: one
    # accepted action, and the creator's refusal was never accepted.
    assert engine.state_document()["accepted_nonces"] == [_NONCE]
    assert engine.state_document()["action_count"] == 1


def test_a_third_same_nonce_request_cannot_touch_the_awarded_handle(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two losing same-nonce executions, both really refused, neither served.

    AMENDED IN THE F2_JOB_HANDLE_TRUTH NARROWING (2026-09-27) — the mirror of
    the review's interleaving keeps every withholding assertion and loses only
    the terminal success: the original ended ``[_RUNNING, _RUNNING,
    _COMPLETED]`` ("the handle ends COMPLETED, because the execution that
    really ran is the one it reports").  Under the narrowed guarantee the handle
    reports no identity-level outcome, so the ending is the absence of any
    served word for that identity.

    The mirror of the review's interleaving with a THIRD request on the same
    ``(session_id, client_nonce)`` handle: the creator and the third request
    are both parked before the governed admission, the second request wins it
    and is really building, and then the other two are each really refused
    with the real ``NONCE_REUSE``.  Neither refusal may touch the awarded
    handle at any point — the served record is byte-identical after each of
    their complete routes.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=931000)
    session_id = engine.session_id()
    entered = threading.Event()
    release = threading.Event()
    turnstile = _Turnstile()
    _gate_first_build(monkeypatch, entered, release)
    _park_before_admission(monkeypatch, turnstile)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    outcomes: dict[str, tuple[int, dict[str, Any]]] = {}
    admitted_status: dict[str, int] = {}
    served_states: list[str] = []
    started: list[threading.Thread] = []
    try:
        voxel, direction = _phantom_seed(40, 60, 20)
        body = _seed_body(
            session_id=session_id,
            voxel=voxel,
            direction=direction,
            nonce=_NONCE,
            parent=None,
        )

        def _post_as(name: str) -> None:
            outcomes[name] = _post_seed(port, session_id, body)

        def _win_the_admission() -> None:
            admitted_status["value"] = _post_seed(port, session_id, body)[0]

        creator = threading.Thread(target=lambda: _post_as("creator"), name="f2job-creator")
        creator.start()
        started.append(creator)
        try:
            turnstile.await_parked(0)
            admitted = threading.Thread(
                target=_win_the_admission, name="f2job-admitted"
            )
            admitted.start()
            started.append(admitted)
            turnstile.await_parked(1)
            third = threading.Thread(target=lambda: _post_as("third"), name="f2job-third")
            third.start()
            started.append(third)
            turnstile.await_parked(2)
            # All THREE executions are validated on one handle: one record.
            assert len(_jobs(_get_state(port, session_id))) == 1
            turnstile.grant(1)
            assert entered.wait(timeout=120), "the admitted build never entered compute"
            awarded = _jobs(_get_state(port, session_id))
            assert awarded[0]["state"] == _RUNNING
            served_states.append(awarded[0]["state"])
            # The creator is really refused; the handle does not move.
            turnstile.grant(0)
            creator.join(timeout=300)
            assert not creator.is_alive(), "the record creator never returned"
            assert outcomes["creator"][0] == 400
            assert (
                cast("dict[str, Any]", outcomes["creator"][1]["error"])["code"]
                == "NONCE_REUSE"
            )
            assert _jobs(_get_state(port, session_id)) == awarded
            # The THIRD same-nonce request is really refused too; still no move.
            turnstile.grant(2)
            third.join(timeout=300)
            assert not third.is_alive(), "the third same-nonce seed never returned"
            assert outcomes["third"][0] == 400
            assert (
                cast("dict[str, Any]", outcomes["third"][1]["error"])["code"]
                == "NONCE_REUSE"
            )
            served = _jobs(_get_state(port, session_id))
            assert served == awarded
            assert served[0]["state"] == _RUNNING
            served_states.append(served[0]["state"])
            assert admitted.is_alive(), "the admitted request already returned"
            # The admitted execution really lands; AMENDED IN THE NARROWING:
            # the handle publishes no terminal identity outcome for it.
            release.set()
            admitted.join(timeout=300)
            assert not admitted.is_alive(), "the admitted seed never returned"
            assert admitted_status["value"] == 200
            landed = _get_state(port, session_id)
            assert _served_states(landed, _NONCE) == ()
            assert _COMPLETED not in _served_states(landed, _NONCE)
        finally:
            turnstile.grant_all()
            release.set()
            for thread in started:
                thread.join(timeout=300)
    finally:
        turnstile.grant_all()
        release.set()
        server.stop()
    assert served_states == [_RUNNING, _RUNNING], served_states
    assert _REFUSED not in served_states
    assert _COMPLETED not in served_states
    assert engine.state_document()["accepted_nonces"] == [_NONCE]
    assert engine.state_document()["action_count"] == 1


def test_the_admitted_execution_refusal_is_served_for_the_handle_it_owns(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A REFUSED handle reports the ADMITTED execution's own real refusal.

    The handle whose admitted execution raises: the second request is granted
    the real F2 admission first, its build really refuses with the typed seam
    code the server raises, and that refusal is served for the handle it was
    awarded — with the EXACT code in its own HTTP body.  The record creator is
    then really refused at admission with ``NONCE_REUSE``, and its refusal may
    neither replace that code nor re-attribute, downgrade or remove the served
    record: a non-owner's real refusal never becomes the handle's state.

    RESTORED TO THE PRODUCT'S TRUTH BY THE F2_JOB_HANDLE_TRUTH SECOND NARROWING
    (manager decision 2026-09-28), which changes WHERE the real refusal is read
    and nothing about what it says.  The earlier rounds had moved the read out
    of the creator's live window because publishing a terminal word while
    another execution of the identity could still be accepted was the P1 under
    review.  That hazard cannot exist now: the record is removed the moment its
    own execution returns, so there is nothing left to withhold and nothing left
    to be overturned.  The real refusal is therefore read while the ADMITTED
    execution is still inside its own route — which is the only instant the
    server can speak for it — and the identity is asserted unreported for good
    the moment that execution returns.  The creator's interference checks are
    unchanged and still run in the middle, where they are hardest: its full
    route, refusal and removal all happen while the handle it does not own is
    still being served.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=932000)
    session_id = engine.session_id()
    turnstile = _Turnstile()
    _park_before_admission(monkeypatch, turnstile)
    admitted_parked = threading.Event()
    admitted_release = threading.Event()
    # Only the ADMITTED execution parks at its route exit; the record creator
    # must run its whole route exactly as the reviewer ordered it.
    _park_at_route_exit(
        monkeypatch, [1000000, 1000000, 1000000], admitted_parked, admitted_release
    )

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    outcomes: dict[str, tuple[int, dict[str, Any]]] = {}
    served_states: list[str] = []
    started: list[threading.Thread] = []
    try:
        good_voxel, good_direction = _phantom_seed(40, 60, 20)
        creator_body = _seed_body(
            session_id=session_id,
            voxel=good_voxel,
            direction=good_direction,
            nonce=_NONCE,
            parent=None,
        )
        # Same handle, and a seed the governed build really refuses: the seed
        # lies far outside every session volume (no clamp, no default).
        refused_body = _seed_body(
            session_id=session_id,
            voxel=[1000000, 1000000, 1000000],
            direction=[0.0, 0.0, 1.0],
            nonce=_NONCE,
            parent=None,
        )

        def _post_as(name: str, request_body: bytes) -> None:
            outcomes[name] = _post_seed(port, session_id, request_body)

        creator = threading.Thread(
            target=lambda: _post_as("creator", creator_body), name="f2job-creator"
        )
        creator.start()
        started.append(creator)
        try:
            turnstile.await_parked(0)
            admitted = threading.Thread(
                target=lambda: _post_as("admitted", refused_body), name="f2job-admitted"
            )
            admitted.start()
            started.append(admitted)
            turnstile.await_parked(1)
            # The SECOND request wins the REAL admission, and its REAL build
            # raises the governed seam refusal.
            turnstile.grant(1)
            assert admitted_parked.wait(timeout=120), (
                "the admitted seed never parked inside its own route"
            )
            # THE AWARDED HANDLE, STILL ITS EXECUTION'S OWN: the refusal is
            # served with the EXACT code its own body carries, from the
            # execution that really raised it.
            refused = _job(_get_state(port, session_id), _NONCE)
            assert refused["state"] == _REFUSED
            assert refused["refusal_code"] == "SEED_OUT_OF_BOUNDS"
            served_states.append(refused["state"])
            # The closed key set is unchanged: no code is borrowed, no extra key.
            assert set(refused) == {"session_id", "client_nonce", "state", "refusal_code"}
            assert creator.is_alive(), "the record creator already returned"
            # The record creator is now really refused at admission; its
            # NONCE_REUSE is a fact about the creator and touches nothing, even
            # though the whole of its route — refusal AND removal — runs while
            # the handle it does not own is still being served.
            turnstile.grant(0)
            creator.join(timeout=300)
            assert not creator.is_alive(), "the record creator never returned"
            assert outcomes["creator"][0] == 400
            assert (
                cast("dict[str, Any]", outcomes["creator"][1]["error"])["code"]
                == "NONCE_REUSE"
            )
            assert _job(_get_state(port, session_id), _NONCE) == refused, (
                "a non-owner's real refusal touched the awarded handle: "
                f"{_jobs(_get_state(port, session_id))}"
            )
            _assert_a_code_belongs_to_its_own_state(_get_state(port, session_id))
            # The admitted execution's route really completes now.
            admitted_release.set()
            admitted.join(timeout=300)
            assert not admitted.is_alive(), "the admitted seed never returned"
            assert outcomes["admitted"][0] == 409
            answered = cast("dict[str, Any]", outcomes["admitted"][1]["error"])["code"]
            assert refused["refusal_code"] == answered
            # THE RETAINED GUARANTEE: the awarded execution has returned, so its
            # refusal is gone with it.  Nothing was accepted, and the frozen
            # document says exactly that.
            returned = _get_state(port, session_id)
            assert _served_states(returned, _NONCE) == (), (
                f"a returned execution is still served: {_reported_jobs(returned)}"
            )
            # F2 (2026-09-28 review, P3): the ``_COMPLETED not in
            # _reported_jobs`` line that used to sit here compared a string
            # against served DICTIONARIES and could never fail, so it is
            # REMOVED: ``_NONCE`` is the only identity this test ever seeds and
            # the absence assertion immediately above already proves the served
            # job list is empty.  No completion coverage is claimed here.
            assert engine.state_document()["accepted_nonces"] == []
            assert engine.state_document()["action_count"] == 0
        finally:
            turnstile.grant_all()
            admitted_release.set()
            for thread in started:
                thread.join(timeout=300)
    finally:
        turnstile.grant_all()
        admitted_release.set()
        server.stop()
    assert served_states == [_REFUSED]


def test_handle_ownership_is_awarded_by_admission_and_never_moves_again(
    volume_only_session,
) -> None:
    """The admission awards the handle; afterwards it cannot move again.

    Driven at the observation API with three DISTINCT request objects on one
    ``client_nonce``, so the whole ownership surface is pinned in-process: the
    handle is awarded by the REAL F2 admission (the unchanged rule, called
    exactly as the route calls it, in the same critical section), the record
    CREATOR is then an ordinary non-holder, the awarded execution advances its
    own handle, and no later award or observation can move, re-attribute or
    remove it.  Only the served document (``planes_state``) is read — the award
    and the binding are enforced, never exposed.

    RESTORED TO THE PRODUCT'S TRUTH BY THE F2_JOB_HANDLE_TRUTH SECOND NARROWING
    (manager decision 2026-09-28).  Two expectations move, both forced by the
    product: the awarded execution's own REAL refusal is served with its exact
    code while that execution is still in flight, and the winner's own route
    exit now REMOVES the record whatever state it holds — including a recorded
    refusal — so that identity is served nothing at all afterwards.  The
    award-never-moves property itself is untouched: no later award, from the
    creator or from a replayed nonce, can take the handle while the awarded
    execution is running.  The real ``_admit_seed_first_build`` call below is
    kept unchanged (it is the F2_SEED_ADMIT rule, byte-identical in this
    attempt); it is no longer consulted by the serve path for anything.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=933000)
    creator = _seed_request(engine, _NONCE)
    admitted = _seed_request(engine, _NONCE)
    replay = _seed_request(engine, _NONCE)

    def served() -> dict[str, Any]:
        """The SERVED document (the only thing these assertions read)."""

        return cast("dict[str, Any]", engine.planes_state())

    engine._seed_job_queued(creator)
    assert _job(served(), _NONCE)["state"] == _QUEUED
    # Arriving with the same nonce never creates a second record.
    engine._seed_job_queued(admitted)
    assert len(_jobs(served())) == 1
    assert _job(served(), _NONCE)["state"] == _QUEUED
    # The REAL F2 admission, called as the route calls it: the winner is
    # ``admitted`` and, in that same critical section, the handle is awarded to
    # it.  The award is not a lifecycle observation — nothing is served yet.
    with engine._seed_admit_lock:
        engine._admit_seed_first_build(admitted)
        engine._seed_job_admit_owner(admitted)
    assert _job(served(), _NONCE)["state"] == _QUEUED
    # The record CREATOR is an ordinary non-holder from here on: its own build
    # states, its own REAL refusal and its own release are all dropped, and the
    # closed key set is unchanged.
    engine._seed_job_running(creator)
    engine._seed_job_refused(creator, "NONCE_REUSE")
    engine._seed_job_release(creator)
    assert _job(served(), _NONCE)["state"] == _QUEUED
    assert set(_jobs(served())[0]) == {"session_id", "client_nonce", "state"}
    # The awarded execution advances its own handle.
    engine._seed_job_running(admitted)
    assert _job(served(), _NONCE)["state"] == _RUNNING
    # OWNERSHIP NEVER MOVES: no later award takes the handle — not the creator,
    # not a replayed nonce — and the creator's real refusal is still dropped.
    engine._seed_job_admit_owner(creator)
    engine._seed_job_admit_owner(replay)
    engine._seed_job_refused(creator, "NONCE_REUSE")
    engine._seed_job_refused(replay, "NONCE_REUSE")
    engine._seed_job_release(creator)
    assert _job(served(), _NONCE)["state"] == _RUNNING
    # The awarded execution's own REAL refusal is served with its exact code
    # while that execution is still in flight, and it is retained through every
    # non-holder effect below.
    engine._seed_job_refused(admitted, "SEED_OUT_OF_BOUNDS")
    terminal = _jobs(served())
    assert terminal[0]["state"] == _REFUSED
    assert terminal[0]["refusal_code"] == "SEED_OUT_OF_BOUNDS"
    engine._seed_job_admit_owner(creator)
    engine._seed_job_admit_owner(replay)
    engine._seed_job_refused(replay, "NONCE_REUSE")
    engine._seed_job_refused(replay, "SEED_OUT_OF_BOUNDS")
    engine._seed_job_release(creator)
    assert _jobs(served()) == terminal
    # A fresh handle on a fresh nonce: the award is the admission's alone, and
    # the holder's own route exit removes its record — the retained guarantee.
    later = _seed_request(engine, _OTHER_NONCE)
    engine._seed_job_queued(later)
    with engine._seed_admit_lock:
        engine._seed_job_admit_owner(later)
    engine._seed_job_running(later)
    assert _job(served(), _OTHER_NONCE)["state"] == _RUNNING
    engine._seed_job_release(_seed_request(engine, _OTHER_NONCE))
    assert _job(served(), _OTHER_NONCE)["state"] == _RUNNING
    engine._seed_job_release(later)
    assert _served_states(served(), _OTHER_NONCE) == (), (
        f"a returned execution is still served: {_jobs(served())}"
    )
    document = served()
    assert [job["client_nonce"] for job in _jobs(document)] == [_NONCE]
    assert _job(document, _NONCE)["state"] == _REFUSED
    # The awarded execution returns: its own refusal goes with it, and the
    # identity is then served nothing at all.
    engine._seed_job_release(admitted)
    assert _served_states(served(), _NONCE) == (), (
        f"a returned execution is still served: {_reported_jobs(served())}"
    )
    # The award is internal: it is never served, and the persisted document
    # still carries no job at all.
    assert _JOBS_KEY not in cast("dict[str, Any]", engine.state_document())


# ---------------------------------------------------------------------------
# ATT0004 (review finding F1, P1): the admission award transfers the HANDLE, and
# it must not inherit a WORD.
#
# The bug these two regressions exist for, in one sentence: the award used to be
# ``replace(record, owner=request, admitted=True)``, which moved ownership and
# left the FORMER execution's ``state`` and ``refusal_code`` in place — and the
# forward-only ordering then blocked the new owner from correcting its own
# handle, while the former owner's route exit became a non-owner no-op.  The
# served word for a live execution could therefore be one that execution never
# earned.  The fix is abstention in the product's own direction: the award
# restarts the handle at the provisional ``QUEUED`` word and clears the former
# owner's ``refusal_code``.
# ---------------------------------------------------------------------------


def test_the_award_never_carries_another_executions_refusal_into_a_live_build(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE REVIEWER'S 13-STEP SCHEDULE, DETERMINISTICALLY, END TO END.

    READ IT AS A SCHEDULE.  One volume-only session (no accepted action yet, no
    centerline), one shared ``client_nonce`` N and one other nonce, three real
    executions over the real route and the real server:

    1.  ``C`` (other nonce) is admitted to the real first-build admission and is
        held INSIDE the governed compute, so its in-flight marker is set and no
        centerline exists yet.
    2.  ``A`` (nonce N) really validates, really creates the record — served
        ``QUEUED``, owned by A — and really reaches the admission rule.
    3.  The real rule refuses A with ``LINEAGE_GAP``, because C's in-flight
        marker says the first-build tip moved.  A leaves the
        ``_seed_admit_lock`` hold and is PARKED in the real window between that
        hold and its own ``except`` clause (see :class:`_RefusalWindowGate`).
        Nothing has been consumed and no ``REFUSED`` exists yet.
    4.  C is released and REALLY fails — its seed lies outside every session
        volume, so the real first build raises its real
        ``SEED_OUT_OF_BOUNDS`` — which clears the in-flight marker.  No action
        has been accepted, so the session is a first-build session again and
        there is still no centerline.  C's whole route really returns.
    5.  ``B`` (the same, still-unused N) really validates.  Its own ``QUEUED``
        observation cannot replace A's record, so B holds no handle of its own.
    6.  B acquires ``_seed_admit_lock`` and the REAL rule admits it: the nonce is
        still unseen, the lineage is the empty-session one, there is no active
        centerline and no in-flight marker.  B is then PARKED, still holding the
        admission lock, immediately before the handle award.
    7.  A resumes.  Its refusal handler runs OUTSIDE ``_seed_admit_lock`` — it
        audits, and it records ``REFUSED``/``LINEAGE_GAP`` on the record A still
        owns.  This is the review's step 8, and it is the step the old award
        docstring wrongly claimed could not happen.  A is then PARKED just
        before its own route exit.
    8.  B resumes and the award runs: ownership of that ``REFUSED`` record moves
        to B.  THE ASSERTION: what is served for N is NOT A's refusal and NOT
        A's code.  It is B's own provisional ``QUEUED`` — the one served word
        that makes no claim about a particular execution's progress, and a fact
        B's own live route has earned.
    9.  B observes ``RUNNING`` and enters the real governed compute, where it is
        held open.
    10. A returns.  Its route-exit removal is now a NON-OWNER NO-OP, so
        whatever the record holds belongs to the live owner B.
    11. THE POLL THE REVIEW ASKED FOR: with A already returned and B still
        building, the served document for N is B's own live word and nothing
        else.  No ``REFUSED``, no ``LINEAGE_GAP``, no word A ever earned.
    12. B's real build really completes and B is really accepted, so the served
        set empties with B's own route exit and the frozen document's own
        ledgers record the one real acceptance.

    On the unreviewed target the award at step 8 kept A's ``REFUSED``, so step 9
    was rejected by the forward-only order and step 11 served
    ``REFUSED``/``LINEAGE_GAP`` for B's entire build: a word published by an
    execution that had already returned, for a build that had not failed.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=933500)
    session_id = engine.session_id()
    good_voxel, good_direction = _phantom_seed(40, 60, 20)
    out_of_bounds = [1000000, 1000000, 1000000]
    # A's own voxel, distinct from B's and C's, so the EXISTING route-exit
    # barrier keys on exactly one request and never on the admitted one.
    a_voxel, _ = _phantom_seed(200, 220, 180)
    c_entered = threading.Event()
    c_release = threading.Event()
    b_entered = threading.Event()
    b_release = threading.Event()
    b_admitted = threading.Event()
    b_resume = threading.Event()
    a_parked = threading.Event()
    a_release = threading.Event()
    _gate_the_first_build_of_one_nonce(monkeypatch, _THIRD_NONCE, c_entered, c_release)
    _gate_the_first_build_of_one_nonce(monkeypatch, _NONCE, b_entered, b_release)
    _hold_an_admission_before_the_handle_award(monkeypatch, _NONCE, b_admitted, b_resume)
    _park_at_route_exit(monkeypatch, a_voxel, a_parked, a_release)
    # Installed LAST, so it wraps the award barrier and therefore sees the real
    # rule's own refusal before that barrier's post-admission park can run.
    refusal_window = _RefusalWindowGate(monkeypatch, engine, _NONCE)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    outcomes: dict[str, tuple[int, dict[str, Any]]] = {}
    started: list[threading.Thread] = []

    def _post_as(name: str, body: bytes) -> None:
        outcomes[name] = _post_seed(port, session_id, body)

    def _body(nonce: str, voxel: list[int] | tuple[int, int, int]) -> bytes:
        return _seed_body(
            session_id=session_id,
            voxel=list(voxel),
            direction=good_direction,
            nonce=nonce,
            parent=None,
        )

    try:
        # STEP 1: C is really admitted and really parked inside the compute.
        c = threading.Thread(
            target=lambda: _post_as("c", _body(_THIRD_NONCE, out_of_bounds)),
            name="f2job-f1-c",
        )
        c.start()
        started.append(c)
        assert c_entered.wait(timeout=120), "C never entered the governed first build"
        # STEP 2/3: A really queues N and is really refused at the admission.
        a = threading.Thread(
            target=lambda: _post_as("a", _body(_NONCE, a_voxel)), name="f2job-f1-a"
        )
        a.start()
        started.append(a)
        assert refusal_window.released.wait(timeout=120), (
            "A never released the admission lock after its real refusal"
        )
        # A owns the record, and no word of A's refusal exists yet: it is parked
        # before the ``except`` clause that would consume the nonce and record it.
        queued = _get_state(port, session_id)
        assert _served_states(queued, _NONCE) == (_QUEUED,), (
            f"the creating execution is not served its own word: {_jobs(queued)}"
        )
        # ``(None,)`` — one served job for this nonce, carrying NO refusal code.
        assert _served_refusal_codes(queued, _NONCE) == (None,)
        # STEP 4: C really fails and really clears the in-flight marker.
        c_release.set()
        c.join(timeout=300)
        assert not c.is_alive(), "C never returned"
        assert outcomes["c"][0] == 409
        assert (
            cast("dict[str, Any]", outcomes["c"][1]["error"])["code"]
            == "SEED_OUT_OF_BOUNDS"
        )
        # STEP 5/6: B is really validated on the still-unused N and really
        # admitted, and is parked holding the admission lock before the award.
        b = threading.Thread(
            target=lambda: _post_as("b", _body(_NONCE, good_voxel)), name="f2job-f1-b"
        )
        b.start()
        started.append(b)
        assert b_admitted.wait(timeout=120), "B was never really admitted"
        # STEP 7: A's refusal handler runs now — outside the admission lock that
        # B is holding — and records A's own real refusal on A's own record.
        refusal_window.resume.set()
        assert a_parked.wait(timeout=120), "A never parked inside its own route"
        refused = _job(_get_state(port, session_id), _NONCE)
        assert refused["state"] == _REFUSED
        assert refused["refusal_code"] == "LINEAGE_GAP"
        # STEP 8: the award runs.  It must move the HANDLE and nothing else.
        b_resume.set()
        assert b_entered.wait(timeout=120), "B never entered its own governed build"
        awarded = _job(_get_state(port, session_id), _NONCE)
        assert awarded["state"] == _RUNNING, (
            "the live owner's own RUNNING observation was rejected, so a word "
            f"another execution earned is still on its handle: {awarded}"
        )
        assert set(awarded) == {"session_id", "client_nonce", "state"}
        # STEP 10: A really returns; its removal is a non-owner no-op.
        a_release.set()
        a.join(timeout=300)
        assert not a.is_alive(), "A never returned"
        assert outcomes["a"][0] == 400
        assert cast("dict[str, Any]", outcomes["a"][1]["error"])["code"] == "LINEAGE_GAP"
        # STEP 11: THE POLL, after A has returned and while B is still building.
        # Nothing that A observed may be served for B's live execution, and
        # nothing may be served that B's own facts have not earned.
        during = _get_state(port, session_id)
        assert b.is_alive(), "B's build already returned before the poll"
        served = _job(during, _NONCE)
        assert served == awarded, (
            "the served word for a live execution is not that execution's own "
            f"latest observation: {_reported_jobs(during)}"
        )
        assert _served_states(during, _NONCE) == (_RUNNING,)
        # ``(None,)`` — the served job carries no refusal code at all, so no code
        # A really raised is being reported against B's live build.
        assert _served_refusal_codes(during, _NONCE) == (None,)
        assert _REFUSED not in _served_states(during, _NONCE), (
            "a returned execution's refusal is published for the live owner's "
            f"whole build: {_reported_jobs(during)}"
        )
        _assert_document_never_contradicts_itself(during)
        # A's nonce really was consumed exactly once, by the execution that
        # really consumed it, and A really was refused.
        assert _nonce_ledger(engine.state_document(), _NONCE) == (1, 0)
        # STEP 12: B's real build really completes and is really accepted.
        b_release.set()
        b.join(timeout=300)
        assert not b.is_alive(), "B's governed build never returned"
        assert outcomes["b"][0] == 200
        landed = _get_state(port, session_id)
        assert _served_states(landed, _NONCE) == (), (
            f"a returned execution is still served: {_reported_jobs(landed)}"
        )
        document = cast("dict[str, Any]", engine.state_document())
        assert document["accepted_nonces"] == [_NONCE]
        assert document["action_count"] == 1
    finally:
        b_resume.set()
        refusal_window.release()
        a_release.set()
        c_release.set()
        b_release.set()
        a_parked.set()
        for thread in started:
            thread.join(timeout=300)
        server.stop()


def test_the_award_never_carries_a_non_queued_record_into_the_new_owner(
    volume_only_session,
) -> None:
    """The transfer of a NON-``QUEUED`` record, at the award itself.

    THE REVIEW'S SECOND VARIANT, DRIVEN AT THE AWARD.  The end-to-end schedule
    above covers the ``REFUSED`` carry-forward over the real route.  This one
    covers the other word the same one-line operation could carry forward —
    ``RUNNING`` — plus ``REFUSED`` for completeness, and it pins the API
    contract directly: whatever state and ``refusal_code`` a former owner left on
    the record, the award hands the new owner a handle whose served word is the
    new owner's OWN provisional ``QUEUED`` and carries no refusal code at all.

    WHY THE ``RUNNING`` VARIANT IS REACHABLE AND WORTH ITS OWN CASE: the review
    named the production path — a request A that owns an in-flight re-seed
    record and has not yet consumed its nonce, whose active centerline is
    invalidated by a real ``REJECT`` action, after which a competitor B with the
    same still-unused nonce and a valid lineage really takes the first-build
    path and is really admitted.  A's ``RUNNING`` word is on the record at that
    moment, and B's own ``RUNNING`` observation is ranked equal, so without the
    fix B could never speak for its own handle at all.

    Driven in-process at the award, exactly as
    ``test_handle_ownership_is_awarded_by_admission_and_never_moves_again`` is,
    so the schedule is deterministic with no threads and no timing.  The REAL
    :meth:`ViewerEngine._admit_seed_first_build` is called as the route calls it,
    inside the real admission lock, and is unchanged.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=933600)

    def served() -> dict[str, Any]:
        """The SERVED document (the only thing these assertions read)."""

        return cast("dict[str, Any]", engine.planes_state())

    def award(new_owner: SeedRequest) -> None:
        """The REAL admission rule, then the real award, in the real hold."""

        with engine._seed_admit_lock:
            engine._admit_seed_first_build(new_owner)
            engine._seed_job_admit_owner(new_owner)

    # --- VARIANT 1: a former owner whose record says ``RUNNING``. ------------
    running_owner = _seed_request(engine, _NONCE)
    engine._seed_job_queued(running_owner)
    engine._seed_job_running(running_owner)
    assert _job(served(), _NONCE)["state"] == _RUNNING
    running_taker = _seed_request(engine, _NONCE)
    award(running_taker)
    # A's ``RUNNING`` is NOT handed to B: B is served its own provisional word,
    # and B's own forward-only ``RUNNING`` observation is then accepted on top.
    handed_over = _job(served(), _NONCE)
    assert handed_over["state"] == _QUEUED, (
        "the award carried the former owner's observed word into the new owner's "
        f"handle: {handed_over}"
    )
    assert set(handed_over) == {"session_id", "client_nonce", "state"}
    engine._seed_job_running(running_taker)
    assert _job(served(), _NONCE)["state"] == _RUNNING
    # The former owner is now an ordinary non-holder: its own refusal and its own
    # release change nothing.
    engine._seed_job_refused(running_owner, "NONCE_REUSE")
    engine._seed_job_release(running_owner)
    assert _job(served(), _NONCE)["state"] == _RUNNING
    engine._seed_job_release(running_taker)
    assert _served_states(served(), _NONCE) == (), (
        f"the admitted execution's own route exit left its record: {_jobs(served())}"
    )

    # --- VARIANT 2: a former owner whose record says ``REFUSED`` + a code. ---
    refusing_owner = _seed_request(engine, _OTHER_NONCE)
    engine._seed_job_queued(refusing_owner)
    engine._seed_job_refused(refusing_owner, "SEED_OUT_OF_BOUNDS")
    refused = _job(served(), _OTHER_NONCE)
    assert refused["state"] == _REFUSED
    assert refused["refusal_code"] == "SEED_OUT_OF_BOUNDS"
    refusing_taker = _seed_request(engine, _OTHER_NONCE)
    # The real rule refuses this one for real, so the award is NOT reached; the
    # award case is driven directly, which is the point — it must be safe even
    # when the real rule would have refused, because its own decision is what
    # routes it here and the ordering of the two is not what this pins.
    with engine._seed_admit_lock:
        engine._seed_job_admit_owner(refusing_taker)
    restarted = _job(served(), _OTHER_NONCE)
    assert restarted["state"] == _QUEUED, (
        "the award carried the former owner's REFUSED word into the new owner's "
        f"handle: {restarted}"
    )
    assert "refusal_code" not in restarted, (
        f"the award carried the former owner's refusal code forward: {restarted}"
    )
    # ``(None,)`` — the restarted job carries no refusal code whatsoever.
    assert _served_refusal_codes(served(), _OTHER_NONCE) == (None,)
    engine._seed_job_refused(refusing_taker, "NONCE_REUSE")
    refused_again = _job(served(), _OTHER_NONCE)
    assert refused_again["state"] == _REFUSED
    assert refused_again["refusal_code"] == "NONCE_REUSE"
    # The former owner cannot take its own refusal back, and the new owner's own
    # route exit removes the record.
    engine._seed_job_refused(refusing_owner, "SEED_OUT_OF_BOUNDS")
    engine._seed_job_release(refusing_owner)
    assert _job(served(), _OTHER_NONCE) == refused_again
    engine._seed_job_release(refusing_taker)
    assert _served_states(served(), _OTHER_NONCE) == ()
    # The award remains internal and the persisted document still carries no job.
    assert _JOBS_KEY not in cast("dict[str, Any]", engine.state_document())


def _await_either(
    first: threading.Event, second: threading.Event, *, timeout: float
) -> bool:
    """True once EITHER event is set (no polling race, no sleep).

    Used where a schedule is deliberately INDEPENDENT of which internal gate a
    particular engine revision reaches: the caller asserts on the served
    document, so which gate fired is not the thing under test.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if first.is_set() or second.is_set():
            return True
        time.sleep(0.01)
    return first.is_set() or second.is_set()


def _park_at_route_exit(
    monkeypatch: pytest.MonkeyPatch,
    voxel: list[int],
    parked: threading.Event,
    release: threading.Event,
) -> None:
    """Park the request with ``voxel`` just before its route-exit record removal.

    THE RETAINED GUARANTEE'S OBSERVATION WINDOW, parked deterministically.  The
    real route-exit hook, held open with nothing acquired: the wrapper sits
    OUTSIDE :meth:`ViewerEngine._seed_job_release`, which takes
    ``_seed_job_lock`` only inside itself.  So a request parked here is a real,
    finished-its-work execution whose record has NOT yet been removed — exactly
    the window in which the server may still speak for it, and exactly the last
    window in which it may be served.  Only the request carrying ``voxel``
    parks, so a competing request proceeds exactly as a reviewer would order it,
    and :meth:`ViewerEngine._seed_job_release` then runs unchanged once the gate
    is released.  Every test that needs to see a served word for a refusal reads
    it through this hook, because that is the only place such a word exists.
    """

    real_release = ViewerEngine._seed_job_release

    def _gated(self, request):  # noqa: ANN001, ANN202
        if tuple(request.voxel) == tuple(voxel):
            parked.set()
            assert release.wait(timeout=120), "the route-exit gate was never released"
        return real_release(self, request)

    monkeypatch.setattr(ViewerEngine, "_seed_job_release", _gated)


def _park_before_the_action_executes(
    monkeypatch: pytest.MonkeyPatch, parked: threading.Event, release: threading.Event
) -> None:
    """Park a validated ordinary action immediately before the engine executes it.

    F2_SEED_SPECIFIC_COMPLETION's R6-F2 schedule needs an action that is REALLY
    validated (so it can never be re-validated against a later ``used_nonces``)
    but has not executed yet.  The wrapper sits on :meth:`ViewerEngine._execute`,
    which the action route calls after the real grammar validation and outside
    every lock, so parking holds nothing and the real execution — with its real
    acceptance bookkeeping — runs unchanged once the gate is released.  No
    product code is replaced to make the schedule deterministic; only WHEN the
    action proceeds is decided.
    """

    real_execute = ViewerEngine._execute

    def _gated(self, parsed):  # noqa: ANN001, ANN202
        parked.set()
        assert release.wait(timeout=120), "the action gate was never released"
        return real_execute(self, parsed)

    monkeypatch.setattr(ViewerEngine, "_execute", _gated)


def _nonce_ledger(document: dict[str, Any], nonce: str) -> tuple[int, int]:
    """``(used, accepted)`` occurrence counts of one nonce in the served ledger."""

    return (
        cast("list[str]", document["used_nonces"]).count(nonce),
        cast("list[str]", document["accepted_nonces"]).count(nonce),
    )


def _post_action(
    port: int,
    session_id: str,
    *,
    nonce: str,
    parent: str | None,
    action: str = "OPEN_QC_PANEL",
    params: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """One ordinary grammar action through the real ``POST /session/:id/action``.

    Used by the F2_SEED_SPECIFIC_COMPLETION cross-route regressions: an action
    is NOT a seed, but it writes the SAME session-wide ``accepted_nonces``
    ledger the serve path used to read, so it is exactly the evidence a seed
    completion must never be inferred from.  The real grammar validates and the
    real engine executes it — nothing here is a stub.
    """

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/session/{session_id}/action",
        data=json.dumps(
            {
                "action": action,
                "client_nonce": nonce,
                "lineage": {"parent_action_nonce": parent, "session_id": session_id},
                "params": params or {},
                "view": {"frame_index": 0, "plane": "AXIAL"},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
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


# ---------------------------------------------------------------------------
# ATT0004 (review round 3): the served handle belongs to the execution that
# OWNS it, and nothing of a returned execution is ever served.
#
# Validation does not consume a nonce, so two validated requests can carry the
# same ``(session_id, client_nonce)`` at the same time.  On the reseed path no
# admission picks a winner, so the record creator keeps the handle — and under
# the second narrowing the creator's own word is served only while the creator
# is still inside its route, which is what makes ownership binding load-bearing
# for the served bytes rather than merely defensive (R3-F1).  R3-F2's masking
# hazard is gone structurally: the record that could mask a later retry is
# removed with the execution that recorded it, so there is nothing left to
# withhold and no cross-execution question to answer.
# ---------------------------------------------------------------------------


def _served_states(document: dict[str, Any], nonce: str) -> tuple[str, ...]:
    """Every state the document serves for ``nonce`` (empty when unreported).

    Tolerant on purpose, and deliberately so: the retained guarantee answers
    "this execution is not in flight" by serving NO entry for the identity, so a
    correct server may report nothing at all for a nonce, in every state of the
    world.  What it may never do is report a word for an execution that has
    returned, or a ``REFUSED`` whose ``refusal_code`` did not come from the same
    record.
    """

    jobs = document.get(_JOBS_KEY)
    if not isinstance(jobs, list):
        return ()
    return tuple(
        str(job["state"])
        for job in cast("list[dict[str, Any]]", jobs)
        if job.get("client_nonce") == nonce
    )


def _reported_jobs(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every served job record, tolerating the ABSENT (abstaining) key.

    ``planes_state`` omits ``seed_jobs`` entirely when nothing is served, so a
    correct server that abstains about every identity reports no key at all.
    That is "the server makes no claim", which these assertions read as an
    empty set — never as an error, and never as a served state.
    """

    jobs = document.get(_JOBS_KEY)
    if not isinstance(jobs, list):
        return []
    return cast("list[dict[str, Any]]", jobs)


def _assert_document_never_contradicts_itself(document: dict[str, Any]) -> None:
    """No served ``COMPLETED`` ever, and no refusal code on a non-refusal.

    Asserted on served bytes only.

    The identity-level completion claim is withdrawn and no record outlives its
    execution, so this holds on every served entry of every served document: the
    served word is one of the three live words, it never claims an identity-level
    completion, a ``refusal_code`` is present if and only if the served state is
    ``REFUSED``, the per-job key set is closed, and the job is keyed to the
    session id of the document that serves it.  An identity a later attempt may
    overturn needs no separate check: the attempt that raised a served refusal
    is by definition still running, and any later attempt gets no handle of its
    own — the ownership binding, not a ledger, is what keeps a served word
    attributable to the execution that observed it.
    """

    jobs = document.get(_JOBS_KEY)
    if not isinstance(jobs, list):
        return
    for job in cast("list[dict[str, Any]]", jobs):
        assert job["state"] in _FROZEN_STATES, f"non-frozen lifecycle word: {job}"
        assert job["state"] != _COMPLETED, (
            f"the narrowed serve path published an identity-level completion: {job}"
        )
        assert ("refusal_code" in job) == (job["state"] == _REFUSED), (
            f"a refusal code is attached to the wrong served state: {job}"
        )
        assert set(job) <= {
            "session_id",
            "client_nonce",
            "state",
            "refusal_code",
        }, f"a served job carries a key outside the closed set: {job}"
        assert str(job["session_id"]) == str(document["session_id"]), (
            f"a job is keyed to a different session than the document: {job}"
        )


def _park_validated_seed(monkeypatch: pytest.MonkeyPatch, turnstile: _Turnstile) -> None:
    """Park each validated seed request AFTER its job exists, before any rule.

    The reseed counterpart of :func:`_park_before_admission`.  The wrapper sits
    on ``_execute_seed``, which :meth:`ViewerEngine.apply_seed_bytes` calls
    immediately after ``_seed_job_queued`` and OUTSIDE ``_seed_admit_lock``, so
    parking never holds a lock and the real reseed rule runs unchanged once the
    turnstile grants the turn.  It decides only WHEN a validated request may
    proceed, which turns the review's decisive interleaving into a
    deterministic test instead of a timing race.
    """

    real_execute = ViewerEngine._execute_seed

    def _gated(self, request):  # noqa: ANN001, ANN202
        turnstile.park()
        return real_execute(self, request)

    monkeypatch.setattr(ViewerEngine, "_execute_seed", _gated)


def _gate_one_reseed(
    monkeypatch: pytest.MonkeyPatch,
    voxel: list[int],
    entered: threading.Event,
    release: threading.Event,
) -> None:
    """Hold the governed re-seed compute of ONE voxel open (no fake result).

    Keyed on the seed voxel, so the same interleaving can let one request's
    real refusal happen while the other request's real derivation is in flight.
    No product code is touched: the real :meth:`ViewerEngine._reseed_centerline`
    still runs afterwards and still decides the real outcome.

    ``entered`` is a NEGATIVE assertion aid, not a handshake: it is set by the
    real product code, the ONLY writer, when the governed compute is genuinely
    entered.  A test that needs to assert the compute was NEVER entered must
    therefore never set it itself — including in a teardown block, which is why
    every ``finally`` here releases ``release`` and leaves ``entered`` alone.
    """

    real_reseed = ViewerEngine._reseed_centerline

    def _gated(self, store, artifact, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if (kwargs["i"], kwargs["j"], kwargs["k"]) == tuple(voxel):
            entered.set()
            assert release.wait(timeout=120), "the re-seed compute gate was never released"
        return real_reseed(self, store, artifact, **kwargs)

    monkeypatch.setattr(ViewerEngine, "_reseed_centerline", _gated)


def _gate_the_first_build_of_one_nonce(
    monkeypatch: pytest.MonkeyPatch,
    nonce: str,
    entered: threading.Event,
    release: threading.Event,
) -> None:
    """Hold the governed first-build compute of ONE ``client_nonce`` open.

    The first-build counterpart of :func:`_gate_one_reseed`, keyed on the nonce
    instead of the voxel because the F1 schedule needs two first-builds in one
    session and they may legitimately share a voxel.  No product code is
    replaced: the real :meth:`ViewerEngine._execute_seed_first_build` still runs
    once the gate is released and still raises the real refusal or produces the
    real centerline, so a gated request that is parked here is a REAL execution
    that has really entered the governed compute and has not returned.

    The wrapper sits at the entry of the compute, so the request has already been
    admitted, awarded and recorded ``RUNNING`` — which is the instant at which
    the served document must show the running execution's own word.
    """

    real_first_build = ViewerEngine._execute_seed_first_build

    def _gated(self, request, store):  # noqa: ANN001, ANN202
        if request.client_nonce == nonce:
            entered.set()
            assert release.wait(timeout=120), "the first-build compute gate was never released"
        return real_first_build(self, request, store)

    monkeypatch.setattr(ViewerEngine, "_execute_seed_first_build", _gated)


def _hold_an_admission_before_the_handle_award(
    monkeypatch: pytest.MonkeyPatch,
    nonce: str,
    admitted: threading.Event,
    resume: threading.Event,
) -> None:
    """Park ONE request after the REAL rule admitted it, before the award.

    The window :meth:`ViewerEngine._seed_job_admit_owner` runs in: the wrapper
    calls the unchanged :meth:`ViewerEngine._admit_seed_first_build` first, so
    the rule has really decided (and really consumed the nonce) and the caller's
    ``with self._seed_admit_lock`` hold is still open — which is precisely the
    hold a competitor's refusal handler runs OUTSIDE of.  Only WHEN a validated
    request reaches the rule is decided, never what the rule decides.
    """

    real_admit = ViewerEngine._admit_seed_first_build

    def _gated(self, request):  # noqa: ANN001, ANN202
        real_admit(self, request)
        if request.client_nonce == nonce:
            admitted.set()
            assert resume.wait(timeout=120), "the admitted request never resumed"

    monkeypatch.setattr(ViewerEngine, "_admit_seed_first_build", _gated)


class _RefusalWindowGate:
    """Park ONE refused request between the admission hold and its own handler.

    THE REVIEWER'S STEP 4, PLACED HONESTLY.  "A releases ``_seed_admit_lock``
    when leaving the ``with`` block.  Pause A **before the exception handler
    consumes N**."  There is no product hook between those two statements, so
    the barrier is built from the two real statements around it: the unchanged
    :meth:`ViewerEngine._admit_seed_first_build` is wrapped to REMEMBER which
    server thread is unwinding a refusal of ``nonce``, and the engine's own
    ``_seed_admit_lock`` is wrapped so that exactly that thread, on exactly that
    thread's refused exit, releases the REAL lock first — as the product's
    ``with`` does — and only then parks.

    The request therefore parks in the real window holding NO lock at all, with
    the real rule's exception still propagating and the real ``except`` clause
    still to run, and its refusal handler then genuinely executes while a
    COMPETITOR holds the admission lock.  That is the whole of the finding: the
    award's docstring used to claim this could not happen.

    The thread is identified by the refusal itself, never by arrival order and
    never by the test's client thread: the governed route runs on a
    ThreadingHTTPServer request thread, which is NOT the thread that posted the
    request.  The window is one-shot — it disarms itself on the refused exit it
    parks — so no later refusal in the same schedule can park behind it.

    Lock ORDER is untouched and never reversed (the parked thread holds nothing),
    and the lock is replaced on ONE engine instance, for ONE test.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, engine: ViewerEngine, nonce: str) -> None:
        self.released = threading.Event()
        self.resume = threading.Event()
        real_lock = engine._seed_admit_lock
        real_admit = ViewerEngine._admit_seed_first_build
        refusing: dict[str, threading.Thread | None] = {"thread": None}
        gate = self

        class _Gate:
            """Context manager over the REAL lock, parking one refused exit."""

            def __enter__(self) -> threading.Lock:
                real_lock.acquire()
                return real_lock

            def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
                real_lock.release()
                if exc_type is not None and refusing["thread"] is threading.current_thread():
                    refusing["thread"] = None
                    gate.released.set()
                    assert gate.resume.wait(timeout=120), (
                        "the refused request never resumed after releasing the "
                        "admission lock"
                    )
                return False

        def _remembering_admit(self, request):  # noqa: ANN001, ANN202
            try:
                real_admit(self, request)
            except BaseException:
                if request.client_nonce == nonce:
                    refusing["thread"] = threading.current_thread()
                raise

        monkeypatch.setattr(ViewerEngine, "_admit_seed_first_build", _remembering_admit)
        monkeypatch.setattr(engine, "_seed_admit_lock", _Gate())

    def release(self) -> None:
        """Never wedge a server thread (teardown safety; idempotent)."""

        self.resume.set()


def _build_the_initial_centerline(port: int, session_id: str) -> None:
    """Give the session an ACTIVE centerline, so the NEXT seed really re-seeds.

    A real first build through the real route with the real derivation; after
    it, ``_execute_seed`` takes the re-seed branch instead of the first-build
    branch, which is the path both R3 findings are about.

    AMENDED IN THE F2_JOB_HANDLE_TRUTH NARROWING (2026-09-27): the helper used
    to assert that the initial build's job was served ``COMPLETED``.  It now
    asserts the narrowed counterpart — the build really succeeded AND the handle
    published no word at all for that identity — so a server that fabricated a
    completion here would still fail every caller that uses this helper.
    """

    voxel, direction = _phantom_seed(40, 60, 20)
    status, _payload = _post_seed(
        port,
        session_id,
        _seed_body(
            session_id=session_id,
            voxel=voxel,
            direction=direction,
            nonce=_NONCE,
            parent=None,
        ),
    )
    assert status == 200, "the initial first build did not succeed"
    built = cast("dict[str, Any]", _get_state(port, session_id))
    assert _served_states(built, _NONCE) == (), (
        f"the accepted first build published a job state: {_reported_jobs(built)}"
    )
    assert cast("list[str]", built["accepted_nonces"]) == [_NONCE]


def test_only_the_first_of_two_same_nonce_reseeds_has_its_refusal_served(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3-F1: the SECOND same-nonce re-seed's refusal is never served at all.

    F3 (2026-09-28 review, P3) — THE NAME IS CORRECTED HERE.  This test used to
    be called ``test_a_losing_reseed_refusal_never_becomes_the_identitys_served_state``,
    which reads as "the losing re-seed's refusal is never served".  Its own exact
    history contains ``(_REFUSED,)``: the FIRST execution really refused, and its
    refusal IS served — with its own real code — for as long as that execution is
    inside its own route.  The property this test actually defends is narrower
    and sharper: the refusal of the execution that never held the handle is never
    served, and nothing at all is served once the owner returns.

    ORIGINAL INTENT (ATT0004, R3-F1, PRESERVED): the reviewer's exact
    interleaving — one identity, two real re-seeds, the first one really
    refused — and the property it defends: a real refusal belongs to the
    execution that raised it, and a DIFFERENT execution of the same identity
    neither publishes a word of its own nor takes the handle away.  What is no
    longer asserted is the withholding rule that the earlier rounds added on
    top: a refusal observed by one execution being withheld for as long as
    another execution of the same identity is live.  That rule existed only to
    protect a record that outlived its execution, and no record outlives its
    execution any more — a refusing execution's word is removed with it, so
    there is nothing to withhold.

    WHY THE ORIGINAL SCENARIO IS IMPOSSIBLE NOW: the original end state was
    ``COMPLETED``, because the second same-nonce re-seed really derived and
    really completed.  That is exactly the DOUBLE-ACCEPT the
    F2_SEED_NONCE_IRREVERSIBILITY guard removed: the loser's real
    ``SEED_OUT_OF_BOUNDS`` refusal CONSUMED that ``client_nonce``, so the
    second request — already validated, since validation consumes nothing —
    reaches the re-seed acceptance path with the nonce already in
    ``used_nonces`` and is refused there with the EXISTING ``NONCE_REUSE`` code
    before any applying statement runs.  One identity is now applied at most
    once, so "a same-nonce re-seed really completes after the loser's refusal"
    can no longer be built at all.  The test was encoding the defect the
    reviewer named, not a reachable product behaviour.

    THE NEW TRUTH PINNED HERE (the turnstile): that same-nonce second attempt
    is refused with ``NONCE_REUSE``; NOTHING is derived for it (the real re-seed
    compute gate armed on its own valid voxel is never entered);
    ``accepted_nonces`` never gains the nonce at all; ``action_count`` advances
    exactly once, for the one real accepted action, and never twice; and once
    BOTH executions have returned, the identity is served NO word at all —
    neither the owner's real refusal nor anything else.  The whole-history check
    is retained on a list that keeps every sample INCLUDING the empty ones, so
    it is a real whole-history claim and not an empty slice (the 2026-09-27
    review's P3).
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=940000)
    session_id = engine.session_id()
    entered = threading.Event()
    release = threading.Event()
    owner_parked = threading.Event()
    owner_release = threading.Event()
    turnstile = _Turnstile()
    good_voxel, good_direction = _phantom_seed(200, 220, 180)
    #: EVERY sample, in order, INCLUDING the empty ones (review P3).
    history: list[tuple[str, ...]] = []

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    loser_outcome: dict[str, tuple[int, dict[str, Any]]] = {}
    winner_outcome: dict[str, tuple[int, dict[str, Any]]] = {}
    started: list[threading.Thread] = []
    try:
        _build_the_initial_centerline(port, session_id)
        # The turnstile and the compute gate are armed only now, so they govern
        # the two re-seeds under test and never the initial first build.  Only
        # the LOSING execution parks at its route exit, so the loser's real
        # refusal can be read at the one instant it may be served.
        _gate_one_reseed(monkeypatch, good_voxel, entered, release)
        _park_validated_seed(monkeypatch, turnstile)
        _park_at_route_exit(
            monkeypatch, [1000000, 1000000, 1000000], owner_parked, owner_release
        )
        # ONE identity, two real executions: the same client_nonce, so the
        # second is a reuse, not a new job.
        losing_body = _seed_body(
            session_id=session_id,
            voxel=[1000000, 1000000, 1000000],
            direction=good_direction,
            nonce=_OTHER_NONCE,
            parent=_NONCE,
        )
        winning_body = _seed_body(
            session_id=session_id,
            voxel=good_voxel,
            direction=good_direction,
            nonce=_OTHER_NONCE,
            parent=_NONCE,
        )

        def _lose() -> None:
            loser_outcome["value"] = _post_seed(port, session_id, losing_body)

        def _win() -> None:
            winner_outcome["value"] = _post_seed(port, session_id, winning_body)

        loser = threading.Thread(target=_lose, name="f2job-reseed-loser")
        loser.start()
        started.append(loser)
        try:
            turnstile.await_parked(0)
            winner = threading.Thread(target=_win, name="f2job-reseed-winner")
            winner.start()
            started.append(winner)
            turnstile.await_parked(1)
            # Both executions are validated on the ONE record A created: no
            # second record, no forged duplicate identity.  The count is 1
            # because the initial first build's own record was removed when that
            # execution returned.
            queued_document = _get_state(port, session_id)
            history.append(_served_states(queued_document, _NONCE))
            history.append(_served_states(queued_document, _OTHER_NONCE))
            assert len(_jobs(queued_document)) == 1, _jobs(queued_document)
            assert _served_states(queued_document, _NONCE) == ()
            assert _served_states(queued_document, _OTHER_NONCE) == (_QUEUED,)
            # THE REVIEW'S ORDER: A runs first and its REAL re-seed refuses.
            turnstile.grant(0)
            assert owner_parked.wait(timeout=120), (
                "the losing re-seed never parked inside its own route"
            )
            # THE OWNING EXECUTION'S OWN REAL REFUSAL, served for exactly as
            # long as that execution is inside its route.  B — the same
            # identity, really validated and really live — holds no handle of
            # its own, so it contributes no word: the served record is A's, with
            # A's own real code.
            refused = _job(_get_state(port, session_id), _OTHER_NONCE)
            history.append(_served_states(_get_state(port, session_id), _OTHER_NONCE))
            assert refused["state"] == _REFUSED
            assert refused["refusal_code"] == "SEED_OUT_OF_BOUNDS", (
                "the served code is the OWNER's own real code, never the other "
                f"same-identity execution's: {refused}"
            )
            _assert_document_never_contradicts_itself(_get_state(port, session_id))
            _assert_a_code_belongs_to_its_own_state(_get_state(port, session_id))
            # A's route really completes: the real wire answer is the real
            # refusal code, and the nonce really was consumed exactly once.
            owner_release.set()
            loser.join(timeout=300)
            assert not loser.is_alive(), "the losing re-seed never returned"
            status, payload = loser_outcome["value"]
            assert status == 409
            assert cast("dict[str, Any]", payload["error"])["code"] == "SEED_OUT_OF_BOUNDS"
            after_refusal = cast("dict[str, Any]", engine.state_document())
            assert _nonce_ledger(after_refusal, _OTHER_NONCE) == (1, 0)
            # THE RETAINED GUARANTEE, IN THE MIDDLE OF A CONCURRENCY SCEDULE:
            # A's execution has returned, so A's refusal is gone even though B
            # is still live.  The identity serves no word at all in that gap —
            # not A's refusal, not a stale ``QUEUED``, not B's guess.
            while_b = _get_state(port, session_id)
            history.append(_served_states(while_b, _OTHER_NONCE))
            assert _served_states(while_b, _OTHER_NONCE) == (), (
                f"a returned execution is still served: {_jobs(while_b)}"
            )
            _assert_document_never_contradicts_itself(while_b)
            _assert_a_code_belongs_to_its_own_state(while_b)
            # B now really runs the re-seed rule — and is refused by the
            # F2_SEED_NONCE_IRREVERSIBILITY guard before ANY applying statement,
            # because the refusal above consumed that very nonce.
            turnstile.grant(1)
            winner.join(timeout=300)
            assert not winner.is_alive(), "the same-nonce re-seed never returned"
            # NOTHING was derived for the duplicate: the real re-seed compute
            # gate, armed on B's own valid voxel, was never entered.  This is
            # asserted FIRST, because it is the falsification point: an engine
            # that still applies the same identity twice really derives here.
            assert not entered.is_set(), (
                "the duplicate re-seed really derived a centerline: the same "
                "identity was applied twice"
            )
            # ... so the duplicate's real wire answer is the EXISTING refusal
            # code, and the server's own ledger records no acceptance for it.
            assert winner_outcome["value"][0] == 400
            winner_code = cast(
                "dict[str, Any]", winner_outcome["value"][1]["error"]
            )["code"]
            assert winner_code == "NONCE_REUSE"
            # B's own execution really fails and returns too.  B never owned a
            # handle, so its refusal is not served at any point — not now, not
            # for a nanosecond — and the identity is served NOTHING at all.
            landed = _get_state(port, session_id)
            history.append(_served_states(landed, _OTHER_NONCE))
            _assert_document_never_contradicts_itself(landed)
            _assert_a_code_belongs_to_its_own_state(landed)
            assert _served_states(landed, _OTHER_NONCE) == (), (
                f"an identity's returned executions are still served: {_jobs(landed)}"
            )
            assert _served_refusal_codes(landed, _OTHER_NONCE) == ()
            # ... and a LATER attempt of that consumed nonce is refused by
            # validation with the same code: nothing of that identity is ever
            # derived twice, whatever the handle does.
            replay_status, replay_payload = _post_seed(port, session_id, winning_body)
            assert replay_status == 400
            assert (
                cast("dict[str, Any]", replay_payload["error"])["code"] == "NONCE_REUSE"
            )
        finally:
            turnstile.grant_all()
            owner_release.set()
            release.set()
            for thread in started:
                thread.join(timeout=300)
    finally:
        turnstile.grant_all()
        owner_release.set()
        release.set()
        server.stop()
    # THE WHOLE HISTORY, INCLUDING EVERY EMPTY SAMPLE: the identity's live word
    # is the creating execution's own ``QUEUED`` and its own ``REFUSED``, each
    # while that execution is in flight, and NOTHING is ever served for an
    # execution that has returned.  A loser's real refusal never becomes the
    # identity's state, and no ``COMPLETED`` is ever emitted.
    assert history == [(), (_QUEUED,), (_REFUSED,), (), ()], history
    assert _COMPLETED not in [word for sample in history for word in sample]
    document = cast("dict[str, Any]", engine.state_document())
    assert cast("list[str]", document["accepted_nonces"]) == [_NONCE]
    assert cast("list[str]", document["used_nonces"]) == [_NONCE, _OTHER_NONCE]
    assert _nonce_ledger(document, _OTHER_NONCE) == (1, 0)
    assert document["action_count"] == 1


def test_a_non_consuming_refusal_never_masks_a_successful_retry_of_that_nonce(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RESTORED: R3-F2 — a non-consuming refusal must not mask a successful retry.

    The re-seed's fail-closed ``ViewerOpenError`` branch records a real refusal
    WITHOUT consuming the nonce, which the server legitimately allows the client
    to retry.  The retry really builds and really completes, so the served handle
    must never report the earlier refusal: not while the retry runs (that is not
    its own state) and not after it (that execution has returned, so its record
    is gone).  This is the test the review of 2026-09-27 cited as F1 evidence: the
    previous amendments had made it require a served ``REFUSED`` that SURVIVES a
    successful same-nonce retry, which is precisely the claim the manager
    withdrew.  The expectation here is the product's own truth.

    WHAT CHANGED, AND WHY IT FOLLOWS FROM THE PRODUCT (not from an expectation
    adjustment).  ORIGINAL INTENT, preserved: the reviewer's exact schedule — a
    real non-consuming refusal, then a legitimate retry of the very same nonce
    that really succeeds — and the property that the earlier refusal never masks
    that retry.  The second narrowing removed the only way the earlier refusal
    could have been served at the end: it is removed from the served document
    with its own execution.  So the ending is re-anchored on the two facts that
    survive, and the whole-history check is made whole again (the review's P3:
    the old ``served_states[:-1]`` was empty, because an empty sample
    contributed no entry and the ``[:-1]`` slice therefore excluded nothing).
    Every sample is now retained, INCLUDING the empty ones, and asserted as a
    list.  The HTTP status, the two ledgers and ``action_count`` are unchanged.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=941000)
    session_id = engine.session_id()
    entered = threading.Event()
    release = threading.Event()
    good_voxel, good_direction = _phantom_seed(200, 220, 180)
    #: EVERY sample this test takes, in order, INCLUDING the empty ones.  A poll
    #: that legitimately abstains contributes an empty tuple, and dropping those
    #: would make a whole-history claim vacuous (the 2026-09-27 review's P3).
    history: list[tuple[str, ...]] = []

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    retry_status: dict[str, int] = {}
    worker: threading.Thread | None = None
    try:
        _build_the_initial_centerline(port, session_id)
        body = _seed_body(
            session_id=session_id,
            voxel=good_voxel,
            direction=good_direction,
            nonce=_OTHER_NONCE,
            parent=_NONCE,
        )
        # THE REAL FAIL-CLOSED CONDITION the guard exists for: the governed
        # build identity is unresolved, so the route's own provenance step
        # raises its real ViewerOpenError.  No code, message or error object is
        # fabricated here; the resolved identity is put straight back, and the
        # re-seed really derives before it refuses.
        resolved_identity = engine._build_commit_sha
        assert resolved_identity is not None, "the fixture resolved no build identity"
        engine._build_commit_sha = None
        try:
            status, payload = _post_seed(port, session_id, body)
        finally:
            engine._build_commit_sha = resolved_identity
        assert status == 409
        answered = cast("dict[str, Any]", payload["error"])["code"]
        assert answered == "SESSION_OPEN_FAILED"
        # The refusal really happened and is still answered verbatim to the
        # client that raised it: a non-served refusal is not a hidden one.
        # That execution has now RETURNED, so its record is gone and the
        # identity is served nothing — not a refusal, and not a claim.
        idle = _get_state(port, session_id)
        history.append(_served_states(idle, _OTHER_NONCE))
        assert _served_states(idle, _OTHER_NONCE) == (), (
            "a returned execution is still served: "
            f"{_reported_jobs(idle)}"
        )
        _assert_document_never_contradicts_itself(idle)
        # ... and the nonce really was NOT consumed, so the retry is legitimate.
        document = engine.state_document()
        assert document["used_nonces"] == [_NONCE]
        assert document["accepted_nonces"] == [_NONCE]
        # The compute gate is armed only now, so it holds the RETRY (and only
        # the retry) inside its real derivation.
        _gate_one_reseed(monkeypatch, good_voxel, entered, release)

        def _retry() -> None:
            retry_status["value"] = _post_seed(port, session_id, body)[0]

        worker = threading.Thread(target=_retry, name="f2job-nonce-retry")
        worker.start()
        try:
            assert entered.wait(timeout=120), "the retried re-seed never built"
            while_running = _get_state(port, session_id)
            history.append(_served_states(while_running, _OTHER_NONCE))
            # While the RETRY is in flight, the handle reports THE RETRY'S OWN
            # live state.  It is not the earlier execution's refusal (that record
            # is gone), and it is not a word about the identity.
            assert _served_states(while_running, _OTHER_NONCE) == (_RUNNING,), (
                "the handle did not report the running retry's own state: "
                f"{_jobs(while_running)}"
            )
            assert _REFUSED not in _served_states(while_running, _OTHER_NONCE), (
                "a non-consuming refusal masked the same nonce's real retry: "
                f"{_jobs(while_running)}"
            )
            assert worker.is_alive(), "the retried re-seed already returned"
            release.set()
            worker.join(timeout=300)
            assert not worker.is_alive(), "the retried re-seed never returned"
            assert retry_status["value"] == 200
        finally:
            release.set()
            worker.join(timeout=300)
    finally:
        release.set()
        server.stop()
    # THE WHOLE HISTORY, INCLUDING EVERY EMPTY SAMPLE: nothing while nothing
    # runs, the retry's own ``RUNNING`` while the retry runs, nothing once the
    # retry has returned.  The earlier refusal never appears at any point, so
    # it never masks the legitimate retry — and the retry, once it returns, is
    # not restated as an identity outcome either.
    landed = cast("dict[str, Any]", engine.planes_state())
    history.append(_served_states(landed, _OTHER_NONCE))
    assert history == [(), (_RUNNING,), ()], history
    assert _COMPLETED not in [word for sample in history for word in sample]
    _assert_document_never_contradicts_itself(landed)
    # The seed really was accepted, and the acceptance is reported where it has
    # always been reported — in the frozen document's own ledgers, not as a
    # served state.
    document = engine.state_document()
    assert document["accepted_nonces"] == [_NONCE, _OTHER_NONCE]
    assert document["action_count"] == 2


def test_a_reseed_whose_only_execution_really_fails_is_served_refused_while_it_runs(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-over-correction pin: a lone real re-seed refusal IS served.

    Exactly ONE execution of this identity runs and it really fails, so that
    execution's own observed refusal is publishable and the server must serve it
    with the exact code it answered the client with — abstention is for a case
    the server cannot support, never a way of never refusing.  Repeated polls
    inside that execution's own route return the byte-identical document, and a
    later replay of the same nonce is refused by the real grammar gate, never
    becomes a job, and leaves the served document untouched.

    RENAMED AND RESTORED BY THE F2_JOB_HANDLE_TRUTH SECOND NARROWING (manager
    decision 2026-09-28) from
    ``test_a_reseed_whose_only_execution_really_fails_is_still_served_refused``.
    The old name and the old ending both relied on the withdrawn terminal claim:
    a late poll, taken long after the only execution had returned, still found
    the ``REFUSED`` there, "settled, not merely un-refuted".  The name now says
    when the refusal is served — while its own execution is in flight — and the
    ending adds the stronger half: the instant that execution returns, the
    identity serves nothing at all.  The served word, its exact code, the closed
    key set and the no-acceptance bookkeeping are all unchanged.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=942000)
    session_id = engine.session_id()
    parked = threading.Event()
    release = threading.Event()
    _park_at_route_exit(monkeypatch, [1000000, 1000000, 1000000], parked, release)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    outcome: dict[str, tuple[int, dict[str, Any]]] = {}
    worker: threading.Thread | None = None
    try:
        _build_the_initial_centerline(port, session_id)
        _, good_direction = _phantom_seed(200, 220, 180)
        body = _seed_body(
            session_id=session_id,
            # Far outside every session volume: the governed re-seed raises its
            # own typed refusal (no clamp, no default, no retry).
            voxel=[1000000, 1000000, 1000000],
            direction=good_direction,
            nonce=_OTHER_NONCE,
            parent=_NONCE,
        )
        worker = threading.Thread(
            target=lambda: outcome.update(value=_post_seed(port, session_id, body)),
            name="f2job-lone-reseed-refusal",
        )
        worker.start()
        try:
            assert parked.wait(timeout=120), (
                "the refused re-seed never parked inside its own route"
            )
            in_flight = _get_state(port, session_id)
            refused = _job(in_flight, _OTHER_NONCE)
            assert refused["state"] == _REFUSED
            assert refused["refusal_code"] == "SEED_OUT_OF_BOUNDS"
            assert set(refused) == {"session_id", "client_nonce", "state", "refusal_code"}
            # Repeated polls inside the same execution's route are
            # byte-identical: the server never reverts the word it is serving.
            assert _job(_get_state(port, session_id), _OTHER_NONCE) == refused
            _assert_document_never_contradicts_itself(in_flight)
            _assert_a_code_belongs_to_its_own_state(in_flight)
            release.set()
            worker.join(timeout=300)
            assert not worker.is_alive(), "the refused re-seed never returned"
            status, payload = outcome["value"]
            assert status == 409
            answered = cast("dict[str, Any]", payload["error"])["code"]
            assert refused["refusal_code"] == answered
            # THE RETAINED GUARANTEE: the only execution of this identity has
            # returned, so nothing of it — not the refusal — is served any more.
            returned = _get_state(port, session_id)
            assert _served_states(returned, _OTHER_NONCE) == (), (
                f"a returned execution is still served: {_reported_jobs(returned)}"
            )
            # F2 (2026-09-28 review, P3): the ``_COMPLETED not in
            # _reported_jobs`` line that used to sit here compared a string
            # against served DICTIONARIES and could never fail.  It is REPLACED
            # by the effective document invariant, which reads every served
            # job's ``state`` and fails on a served ``COMPLETED``: unlike the
            # removed line this document's served set is not proven empty by the
            # absence assertion above (which constrains ``_OTHER_NONCE`` only),
            # so the check is kept rather than deleted.
            _assert_document_never_contradicts_itself(returned)
            # A later replay of the same nonce is refused by the real grammar
            # gate, never becomes a job and leaves the served document untouched.
            replay_status, replay_payload = _post_seed(port, session_id, body)
            assert replay_status == 400
            assert (
                cast("dict[str, Any]", replay_payload["error"])["code"] == "NONCE_REUSE"
            )
            assert _served_states(_get_state(port, session_id), _OTHER_NONCE) == ()
            # No fabricated success: the only accepted action is the initial build.
            document = cast("dict[str, Any]", engine.state_document())
            assert document["accepted_nonces"] == [_NONCE]
            assert document["action_count"] == 1
        finally:
            release.set()
            parked.set()
            if worker is not None:
                worker.join(timeout=300)
    finally:
        release.set()
        server.stop()


def test_an_admission_is_not_an_acceptance_so_a_real_refusal_is_still_published(
    volume_only_session,
) -> None:
    """The award still owns the handle, and a real refusal is still published.

    RENAMED IN THE F2_JOB_HANDLE_TRUTH NARROWING (2026-09-27) from
    ``test_the_admitted_executions_real_refusal_is_still_the_identitys_truth``.
    The name no longer promises the identity's truth — the identity-level
    terminal claim was withdrawn, and under the second narrowing (manager
    decision 2026-09-28) it is withdrawn again in the only form left: the
    refused execution's word is gone the moment that execution returns.  What
    the test now states is exactly what the product gives: an admission is not
    an acceptance, the awarded execution's own real refusal is served with its
    exact code WHILE that execution is in flight, no other execution of the
    identity can put a word on that handle, and the awarded execution's own
    route exit removes the record for good.

    Driven at the observation API with three DISTINCT request objects on one
    ``client_nonce``: the handle is awarded by the REAL F2 admission (the
    unchanged rule, called exactly as the route calls it, in the same critical
    section), and the AWARDED execution then really raises the governed typed
    refusal.  An admission CONSUMED the nonce but did not ACCEPT the work, so
    while it is running the handle reports that real refusal with its exact
    code, and nothing turns an admitted-but-failed execution into a completion.
    A later same-nonce loser, really refused and really released, can neither
    publish a word of its own nor re-attribute, downgrade or remove the served
    record.  Only the served document is read.

    CHANGES FOR THE SECOND NARROWING, each forced by the product and each
    marked: (i) the non-owner effects are still exercised BEFORE the owner
    returns, which is now the hard case — a non-owner's real refusal must not
    touch a handle that is being served, and a non-owner's route exit must not
    remove it; (ii) the "settled" ending is replaced by the owner returning,
    after which the identity serves nothing at all.  The 2026-09-27 review's P3
    (two identical assertions back to back with no intervening operation) is
    fixed here: each of the two remaining reads is separated by a real
    non-owner operation, and each asserts a distinct fact.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=943000)
    creator = _seed_request(engine, _NONCE)
    admitted = _seed_request(engine, _NONCE)
    loser = _seed_request(engine, _NONCE)

    def served() -> dict[str, Any]:
        """The SERVED document (the only thing these assertions read)."""

        return cast("dict[str, Any]", engine.planes_state())

    engine._seed_job_queued(creator)
    with engine._seed_admit_lock:
        # The REAL F2 admission, called as the route calls it: it consumes the
        # nonce into used_nonces and then the handle is awarded to `admitted`.
        engine._admit_seed_first_build(admitted)
        engine._seed_job_admit_owner(admitted)
    # The awarded execution really raises the governed seam refusal, and is
    # still inside its route: its own real refusal is served, verbatim.
    engine._seed_job_running(admitted)
    engine._seed_job_refused(admitted, "SEED_OUT_OF_BOUNDS")
    refused = _job(served(), _NONCE)
    assert refused["state"] == _REFUSED
    assert refused["refusal_code"] == "SEED_OUT_OF_BOUNDS"
    assert set(refused) == {"session_id", "client_nonce", "state", "refusal_code"}
    # An admission is not an acceptance: the work never returned accepted, so
    # the handle reports the refusal its own execution really observed.
    document = cast("dict[str, Any]", engine.state_document())
    assert document["used_nonces"] == [_NONCE]
    assert document["accepted_nonces"] == []
    assert document["action_count"] == 0
    # The record CREATOR never held the handle (the award moved it), so its own
    # route exit removes nothing: the awarded handle keeps being served.
    engine._seed_job_release(creator)
    assert _job(served(), _NONCE) == refused
    # A same-identity execution is now live and running, and it is a NON-OWNER:
    # its observations are all dropped, so it can publish no word of its own and
    # cannot re-attribute the handle that is already being served.
    engine._seed_job_queued(loser)
    while_live = served()
    assert _job(while_live, _NONCE) == refused, (
        f"a non-owner took the served handle: {_reported_jobs(while_live)}"
    )
    _assert_document_never_contradicts_itself(while_live)
    engine._seed_job_refused(loser, "NONCE_REUSE")
    engine._seed_job_running(loser)
    assert _job(served(), _NONCE) == refused, (
        "a non-owner's real refusal reached the served handle: "
        f"{_reported_jobs(served())}"
    )
    # ... and a non-owner's own route exit removes nothing either.
    engine._seed_job_release(loser)
    assert _job(served(), _NONCE) == refused
    # THE OWNER RETURNS: its record goes with it, and the identity then serves
    # nothing at all.  Still nothing was accepted, and the persisted document
    # carries no job.
    engine._seed_job_release(admitted)
    assert _served_states(served(), _NONCE) == (), (
        f"a returned execution is still served: {_reported_jobs(served())}"
    )
    assert cast("dict[str, Any]", engine.state_document())["accepted_nonces"] == []
    assert _JOBS_KEY not in cast("dict[str, Any]", engine.state_document())



# ---------------------------------------------------------------------------
# ATT0005 (review round 4) — HISTORICAL RECORD.  The round's findings, verbatim
# in substance:
#
# F1: the exemption for an admission-AWARDED handle rested on a FALSE premise
#     (a consumed nonce rules out a later same-nonce acceptance) — the re-seed
#     path never re-checks the nonce — so a terminal REFUSED could be served and
#     then overturned.  F2: a validated request was registered live only after
#     the used_nonces check.  F3: accepted_nonces was read before the live
#     snapshot, so a poll could serve a false REFUSED.  F4: the served refusal
#     code could belong to a refusal the served state does not report.
#
# WHICH OF THOSE MECHANISMS SURVIVE, under the F2_JOB_HANDLE_TRUTH SECOND
# NARROWING (manager decision 2026-09-28): none of the serve-time reads and
# none of the withholding rules.  F1's exemption, F2's live registry, F3's
# acceptance and snapshot reads and the whole class of "read order" hazards are
# gone, because nothing outlives its execution.  What is left, and what the
# tests below now pin, is exactly: the handle is bound to the execution that
# holds it (F4, structurally — the served code is read off the very record the
# serving execution recorded), and an execution that has returned contributes
# nothing at all.
#
# Every test below drives the REAL route over the real server and reads only
# the SERVED document.  The turnstiles and gates decide only WHEN a validated
# request proceeds and WHEN a real compute is held open; the real rules run
# unchanged and produce the real refusals and the real acceptances asserted
# here.  No product code is replaced to make a test deterministic.
# ---------------------------------------------------------------------------


def _served_refusal_codes(document: dict[str, Any], nonce: str) -> tuple[str | None, ...]:
    """Every ``refusal_code`` the document serves for ``nonce`` (served truth)."""

    return tuple(
        job.get("refusal_code")
        for job in _reported_jobs(document)
        if job["client_nonce"] == nonce
    )


def _assert_a_code_belongs_to_its_own_state(document: dict[str, Any]) -> None:
    """A refusal code is served on a refusal, and on nothing else (F4).

    The mechanical half of F4: a ``refusal_code`` key is present if and only if
    the served state is ``REFUSED``, so no code can ride on a live word and no
    refusal can be served without the code the server really raised.  The code is
    read straight off the record the serving execution recorded, so it cannot be
    borrowed from another attempt.
    """

    for job in _reported_jobs(document):
        assert ("refusal_code" in job) == (job["state"] == _REFUSED), (
            f"a refusal code is attached to the wrong served state: {job}"
        )


def _park_before_registration(
    monkeypatch: pytest.MonkeyPatch,
    voxel: list[int],
    parked: threading.Event,
    release: threading.Event,
) -> None:
    """Park the request with ``voxel`` between validation and its first record.

    ATT0005/F2's window, made deterministic instead of a thread switch.  The
    wrapper sits on ``_seed_job_queued``, which :meth:`apply_seed_bytes` calls
    right after the ``used_nonces`` check and OUTSIDE every lock, so parking
    holds nothing; the real registration and then the real rules run unchanged
    once the gate is released.  Only the request carrying ``voxel`` parks, so a
    competing same-identity request proceeds exactly as the reviewer ordered it.

    This was the window in which a validated execution had to be VISIBLE to the
    serve path, because a record that outlived its own execution had to be
    withheld while another execution was live.  It is kept as a window because it
    is a real one: an execution that has not observed anything holds no record,
    so the identity is served NOTHING for it — which is the retained guarantee,
    asserted by the test that uses this hook.
    """

    real_queued = ViewerEngine._seed_job_queued

    def _gated(self, request, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if tuple(request.voxel) == tuple(voxel):
            parked.set()
            assert release.wait(timeout=120), "the registration gate was never released"
        return real_queued(self, request, **kwargs)

    monkeypatch.setattr(ViewerEngine, "_seed_job_queued", _gated)


def test_a_refusal_a_later_attempt_could_overturn_is_served_only_while_its_execution_runs(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4-F1, restored: a returned execution's refusal can never be overturned,
    because it is no longer there to be overturned.

    ORIGINAL INTENT (ATT0005, R4-F1, PRESERVED): the reviewer's own reproducer
    in the reviewer's own order — B wins the real first-build admission, really
    consumes the nonce and is really refused ``SEED_OUT_OF_BOUNDS``; a different
    identity M then really builds; C, validated before that consumption, resumes
    on the re-seed branch.  The property defended is unchanged in substance: the
    handle must never report something a later attempt of that identity
    contradicts.  Under the second narrowing that is guaranteed STRUCTURALLY
    rather than by a withholding rule: B's record is removed when B's own
    execution returns, so there is nothing left for C — or for any later
    attempt — to contradict, in either order.  The old name promised a
    "terminal" word that is withheld until the identity is settled; the name now
    states what is true.

    WHY THE ORIGINAL SCENARIO IS IMPOSSIBLE NOW: the original end state was C
    being really ACCEPTED under the very nonce B's refusal consumed — the second
    application of one identity, which the F2_SEED_NONCE_IRREVERSIBILITY guard
    refuses.  C is already validated (validation consumes nothing), so it reaches
    the re-seed acceptance path with the nonce in ``used_nonces`` and is refused
    there with the EXISTING ``NONCE_REUSE`` code before any applying statement.

    THE NEW TRUTH PINNED HERE (the turnstile): C's real outcome is the guard's
    ``NONCE_REUSE`` refusal, not an acceptance — ``accepted_nonces`` never gains
    that nonce, ``action_count`` counts only M's one real accepted action, and
    a fresh attempt of the same consumed nonce is refused by validation with the
    same code.  B's own real refusal IS served, verbatim with its own code, but
    only while B's own execution is still inside its route (parked at the real
    route-exit hook); from the moment B returns — including the whole window in
    which C is still live — the identity serves nothing at all.  The review's
    P3 about empty whole-history slices is fixed: every sample is kept, empty
    ones included, and asserted as a list.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=944000)
    session_id = engine.session_id()
    turnstile = _Turnstile()
    _, first_build_direction = _phantom_seed(40, 60, 20)
    reseed_voxel, reseed_direction = _phantom_seed(200, 220, 180)
    refused_parked = threading.Event()
    refused_release = threading.Event()
    #: EVERY sample, in order, INCLUDING the empty ones (review P3).
    history: list[tuple[str, ...]] = []

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    outcomes: dict[str, tuple[int, dict[str, Any]]] = {}
    started: list[threading.Thread] = []
    try:
        # Every validated seed parks at the turnstile, outside every lock, so
        # both same-nonce requests are validated BEFORE either reaches a rule.
        _park_validated_seed(monkeypatch, turnstile)
        # Only B (the execution that really refuses) parks at its route exit:
        # its own real refusal is then readable at the one instant it may be.
        _park_at_route_exit(
            monkeypatch, [1000000, 1000000, 1000000], refused_parked, refused_release
        )
        refused_body = _seed_body(
            session_id=session_id,
            voxel=[1000000, 1000000, 1000000],
            direction=first_build_direction,
            nonce=_NONCE,
            parent=None,
        )
        later_acceptance_body = _seed_body(
            session_id=session_id,
            voxel=reseed_voxel,
            direction=reseed_direction,
            nonce=_NONCE,
            parent=None,
        )
        other_body = _seed_body(
            session_id=session_id,
            voxel=_phantom_seed(40, 60, 20)[0],
            direction=first_build_direction,
            nonce=_THIRD_NONCE,
            parent=None,
        )

        def _post_as(name: str, body: bytes) -> None:
            outcomes[name] = _post_seed(port, session_id, body)

        refused = threading.Thread(
            target=lambda: _post_as("refused", refused_body), name="f2job-r4f1-refused"
        )
        refused.start()
        started.append(refused)
        try:
            turnstile.await_parked(0)
            later = threading.Thread(
                target=lambda: _post_as("later", later_acceptance_body),
                name="f2job-r4f1-later",
            )
            later.start()
            started.append(later)
            turnstile.await_parked(1)
            # BOTH same-nonce executions are validated and inside their routes;
            # the handle exists as one QUEUED record owned by the creator.
            both_parked = _get_state(port, session_id)
            history.append(_served_states(both_parked, _NONCE))
            assert _served_states(both_parked, _NONCE) == (_QUEUED,)
            # B really wins the admission, really consumes the nonce, really
            # refuses, and is then held open INSIDE its own route.
            turnstile.grant(0)
            assert refused_parked.wait(timeout=120), (
                "the admitted-and-refused seed never parked inside its own route"
            )
            # B'S OWN REAL REFUSAL, served because B is still running: its own
            # code, the closed key set, and nothing claimed about the identity.
            in_flight = _get_state(port, session_id)
            history.append(_served_states(in_flight, _NONCE))
            served = _job(in_flight, _NONCE)
            assert served["state"] == _REFUSED
            assert served["refusal_code"] == "SEED_OUT_OF_BOUNDS"
            _assert_document_never_contradicts_itself(in_flight)
            _assert_a_code_belongs_to_its_own_state(in_flight)
            # B's route really returns now.  Its refusal goes with it — even
            # though C, a same-nonce execution, is still inside its own route.
            # THIS is the property under review, now structural: there is no
            # served word left for a later attempt of that identity to
            # contradict.
            refused_release.set()
            refused.join(timeout=300)
            assert not refused.is_alive(), "the admitted-and-refused seed never returned"
            status, payload = outcomes["refused"]
            assert status == 409
            owner_code = cast("dict[str, Any]", payload["error"])["code"]
            assert owner_code == "SEED_OUT_OF_BOUNDS"
            while_c_live = _get_state(port, session_id)
            history.append(_served_states(while_c_live, _NONCE))
            assert _served_states(while_c_live, _NONCE) == (), (
                f"a returned execution is still served: {_jobs(while_c_live)}"
            )
            _assert_document_never_contradicts_itself(while_c_live)
            # M (a DIFFERENT identity) really builds the initial centerline.
            other = threading.Thread(
                target=lambda: _post_as("other", other_body), name="f2job-r4f1-other"
            )
            other.start()
            started.append(other)
            turnstile.await_parked(2)
            turnstile.grant(2)
            other.join(timeout=300)
            assert not other.is_alive(), "the other first build never returned"
            assert outcomes["other"][0] == 200
            with_centerline = _get_state(port, session_id)
            history.append(_served_states(with_centerline, _NONCE))
            history.append(_served_states(with_centerline, _THIRD_NONCE))
            assert _served_states(with_centerline, _NONCE) == ()
            # M's own accepted first build is not reported either: M's execution
            # has returned too.
            assert _served_states(with_centerline, _THIRD_NONCE) == ()
            # C resumes and really takes the re-seed branch.  It must be refused
            # by the F2_SEED_NONCE_IRREVERSIBILITY guard under the very nonce B's
            # refusal consumed: nothing of C is derived, and no acceptance of
            # that identity ever happens.  The ledger is read FIRST, because it
            # is the falsification point — an engine that still accepts the
            # duplicate has already advanced the accepted chain by then, and
            # this fails on the server's own bookkeeping rather than on which
            # wire status the duplicate happened to get.
            turnstile.grant(1)
            later.join(timeout=300)
            assert not later.is_alive(), "the later same-nonce seed never returned"
            after_c = cast("dict[str, Any]", engine.state_document())
            assert _nonce_ledger(after_c, _NONCE) == (1, 0), (
                "the later same-nonce seed really advanced the accepted chain: "
                f"{after_c['accepted_nonces']}"
            )
            assert after_c["action_count"] == 1, (
                f"a second action was really accepted: {after_c['action_count']}"
            )
            assert outcomes["later"][0] == 400
            later_code = cast("dict[str, Any]", outcomes["later"][1]["error"])["code"]
            assert later_code == "NONCE_REUSE"
            # C never owned a handle, so its own real refusal is never served
            # either: the identity serves NOTHING, in either direction.
            landed = _get_state(port, session_id)
            history.append(_served_states(landed, _NONCE))
            _assert_document_never_contradicts_itself(landed)
            _assert_a_code_belongs_to_its_own_state(landed)
            assert _served_states(landed, _NONCE) == (), (
                f"a returned execution is still served: {_jobs(landed)}"
            )
            assert _served_refusal_codes(landed, _NONCE) == ()
            # A fresh attempt of that consumed nonce fails closed at validation,
            # with the same code: the identity can never be accepted again.
            replay = threading.Thread(
                target=lambda: _post_as("replay", later_acceptance_body),
                name="f2job-r4f1-replay",
            )
            replay.start()
            started.append(replay)
            replay.join(timeout=300)
            assert not replay.is_alive(), "the replayed seed never returned"
            assert outcomes["replay"][0] == 400
            assert (
                cast("dict[str, Any]", outcomes["replay"][1]["error"])["code"]
                == "NONCE_REUSE"
            )
        finally:
            turnstile.grant_all()
            refused_release.set()
            for thread in started:
                thread.join(timeout=300)
    finally:
        turnstile.grant_all()
        refused_release.set()
        server.stop()
    # THE WHOLE HISTORY, INCLUDING EVERY EMPTY SAMPLE: this identity's served
    # word is B's own ``REFUSED`` and only while B's own execution is inside
    # its route.  From the instant B returns — through M's real build, through
    # C's real refusal and after it — the identity serves nothing at all, so no
    # later attempt of that nonce can ever contradict a served word.
    assert history == [(_QUEUED,), (_REFUSED,), (), (), (), ()], history
    assert _COMPLETED not in [word for sample in history for word in sample]
    document = cast("dict[str, Any]", engine.state_document())
    assert cast("list[str]", document["accepted_nonces"]) == [_THIRD_NONCE]
    assert _nonce_ledger(document, _NONCE) == (1, 0)
    assert document["action_count"] == 1
    # M's own accepted identity is not reported either, and B's refusal is
    # gone rather than re-published: the ledger is the only place either
    # outcome survives.
    settled = cast("dict[str, Any]", engine.planes_state())
    assert _served_states(settled, _NONCE) == ()
    assert _served_refusal_codes(settled, _NONCE) == ()
    assert _served_states(settled, _THIRD_NONCE) == ()
    # F2 (2026-09-28 review, P3): the ``_COMPLETED not in _reported_jobs`` line
    # that used to sit here compared a string against served DICTIONARIES and
    # could never fail, so it is REMOVED: the three absence assertions above
    # already cover every identity this test ever creates (``_NONCE`` and
    # ``_THIRD_NONCE``), so the served job list is empty.  No completion coverage
    # is claimed from this site.


def test_a_same_identity_execution_is_served_only_from_its_first_observation(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4-F2: the served window is FIRST OBSERVATION -> ROUTE RETURN, no wider.

    F3 (2026-09-28 review, P3) — THE NAME IS CORRECTED HERE.  This test used to
    be called ``test_no_same_identity_execution_is_invisible_between_validation_and_registration``
    and to claim "an execution is invisible in NO window of its own route".  That
    was false on its own evidence: the first two samples of the asserted history
    are ``()`` — the window between validation and the execution's FIRST
    observation really does serve nothing, because a validated request that has
    observed nothing has no record at all.  The retained claim is the honest
    one, and it is what the whole history below pins: an execution is served its
    OWN live state from its first observation until its route returns, and is
    served nothing at any other instant.

    ORIGINAL INTENT (ATT0005, R4-F2, PRESERVED), and it is now trivially
    satisfiable, which is the point of pinning it.  The reviewer's F2
    interleaving is still built exactly: B VALIDATES first and is parked in the
    window between the ``used_nonces`` check and its own first observation; A (a
    seed the real re-seed refuses) then runs, really refuses, really consumes the
    nonce and returns.  The original property was that A's terminal refusal could
    not be published while B was live, which needed a LIVE REGISTRY to know B was
    there.  There is no registry any more and none is needed: B has no record at
    all until it observes one, and A's record is gone the moment A returns, so
    the identity serves nothing in this window — not a stale refusal, not a
    borrowed word.  What the test now pins is the RETAINED guarantee at each of
    the two windows that exist: an execution is unreported before its first
    observation, reported as its own live state while it is inside its route, and
    unreported again the instant that route returns.

    WHY THE ORIGINAL SCENARIO IS IMPOSSIBLE NOW: the original end state was B
    really deriving and really completing after A's refusal — the second
    application of one identity, which the F2_SEED_NONCE_IRREVERSIBILITY guard
    refuses.  B is validated (validation consumes nothing), so when it reaches
    the re-seed acceptance path the nonce is already in ``used_nonces`` and the
    guard refuses it with the EXISTING ``NONCE_REUSE`` code before any applying
    statement, so B never derives.

    THE NEW TRUTH PINNED HERE: B is refused ``NONCE_REUSE``; NOTHING is derived
    for it (the real re-seed compute gate armed on its own valid voxel is never
    entered); ``accepted_nonces`` never gains the nonce; ``action_count`` advances
    exactly once for the initial build; and the whole served history is B's own
    live ``QUEUED`` while B is inside its route and NOTHING at any other instant.
    The review's P3 about empty whole-history slices is fixed: every sample is
    kept, empty ones included, and asserted as a list.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=945000)
    session_id = engine.session_id()
    good_voxel, good_direction = _phantom_seed(200, 220, 180)
    parked = threading.Event()
    release = threading.Event()
    entered = threading.Event()
    reseed_release = threading.Event()
    exit_parked = threading.Event()
    exit_release = threading.Event()
    #: EVERY sample, in order, INCLUDING the empty ones (review P3).
    history: list[tuple[str, ...]] = []

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    retry_outcome: dict[str, tuple[int, dict[str, Any]]] = {}
    retry: threading.Thread | None = None
    try:
        _build_the_initial_centerline(port, session_id)
        # Arm all three gates, so the same-nonce retry is parked in the window
        # the review names, its real derive would be held open afterwards, and
        # its route exit is held open once the guard has answered it.
        _park_before_registration(monkeypatch, good_voxel, parked, release)
        _gate_one_reseed(monkeypatch, good_voxel, entered, reseed_release)
        _park_at_route_exit(monkeypatch, good_voxel, exit_parked, exit_release)
        retry_body = _seed_body(
            session_id=session_id,
            voxel=good_voxel,
            direction=good_direction,
            nonce=_OTHER_NONCE,
            parent=_NONCE,
        )

        def _retry() -> None:
            retry_outcome["value"] = _post_seed(port, session_id, retry_body)

        # B VALIDATES first (validation does not consume a nonce) and is parked
        # between the nonce check and its first observation.
        retry = threading.Thread(target=_retry, name="f2job-r4f2-retry")
        retry.start()
        try:
            assert parked.wait(timeout=120), (
                "the same-nonce retry never parked between validation and its "
                "first observation"
            )
            # B is inside its route but has observed nothing yet, so it holds no
            # record.  A has not run at all.
            before_a = _get_state(port, session_id)
            history.append(_served_states(before_a, _OTHER_NONCE))
            assert _served_states(before_a, _OTHER_NONCE) == (), (
                f"an execution that has observed nothing is already served: "
                f"{_reported_jobs(before_a)}"
            )
            # A (the real out-of-bounds re-seed) then really refuses, really
            # consumes that nonce and returns: it owned the one record, and its
            # route exit removed it.
            refusal_status, refusal_payload = _post_seed(
                port,
                session_id,
                _seed_body(
                    session_id=session_id,
                    voxel=[1000000, 1000000, 1000000],
                    direction=good_direction,
                    nonce=_OTHER_NONCE,
                    parent=_NONCE,
                ),
            )
            assert refusal_status == 409
            owner_code = cast("dict[str, Any]", refusal_payload["error"])["code"]
            assert owner_code == "SEED_OUT_OF_BOUNDS"
            # THE WINDOW, RESTATED FOR THE RETAINED GUARANTEE: A has returned
            # and B is validated but has not yet observed anything, so the
            # identity serves NOTHING — the returned A's refusal is not there to
            # be mistaken for B's, and B has no word of its own to show.
            unregistered = _get_state(port, session_id)
            history.append(_served_states(unregistered, _OTHER_NONCE))
            assert _served_states(unregistered, _OTHER_NONCE) == (), (
                "a validated same-identity execution that has not observed "
                f"anything was served a word: {_reported_jobs(unregistered)}"
            )
            _assert_document_never_contradicts_itself(unregistered)
            # B observes its own ``QUEUED`` and is then inside its route at
            # exactly one of the two gates — deterministically, whichever the
            # engine under test really reaches:
            #
            #   * a CORRECT engine refuses the duplicate at the re-seed rule, so
            #     the governed compute is never entered and the execution parks
            #     at its route exit (``exit_parked``);
            #   * an engine that still double-accepts really derives, so the
            #     governed compute is entered (``entered``) and is held open
            #     there.  That is the same property — inside its route, not
            #     finished — reached by a different route, and the served-bytes
            #     assertions below decide the verdict either way.  This test must
            #     falsify on the served document, not on which gate fired.
            release.set()
            assert _await_either(
                entered, exit_parked, timeout=120
            ), "the same-nonce retry reached neither gate"
            if entered.is_set():
                reseed_release.set()
            registered = _get_state(port, session_id)
            history.append(_served_states(registered, _OTHER_NONCE))
            # INSIDE ITS OWN ROUTE, the identity is served B's OWN live state.
            # It is never A's returned refusal: A's record was removed when A
            # returned, and the handle is bound to the execution that holds it.
            assert _served_states(registered, _OTHER_NONCE) == (_QUEUED,), (
                f"the handle did not report the running execution's own state: "
                f"{_reported_jobs(registered)}"
            )
            assert _REFUSED not in _served_states(registered, _OTHER_NONCE), (
                f"a returned execution's refusal was published for a registered, "
                f"still-running same-identity execution: {_reported_jobs(registered)}"
            )
            _assert_document_never_contradicts_itself(registered)
            assert retry.is_alive(), "the same-nonce retry already returned"
            # B's route returns, and the identity is unreported again.
            exit_release.set()
            retry.join(timeout=300)
            assert not retry.is_alive(), "the same-nonce retry never returned"
            # NOTHING was derived for the duplicate: the real re-seed compute
            # was never entered at all.  Asserted before the wire answer, because
            # this is the falsification point for a double-accepting engine.
            assert not entered.is_set(), (
                "the duplicate re-seed really derived a centerline: the same "
                "identity was applied twice"
            )
            # ... so its real outcome is the guard's refusal with the EXISTING
            # code, and the server's own ledger records no acceptance for it.
            assert retry_outcome["value"][0] == 400
            retry_code = cast("dict[str, Any]", retry_outcome["value"][1]["error"])[
                "code"
            ]
            assert retry_code == "NONCE_REUSE"
            landed = _get_state(port, session_id)
            history.append(_served_states(landed, _OTHER_NONCE))
            _assert_document_never_contradicts_itself(landed)
            _assert_a_code_belongs_to_its_own_state(landed)
            # BOTH executions of this identity have now returned, so nothing of
            # either is served: neither the returned A's real refusal nor B's
            # own live word survives B's route exit.
            assert _served_states(landed, _OTHER_NONCE) == (), (
                f"a returned execution is still served: {_reported_jobs(landed)}"
            )
            assert _served_refusal_codes(landed, _OTHER_NONCE) == ()
        finally:
            # NOTE: ``entered`` is deliberately NOT set here — it is the
            # negative assertion aid for "the governed re-seed compute was
            # never entered", and only the real product code may set it.
            parked.set()
            release.set()
            reseed_release.set()
            exit_parked.set()
            exit_release.set()
            if retry is not None:
                retry.join(timeout=300)
    finally:
        parked.set()
        release.set()
        reseed_release.set()
        exit_parked.set()
        exit_release.set()
        server.stop()
    # THE WHOLE HISTORY, INCLUDING EVERY EMPTY SAMPLE: this identity is served a
    # word at exactly one instant in the whole schedule — while B itself is
    # inside its own route — and it is B's own ``QUEUED``.  No refusal is ever
    # published, no ``COMPLETED`` is ever emitted, and no same-nonce attempt was
    # ever accepted.
    assert history == [(), (), (_QUEUED,), ()], history
    assert _COMPLETED not in [word for sample in history for word in sample]
    document = cast("dict[str, Any]", engine.state_document())
    assert cast("list[str]", document["accepted_nonces"]) == [_NONCE]
    assert _nonce_ledger(document, _OTHER_NONCE) == (1, 0)
    assert document["action_count"] == 1




def test_a_poll_serves_a_word_only_while_that_execution_is_in_flight(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4-F3, restored and re-scoped: a poll sees a running execution, nothing else.

    ORIGINALLY NAMED
    ``test_a_poll_never_serves_a_refusal_from_a_stale_accepted_nonces_read``, and
    before that it promised that a poll must never serve a terminal ``REFUSED``
    out of a STALE read of the acceptance fact — the reviewer's F3 interleaving,
    in which a poll's acceptance read is held open while a same-nonce execution
    is really accepted.  That read ORDER is no longer a choice, because the read
    does not exist: the seed-scoped acceptance evidence was removed with the
    identity-level completion claim it licensed, and under the second narrowing
    the serve path takes NO read at all beyond one ``_seed_job_lock`` hold over
    the records the running executions hold.  A server that has nothing to read
    cannot be stale about anything, so the hazard class is gone with the claim.

    WHY THAT MEANS THE INSTRUMENTATION IS DELETED, NOT KEPT (review P3): the
    earlier version installed a gate on the engine's seed-acceptance attribute
    under a ``hasattr`` guard and then asserted that the poll had read it.  That
    assertion was already unreachable, and under this product it is
    unreachable BY CONSTRUCTION rather than by accident.  It is removed, along
    with the instrument class itself.  HONESTY NOTE, stated here rather than
    claimed as a falsification: this test can no longer falsify the dropped
    mechanism, because that mechanism does not exist.  What it DOES falsify is
    the retained guarantee, on the reviewer's own schedule, with no gate at all.

    THE NEW TRUTH PINNED HERE: a poll taken while a same-identity execution is
    inside its own route serves that execution's OWN live state and nothing
    else — never a returned execution's refusal, never a borrowed code, never a
    completion.  Then the same-nonce execution's real outcome is the guard's
    ``NONCE_REUSE`` refusal with NOTHING derived for it, ``accepted_nonces`` never
    gains the nonce and ``action_count`` advances only for the initial build; and
    after that execution returns, the identity is served nothing at all.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=946000)
    session_id = engine.session_id()
    good_voxel, good_direction = _phantom_seed(200, 220, 180)
    entered = threading.Event()
    reseed_release = threading.Event()
    parked = threading.Event()
    register = threading.Event()
    exit_parked = threading.Event()
    exit_release = threading.Event()

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    polled: dict[str, dict[str, Any]] = {}
    outcome: dict[str, tuple[int, dict[str, Any]]] = {}
    worker: threading.Thread | None = None
    try:
        _build_the_initial_centerline(port, session_id)
        # Arm the reviewer's three windows, all keyed on B's own valid voxel so
        # the initial first build and the out-of-bounds loser are never gated.
        _park_before_registration(monkeypatch, good_voxel, parked, register)
        _gate_one_reseed(monkeypatch, good_voxel, entered, reseed_release)
        _park_at_route_exit(monkeypatch, good_voxel, exit_parked, exit_release)
        body = _seed_body(
            session_id=session_id,
            voxel=good_voxel,
            direction=good_direction,
            nonce=_OTHER_NONCE,
            parent=_NONCE,
        )

        def _retry() -> None:
            outcome["value"] = _post_seed(port, session_id, body)

        # The same-nonce execution VALIDATES first (validation does not consume
        # a nonce) and is parked between the nonce check and its first
        # observation.
        worker = threading.Thread(target=_retry, name="f2job-r4f3-retry")
        worker.start()
        try:
            assert parked.wait(timeout=120), (
                "the same-nonce execution never parked between validation and "
                "its first observation"
            )
            # The out-of-bounds re-seed then really refuses, really consumes
            # that nonce and RETURNS — which is the whole point: its real
            # refusal is removed with it, before the same-nonce execution has
            # even observed anything.
            refusal_status, refusal_payload = _post_seed(
                port,
                session_id,
                _seed_body(
                    session_id=session_id,
                    voxel=[1000000, 1000000, 1000000],
                    direction=good_direction,
                    nonce=_OTHER_NONCE,
                    parent=_NONCE,
                ),
            )
            assert refusal_status == 409
            owner_code = cast("dict[str, Any]", refusal_payload["error"])["code"]
            assert owner_code == "SEED_OUT_OF_BOUNDS"
            # The same-nonce execution then observes its own state and is inside
            # its route at exactly one of the two gates — deterministically,
            # whichever the engine under test really reaches:
            #
            #   * a CORRECT engine refuses the duplicate at the re-seed rule, so
            #     the governed compute is never entered and the execution parks
            #     at its route exit (``exit_parked``);
            #   * an engine that still double-accepts really derives, so the
            #     governed compute is entered (``entered``) and is held open
            #     there.  That is the same property — inside its route, not
            #     finished — reached by a different route, and the served-bytes
            #     assertions below decide the verdict either way.  This test must
            #     falsify on the served document, not on which gate fired.
            register.set()
            assert _await_either(
                entered, exit_parked, timeout=120
            ), "the same-nonce execution reached neither gate"
            if entered.is_set():
                reseed_release.set()
            assert worker.is_alive(), "the same-nonce execution already returned"
            # THE POLL, with no gate on anything: it must see the running
            # execution's own live state, and nothing that has already returned.
            poller = threading.Thread(
                target=lambda: polled.update(value=_get_state(port, session_id)),
                name="f2job-r4f3-poll",
            )
            poller.start()
            poller.join(timeout=300)
            assert not poller.is_alive(), "the poll never returned"
            document = cast("dict[str, Any]", polled["value"])
            assert _served_states(document, _OTHER_NONCE) == (_QUEUED,), (
                "a poll did not report the running execution's own live state: "
                f"{_reported_jobs(document)}"
            )
            assert _REFUSED not in _served_states(document, _OTHER_NONCE), (
                "a poll served the RETURNED execution's refusal for a "
                f"still-running same-identity execution: {_reported_jobs(document)}"
            )
            assert _COMPLETED not in _served_states(document, _OTHER_NONCE), (
                f"a refused duplicate seed was reported as COMPLETED: "
                f"{_reported_jobs(document)}"
            )
            _assert_document_never_contradicts_itself(document)
            _assert_a_code_belongs_to_its_own_state(document)
            # The same-nonce execution's route really returns now.
            exit_release.set()
            worker.join(timeout=300)
            assert not worker.is_alive(), "the same-nonce execution never returned"
        finally:
            # NOTE: ``entered`` is deliberately NOT set here — it is the
            # negative assertion aid for "the governed re-seed compute was
            # never entered", and only the real product code may set it.
            parked.set()
            register.set()
            reseed_release.set()
            exit_parked.set()
            exit_release.set()
            if worker is not None:
                worker.join(timeout=300)
    finally:
        parked.set()
        register.set()
        reseed_release.set()
        exit_parked.set()
        exit_release.set()
        server.stop()
    # THE SAME-NONCE EXECUTION WAS NEVER ACCEPTED: the server's own ledger says
    # so, and the wire answer to the duplicate is the EXISTING refusal code.
    # Every one of these is asserted on the served document and the server's own
    # ledger, so an engine that still double-accepts this identity fails HERE,
    # on served bytes, whether it did so through the re-seed compute or through
    # any other route.
    settled = cast("dict[str, Any]", engine.planes_state())
    assert _served_states(settled, _OTHER_NONCE) == (), (
        f"a returned execution is still served: {_reported_jobs(settled)}"
    )
    assert _served_refusal_codes(settled, _OTHER_NONCE) == ()
    ledger = cast("dict[str, Any]", engine.state_document())
    assert _nonce_ledger(ledger, _OTHER_NONCE) == (1, 0), (
        f"the duplicate really advanced the accepted chain: {ledger['accepted_nonces']}"
    )
    assert ledger["action_count"] == 1
    assert outcome["value"][0] == 400
    assert cast("dict[str, Any]", outcome["value"][1]["error"])["code"] == "NONCE_REUSE"


def test_no_refusal_code_is_served_for_an_execution_that_has_returned(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4-F4, restored: the served code belongs to the refusal the state reports.

    The reviewer's F4 shape, with a later attempt that has its OWN real refusal
    code.  Fresh session: B wins the real admission, consumes the nonce and is
    really refused with ``SEED_OUT_OF_BOUNDS``; C — validated before that
    consumption and therefore still inside its own route — is then really
    refused too, with the admission's own ``NONCE_REUSE``, a DIFFERENT code.

    RESTORED TO THE PRODUCT'S TRUTH BY THE F2_JOB_HANDLE_TRUTH SECOND NARROWING
    (manager decision 2026-09-28).  The property is unchanged and is in fact
    stronger: the owner's own real refusal IS served, with the owner's own real
    code, for as long as the owner's own execution is inside its route (parked
    at the real route-exit hook) — and C's different code is never borrowed for
    it, because C never held the handle and ownership drops a non-owner before
    any state is even considered.  When the owner's execution returns, the
    refusal goes with it, and the identity is then served NOTHING: C's
    ``NONCE_REUSE`` — a real refusal, of a real attempt — is never published for
    an identity it does not own, in either window.  The name is corrected: no
    code is served for "an identity a later attempt may overturn", because no
    served word outlives the execution that produced it.
    """
    engine = _open_volume_only_engine(volume_only_session, ulid_start=947000)
    session_id = engine.session_id()
    turnstile = _Turnstile()
    _, first_build_direction = _phantom_seed(40, 60, 20)
    good_voxel, good_direction = _phantom_seed(200, 220, 180)
    owner_parked = threading.Event()
    owner_release = threading.Event()
    # Only the owner parks at its route exit: its own real refusal is readable
    # there and nowhere else.
    _park_at_route_exit(
        monkeypatch, [1000000, 1000000, 1000000], owner_parked, owner_release
    )

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    outcomes: dict[str, tuple[int, dict[str, Any]]] = {}
    started: list[threading.Thread] = []
    try:
        _park_validated_seed(monkeypatch, turnstile)
        refused_body = _seed_body(
            session_id=session_id,
            voxel=[1000000, 1000000, 1000000],
            direction=first_build_direction,
            nonce=_NONCE,
            parent=None,
        )
        later_body = _seed_body(
            session_id=session_id,
            voxel=good_voxel,
            direction=good_direction,
            nonce=_NONCE,
            parent=None,
        )

        def _post_as(name: str, body: bytes) -> None:
            outcomes[name] = _post_seed(port, session_id, body)

        owner = threading.Thread(
            target=lambda: _post_as("owner", refused_body), name="f2job-r4f4-owner"
        )
        owner.start()
        started.append(owner)
        try:
            turnstile.await_parked(0)
            later = threading.Thread(
                target=lambda: _post_as("later", later_body), name="f2job-r4f4-later"
            )
            later.start()
            started.append(later)
            turnstile.await_parked(1)
            turnstile.grant(0)
            assert owner_parked.wait(timeout=120), (
                "the awarded seed never parked inside its own route"
            )
            # THE OWNER'S OWN REAL REFUSAL, served with the OWNER'S OWN real
            # code, while the owner's own execution is still inside its route.
            in_flight = _get_state(port, session_id)
            _assert_a_code_belongs_to_its_own_state(in_flight)
            served = _job(in_flight, _NONCE)
            assert served["state"] == _REFUSED
            assert served["refusal_code"] == "SEED_OUT_OF_BOUNDS"
            assert set(served) == {"session_id", "client_nonce", "state", "refusal_code"}
            # The later attempt is really refused too, with its own real code —
            # and that DIFFERENT code may not borrow the handle it does not own,
            # even while the owner is still being served.
            turnstile.grant(1)
            later.join(timeout=300)
            assert not later.is_alive(), "the later same-nonce seed never returned"
            assert outcomes["later"][0] == 400
            later_code = cast("dict[str, Any]", outcomes["later"][1]["error"])["code"]
            assert later_code == "NONCE_REUSE"
            assert later_code not in _served_refusal_codes(in_flight, _NONCE)
            assert _job(_get_state(port, session_id), _NONCE) == served, (
                "a non-owner's real refusal reached the owner's served handle: "
                f"{_jobs(_get_state(port, session_id))}"
            )
            # The owner's execution now really returns, and its refusal goes with
            # it.  C's own real ``NONCE_REUSE`` is still never served: C never
            # held the handle, so it has no word to serve, and once the owner's
            # execution has returned there is no record left for anyone to speak
            # for.
            owner_release.set()
            owner.join(timeout=300)
            assert not owner.is_alive(), "the awarded seed never returned"
            assert outcomes["owner"][0] == 409
            owner_code = cast("dict[str, Any]", outcomes["owner"][1]["error"])["code"]
            assert owner_code == "SEED_OUT_OF_BOUNDS"
            # ... and it is the very code the server answered the refusing
            # client with, which it could only be because the served record IS
            # that execution's own record.
            assert served["refusal_code"] == owner_code
            returned = _get_state(port, session_id)
            _assert_a_code_belongs_to_its_own_state(returned)
            assert _served_refusal_codes(returned, _NONCE) == (), (
                f"a code was served for an execution that has returned: "
                f"{_reported_jobs(returned)}"
            )
            assert _served_states(returned, _NONCE) == ()
            _assert_document_never_contradicts_itself(returned)
        finally:
            turnstile.grant_all()
            owner_release.set()
            for thread in started:
                thread.join(timeout=300)
    finally:
        turnstile.grant_all()
        owner_release.set()
        server.stop()
    # NOTHING WAS ACCEPTED — neither attempt of this identity ever applied — and
    # the handle claimed no outcome at any point after its executions returned.
    document = cast("dict[str, Any]", engine.state_document())
    assert document["accepted_nonces"] == []
    assert document["action_count"] == 0


# ---------------------------------------------------------------------------
# ATT0006 (review round 5) as RESTORED by the F2_JOB_HANDLE_TRUTH SECOND
# NARROWING (manager decision 2026-09-28).
#
# R5-F1 (P1) was: a re-seed's fail-closed ``ViewerOpenError`` refusal does NOT
#     consume the nonce, so the identity is retryable, yet the serve path could
#     still publish it — and the reviewer's schedules for that were an
#     execution APPEARING between the poll's cut and its irreversibility read.
#     The second narrowing removes the need for the whole apparatus: a recorded
#     refusal is removed with the execution that recorded it, so the sequential
#     hole, the interleaved hole and the "final pass" hole are the same hole —
#     a record that outlived its execution — and it is closed at the root.
# R5-F2 (P3) was: an exception raised while the ``QUEUED`` observation was being
#     recorded could leave bookkeeping behind for ever.  With the live registry
#     gone, a crashed execution leaves NOTHING behind: the record it would have
#     created is never created, and the route-exit hook that would have removed
#     it runs anyway and removes nothing.
#
# Every test below drives the REAL route (real refusals, real acceptances, real
# derivations) and asserts only the SERVED document — never ``_seed_jobs``,
# never a record's ``owner`` or ``admitted``.  The gates decide only WHEN a
# request proceeds and WHEN a real compute is held open; no product code is
# replaced to make a test deterministic.  No gate is installed on a read the
# serve path no longer performs, because there is none.
# ---------------------------------------------------------------------------


def _park_the_nonce_check(
    monkeypatch: pytest.MonkeyPatch,
    voxel: list[int],
    parked: threading.Event,
    release: threading.Event,
) -> None:
    """Park the request with ``voxel`` before its nonce check.

    ATT0006's schedule needs an execution of an identity that is INSIDE its route
    but has not finished validating, and has therefore observed nothing: the
    identity must be unreported for it, because it holds no record yet.  The
    wrapper sits on :meth:`ViewerEngine._validate_seed_request`, which
    :meth:`ViewerEngine.apply_seed_bytes` calls BEFORE
    :meth:`ViewerEngine._seed_job_queued`, so parking holds no lock and the real
    rule runs unchanged (and really refuses with its real code) once the gate is
    released.  Only the request carrying ``voxel`` parks, so the request that
    owns the record under test proceeds exactly as the reviewer ordered it.

    The earlier version of this helper parked the request where the live registry
    already contained it, because a validated-but-unregistered execution had to
    be VISIBLE to the serve path.  Nothing has to be visible now: an execution
    that has not observed anything holds no record, and the identity is served
    nothing — which is exactly the property this test pins.
    """

    real_validate = ViewerEngine._validate_seed_request

    def _gated(self, parsed, *, path_session_id):  # noqa: ANN001, ANN003, ANN202
        if list(cast("list[int]", parsed["voxel"])) == list(voxel):
            parked.set()
            assert release.wait(timeout=120), "the nonce-check gate was never released"
        return real_validate(self, parsed, path_session_id=path_session_id)

    monkeypatch.setattr(ViewerEngine, "_validate_seed_request", _gated)


def test_a_retryable_refusal_serves_nothing_while_its_nonce_is_unused(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5-F1, the reviewer's interleaved schedule, restored to the product's truth.

    Step 1 is a re-seed whose fail-closed ``ViewerOpenError`` really fires (the
    governed build identity is put unresolved for the duration of that one
    request, so the route's own provenance step raises its real
    ``SESSION_OPEN_FAILED``; the identity is then put straight back).  That
    refusal does NOT consume the nonce, which the assertions below check from
    the server's own bookkeeping — the local verifiable fact that made it
    retryable.

    Step 2 is the reviewer's schedule, KEPT: a legitimate retry of the very same
    nonce really validates (the nonce was never consumed) and is really inside
    its re-seed derive, held open by a gate on the real compute.  Step 3 is a
    poll taken while that retry is in flight.  A server that had kept A's
    recorded refusal would tell a client here that the seed is refused while it
    is about to be accepted.

    RESTORED TO THE PRODUCT'S TRUTH BY THE F2_JOB_HANDLE_TRUTH SECOND NARROWING
    (manager decision 2026-09-28).  The test-side gate on the serve path's
    consumed-nonce read is DELETED, not re-pointed, because the serve path
    performs no such read any more: with no irreversibility predicate there is
    nothing for a poll to be stale about, and a poll can be asked the question
    directly, with no instrumentation in between at all.  HONESTY NOTE, stated
    here rather than claimed as a falsification: this test can no longer
    falsify the dropped read-order mechanism, because that mechanism does not
    exist.  What it DOES falsify, on the reviewer's own schedule, is the
    retained guarantee: the poll sees the RETRY's own live state and nothing of
    the returned A, and after the retry returns the identity is served nothing
    at all.  The ending the previous amendment required — a served ``REFUSED``
    surviving a successful same-nonce retry — is the claim the review of
    2026-09-27 cited as its F1 evidence, and it is gone with the product.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=948000)
    session_id = engine.session_id()
    good_voxel, good_direction = _phantom_seed(200, 220, 180)
    entered = threading.Event()
    release = threading.Event()

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    polled: dict[str, dict[str, Any]] = {}
    retry_status: dict[str, int] = {}
    retry: threading.Thread | None = None
    try:
        _build_the_initial_centerline(port, session_id)
        body = _seed_body(
            session_id=session_id,
            voxel=good_voxel,
            direction=good_direction,
            nonce=_OTHER_NONCE,
            parent=_NONCE,
        )
        # STEP 1: the REAL non-consuming refusal.
        resolved_identity = engine._build_commit_sha
        assert resolved_identity is not None, "the fixture resolved no build identity"
        engine._build_commit_sha = None
        try:
            status, payload = _post_seed(port, session_id, body)
        finally:
            engine._build_commit_sha = resolved_identity
        assert status == 409
        answered = cast("dict[str, Any]", payload["error"])["code"]
        assert answered == "SESSION_OPEN_FAILED"
        # THE LOCAL VERIFIABLE FACT: the nonce was NOT consumed, so this refusal
        # was retryable.  Nothing else in this test fabricates an outcome.
        after_refusal = cast("dict[str, Any]", engine.state_document())
        assert _OTHER_NONCE not in cast("list[str]", after_refusal["used_nonces"])
        assert _OTHER_NONCE not in cast("list[str]", after_refusal["accepted_nonces"])
        # A's execution has returned, so nothing of it is served.
        assert _served_states(_get_state(port, session_id), _OTHER_NONCE) == ()
        # The retry's real derive is held open, so the retry is really inside
        # the governed build while the poll decides.
        _gate_one_reseed(monkeypatch, good_voxel, entered, release)

        def _retry() -> None:
            retry_status["value"] = _post_seed(port, session_id, body)[0]

        # STEP 2: the retry really validates the STILL-UNUSED nonce and really
        # enters its re-seed derive.
        retry = threading.Thread(target=_retry, name="f2job-r5f1-retry")
        retry.start()
        try:
            assert entered.wait(timeout=120), (
                "the same-nonce retry never entered its re-seed derive, so "
                "the reviewer's schedule was not reproduced"
            )
            assert retry.is_alive(), "the same-nonce retry already returned"
            # STEP 3: the poll, with NO instrumentation between it and the
            # product.  It must report the running retry's own live state — and
            # never the refusal of the execution that has already returned.
            poller = threading.Thread(
                target=lambda: polled.update(value=_get_state(port, session_id)),
                name="f2job-r5f1-poll",
            )
            poller.start()
            poller.join(timeout=300)
            assert not poller.is_alive(), "the poll never returned"
            document = cast("dict[str, Any]", polled["value"])
            assert _served_states(document, _OTHER_NONCE) == (_RUNNING,), (
                "a poll did not report the running retry's own live state: "
                f"{_reported_jobs(document)}"
            )
            assert _REFUSED not in _served_states(document, _OTHER_NONCE), (
                "a returned execution's refusal was served while the same "
                f"identity was really retrying: {_reported_jobs(document)}"
            )
            assert _served_refusal_codes(document, _OTHER_NONCE) == (None,), (
                "a returned execution's refusal code was served: "
                f"{_reported_jobs(document)}"
            )
            _assert_document_never_contradicts_itself(document)
            _assert_a_code_belongs_to_its_own_state(document)
            # The same read again, from the serving thread, while the retry is
            # still in flight: still the retry's own state, never A's refusal.
            while_running = _get_state(port, session_id)
            assert _served_states(while_running, _OTHER_NONCE) == (_RUNNING,)
            assert retry is not None
            release.set()
            retry.join(timeout=300)
            assert not retry.is_alive(), "the retried re-seed never returned"
            assert retry_status["value"] == 200
        finally:
            release.set()
            entered.set()
            if retry is not None:
                retry.join(timeout=300)
        # The retry really was accepted, and the handle publishes NO word for it:
        # not a completion, and not the earlier execution's refusal either.  The
        # accepted chain remains the authority on acceptance.
        landed = _get_state(port, session_id)
        assert _served_states(landed, _OTHER_NONCE) == (), (
            f"a returned execution is still served: {_reported_jobs(landed)}"
        )
        assert _served_refusal_codes(landed, _OTHER_NONCE) == ()
        # F2 (2026-09-28 review, P3): the ``_COMPLETED not in _reported_jobs``
        # line that used to sit here could never fail (a string compared against
        # served DICTIONARIES), so it is REMOVED: the document invariant on the
        # next line is the effective form of the same claim on the same
        # document, and the absence assertion above already proves this document
        # is empty.
        _assert_document_never_contradicts_itself(landed)
        document = cast("dict[str, Any]", engine.state_document())
        assert document["used_nonces"] == [_NONCE, _OTHER_NONCE]
        assert document["accepted_nonces"] == [_NONCE, _OTHER_NONCE]
        assert document["action_count"] == 2
    finally:
        release.set()
        if retry is not None:
            retry.join(timeout=300)
        server.stop()



def test_a_refusal_serves_nothing_while_its_nonce_is_unused_and_never_after_a_retry(
    volume_only_session,
) -> None:
    """R5-F1, the SEQUENTIAL case: no execution is running at all.

    The same non-consuming refusal, with nothing in flight and no race: the
    request has returned, and the server still accepts a retry of that very
    nonce, so the recorded refusal was one execution's observation.  This is the
    hole with no interleaving in it, which is why it is asserted separately: a
    fix that only re-checked liveness at serve time would still have failed
    here.  Both ends are asserted on real facts: the attempt that failed is
    answered with the exact code the server raised, and the legitimate retry
    really builds and is really accepted (HTTP 200, the nonce in both ledgers,
    ``action_count`` 2).

    RESTORED TO THE PRODUCT'S TRUTH BY THE F2_JOB_HANDLE_TRUTH SECOND NARROWING
    (manager decision 2026-09-28).  Two names had to change and one assertion
    had to change.  The original name promised that the handle "publishes
    nothing until its nonce is really spent", i.e. that a state appears after a
    later execution consumes the nonce; no such state exists, because the record
    is removed with the execution that made it.  The name now says what is
    actually asserted.  The ending changes from ``(_REFUSED,)`` with the old
    code to NOTHING: the earlier execution returned long ago, so its refusal is
    gone, and the retry's own execution has returned too, so its success is not
    restated as a state — while ``accepted_nonces`` and ``action_count`` record
    the real acceptance in full.  This is the F1 evidence the review of
    2026-09-27 named, and the expectation is now the product's.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=949000)
    session_id = engine.session_id()
    good_voxel, good_direction = _phantom_seed(200, 220, 180)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    try:
        _build_the_initial_centerline(port, session_id)
        body = _seed_body(
            session_id=session_id,
            voxel=good_voxel,
            direction=good_direction,
            nonce=_OTHER_NONCE,
            parent=_NONCE,
        )
        resolved_identity = engine._build_commit_sha
        assert resolved_identity is not None, "the fixture resolved no build identity"
        engine._build_commit_sha = None
        try:
            status, payload = _post_seed(port, session_id, body)
        finally:
            engine._build_commit_sha = resolved_identity
        assert status == 409
        answered = cast("dict[str, Any]", payload["error"])["code"]
        assert answered == "SESSION_OPEN_FAILED"
        # Nothing of this identity is served, with no live execution anywhere:
        # that execution has returned, so its record is gone.
        idle = _get_state(port, session_id)
        assert _served_states(idle, _OTHER_NONCE) == (), (
            f"a returned execution is still served: {_reported_jobs(idle)}"
        )
        assert _served_refusal_codes(idle, _OTHER_NONCE) == ()
        _assert_document_never_contradicts_itself(idle)
        # The locally verifiable facts that made the refusal retryable, and no
        # success was fabricated for it.
        after_refusal = cast("dict[str, Any]", engine.state_document())
        assert _OTHER_NONCE not in cast("list[str]", after_refusal["used_nonces"])
        assert after_refusal["accepted_nonces"] == [_NONCE]
        assert after_refusal["action_count"] == 1
        # The legitimate retry of the SAME nonce really builds and is really
        # accepted — and the handle publishes no word for it: not a completion,
        # and not the earlier execution's refusal either.  Nothing outlives its
        # own execution, in either direction.
        status, _payload = _post_seed(port, session_id, body)
        assert status == 200
        landed = _get_state(port, session_id)
        assert _served_states(landed, _OTHER_NONCE) == (), (
            f"a returned execution is still served: {_reported_jobs(landed)}"
        )
        assert _served_refusal_codes(landed, _OTHER_NONCE) == ()
        # F2 (2026-09-28 review, P3): REMOVED, and not replaced.  The
        # ``_COMPLETED not in _reported_jobs`` line that used to sit here
        # compared a string against served DICTIONARIES and could never fail;
        # the document invariant immediately below is the effective form of the
        # same claim on the same document, and the absence assertion above
        # already proves this document is empty.
        _assert_document_never_contradicts_itself(landed)
        document = cast("dict[str, Any]", engine.state_document())
        assert document["used_nonces"] == [_NONCE, _OTHER_NONCE]
        assert document["accepted_nonces"] == [_NONCE, _OTHER_NONCE]
        assert document["action_count"] == 2
    finally:
        server.stop()

def test_a_refusal_is_served_while_its_own_execution_is_in_flight_and_not_after(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-over-correction pin: a real refusal IS served while it is in flight.

    RESTORED AND RENAMED BY THE F2_JOB_HANDLE_TRUTH SECOND NARROWING (manager
    decision 2026-09-28) from
    ``test_an_irreversible_refusal_that_consumed_its_nonce_is_still_published``.
    The name promised a distinction that no longer exists in the product — an
    "irreversible" refusal, i.e. one whose nonce was consumed, is now served on
    exactly the same footing as any other: for as long as its own execution is
    inside its route, and not one instant longer.  The PINNED PROPERTY is
    unchanged and is in fact the important half of the whole suite: abstention
    is for a case the server cannot support, never a way of never refusing.

    The governed re-seed really raises its typed ``SEED_OUT_OF_BOUNDS`` (the seed
    lies far outside every session volume — no clamp, no default, no retry) and
    really consumes the nonce, which the assertions read from the server's own
    bookkeeping.  The refusal is then served, with the exact code it answered the
    client with, while that execution is parked at the real route-exit hook.

    The second half is the new and stronger half: a same-nonce attempt that is
    inside its own route but has not yet observed anything contributes NO word,
    so it never displaces or duplicates the owner's real refusal — and the
    instant the owner's execution returns, the identity is served nothing at
    all.  Nothing is ever accepted here, before or after.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=950000)
    session_id = engine.session_id()
    good_voxel, good_direction = _phantom_seed(200, 220, 180)
    parked = threading.Event()
    release = threading.Event()
    owner_parked = threading.Event()
    owner_release = threading.Event()
    # The owner parks at its route exit; the competing attempt parks at the
    # nonce check.  Both hold nothing, so the real rules run unchanged.
    _park_the_nonce_check(monkeypatch, good_voxel, parked, release)
    _park_at_route_exit(
        monkeypatch, [1000000, 1000000, 1000000], owner_parked, owner_release
    )

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    competing: dict[str, Any] = {}
    owner_outcome: dict[str, tuple[int, dict[str, Any]]] = {}
    worker: threading.Thread | None = None
    owner: threading.Thread | None = None
    try:
        _build_the_initial_centerline(port, session_id)
        refusing_body = _seed_body(
            session_id=session_id,
            voxel=[1000000, 1000000, 1000000],
            direction=good_direction,
            nonce=_OTHER_NONCE,
            parent=_NONCE,
        )

        def _refuse() -> None:
            owner_outcome["value"] = _post_seed(port, session_id, refusing_body)

        owner = threading.Thread(target=_refuse, name="f2job-r5f1-owner")
        owner.start()
        try:
            assert owner_parked.wait(timeout=120), (
                "the refused re-seed never parked inside its own route"
            )
            in_flight = _get_state(port, session_id)
            refused = _job(in_flight, _OTHER_NONCE)
            assert refused["state"] == _REFUSED
            assert set(refused) == {"session_id", "client_nonce", "state", "refusal_code"}
            _assert_a_code_belongs_to_its_own_state(in_flight)
            # A same-nonce attempt is now inside its own route, parked at the
            # nonce check.  It has observed nothing, so it holds no record and
            # contributes no word: the owner's own real refusal is still the
            # served state, unchanged, and never duplicated or downgraded.
            worker = threading.Thread(
                target=lambda: competing.update(value=_post_seed(port, session_id, _seed_body(
                    session_id=session_id,
                    voxel=good_voxel,
                    direction=good_direction,
                    nonce=_OTHER_NONCE,
                    parent=_NONCE,
                ))),
                name="f2job-r5f1-competing",
            )
            worker.start()
            try:
                assert parked.wait(timeout=120), (
                    "the same-nonce attempt never parked at the nonce check, so it "
                    "was never inside its own route"
                )
                while_competing = _get_state(port, session_id)
                assert _job(while_competing, _OTHER_NONCE) == refused, (
                    "an attempt that has observed nothing changed the served "
                    f"refusal: {_reported_jobs(while_competing)}"
                )
                # ... and it is released: the real grammar gate refuses the
                # consumed nonce, so the competing execution really returns and
                # really is refused with its own real code.
                release.set()
                worker.join(timeout=300)
                assert (
                    not worker.is_alive()
                ), "the competing same-nonce seed never returned"
                assert competing["value"][0] == 400
                assert (
                    cast("dict[str, Any]", competing["value"][1]["error"])["code"]
                    == "NONCE_REUSE"
                )
                assert _job(_get_state(port, session_id), _OTHER_NONCE) == refused, (
                    "a non-owner's real refusal reached the owner's served "
                    f"handle: {_jobs(_get_state(port, session_id))}"
                )
            finally:
                release.set()
                if worker is not None:
                    worker.join(timeout=300)
            # THE OWNER'S OWN EXECUTION RETURNS: the refusal goes with it, and
            # the identity is then served nothing at all — not the refusal, not
            # a live word, not a completion.
            owner_release.set()
            owner.join(timeout=300)
            assert not owner.is_alive(), "the refused re-seed never returned"
            status, payload = owner_outcome["value"]
            assert status == 409
            answered = cast("dict[str, Any]", payload["error"])["code"]
            assert answered == "SEED_OUT_OF_BOUNDS"
            assert refused["refusal_code"] == answered
            # The fact a reader would have used to call this outcome
            # irreversible, read from the server's own bookkeeping.
            consumed = cast("dict[str, Any]", engine.state_document())
            assert _OTHER_NONCE in cast("list[str]", consumed["used_nonces"])
            returned = _get_state(port, session_id)
            assert _served_states(returned, _OTHER_NONCE) == (), (
                f"a returned execution is still served: {_reported_jobs(returned)}"
            )
            assert _served_refusal_codes(returned, _OTHER_NONCE) == ()
            _assert_document_never_contradicts_itself(returned)
            # No fabricated success: only the initial build was ever accepted.
            document = cast("dict[str, Any]", engine.state_document())
            assert document["accepted_nonces"] == [_NONCE]
            assert document["action_count"] == 1
        finally:
            owner_release.set()
            release.set()
            owner_parked.set()
            if owner is not None:
                owner.join(timeout=300)
    finally:
        owner_release.set()
        release.set()
        if worker is not None:
            worker.join(timeout=300)
        server.stop()



def test_an_exception_before_the_queue_observation_leaves_nothing_behind(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5-F2: a crash BEFORE the record exists leaves NOTHING behind at all.

    F3 (2026-09-28 review, P3) — THE NAME AND THE FAULT PLACEMENT ARE CORRECTED
    HERE.  This test used to be called
    ``test_an_exception_while_the_queue_observation_runs_leaves_nothing_behind``
    and to speak of "post-insertion fault coverage".  It is not: the injector
    below raises BEFORE it calls the real observer, so no record is ever
    inserted and there is nothing to clean up.  The name now says where the
    fault is.  The missing placement — an exception escaping AFTER the record has
    really been written — is covered by its own regression,
    ``test_an_exception_after_the_record_is_inserted_is_removed_at_route_exit``,
    added with the F1 fix; the two halves of the claim are now both real.

    The reviewer's fault injection: an unexpected exception escapes
    ``_seed_job_observe`` before the ``QUEUED`` observation of a validated
    request has recorded anything.  The earlier rounds defended against that
    fault with a LIVE REGISTRY that had to be de-registered on every path,
    because a leaked entry withheld the identity for ever.  Under the second
    narrowing there is no registry to leak: the record the crashed execution
    would have created is never created, and the route-exit hook that would have
    removed it runs anyway and removes nothing.  The name is corrected
    accordingly — "leaves no live entry behind" promised a registry that does
    not exist.

    The assertion is made entirely in served terms, and is now STRONGER: after
    the fault, a legitimate same-nonce attempt that really fails must have its
    OWN real refusal served, with its exact real code, while its own execution
    is in flight — and the identity must be served nothing at all the moment
    that execution returns.  A server that leaked anything from the crashed
    execution fails on the first half; one that retained the second execution's
    refusal fails on the second.

    The faulted request is driven through :meth:`ViewerEngine.apply_seed_bytes`
    directly, exactly as the reviewer drove it, because an unexpected exception
    has no typed HTTP mapping: inventing a response for it would be a
    fabricated answer.  Everything else in this test is the real route over the
    real server, and the fault is injected once, on the first observation only.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=951000)
    session_id = engine.session_id()
    good_voxel, good_direction = _phantom_seed(200, 220, 180)
    fired: list[str] = []
    parked = threading.Event()
    release = threading.Event()

    real_observe = ViewerEngine._seed_job_observe

    def _failing_observe(self, request, state, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if state == _QUEUED and not fired:
            fired.append(state)
            raise RuntimeError("injected fault while recording the queue observation")
        return real_observe(self, request, state, **kwargs)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    outcome: dict[str, tuple[int, dict[str, Any]]] = {}
    worker: threading.Thread | None = None
    try:
        _build_the_initial_centerline(port, session_id)
        # The fault is armed only now, so it governs the next queue observation
        # and nothing that came before it.
        monkeypatch.setattr(ViewerEngine, "_seed_job_observe", _failing_observe)
        faulted_body = _seed_body(
            session_id=session_id,
            voxel=good_voxel,
            direction=good_direction,
            nonce=_OTHER_NONCE,
            parent=_NONCE,
        )
        with pytest.raises(RuntimeError, match="injected fault"):
            engine.apply_seed_bytes(faulted_body, path_session_id=session_id)
        assert fired == [_QUEUED], "the fault never reached the queue observation"
        # The crashed execution's route EXITED through the hook even though its
        # observation raised, so there is nothing of it anywhere.
        assert _served_states(engine.planes_state(), _OTHER_NONCE) == (), (
            "a crashed execution left a served record behind"
        )
        monkeypatch.undo()
        # The crash has already happened, so the route-exit hook is armed only
        # now: it belongs to the NEXT execution and is undone with the rest of
        # this test, never with the injected fault above.
        _park_at_route_exit(monkeypatch, [1000000, 1000000, 1000000], parked, release)
        # The crashed execution consumed nothing and accepted nothing: it never
        # reached a rule, so its identity is untouched and legitimately open.
        after_fault = cast("dict[str, Any]", engine.state_document())
        assert _OTHER_NONCE not in cast("list[str]", after_fault["used_nonces"])
        assert after_fault["accepted_nonces"] == [_NONCE]
        # THE SERVED CONSEQUENCE: a legitimate same-nonce attempt now really
        # fails (far outside every session volume), consuming its nonce, and its
        # own real refusal must be served while its own execution is in flight.
        worker = threading.Thread(
            target=lambda: outcome.update(value=_post_seed(
                port,
                session_id,
                _seed_body(
                    session_id=session_id,
                    voxel=[1000000, 1000000, 1000000],
                    direction=good_direction,
                    nonce=_OTHER_NONCE,
                    parent=_NONCE,
                ),
            )),
            name="f2job-after-fault",
        )
        worker.start()
        try:
            assert parked.wait(timeout=120), (
                "the post-fault refusal never parked inside its own route"
            )
            served = _get_state(port, session_id)
            published = _job(served, _OTHER_NONCE)
            assert published["state"] == _REFUSED
            assert set(published) == {
                "session_id",
                "client_nonce",
                "state",
                "refusal_code",
            }
            _assert_document_never_contradicts_itself(served)
            _assert_a_code_belongs_to_its_own_state(served)
            # It is served byte-identically on a later read inside the same
            # route: the crashed execution contributed nothing that could
            # disturb it.
            assert _job(_get_state(port, session_id), _OTHER_NONCE) == published
            release.set()
            worker.join(timeout=300)
            assert not worker.is_alive(), "the post-fault refusal never returned"
            status, payload = outcome["value"]
            assert status == 409
            answered = cast("dict[str, Any]", payload["error"])["code"]
            assert answered == "SEED_OUT_OF_BOUNDS"
            assert published["refusal_code"] == answered
        finally:
            release.set()
            parked.set()
            if worker is not None:
                worker.join(timeout=300)
        # And the instant that execution returns, the identity is served
        # nothing: nothing outlives its execution, crashed or not.
        returned = _get_state(port, session_id)
        assert _served_states(returned, _OTHER_NONCE) == (), (
            f"a returned execution is still served: {_reported_jobs(returned)}"
        )
        document = cast("dict[str, Any]", engine.state_document())
        assert document["accepted_nonces"] == [_NONCE]
        assert document["action_count"] == 1
    finally:
        release.set()
        monkeypatch.undo()
        server.stop()


def test_an_exception_after_the_record_is_inserted_is_removed_at_route_exit(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F3 (P3): the POST-INSERTION fault, which the sibling test never covered.

    THE MISSING PLACEMENT, NOW COVERED.  The review's P3 on
    ``test_an_exception_while_the_queue_observation_runs_leaves_nothing_behind``
    was exactly this: its injector raised BEFORE it called the real observer, so
    no record was ever written and the test could not say anything about a crash
    that happens once a record EXISTS.  That sibling has been renamed and
    narrowed to its real claim; this is the other half, and it is the half that
    matters most, because a record that really exists is the one that has to be
    cleaned up.

    THE SCHEDULE, in full: the injector calls the REAL
    :meth:`ViewerEngine._seed_job_observe` first — so the ``QUEUED`` record is
    genuinely inserted — and only then raises.  The escaped exception therefore
    leaves a real, owned record behind at a point where the execution is already
    crashed.  The route's ``finally`` hook
    (:meth:`ViewerEngine._seed_job_release`) must still remove it, and the served
    document must show nothing for that identity afterwards.

    Driven through :meth:`ViewerEngine.apply_seed_bytes` directly, exactly as the
    sibling does, because an unexpected exception has no typed HTTP mapping:
    inventing a response for it would be a fabricated answer.  The record is
    PROVEN to have existed first — the test reads the served document from
    inside the fault, between the insertion and the raise — so a server that
    never inserted anything cannot pass this by accident.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=951500)
    session_id = engine.session_id()
    real_observe = ViewerEngine._seed_job_observe
    seen_inside_fault: dict[str, Any] = {}

    def _failing_after_insertion(self, request, state, **kwargs):  # noqa: ANN001, ANN003, ANN202
        result = real_observe(self, request, state, **kwargs)
        if state == _QUEUED and _JOBS_KEY not in seen_inside_fault:
            # The record is REALLY there, read from the served document, before
            # the fault is raised.  This is what the sibling test never showed.
            seen_inside_fault.update(self.planes_state())
            raise RuntimeError("injected fault after the record was inserted")
        return result

    good_voxel, good_direction = _phantom_seed(200, 220, 180)
    body = _seed_body(
        session_id=session_id,
        voxel=good_voxel,
        direction=good_direction,
        nonce=_OTHER_NONCE,
        parent=None,
    )
    monkeypatch.setattr(ViewerEngine, "_seed_job_observe", _failing_after_insertion)
    with pytest.raises(RuntimeError, match="after the record was inserted"):
        engine.apply_seed_bytes(body, path_session_id=session_id)
    # PROVEN: the record existed, really, and it was the crashed execution's own.
    inserted = _job(cast("dict[str, Any]", seen_inside_fault), _OTHER_NONCE)
    assert inserted["state"] == _QUEUED
    assert _reported_jobs(cast("dict[str, Any]", seen_inside_fault)) == [inserted]
    # AND IT IS GONE: the route-exit hook ran on the crashed path and removed the
    # record its owner had really created, so nothing of it is served at all.
    assert _reported_jobs(engine.planes_state()) == []
    assert _served_states(engine.planes_state(), _OTHER_NONCE) == ()
    _assert_document_never_contradicts_itself(engine.planes_state())
    # The crashed execution consumed nothing and accepted nothing: it never
    # reached a rule, so its identity is untouched and legitimately open, and a
    # later attempt of the very same nonce is still really admissible.
    document = cast("dict[str, Any]", engine.state_document())
    assert document["accepted_nonces"] == []
    assert document["action_count"] == 0
    assert _OTHER_NONCE not in cast("list[str]", document["used_nonces"])
    assert _JOBS_KEY not in document


def test_a_poll_concurrent_with_a_late_execution_never_serves_a_returned_one(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stale-decision hazard, on the schedule it existed for — now structural.

    The hazard this test was built for: a poll decides to serve a terminal
    refusal from a cut that is RELEASED before the decision is served, so an
    execution of the same identity can REGISTER in between and the poll can
    serve a decision it had already made from an obsolete snapshot.  Under the
    F2_JOB_HANDLE_TRUTH SECOND NARROWING (manager decision 2026-09-28) the
    decision itself is gone.  The serve path takes ONE ``_seed_job_lock`` hold,
    reads nothing else, and serves exactly the records the running executions
    hold; a record's presence is a fact about an execution that has not returned
    yet, not a snapshot of one.  So there is no cut, no released window, no
    second pass and no read order that could go stale.

    HONESTY NOTE, stated here rather than claimed as a falsification: this test
    can no longer falsify the dropped re-read mechanism, because that mechanism
    does not exist, and the test-side gate on the serve path's consumed-nonce
    read is DELETED with it (it would simply never fire).  It also cannot be
    FALSIFIED INTO a false pass by the hazard it used to stage: a poll running
    concurrently with a late-registering same-nonce execution is asserted here
    directly, with no instrumentation in between at all, and the poll must serve
    that late execution's own live state — and nothing of the execution that has
    already returned.

    What IS pinned: A really fails, really consumes its nonce, and is then held
    open at its real route-exit hook, so its own real refusal is readable.  A
    same-nonce attempt then arrives, is really validated and parks at the nonce
    check (inside its route, having observed nothing).  A poll taken in exactly
    that state must serve A's own real refusal — because A is still running —
    and must not serve a borrowed code or a completion.  When A's execution
    returns, the identity is served nothing at all, whatever B does next.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=952000)
    session_id = engine.session_id()
    good_voxel, good_direction = _phantom_seed(200, 220, 180)
    parked = threading.Event()
    release = threading.Event()
    owner_parked = threading.Event()
    owner_release = threading.Event()
    _park_the_nonce_check(monkeypatch, good_voxel, parked, release)
    _park_at_route_exit(
        monkeypatch, [1000000, 1000000, 1000000], owner_parked, owner_release
    )

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    polled: dict[str, dict[str, Any]] = {}
    competing: dict[str, Any] = {}
    owner_outcome: dict[str, tuple[int, dict[str, Any]]] = {}
    worker: threading.Thread | None = None
    owner: threading.Thread | None = None
    try:
        _build_the_initial_centerline(port, session_id)
        # A really fails and really consumes its nonce, and is held open inside
        # its own route.
        refusing_body = _seed_body(
            session_id=session_id,
            voxel=[1000000, 1000000, 1000000],
            direction=good_direction,
            nonce=_OTHER_NONCE,
            parent=_NONCE,
        )
        owner = threading.Thread(
            target=lambda: owner_outcome.update(
                value=_post_seed(port, session_id, refusing_body)
            ),
            name="f2job-stale-owner",
        )
        owner.start()
        try:
            assert owner_parked.wait(timeout=120), (
                "the refused re-seed never parked inside its own route"
            )
            published = _job(_get_state(port, session_id), _OTHER_NONCE)
            assert published["state"] == _REFUSED
            assert published["refusal_code"] == "SEED_OUT_OF_BOUNDS"
            # An execution of the SAME identity arrives and parks at the nonce
            # check: inside its route, having observed nothing.
            worker = threading.Thread(
                target=lambda: competing.update(value=_post_seed(port, session_id, _seed_body(
                    session_id=session_id,
                    voxel=good_voxel,
                    direction=good_direction,
                    nonce=_OTHER_NONCE,
                    parent=_NONCE,
                ))),
                name="f2job-stale-execution",
            )
            worker.start()
            assert parked.wait(timeout=120), (
                "the same-nonce execution never parked at the nonce check, so it "
                "was not inside its route when the poll finished"
            )
            # THE POLL, concurrent with the late execution, with no
            # instrumentation between it and the product.
            poller = threading.Thread(
                target=lambda: polled.update(value=_get_state(port, session_id)),
                name="f2job-stale-poll",
            )
            poller.start()
            poller.join(timeout=300)
            assert not poller.is_alive(), "the poll never returned"
            document = cast("dict[str, Any]", polled["value"])
            # A is still running, so A's own real refusal is served, verbatim:
            # no borrowed code, no completion, no stale decision.
            assert _job(document, _OTHER_NONCE) == published, (
                "a poll did not serve the running execution's own refusal: "
                f"{_reported_jobs(document)}"
            )
            # F2 (2026-09-28 review, P3): the ``_COMPLETED not in
            # _reported_jobs`` line that used to sit here could never fail (a
            # string against served DICTIONARIES).  It is REMOVED, not replaced,
            # because this document is NOT empty — it really does serve A's live
            # refusal — and the two helpers on the next lines already assert the
            # effective form of the claim on this very document: the invariant
            # reads every served job's ``state`` and fails on ``COMPLETED``.
            _assert_a_code_belongs_to_its_own_state(document)
            _assert_document_never_contradicts_itself(document)
            # A's execution really returns now.
            owner_release.set()
            owner.join(timeout=300)
            assert not owner.is_alive(), "the refused re-seed never returned"
            assert owner_outcome["value"][0] == 409
            # ... and the identity is served NOTHING, so the late execution — and
            # anything after it — has no returned word to contradict.
            after_owner = _get_state(port, session_id)
            assert _served_states(after_owner, _OTHER_NONCE) == (), (
                f"a returned execution is still served: {_reported_jobs(after_owner)}"
            )
            # The late execution really returns too, really refused for its own
            # real reason, and contributes nothing to the served document.
            assert worker is not None
            release.set()
            worker.join(timeout=300)
            assert not worker.is_alive(), "the same-nonce execution never returned"
            assert competing["value"][0] == 400
            assert (
                cast("dict[str, Any]", competing["value"][1]["error"])["code"] == "NONCE_REUSE"
            )
            resettled = _get_state(port, session_id)
            assert _served_states(resettled, _OTHER_NONCE) == (), (
                f"a returned execution is still served: {_reported_jobs(resettled)}"
            )
            _assert_document_never_contradicts_itself(resettled)
            document_after = cast("dict[str, Any]", engine.state_document())
            assert document_after["accepted_nonces"] == [_NONCE]
            assert document_after["action_count"] == 1
        finally:
            owner_release.set()
            release.set()
            if owner is not None:
                owner.join(timeout=300)
    finally:
        owner_release.set()
        release.set()
        if worker is not None:
            worker.join(timeout=300)
        server.stop()


# ---------------------------------------------------------------------------
# F2_SEED_SPECIFIC_COMPLETION (review round 6, R6-F2): a seed COMPLETED could
# only ever be licensed by the SEED ROUTE'S OWN acceptance evidence, never by
# membership of the session-wide ``accepted_nonces`` ledger — that ledger is an
# ALL-ACTION one and ordinary grammar actions append to it, so ledger
# membership was never evidence that a SEED was accepted.
#
# THE F2_JOB_HANDLE_TRUTH NARROWING (2026-09-27) then WITHDREW the completion
# claim itself, and with it the seed-scoped evidence that licensed it, so both
# regressions below are retained with their original reproducers and their
# served-byte assertions intact — and both now also witness the stronger,
# narrowed property that NO completion is published for any identity, neither
# one fabricated by a grammar action nor one earned by a real seed acceptance.
#
# The F2_JOB_HANDLE_TRUTH SECOND NARROWING (manager decision 2026-09-28) goes
# further for the same two reproducers: because nothing outlives its execution,
# a returned seed refusal is not merely immune to being overturned by a grammar
# action, it is GONE.  Neither test can therefore stage the original "published
# refusal overturned" moment at all, and both now assert what the product really
# guarantees — that an ordinary action can fabricate no state for a seed, and
# that a seed identity is unreported as soon as its own execution has returned,
# whatever the all-action ledger says afterwards.
#
# Both regressions are the reviewer's own reproducers, in its own order,
# over the real grammar validator and the real engine action execution.  Neither
# needs any seed-live snapshot race: the competing action is never represented
# in the seed live registry at all, which is exactly why the defect was
# reachable.
# ---------------------------------------------------------------------------


def test_a_grammar_action_can_never_fabricate_a_seed_completion(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R6-F2 reproducer A, the sequential false completion.

    Step 1: a re-seed really raises the non-consuming ``ViewerOpenError``
    (``SESSION_OPEN_FAILED``), so the identity's nonce stays UNUSED and the
    server itself still accepts a retry of it.  Step 2: polling correctly
    abstains — nothing is served for that identity.  Step 3: an ordinary
    ``OPEN_QC_PANEL`` carrying the SAME ``client_nonce`` really validates and
    really executes, which appends that nonce to the session-wide
    ``accepted_nonces`` — the all-action ledger.  Step 4: the seed record is
    polled again.

    Before the fix the seed handle reported ``COMPLETED`` here: the reviewer's
    observed output was ``seed publication: [] -> COMPLETED``, ``qc_panel_open:
    True``, ``successful seed execution: none``.  No seed was ever accepted, so
    a completion is a fabricated claim.  F2_SEED_SPECIFIC_COMPLETION then
    licensed a completion only by the seed route's own evidence, which no
    grammar action can write, so the handle kept abstaining — and the panel
    really did open, so the action's own effect is untouched.

    AMENDED IN THE F2_JOB_HANDLE_TRUTH NARROWING (2026-09-27) — the CONTROL at
    the end is the only part that changes, and it changes from "a fresh seed
    really completes, so the word is reachable" to the stronger property of the
    narrowed guarantee: even a genuinely accepted seed publishes NO completion,
    so the withheld word is not "merely unreachable" — it is never emitted, and
    the control asserts that on served bytes while still asserting the real
    acceptance (HTTP 200 and the accepted chain) that the handle declines to
    restate.  Every step above the control is unchanged.

    RESTORED IN THE F2_JOB_HANDLE_TRUTH SECOND NARROWING (manager decision
    2026-09-28) at steps 2, 4 and after the replay.  Steps 2 and the ``if``-guarded
    block of step 4 were conditional or skipped once the word stopped being
    reachable, which the 2026-09-27 review rightly flagged as a vacuity risk; the
    ``if`` is now GONE and replaced by an UNCONDITIONAL absence assertion, which is
    stronger than the branch it replaced, because the branch it replaced could be
    skipped entirely.  Nothing of step 1's execution is served at any point,
    because that execution has returned: the earlier amendment's
    "irreversibly so" reading — where the ordinary action's consumption of the
    nonce turned the returned seed's refusal into a serveable terminal state — is
    exactly the identity-level claim this narrowing removes.  The real refusal
    code is still asserted, on the wire, where the seed route really raised it.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=953000)
    session_id = engine.session_id()
    good_voxel, good_direction = _phantom_seed(200, 220, 180)

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    try:
        _build_the_initial_centerline(port, session_id)
        body = _seed_body(
            session_id=session_id,
            voxel=good_voxel,
            direction=good_direction,
            nonce=_OTHER_NONCE,
            parent=_NONCE,
        )
        # STEP 1: the REAL non-consuming refusal (see the R5 schedule above for
        # why the unresolved build identity produces the route's real code).
        resolved_identity = engine._build_commit_sha
        assert resolved_identity is not None, "the fixture resolved no build identity"
        engine._build_commit_sha = None
        try:
            status, payload = _post_seed(port, session_id, body)
        finally:
            engine._build_commit_sha = resolved_identity
        assert status == 409
        answered = cast("dict[str, Any]", payload["error"])["code"]
        assert answered == "SESSION_OPEN_FAILED"
        # STEP 2: that execution has returned, so nothing of it is served.
        before = _get_state(port, session_id)
        assert _served_states(before, _OTHER_NONCE) == (), (
            f"a returned execution is still served: {_reported_jobs(before)}"
        )
        assert _served_refusal_codes(before, _OTHER_NONCE) == ()
        # STEP 3: the ordinary action really validates and really executes.  It
        # is validated while the nonce is UNUSED, which is the only window it has.
        action_status, action_payload = _post_action(
            port, session_id, nonce=_OTHER_NONCE, parent=_NONCE
        )
        assert action_status == 200, f"the ordinary action was not accepted: {action_payload}"
        after_action = cast("dict[str, Any]", engine.state_document())
        assert after_action["qc_panel_open"] is True
        # The all-action ledger really did gain the nonce ...
        assert _nonce_ledger(after_action, _OTHER_NONCE) == (1, 1)
        # STEP 4: ... and the seed handle must NOT read COMPLETED for it.  No
        # seed was ever accepted, so there is no seed completion to report — and
        # the seed's own execution has returned long ago, so there is no refusal
        # to re-publish either.  Unconditionally: nothing is served.
        #
        # F2 (2026-09-28 review, P3): the assertion that used to sit here read
        # ``_COMPLETED not in _reported_jobs(after)``, which compares a STRING
        # against the served job DICTIONARIES and therefore cannot fail, so it
        # is REMOVED rather than presented as completion coverage.  The effective
        # check on this same document is
        # ``_assert_document_never_contradicts_itself(after)`` further down,
        # which reads every served job's ``state``; the absence assertion here
        # independently guarantees this document is empty.
        after = _get_state(port, session_id)
        assert _served_states(after, _OTHER_NONCE) == (), (
            "an ordinary action's ledger append re-published a returned seed "
            f"execution: {_reported_jobs(after)}"
        )
        assert _served_refusal_codes(after, _OTHER_NONCE) == ()
        _assert_a_code_belongs_to_its_own_state(after)
        _assert_document_never_contradicts_itself(after)
        # THE POINT OF THE SCHEDULE, unchanged: the ordinary action consumed the
        # identity's nonce, so a seed retry of that very nonce can now never be
        # accepted either.  The seed never said COMPLETED at any point, which is
        # the fabrication the review caught, and it says nothing now either.
        replay_status, replay_payload = _post_seed(port, session_id, body)
        assert replay_status == 400
        assert (
            cast("dict[str, Any]", replay_payload["error"])["code"] == "NONCE_REUSE"
        )
        final = _get_state(port, session_id)
        assert _served_states(final, _OTHER_NONCE) == (), (
            f"a returned execution is still served: {_reported_jobs(final)}"
        )
        assert _served_refusal_codes(final, _OTHER_NONCE) == ()
        _assert_document_never_contradicts_itself(final)
        _assert_a_code_belongs_to_its_own_state(final)
        # THE CONTROL, RE-ENCODED BY THE NARROWING: a seed on a genuinely FRESH
        # nonce really completes — and STILL no completion is ever published, so
        # the word is not merely unreachable in these schedules: it is never
        # emitted at all.  The real acceptance is asserted on the wire and on
        # the server's own accepted chain, which is the only place a success is
        # reported.
        control_status, _control_payload = _post_seed(
            port,
            session_id,
            _seed_body(
                session_id=session_id,
                voxel=good_voxel,
                direction=good_direction,
                nonce=_THIRD_NONCE,
                parent=_NONCE,
            ),
        )
        assert control_status == 200
        controlled = _get_state(port, session_id)
        assert _served_states(controlled, _THIRD_NONCE) == (), (
            f"a really accepted seed published a completion: "
            f"{_reported_jobs(controlled)}"
        )
        assert _COMPLETED not in _served_states(controlled, _THIRD_NONCE)
        _assert_document_never_contradicts_itself(controlled)
        _assert_a_code_belongs_to_its_own_state(controlled)
        control_document = cast("dict[str, Any]", engine.state_document())
        assert _THIRD_NONCE in cast("list[str]", control_document["accepted_nonces"])
        # The initial first build, the ordinary OPEN_QC_PANEL action and this
        # control seed are the three real acceptances of the session.
        assert control_document["action_count"] == 3
    finally:
        server.stop()


def test_a_grammar_action_can_never_overturn_a_returned_seed_refusal(
    volume_only_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R6-F2 reproducer B, a served seed refusal overtaken by a grammar action.

    Step 1: an ordinary ``OPEN_QC_PANEL`` carrying nonce ``N`` really validates
    while ``N`` is unused, and is held open just before it executes.  Step 2: the
    seed really runs under that same ``N`` and really raises the CONSUMING
    ``SEED_OUT_OF_BOUNDS``.  Step 3: while that seed execution is still inside
    its route, the handle really does serve ``REFUSED / SEED_OUT_OF_BOUNDS`` with
    its own exact real code.  Step 4: the seed execution really returns, and the
    already-validated ordinary action really executes, appending ``N`` to the
    all-action ledger.  Step 5: the handle is polled again.

    Before the original fix the handle reported ``COMPLETED`` at step 5 — a
    served terminal state overturned by an action that has nothing to do with
    seeding.  The 2026-09-27 narrowing kept the refusal instead.  RESTORED AND
    RENAMED BY THE F2_JOB_HANDLE_TRUTH SECOND NARROWING (manager decision
    2026-09-28), which removes the case rather than adjudicating it: the seed
    execution returned at step 4, so its record is gone, and there is no served
    word left for the all-action ledger to overturn in either direction.  The
    name is corrected accordingly — nothing is served to be overturned, and no
    ``COMPLETED`` may appear at any step of this schedule.  No seed-live
    snapshot race is involved: the competing action is never represented in the
    seed's records at all, which is exactly why the original defect was
    reachable.
    """

    engine = _open_volume_only_engine(volume_only_session, ulid_start=954000)
    session_id = engine.session_id()
    good_voxel, good_direction = _phantom_seed(200, 220, 180)
    action_parked = threading.Event()
    action_release = threading.Event()
    seed_parked = threading.Event()
    seed_release = threading.Event()
    _park_at_route_exit(
        monkeypatch, [1000000, 1000000, 1000000], seed_parked, seed_release
    )

    server = ViewerServer(audit_sink=None)
    server.add_engine(engine)
    _, port = server.start_background()
    action_outcome: dict[str, Any] = {}
    seed_outcome: dict[str, Any] = {}
    action: threading.Thread | None = None
    seed: threading.Thread | None = None
    try:
        _build_the_initial_centerline(port, session_id)
        # STEP 1: the ordinary action validates (its nonce is UNUSED) and is held
        # open immediately before the real engine executes it.  Parking is on the
        # real action route, outside every lock.
        _park_before_the_action_executes(monkeypatch, action_parked, action_release)
        action = threading.Thread(
            target=lambda: action_outcome.update(
                value=_post_action(port, session_id, nonce=_OTHER_NONCE, parent=_NONCE)
            ),
            name="f2job-r6f2-action",
        )
        action.start()
        assert action_parked.wait(timeout=120), (
            "the ordinary action never validated and parked before execution"
        )
        # STEP 2: the seed really runs under that same nonce and really refuses,
        # consuming it — and is then held open inside its own route.
        seed_body = _seed_body(
            session_id=session_id,
            voxel=[1000000, 1000000, 1000000],
            direction=good_direction,
            nonce=_OTHER_NONCE,
            parent=_NONCE,
        )
        seed = threading.Thread(
            target=lambda: seed_outcome.update(
                value=_post_seed(port, session_id, seed_body)
            ),
            name="f2job-r6f2-seed",
        )
        seed.start()
        assert seed_parked.wait(timeout=120), (
            "the refused seed never parked inside its own route"
        )
        # STEP 3: while that execution is still inside its route, the handle
        # really is served ``REFUSED`` with its own exact real code.
        published_document = _get_state(port, session_id)
        published = _job(published_document, _OTHER_NONCE)
        assert published["state"] == _REFUSED
        assert published["refusal_code"] == "SEED_OUT_OF_BOUNDS"
        _assert_a_code_belongs_to_its_own_state(published_document)
        _assert_document_never_contradicts_itself(published_document)
        # STEP 4: the seed's execution really returns (its record goes with it)
        # and the already-validated ordinary action really executes and really
        # appends that nonce to the all-action ledger.
        seed_release.set()
        assert seed is not None
        seed.join(timeout=300)
        assert not seed.is_alive(), "the refused seed never returned"
        status, payload = seed_outcome["value"]
        assert status == 409
        refused_code = cast("dict[str, Any]", payload["error"])["code"]
        assert refused_code == "SEED_OUT_OF_BOUNDS"
        assert published["refusal_code"] == refused_code
        action_release.set()
        assert action is not None
        action.join(timeout=300)
        assert not action.is_alive(), "the ordinary action never returned"
        assert cast("tuple[int, dict[str, Any]]", action_outcome["value"])[0] == 200
        ledger = cast("dict[str, Any]", engine.state_document())
        assert ledger["qc_panel_open"] is True
        # The all-action ledger really does name that nonce as ACCEPTED — the
        # exact fact the old resolver read as seed completion.  (``used_nonces``
        # holds it twice: the seed's consuming refusal appended it, and the
        # action's own bookkeeping appends unconditionally — pre-existing
        # behaviour of the action route, untouched here.)
        assert _nonce_ledger(ledger, _OTHER_NONCE) == (2, 1)
        # STEP 5: the identity is served NOTHING.  The seed's execution returned
        # before the ordinary action ran, so there is no served word to overturn,
        # and no ``COMPLETED`` may be manufactured from the ledger either.
        after = _get_state(port, session_id)
        assert _served_states(after, _OTHER_NONCE) == (), (
            f"a returned execution is still served: {_reported_jobs(after)}"
        )
        assert _served_refusal_codes(after, _OTHER_NONCE) == ()
        # F2 (2026-09-28 review, P3): the ``_COMPLETED not in _reported_jobs``
        # line that used to sit here could never fail (a string compared against
        # served DICTIONARIES), so it is REMOVED: the document invariant on the
        # next line is the effective form of the same claim on the same
        # document, and the absence assertion above already proves this document
        # is empty.
        _assert_document_never_contradicts_itself(after)
        _assert_a_code_belongs_to_its_own_state(after)
    finally:
        action_release.set()
        seed_release.set()
        if action is not None:
            action.join(timeout=300)
        if seed is not None:
            seed.join(timeout=300)
        server.stop()
