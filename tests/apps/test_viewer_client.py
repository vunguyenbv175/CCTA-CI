"""Real-loopback smoke tests for the closed stdlib physician client seam."""

from __future__ import annotations

import json
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest

from apps.viewer.engine import ViewerEngine
from apps.viewer.grammar import RejectionCode
from apps.viewer.server import ViewerServer
from apps.workstation.round_trip import RoundTripVerificationEngine
from apps.workstation.session_store import RestoredSession
from tests.apps.test_viewer_e2e import _BoundSession, _deterministic_ulids
from tests.apps.viewer_pipeline import build_viewer_session


@pytest.fixture(autouse=True)
def _run_inside_bound_checkout(clean_checkout: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(clean_checkout)


@pytest.fixture()
def client_server(tmp_path: Path, clean_checkout: Path):
    fixture = _BoundSession(build_viewer_session(tmp_path / "ws"), clean_checkout)
    restored = fixture.session
    assert isinstance(restored, RestoredSession)
    engine = ViewerEngine(
        restored,
        id_factory=_deterministic_ulids(),
        now=lambda: "2026-09-07T00:00:00Z",
        repo_root=fixture.repo_root,
    )
    engine.open()
    server = ViewerServer()
    server.add_engine(engine)
    _host, port = server.start_background()
    yield fixture, engine, server, port
    server.stop()


def _request(
    port: int,
    path: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method="POST" if data is not None else "GET",
    )
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def _action(
    session_id: str,
    action: str,
    params: dict[str, object],
    *,
    nonce: str,
    parent: str | None = None,
    view: dict[str, object] | None = None,
) -> bytes:
    document = {
        "action": action,
        "params": params,
        "client_nonce": nonce,
        "view": view or {"plane": "AXIAL", "frame_index": 0},
        "lineage": {"session_id": session_id, "parent_action_nonce": parent},
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def test_client_assets_and_reader_loop_over_real_http(client_server) -> None:
    fixture, engine, _server, port = client_server
    status, headers, html = _request(port, "/")
    assert status == 200
    assert headers["Content-Security-Policy"] == "default-src 'self'"
    assert b"/static/app.js" in html
    status, _headers, listing = _request(port, "/sessions")
    assert status == 200
    assert json.loads(listing)["sessions"] == [engine.session_id()]
    sid = engine.session_id()
    status, _headers, frames = _request(port, f"/session/{sid}/frames/AXIAL")
    assert status == 200
    record = json.loads(frames)["frames"][0]
    frame_id = record["provenance_ref"].split(":")[1]
    status, frame_headers, png = _request(port, f"/frame/{frame_id}.png")
    assert status == 200 and png.startswith(b"\x89PNG\r\n\x1a\n")
    status, _headers, _body = _request(
        port,
        f"/frame/{frame_id}.png",
        headers={"If-None-Match": frame_headers["ETag"]},
    )
    assert status == 304

    parent = None
    actions: tuple[tuple[str, dict[str, object]], ...] = (
        ("SET_SLICE", {"plane": "AXIAL", "frame_index": 1}),
        ("SET_WL_PRESET", {"preset_id": "CT_VESSEL_CONTRAST"}),
        ("CLEAR_WL", {}),
        ("SET_VIEWPORT_MATRIX", {"zoom": 2.0, "pan_x": 0.5, "pan_y": -0.5}),
        ("OPEN_QC_PANEL", {}),
    )
    for number, (action, params) in enumerate(actions):
        nonce = f"client-smoke-{number:016d}"
        body = _action(sid, action, params, nonce=nonce, parent=parent)
        status, _headers, response = _request(
            port,
            f"/session/{sid}/action",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Origin": f"http://127.0.0.1:{port}",
            },
        )
        assert status == 200, response
        parent = nonce
    status, _headers, qc_body = _request(port, f"/session/{sid}/qc")
    assert status == 200
    assert json.loads(qc_body) == engine.qc_panel_document()
    assert json.loads(qc_body)["outcome"]["decision"] == "ABSTAIN"
    assert fixture.study_id == engine.session_document()["study_id"]


def test_client_contract_carries_restored_lineage_qc_viewport_and_typed_png_errors(
    client_server,
) -> None:
    """Executable HTTP seam evidence for the browser state-machine contract.

    The repository intentionally has no JS runtime dependency.  This test
    verifies every response shape consumed by app.js: state restoration,
    verbatim QC, immutable PNG revalidation, and typed PNG rejection.
    """
    _fixture, engine, _server, port = client_server
    sid = engine.session_id()
    status, _headers, state_body = _request(port, f"/session/{sid}/state")
    assert status == 200
    restored = json.loads(state_body)
    assert restored["accepted_nonces"] == []
    assert set(restored["viewport_by_plane"]) == {"AXIAL", "CORONAL", "SAGITTAL"}
    qc_status, _headers, qc_body = _request(port, f"/session/{sid}/qc")
    assert qc_status == 200
    assert json.loads(qc_body) == engine.qc_panel_document()

    viewport_nonce = "client-state-restore-000001"
    action_status, _headers, action_body = _request(
        port,
        f"/session/{sid}/action",
        data=_action(
            sid,
            "SET_VIEWPORT_MATRIX",
            {"zoom": 2.0, "pan_x": 0.5, "pan_y": -0.5},
            nonce=viewport_nonce,
        ),
        headers={"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"},
    )
    assert action_status == 200
    assert json.loads(action_body)["accepted_nonces"][-1] == viewport_nonce
    restored_status, _headers, restored_body = _request(port, f"/session/{sid}/state")
    assert restored_status == 200
    restored_after_action = json.loads(restored_body)
    assert restored_after_action["accepted_nonces"][-1] == viewport_nonce
    assert restored_after_action["viewport_by_plane"]["AXIAL"] == {
        "zoom": 2.0, "pan_x": 0.5, "pan_y": -0.5
    }

    frame = json.loads(_request(port, f"/session/{sid}/frames/AXIAL")[2])["frames"][0]
    frame_id = frame["provenance_ref"].split(":")[1]
    png_status, png_headers, _png = _request(port, f"/frame/{frame_id}.png")
    assert png_status == 200
    cached_status, _headers, cached_body = _request(
        port, f"/frame/{frame_id}.png", headers={"If-None-Match": png_headers["ETag"]}
    )
    assert cached_status == 304 and cached_body == b""
    missing_status, _headers, missing_body = _request(port, "/frame/NO_SUCH_FRAME.png")
    assert missing_status == 404
    assert json.loads(missing_body)["error"]["code"] == "FRAME_NOT_FOUND"


def _run_node_generation_race(port: int, *, late_success: bool) -> None:
    """Execute the preserved v0 app.js bytes against a deferred real HTTP action response.

    CS-001 ships the Cornerstone bundle as the served /static/app.js; the exact
    v0 vanilla-client bytes live under webapp/v0-fallback/ (reference copy, not
    served) and keep this generation-race coverage green.  The NEW served bundle
    is exercised by the real-Chrome gate
    (tests/apps/test_viewer_client_cs001_chrome.py).
    """
    script = r'''
const fs = require("fs");
const base = "__BASE_URL__";
const elements = {};
class Element {
  constructor() {
    this.listeners = {}; this.children = []; this.style = {}; this._value = ""; this.src = "";
  }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  dispatch(type) { if (this.listeners[type]) this.listeners[type]({target: this}); }
  appendChild(child) {
    this.children.push(child);
    if (this === elements["session-select"] && this.children.length === 1) this.value = child.value;
  }
  replaceChildren() { this.children = []; }
  removeAttribute(name) { if (name === "src") this.src = ""; }
  set value(value) { this._value = String(value); }
  get value() { return this._value; }
}
for (const id of ["status", "session-select", "plane-select", "zoom", "pan-x", "pan-y",
  "frame", "frame-caption", "qc-panel", "qc-text", "slice-label", "previous", "next",
  "presets", "clear-wl", "apply-viewport", "open-qc"]) elements[id] = new Element();
global.document = {
  getElementById: (id) => elements[id],
  createElement: () => new Element(),
};
const nativeFetch = global.fetch;
global.fetch = (input, init) => nativeFetch(
  typeof input === "string" && input.startsWith("/") ? `${base}${input}` : input, init,
);
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function waitFor(predicate) {
  for (let i = 0; i < 200; i += 1) {
    if (predicate()) return;
    await wait(10);
  }
  throw new Error(`timed out waiting for viewer state ${JSON.stringify(snapshot())}`);
}
async function probe(path) {
  const response = await fetch(`${base}${path}`);
  return response.json();
}
function snapshot() {
  return {
    plane: elements["plane-select"].value,
    zoom: elements.zoom.value,
    panX: elements["pan-x"].value,
    panY: elements["pan-y"].value,
    transform: elements.frame.style.transform,
    frame: elements.frame.src,
    caption: elements["frame-caption"].textContent,
    status: elements.status.textContent,
  };
}
(async () => {
  eval(fs.readFileSync("webapp/v0-fallback/app.js", "utf8"));
  await waitFor(() => elements["session-select"].value === "A"
    && elements.frame.src.endsWith("/frame/A-frame.png"));
  elements.zoom.value = "7";
  elements["apply-viewport"].listeners.click();
  await waitFor(() => probe("/probe/a-started").then((body) => body.started));
  elements["session-select"].value = "B";
  elements["session-select"].listeners.change({target: elements["session-select"]});
  await waitFor(() => elements["session-select"].value === "B"
    && elements["plane-select"].value === "SAGITTAL"
    && elements.frame.src.endsWith("/frame/B-frame.png"));
  const before = snapshot();
  await fetch(`${base}/release`);
  await wait(100);
  const after = snapshot();
  if (JSON.stringify(before) !== JSON.stringify(after)) {
    throw new Error("obsolete A response changed B display");
  }
  elements.zoom.value = "2";
  elements["apply-viewport"].listeners.click();
  const next = await (async () => {
    for (let i = 0; i < 200; i += 1) {
      const body = await probe("/probe/b-action");
      if (body.parent) return body;
      await wait(10);
    }
    throw new Error("B next action was not submitted");
  })();
  if (next.parent !== "B-parent") throw new Error(`wrong B parent: ${next.parent}`);
  console.log(JSON.stringify({before, after, nextParent: next.parent}));
})().catch((error) => { console.error(error.stack || error); process.exit(1); });
'''
    script = script.replace("__BASE_URL__", f"http://127.0.0.1:{port}")
    result = subprocess.run(
        ["node", "-e", script],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["nextParent"] == "B-parent"
    assert output["before"] == output["after"]


def test_client_deferred_a_to_b_response_keeps_b_state_and_lineage() -> None:
    """Late A success and failure responses are ignored after B is displayed."""
    sessions = {
        "A": {
            "accepted_nonces": ["A-parent"],
            "current_plane": "CORONAL",
            "frame_index_by_plane": {"AXIAL": 0, "CORONAL": 2, "SAGITTAL": 0},
            "viewport_by_plane": {
                "AXIAL": {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0},
                "CORONAL": {"zoom": 7.0, "pan_x": 0.0, "pan_y": 0.0},
                "SAGITTAL": {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0},
            },
        },
        "B": {
            "accepted_nonces": ["B-parent"],
            "current_plane": "SAGITTAL",
            "frame_index_by_plane": {"AXIAL": 1, "CORONAL": 0, "SAGITTAL": 3},
            "viewport_by_plane": {
                "AXIAL": {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0},
                "CORONAL": {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0},
                "SAGITTAL": {"zoom": 1.5, "pan_x": 0.25, "pan_y": -0.5},
            },
        },
    }
    for late_success in (True, False):
        a_release = threading.Event()
        a_started = threading.Event()
        received: list[dict[str, object]] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                pass

            def _write(self, status: int, document: object) -> None:
                payload = json.dumps(document, sort_keys=True).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/sessions":
                    self._write(200, {"sessions": ["A", "B"]})
                elif self.path == "/probe/a-started":
                    self._write(200, {"started": a_started.is_set()})
                elif self.path == "/probe/b-action":
                    parent = (
                        cast(dict[str, object], received[-1]["lineage"])["parent_action_nonce"]
                        if received and len(received) > 1 else None
                    )
                    self._write(200, {"parent": parent})
                elif self.path == "/release":
                    a_release.set()
                    self._write(200, {"released": True})
                elif self.path.endswith("/state"):
                    self._write(200, sessions[self.path.split("/")[2]])
                elif "/frames/" in self.path:
                    session_id = self.path.split("/")[2]
                    self._write(
                        200,
                        {
                            "frames": [
                                {"provenance_ref": f"mpr:{session_id}-frame", "frame_index": 0}
                            ]
                        },
                    )
                elif self.path.startswith("/frame/"):
                    self._write(200, {"ok": True})
                else:
                    self._write(200, {"qc": "B"})

            def do_POST(self) -> None:  # noqa: N802
                size = int(self.headers["Content-Length"] or 0)
                document = json.loads(self.rfile.read(size))
                received.append(document)
                if self.path == "/session/A/action":
                    a_started.set()
                    assert a_release.wait(timeout=5)
                    if late_success:
                        self._write(200, sessions["A"])
                    else:
                        self._write(409, {"error": {"code": "LINEAGE_GAP", "message": "stale"}})
                else:
                    self._write(200, sessions["B"])

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            _run_node_generation_race(server.server_address[1], late_success=late_success)
            first = cast(dict[str, object], received[0])
            last = cast(dict[str, object], received[-1])
            assert cast(dict[str, object], first["lineage"])["parent_action_nonce"] == "A-parent"
            assert cast(dict[str, object], last["lineage"])["parent_action_nonce"] == "B-parent"
        finally:
            server.shutdown()
            server.server_close()


def test_client_surfaces_typed_rejection_and_seventh_check_after_reopen(client_server) -> None:
    fixture, engine, _server, port = client_server
    sid = engine.session_id()
    bad_params: dict[str, object] = {"preset_id": "CUSTOM"}
    bad = _action(
        sid, "SET_WL_PRESET", bad_params,
        nonce="client-rejection-000001", view={"plane": "AXIAL", "frame_index": 0},
    )
    status, _headers, body = _request(
        port,
        f"/session/{sid}/action",
        data=bad,
        headers={
            "Content-Type": "application/json",
            "Origin": f"http://127.0.0.1:{port}",
        },
    )
    assert status == 400
    assert json.loads(body)["error"]["code"] == RejectionCode.UNKNOWN_PRESET.value
    engine.persist(fixture.store)
    reopened = fixture.store.reopen()
    report = RoundTripVerificationEngine(
        fixture.workspace, viewer_repo_root=fixture.repo_root
    ).verify(reopened)
    assert report.verdict == "PASS", report.to_dict()
