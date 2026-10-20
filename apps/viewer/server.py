"""Loopback-only stdlib HTTP server (VIEWER_001_SPEC Part 6).

Endpoints: ``GET /sessions``, ``GET /session/:id/state``, ``GET
/session/:id/frames/:plane``, ``GET /frame/:id.png`` (ETag = ``png_sha256``,
strong, quoted per RFC 9110; a matching ``If-None-Match`` returns 304 with
no body; ``Cache-Control: no-store`` on state and action responses,
``max-age=31536000, immutable`` on frame PNGs), ``GET /session/:id/qc``,
``GET /session/:id/coronary`` (the closed single-path coronary overlay
document), ``GET /session/:id/cpr`` (the closed CPR view document), ``GET
/session/:id/cpr.png`` (the frozen CPR PNG transport), ``GET
/session/:id/tree-schematic`` (the closed read-only tree-schematic navigator
document), ``GET /session/:id/sync-point`` (the closed read-only synchronized
MPR crosshair cursor derivation — plane + frame + row + column query, no
stored state), ``POST /session/:id/action``, ``POST /session/:id/seed`` (the
dedicated T2 physician-seed route — NOT a grammar action), and the closed
static client asset route.

Host/origin authorization (ADAPT of the accepted workstation skeleton
discipline): GET requests require exactly one valid Host in the configured
authority allow-list; mutation (POST) additionally requires exactly one
Origin equal to the Host authority.  The server runs under the
``core.network_guard.py`` loopback allowlist (no egress, no external DNS);
the guard's sink is the WS-02 ``PhiSafeAuditSink``.  No new runtime
dependency: stdlib only (like WS-02).  Any client is out of scope for v1.
"""

from __future__ import annotations

import hashlib
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final
from urllib.parse import unquote, urlsplit

from apps.viewer.cpr_view import CPRRefusalError
from apps.viewer.engine import (
    HumanSeamRefusalError,
    ViewerEngine,
    ViewerOpenError,
)
from apps.viewer.grammar import RejectionCode, ViewerRejectionError
from apps.viewer.overlay import CoronaryOverlayRefusalError
from apps.viewer.schemas import PLANES
from apps.viewer.sync_point import SyncPointRefusalError
from apps.viewer.tree_schematic import TreeSchematicRefusalError
from core.network_guard import (
    NetworkGuardHandle,
    install_import_guard,
    install_network_guard,
)

_ALLOWED_ORIGINS_SCHEME: Final[str] = "http"
_SESSION_RE: Final[re.Pattern[str]] = re.compile(
    r"\A/sessions?\Z"
)
_STATE_PATH_RE: Final[re.Pattern[str]] = re.compile(r"\A/session/([A-Za-z0-9_-]+)/state\Z")
_FRAMES_PATH_RE: Final[re.Pattern[str]] = re.compile(
    rf"\A/session/([A-Za-z0-9_-]+)/frames/({'|'.join(PLANES)})\Z"
)
_QC_PATH_RE: Final[re.Pattern[str]] = re.compile(r"\A/session/([A-Za-z0-9_-]+)/qc\Z")
_CORONARY_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"\A/session/([A-Za-z0-9_-]+)/coronary\Z"
)
_CPR_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"\A/session/([A-Za-z0-9_-]+)/cpr\Z"
)
_CPR_PNG_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"\A/session/([A-Za-z0-9_-]+)/cpr\.png\Z"
)
_TREE_SCHEMATIC_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"\A/session/([A-Za-z0-9_-]+)/tree-schematic\Z"
)
_SYNC_POINT_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"\A/session/([A-Za-z0-9_-]+)/sync-point\Z"
)
_ACTION_PATH_RE: Final[re.Pattern[str]] = re.compile(r"\A/session/([A-Za-z0-9_-]+)/action\Z")
_SEED_PATH_RE: Final[re.Pattern[str]] = re.compile(r"\A/session/([A-Za-z0-9_-]+)/seed\Z")
_FRAME_PNG_PATH_RE: Final[re.Pattern[str]] = re.compile(r"\A/frame/([A-Za-z0-9_-]+)\.png\Z")
_STATIC_ASSETS: Final[frozenset[str]] = frozenset({"index.html", "app.js", "style.css"})
_STATIC_MIME: Final[dict[str, str]] = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}
_STATIC_ROOT: Final[Path] = Path(__file__).resolve().parent / "static"


class AuthorizationError(ValueError):
    """A request does not satisfy the frozen Host/Origin authority policy."""


def _normalize_authority(authority: str, *, require_port: bool) -> str:
    """Normalize one host[:port] HTTP authority (loopback DNS/IP form)."""

    text = authority.strip().strip("[]")
    if not text:
        raise AuthorizationError("authority is empty")
    host = text
    port = ""
    if ":" in text and not text.startswith("["):
        host, _, maybe_port = text.rpartition(":")
        if maybe_port.isdigit():
            port = maybe_port
        else:
            raise AuthorizationError("invalid authority port")
    if not host or host != host.strip():
        raise AuthorizationError("invalid authority host")
    if host not in {"localhost", "127.0.0.1", "::1"}:
        raise AuthorizationError(f"non-loopback authority host {host!r} is forbidden")
    if require_port and not port:
        raise AuthorizationError("authority port is required")
    return f"{host}:{port}" if port else host


def authorize_host(headers: Any, allowed_authorities: frozenset[str]) -> str:
    """Require exactly one valid loopback Host in the allow-list (GET)."""

    host_values = _header_values(headers, "Host")
    if len(host_values) != 1:
        raise AuthorizationError("exactly one Host header is required")
    try:
        normalized = _normalize_authority(host_values[0], require_port=False)
    except AuthorizationError:
        raise AuthorizationError("Host authority is malformed") from None
    if normalized not in allowed_authorities:
        raise AuthorizationError("Host authority is not allowed")
    return normalized


def _authorize_origin(headers: Any, host: str) -> None:
    """Validate one supplied Origin against an already-authorized Host."""

    origins = _header_values(headers, "Origin")
    if len(origins) != 1:
        raise AuthorizationError("exactly one non-null Origin header is required")
    origin = origins[0]
    if not origin.lower().startswith(f"{_ALLOWED_ORIGINS_SCHEME}://"):
        raise AuthorizationError("Origin must be a lowercase http origin")
    try:
        normalized_origin = _normalize_authority(
            origin.split("://", 1)[1].split("/", 1)[0], require_port=False
        )
    except AuthorizationError:
        raise AuthorizationError("Origin authority is malformed") from None
    if normalized_origin != host:
        raise AuthorizationError("Origin authority does not match Host")


def authorize_mutation(headers: Any, allowed_authorities: frozenset[str]) -> str:
    """Authorize a POST via the exact Host/Origin equality rule."""

    host = authorize_host(headers, allowed_authorities)
    _authorize_origin(headers, host)
    return host


def authorize_static(
    headers: Any, allowed_authorities: frozenset[str]
) -> str:
    """Authorize static GETs with optional same-origin validation.

    Ordinary top-level navigation commonly omits Origin, so absence is
    allowed.  If a browser supplies Origin, the governed mutation policy is
    applied to it; null and duplicate values fail closed.
    """

    host = authorize_host(headers, allowed_authorities)
    if _header_values(headers, "Origin"):
        _authorize_origin(headers, host)
    return host


def _header_values(headers: Any, name: str) -> list[str]:
    values: list[str] = []
    for key, value in headers.items():
        if key.lower() == name.lower():
            values.append(str(value))
    return values


class ViewerServer:
    """The governed viewer server over one workspace of sessions.

    Sessions are opened lazily from restored WS-02 sessions supplied by the
    caller (the server NEVER opens its own volume path) and kept bound for
    their lifetime; an open failure surfaces the Part 6 fail-closed table.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        allowed_authorities: frozenset[str] | None = None,
        engines: dict[str, ViewerEngine] | None = None,
        audit_sink: Any = None,
    ) -> None:
        self._host = host
        self._port = port
        if allowed_authorities is None:
            allowed = {"localhost", "127.0.0.1", "::1", f"{host}:{port}"}
            allowed |= {f"{host}:{port}" for port in (port,)}
            self._allowed_authorities = frozenset(allowed)
        else:
            self._allowed_authorities = frozenset(allowed_authorities)
        self._engines: dict[str, ViewerEngine] = dict(engines or {})
        self._audit_sink = audit_sink
        self._server: ThreadingHTTPServer | None = None
        self._guard: NetworkGuardHandle | None = None
        self._import_guard_installed = False

    # -- lifecycle --------------------------------------------------------

    def start(self) -> tuple[str, int]:
        """Bind loopback-only and arm the network guard; returns (host, port)."""

        if self._server is not None:
            raise RuntimeError("server is already running")
        self._guard = install_network_guard(sink=self._audit_sink, allow_loopback=True)
        install_import_guard()
        self._import_guard_installed = True
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer((self._host, self._port), handler)
        self._port = self._server.server_address[1]
        # Re-derive the authority allow-list with the concrete port.
        self._allowed_authorities = frozenset(
            {
                "localhost",
                "127.0.0.1",
                "::1",
                f"localhost:{self._port}",
                f"127.0.0.1:{self._port}",
            }
        )
        self._server.serve_forever(poll_interval=0.05)
        return (self._host, self._port)

    def start_background(self) -> tuple[str, int]:
        """Bind + serve on a daemon thread (tests + CLI use)."""

        import threading

        if self._server is not None:
            return (self._host, self._port)
        self._guard = install_network_guard(sink=self._audit_sink, allow_loopback=True)
        install_import_guard()
        self._import_guard_installed = True
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer((self._host, self._port), handler)
        self._port = self._server.server_address[1]
        self._allowed_authorities = frozenset(
            {
                "localhost",
                "127.0.0.1",
                "::1",
                f"localhost:{self._port}",
                f"127.0.0.1:{self._port}",
            }
        )
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        return (self._host, self._port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def allowed_authorities(self) -> frozenset[str]:
        return self._allowed_authorities

    def engine(self, session_id: str) -> ViewerEngine | None:
        return self._engines.get(session_id)

    def add_engine(self, engine: ViewerEngine) -> str:
        session_id = engine.session_id()
        self._engines[session_id] = engine
        return session_id

    def session_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._engines))


def _make_handler(server: ViewerServer) -> type[BaseHTTPRequestHandler]:
    class ViewerRequestHandler(BaseHTTPRequestHandler):
        server_version = "CCTAViewer/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            del format, args  # PHI-safe: no request logging

        def _send_json(
            self,
            status: int,
            document: dict[str, object],
            *,
            headers: dict[str, str] | None = None,
        ) -> None:
            import json

            body = json.dumps(document, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_rejection(self, code: str, message: str, status: int) -> None:
            self._send_json(
                status, {"error": {"code": code, "message": message}}
            )

        def _send_png(
            self, png: bytes, etag: str, *, immutable: bool = True
        ) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.send_header("ETag", etag)
            if immutable:
                self.send_header("Cache-Control", "max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(png)

        def _send_static_asset(self, name: str) -> None:
            """Serve one closed, digest-addressed physician client asset."""

            if name not in _STATIC_ASSETS:
                self._send_rejection("NOT_FOUND", "no such static asset", 404)
                return
            # The allowlist check is intentionally before path construction: a
            # request can never turn an arbitrary URL path into a filesystem
            # read, including encoded traversal or absolute-path spellings.
            try:
                static_root = _STATIC_ROOT.resolve()
                asset_path = (_STATIC_ROOT / name).resolve()
                if asset_path.parent != static_root or not asset_path.is_file():
                    raise FileNotFoundError(name)
                body = asset_path.read_bytes()
            except OSError:
                self._send_rejection("NOT_FOUND", "static asset is unavailable", 404)
                return
            digest = hashlib.sha256(body).hexdigest()
            etag = f'"{digest}"'
            if _etag_matches(self.headers, digest):
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Content-Security-Policy", "default-src 'self'")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", _STATIC_MIME[name])
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", etag)
            self.send_header("Content-Security-Policy", "default-src 'self'")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        # -- HTTP verbs ----------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            parsed_path = urlsplit(self.path).path
            if parsed_path == "/" or parsed_path.startswith("/static/"):
                try:
                    authorize_static(self.headers, server.allowed_authorities)
                except AuthorizationError as exc:
                    self._send_rejection("AUTHORIZATION_DENIED", str(exc), 403)
                    return
                static_path = unquote(parsed_path)
                if "\x00" in static_path or "\\" in static_path:
                    self._send_rejection("NOT_FOUND", "no such viewer resource", 404)
                    return
                if static_path == "/":
                    self._send_static_asset("index.html")
                    return
                asset_name = static_path[len("/static/") :]
                if "/" in asset_name or not asset_name:
                    self._send_rejection("NOT_FOUND", "no such static asset", 404)
                    return
                self._send_static_asset(asset_name)
                return

            try:
                authorize_host(self.headers, server.allowed_authorities)
            except AuthorizationError as exc:
                self._send_rejection("AUTHORIZATION_DENIED", str(exc), 403)
                return
            path = parsed_path

            if path == "/sessions":
                self._send_json(
                    200,
                    {"sessions": list(server.session_ids())},
                )
                return

            match = _SESSION_RE.match(path)
            if match and path != "/sessions":
                self._send_json(200, {"sessions": list(server.session_ids())})
                return

            state_match = _STATE_PATH_RE.match(path)
            if state_match:
                session_id = unquote(state_match.group(1))
                engine = server.engine(session_id)
                if engine is None:
                    self._send_rejection(
                        RejectionCode.SESSION_NOT_FOUND.value,
                        "no open viewer session for this id",
                        404,
                    )
                    return
                try:
                    self._send_json(200, engine.planes_state())
                except ViewerOpenError as exc:
                    self._send_rejection(
                        exc.code.value, "; ".join(exc.problems), 409
                    )
                return

            frames_match = _FRAMES_PATH_RE.match(path)
            if frames_match:
                session_id = unquote(frames_match.group(1))
                plane = frames_match.group(2)
                engine = server.engine(session_id)
                if engine is None:
                    self._send_rejection(
                        RejectionCode.SESSION_NOT_FOUND.value,
                        "no open viewer session for this id",
                        404,
                    )
                    return
                self._send_json(200, engine.frames_document(plane))
                return

            qc_match = _QC_PATH_RE.match(path)
            if qc_match:
                session_id = unquote(qc_match.group(1))
                engine = server.engine(session_id)
                if engine is None:
                    self._send_rejection(
                        RejectionCode.SESSION_NOT_FOUND.value,
                        "no open viewer session for this id",
                        404,
                    )
                    return
                self._send_json(200, engine.qc_panel_document())
                return

            coronary_match = _CORONARY_PATH_RE.match(path)
            if coronary_match:
                session_id = unquote(coronary_match.group(1))
                engine = server.engine(session_id)
                if engine is None:
                    self._send_rejection(
                        RejectionCode.SESSION_NOT_FOUND.value,
                        "no open viewer session for this id",
                        404,
                    )
                    return
                try:
                    query = urlsplit(self.path).query
                    params: dict[str, str] = {}
                    for pair in query.split("&"):
                        if not pair:
                            continue
                        if "=" in pair:
                            key, _, value = pair.partition("=")
                            params[key] = value
                        else:
                            params[pair] = ""
                    requested_mode = unquote(params.get("mode", "")).strip()
                    requested_vessel = unquote(params.get("vessel", "")).strip()
                    self._send_json(
                        200,
                        engine.coronary_overlay_document(
                            mode=requested_mode if requested_mode else None,
                            vessel_label=requested_vessel if requested_vessel else None,
                        ),
                    )
                except CoronaryOverlayRefusalError as refusal:
                    # Typed overlay refusal: visible, closed-vocabulary,
                    # fail-closed (HTTP 409; the 16 grammar codes stay
                    # reserved to grammar violations).
                    self._send_rejection(refusal.code, refusal.message, refusal.http_status)
                return

            cpr_match = _CPR_PATH_RE.match(path)
            if cpr_match:
                session_id = unquote(cpr_match.group(1))
                engine = server.engine(session_id)
                if engine is None:
                    self._send_rejection(
                        RejectionCode.SESSION_NOT_FOUND.value,
                        "no open viewer session for this id",
                        404,
                    )
                    return
                try:
                    query = urlsplit(self.path).query
                    cpr_params: dict[str, str] = {}
                    for pair in query.split("&"):
                        if not pair:
                            continue
                        if "=" in pair:
                            key, _, value = pair.partition("=")
                            cpr_params[key] = value
                        else:
                            cpr_params[pair] = ""
                    requested_vessel = unquote(cpr_params.get("vessel", "")).strip()
                    self._send_json(
                        200,
                        engine.cpr_view_document(
                            vessel_label=requested_vessel if requested_vessel else None
                        ),
                    )
                except CPRRefusalError as refusal:
                    self._send_rejection(refusal.code, refusal.message, refusal.http_status)
                return

            cpr_png_match = _CPR_PNG_PATH_RE.match(path)
            if cpr_png_match:
                session_id = unquote(cpr_png_match.group(1))
                engine = server.engine(session_id)
                if engine is None:
                    self._send_rejection(
                        RejectionCode.SESSION_NOT_FOUND.value,
                        "no open viewer session for this id",
                        404,
                    )
                    return
                try:
                    query = urlsplit(self.path).query
                    png_params: dict[str, str] = {}
                    for pair in query.split("&"):
                        if not pair:
                            continue
                        if "=" in pair:
                            key, _, value = pair.partition("=")
                            png_params[key] = value
                    requested_vessel = unquote(png_params.get("vessel", "")).strip()
                    png, _view = engine.cpr_png_bytes(
                        vessel_label=requested_vessel if requested_vessel else None
                    )
                except CPRRefusalError as refusal:
                    self._send_rejection(refusal.code, refusal.message, refusal.http_status)
                    return
                import hashlib as _hashlib

                self._send_png(png, f'"{_hashlib.sha256(png).hexdigest()}"')
                return

            tree_match = _TREE_SCHEMATIC_PATH_RE.match(path)
            if tree_match:
                # TREE_SCHEMATIC_NAV read-only navigator route: mirrors the
                # #28 CPR GET precedent - session lookup, read-only engine
                # derivation, typed refusal mapping (HTTP 409).  No grammar
                # action is involved (``ACTIONS`` stays closed).
                session_id = unquote(tree_match.group(1))
                engine = server.engine(session_id)
                if engine is None:
                    self._send_rejection(
                        RejectionCode.SESSION_NOT_FOUND.value,
                        "no open viewer session for this id",
                        404,
                    )
                    return
                try:
                    self._send_json(200, engine.tree_schematic_document())
                except TreeSchematicRefusalError as refusal:
                    self._send_rejection(refusal.code, refusal.message, refusal.http_status)
                return

            sync_match = _SYNC_POINT_PATH_RE.match(path)
            if sync_match:
                # MPR_SYNC_CURSOR read-only derivation route: mirrors the
                # tree-schematic GET precedent - session lookup, read-only
                # engine derivation, typed refusal mapping (HTTP 409).  The
                # cursor pick rides the query string (plane + frame + row +
                # column); nothing is stored server-side (Rule F).  No grammar
                # action is involved (``ACTIONS`` stays closed).
                session_id = unquote(sync_match.group(1))
                engine = server.engine(session_id)
                if engine is None:
                    self._send_rejection(
                        RejectionCode.SESSION_NOT_FOUND.value,
                        "no open viewer session for this id",
                        404,
                    )
                    return
                try:
                    query = urlsplit(self.path).query
                    sync_params: dict[str, str] = {}
                    for pair in query.split("&"):
                        if not pair:
                            continue
                        if "=" in pair:
                            key, _, value = pair.partition("=")
                            sync_params[key] = value
                        else:
                            sync_params[pair] = ""
                    requested_plane = unquote(sync_params.get("plane", "")).strip()
                    try:
                        requested_frame = int(sync_params.get("frame", "").strip())
                        requested_row = float(sync_params.get("row", "").strip())
                        requested_column = float(sync_params.get("column", "").strip())
                    except ValueError:
                        raise SyncPointRefusalError(
                            "SYNC_POINT_INVALID",
                            "sync-point query must carry frame (int) + row/column "
                            "(finite numbers)",
                        ) from None
                    self._send_json(
                        200,
                        engine.sync_point_document(
                            plane=requested_plane,
                            frame_index=requested_frame,
                            row=requested_row,
                            column=requested_column,
                        ),
                    )
                except SyncPointRefusalError as refusal:
                    self._send_rejection(refusal.code, refusal.message, refusal.http_status)
                return

            png_match = _FRAME_PNG_PATH_RE.match(path)
            if png_match:
                frame_id = unquote(png_match.group(1))
                for session_id in server.session_ids():
                    engine = server.engine(session_id)
                    if engine is None:
                        continue
                    found = engine.png_for_frame(frame_id)
                    if found is not None:
                        png, record = found
                        etag = f'"{record.png_sha256}"'
                        client_tags = _header_values(self.headers, "If-None-Match")
                        if any(
                            tag.strip().strip("W/").strip('"') == record.png_sha256
                            for tag in client_tags
                        ):
                            self.send_response(304)
                            self.send_header("ETag", etag)
                            self.send_header(
                                "Cache-Control", "max-age=31536000, immutable"
                            )
                            self.send_header("Content-Length", "0")
                            self.end_headers()
                            return
                        self._send_png(png, etag)
                        return
                self._send_rejection(
                    "FRAME_NOT_FOUND", "no frame record with this id", 404
                )
                return

            self._send_rejection("NOT_FOUND", "no such viewer resource", 404)

        def do_POST(self) -> None:  # noqa: N802
            try:
                authorize_mutation(self.headers, server.allowed_authorities)
            except AuthorizationError as exc:
                self._send_rejection("AUTHORIZATION_DENIED", str(exc), 403)
                return
            path = urlsplit(self.path).path
            seed_match = _SEED_PATH_RE.match(path)
            if seed_match:
                # T2 physician/manual seed route (Phase A): a dedicated
                # session route following the #28 CPR GET precedent — NOT a
                # grammar action (``ACTIONS`` stays the closed 11).  The
                # authorization, session lookup, body framing, and typed
                # refusal mapping mirror the action route byte-for-byte.
                session_id = unquote(seed_match.group(1))
                engine = server.engine(session_id)
                if engine is None:
                    self._send_rejection(
                        RejectionCode.SESSION_NOT_FOUND.value,
                        "no open viewer session for this id",
                        404,
                    )
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._send_rejection(
                        "NON_CANONICAL_ENCODING", "invalid Content-Length", 400
                    )
                    return
                body = self.rfile.read(length) if length > 0 else b""
                try:
                    document = engine.apply_seed_bytes(
                        body, path_session_id=session_id
                    )
                except ViewerRejectionError as rejection:
                    self._send_rejection(
                        rejection.code.value, rejection.message, rejection.http_status
                    )
                    return
                except HumanSeamRefusalError as refusal:
                    # Typed seam refusal: visible, closed-vocabulary,
                    # fail-closed (HTTP 409 with the coronary refusal code;
                    # the 16 grammar codes stay reserved to grammar
                    # violations).
                    self._send_rejection(refusal.code, refusal.message, 409)
                    return
                except ViewerOpenError as exc:
                    self._send_rejection(exc.code.value, "; ".join(exc.problems), 409)
                    return
                self._send_json(200, document)
                return
            action_match = _ACTION_PATH_RE.match(path)
            if not action_match:
                self._send_rejection("NOT_FOUND", "no such viewer resource", 404)
                return
            session_id = unquote(action_match.group(1))
            engine = server.engine(session_id)
            if engine is None:
                self._send_rejection(
                    RejectionCode.SESSION_NOT_FOUND.value,
                    "no open viewer session for this id",
                    404,
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_rejection(
                    "NON_CANONICAL_ENCODING", "invalid Content-Length", 400
                )
                return
            body = self.rfile.read(length) if length > 0 else b""
            try:
                document = engine.apply_action_bytes(body, path_session_id=session_id)
            except ViewerRejectionError as rejection:
                self._send_rejection(
                    rejection.code.value, rejection.message, rejection.http_status
                )
                return
            except HumanSeamRefusalError as refusal:
                # Typed seam refusal: visible, closed-vocabulary, fail-closed
                # (HTTP 409 conflict with the seam's frozen matrix; the 16
                # grammar codes stay reserved to grammar violations).
                self._send_rejection(refusal.code, refusal.message, 409)
                return
            except ViewerOpenError as exc:
                self._send_rejection(exc.code.value, "; ".join(exc.problems), 409)
                return
            self._send_json(200, document)

    return ViewerRequestHandler


def _etag_matches(headers: Any, digest: str) -> bool:
    """RFC 9110 strong-ETag match: exact quoted form or a list member."""

    for tag in _header_values(headers, "If-None-Match"):
        for candidate in tag.split(","):
            cleaned = candidate.strip()
            if cleaned == f'"{digest}"':
                return True
            if cleaned.startswith("W/"):
                cleaned = cleaned[2:]
            if cleaned.strip('"') == digest:
                return True
    return False


__all__ = [
    "AuthorizationError",
    "ViewerServer",
    "authorize_host",
    "authorize_mutation",
    "authorize_static",
]
