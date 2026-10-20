"""Loopback launcher for the stdlib physician viewer client."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from apps.viewer.engine import ViewerEngine
from apps.viewer.server import ViewerServer
from apps.workstation.cli import _arm_guard, _disarm, _load_session
from apps.workstation.session_store import WorkstationPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m apps.viewer",
        description="Serve a saved CCTA viewer session on loopback only.",
    )
    parser.add_argument("store_path", type=Path, help="saved WS-02 workspace path")
    parser.add_argument(
        "engine_id_positional", nargs="?", help="optional session id label"
    )
    parser.add_argument(
        "--engine-id",
        dest="engine_id_option",
        default=None,
        help="optional session id label",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = WorkstationPaths.create(args.store_path)
    sink = _arm_guard(paths, disabled=False)
    server: ViewerServer | None = None
    try:
        restored = _load_session(paths)
        engine = ViewerEngine(restored, audit_sink=sink)
        engine.open()
        engine_id = args.engine_id_option or args.engine_id_positional
        if engine_id is not None and engine_id != engine.session_id():
            raise SystemExit("engine id does not match the restored session")
        server = ViewerServer(host="127.0.0.1", port=0, audit_sink=sink)
        server.add_engine(engine)
        host, port = server.start_background()
        print(f"http://{host}:{port}/", flush=True)
        # The server thread is daemonized by ViewerServer; park the launcher
        # without reading stdin so it remains usable under a service manager.
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if server is not None:
            server.stop()
        _disarm()


if __name__ == "__main__":
    sys.exit(main())
