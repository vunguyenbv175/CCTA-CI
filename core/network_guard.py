"""No-network personal-mode guard (WS02_DESIGN.md Part B §B7).

Defence-in-depth, ordered by portability:

1. PEP 578 audit hook blocking ``socket.connect`` / ``socket.bind`` /
   ``socket.sendto`` / ``socket.sendmsg`` / ``socket.getaddrinfo`` outside an
   explicit loopback allowlist -> raises :class:`NetworkEgressBlockedError`
   and appends exactly one sealed audit event per blocked attempt.
2. Egress mediator seam: every future remote adapter must route through an
   :class:`EgressMediator`; the default is :class:`DenyAllEgressMediator`
   (F4 policy, deny-by-default).
3. ``sys.meta_path`` import guard refusing networking libraries (``requests``,
   ``httpx``, ``aiohttp``, ``urllib3``) in product code paths.
4. OS-level isolation is OPERATOR GUIDANCE ONLY and deliberately NOT
   implemented here (out of process scope; see threat-model notes).

Known bypass classes recorded honestly (pre-research §5.1): C extensions that
open sockets without raising audit events (ctypes/cffi calling WinSock or
libcurl directly), and subprocesses executing external network tools. Layer 4
(firewall/netns) is the operator's mitigation for those classes.

The audit hook cannot be removed once installed (PEP 578); deactivation flips
an internal flag so the hook becomes inert — the honest runtime state stays
observable via :func:`is_network_guard_active`.
"""

from __future__ import annotations

import ipaddress
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Final, Protocol

BLOCKED_AUDIT_EVENTS: Final[frozenset[str]] = frozenset(
    {
        "socket.connect",
        "socket.bind",
        "socket.sendto",
        "socket.sendmsg",
        "socket.getaddrinfo",
    }
)

_BLOCKED_IMPORT_ROOTS: Final[frozenset[str]] = frozenset(
    {"requests", "httpx", "aiohttp", "urllib3"}
)
_LOCAL_HOSTNAMES: Final[frozenset[str]] = frozenset({"localhost"})


class NetworkEgressBlockedError(RuntimeError):
    """A network egress attempt violated the deny-by-default personal mode."""


# ---------------------------------------------------------------------------
# Layer 2 seam: egress mediator (F4 deny-by-default)
# ---------------------------------------------------------------------------


class EgressMediator(Protocol):
    """Seam every future remote adapter must route through (§B7 layer 2)."""

    def authorize(self, operation: str, destination: str) -> None:
        """Return silently to allow, raise to deny."""
        ...


class DenyAllEgressMediator:
    """Default mediator: every egress request is denied before socket setup."""

    def authorize(self, operation: str, destination: str) -> None:
        raise NetworkEgressBlockedError(
            f"egress denied by DenyAllEgressMediator policy: {operation} -> {destination}"
        )


# ---------------------------------------------------------------------------
# Layer 1: PEP 578 audit hook
# ---------------------------------------------------------------------------


class EgressAuditSink(Protocol):
    """Structural sink used for sealed egress-block records (see B8)."""

    def record_egress_block(self, *, event: str, destination: str) -> object:
        """Record exactly one tamper-evident event per blocked attempt."""
        ...


class InMemoryEgressAuditSink:
    """Non-persistent fallback sink; production callers pass PhiSafeAuditSink."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def record_egress_block(self, *, event: str, destination: str) -> object:
        self.events.append((event, destination))
        return len(self.events)


@dataclass(frozen=True)
class _Destination:
    host: str
    port: int | None

    def __str__(self) -> str:
        return f"{self.host}:{self.port}" if self.port is not None else self.host


def _first_address_arg(args: Sequence[object]) -> _Destination | None:
    """Extract a host/port destination from audit-hook args, defensively.

    CPython passes ``(socket, address)`` for connect/bind/sendto/sendmsg and
    ``(host, port, ...)`` for getaddrinfo, but the exact shapes have varied
    across versions; anything we cannot confidently parse is treated as NOT
    loopback (fail-closed default deny).
    """
    for index, arg in enumerate(args):
        if isinstance(arg, str):
            port = args[index + 1] if index + 1 < len(args) else None
            return _Destination(host=arg, port=port if isinstance(port, int) else None)
        if isinstance(arg, bytes):
            try:
                return _Destination(host=arg.decode("idna"), port=None)
            except UnicodeDecodeError:
                return None
        if isinstance(arg, (tuple, list)) and arg:
            head = arg[0]
            if isinstance(head, str):
                port = arg[1] if len(arg) > 1 and isinstance(arg[1], int) else None
                return _Destination(host=head, port=port)
            if isinstance(head, bytes):
                try:
                    return _Destination(host=head.decode("idna"), port=None)
                except UnicodeDecodeError:
                    return None
    return None


def _is_loopback_host(host: str) -> bool:
    """True only for literal loopback IPs or the ``localhost`` name.

    Hostnames other than ``localhost`` are never resolved here: resolution
    itself would be the egress we are blocking.
    """
    if host.casefold() in _LOCAL_HOSTNAMES:
        return True
    candidate = host.strip("[]")
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


@dataclass
class _GuardState:
    active: bool = False
    allow_loopback: bool = True
    sink: EgressAuditSink | None = None
    evicted_imports: dict[str, ModuleType] = field(default_factory=dict)


_STATE = _GuardState()


def _network_guard_hook(event: str, args: tuple[object, ...]) -> None:
    if not _STATE.active or event not in BLOCKED_AUDIT_EVENTS:
        return
    destination = _first_address_arg(args)
    if (
        destination is not None
        and _STATE.allow_loopback
        and _is_loopback_host(destination.host)
    ):
        return
    described = str(destination) if destination is not None else "<unresolved-destination>"
    sink = _STATE.sink
    if sink is not None:
        sink.record_egress_block(event=event, destination=described)
    raise NetworkEgressBlockedError(
        f"network egress blocked by no-network personal mode: {event} -> {described}"
    )


@dataclass(frozen=True)
class NetworkGuardHandle:
    """Opaque handle returned by :func:`install_network_guard`."""

    allow_loopback: bool


def install_network_guard(
    *, sink: EgressAuditSink | None = None, allow_loopback: bool = True
) -> NetworkGuardHandle:
    """Activate the PEP 578 egress-blocking hook (idempotent re-install)."""
    _STATE.active = True
    _STATE.allow_loopback = allow_loopback
    _STATE.sink = sink if sink is not None else InMemoryEgressAuditSink()
    sys.addaudithook(_network_guard_hook)
    return NetworkGuardHandle(allow_loopback=allow_loopback)


def deactivate_network_guard(handle: NetworkGuardHandle | None = None) -> None:
    """Make the hook inert. The hook itself cannot be uninstalled (PEP 578)."""
    _STATE.active = False


def is_network_guard_active() -> bool:
    return _STATE.active


# ---------------------------------------------------------------------------
# Layer 3: meta_path import guard
# ---------------------------------------------------------------------------


class BlockedNetworkImportFinder(MetaPathFinder):
    """meta_path finder raising ImportError for banned networking stacks."""

    BLOCKED_ROOTS: Final[frozenset[str]] = _BLOCKED_IMPORT_ROOTS

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
        /,
    ) -> ModuleSpec | None:
        root = fullname.split(".", 1)[0]
        if root in self.BLOCKED_ROOTS:
            raise ImportError(
                f"import of networking library {fullname!r} is forbidden in "
                "no-network personal mode (WS02_DESIGN.md Part B7 layer 3)"
            )
        return None


_IMPORT_FINDER: Final[BlockedNetworkImportFinder] = BlockedNetworkImportFinder()


def install_import_guard() -> None:
    """Insert the blocking finder at the front of ``sys.meta_path``.

    Also evicts already-imported banned networking modules (root packages
    and every submodule) from ``sys.modules``, recording each entry under
    its own name. A loaded module bypasses every meta_path finder by
    Python semantics, so eviction is required for the ban to be truthful
    when the guard is installed after those roots were pulled in (e.g. a
    hybrid environment where an optional ``requests``/``urllib3`` was
    importable before the guard — pydicom's networking helpers are
    try/except-optional, so the eviction does not break it).
    ``remove_import_guard`` restores every evicted entry losslessly
    (review F2 correction, 2026-08-28: per-name restore keeps root
    packages real packages and submodule imports working afterwards).
    """
    evicted: dict[str, ModuleType] = {}
    for name in list(sys.modules):
        root = name.split(".", 1)[0]
        if root in _BLOCKED_IMPORT_ROOTS:
            evicted[name] = sys.modules.pop(name)
    _STATE.evicted_imports = evicted
    if _IMPORT_FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _IMPORT_FINDER)


def remove_import_guard() -> None:
    """Remove the blocking finder (tests / operator override).

    Restores EVERY module the guard install evicted, under the same names
    (lossless: root packages, their submodules, and import aliases all
    return; nothing is rebound to a non-package object).
    """
    if _IMPORT_FINDER in sys.meta_path:
        sys.meta_path.remove(_IMPORT_FINDER)
    for name, module in _STATE.evicted_imports.items():
        sys.modules.setdefault(name, module)
    _STATE.evicted_imports.clear()


__all__ = [
    "BLOCKED_AUDIT_EVENTS",
    "BlockedNetworkImportFinder",
    "DenyAllEgressMediator",
    "EgressAuditSink",
    "EgressMediator",
    "InMemoryEgressAuditSink",
    "NetworkEgressBlockedError",
    "NetworkGuardHandle",
    "deactivate_network_guard",
    "install_import_guard",
    "install_network_guard",
    "is_network_guard_active",
    "remove_import_guard",
]
