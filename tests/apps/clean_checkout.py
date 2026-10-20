"""F11 accepted test pattern: materialize the exact bytes under test into a
clean temp git checkout.

The dev tree is DIRTY while the implementer edits (uncommitted changes), and
the governed build-identity resolver is REQUIRED to fail closed on a dirty
tree — so the happy-path identity equality cannot run against the dirty
working tree directly.  The accepted pattern (r2 prescription, F11.4 /
dispatch practical-testing note): clone/worktree at HEAD, copy the working
tree's ``apps/`` + ``tests/`` over it, and COMMIT there; run the engine
against that bound checkout; and independently compare minted provenance
against ``git rev-parse HEAD`` executed in the temp checkout.  The expected
value is whatever that commit IS — computed independently, never hard-coded.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_COPY_PACKAGES: tuple[str, ...] = ("apps", "tests")
_COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
_COMMIT_ENV = {
    "GIT_AUTHOR_NAME": "VIEWER_001 att0003 test materialization",
    "GIT_AUTHOR_EMAIL": "att0003-test@invalid",
    "GIT_COMMITTER_NAME": "VIEWER_001 att0003 test materialization",
    "GIT_COMMITTER_EMAIL": "att0003-test@invalid",
}


def repository_root() -> Path:
    """The real repository root (this file lives at ``tests/apps/``)."""

    return Path(__file__).resolve().parents[2]


def materialize_clean_checkout(destination: Path, *, source_root: Path | None = None) -> Path:
    """Clone at HEAD, copy the working tree's ``apps/``+``tests/``, commit.

    Returns the checkout root.  The checkout is CLEAN and carries exactly
    the source bytes under test; its HEAD commit is the governed build
    identity the engine must resolve when rooted there.
    """

    root = source_root if source_root is not None else repository_root()
    destination = Path(destination)
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(root), str(destination)],
        check=True,
    )
    for package in _COPY_PACKAGES:
        target = destination / package
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(root / package, target, ignore=_COPY_IGNORE)
    subprocess.run(["git", "add", "-A"], cwd=destination, check=True)
    environment = dict(os.environ)
    environment.update(_COMMIT_ENV)
    committed = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--quiet",
            "--no-verify",
            "-m",
            "VIEWER_001 att0003: materialize the exact bytes under test",
        ],
        cwd=destination,
        env=environment,
        capture_output=True,
        text=True,
    )
    if committed.returncode != 0 and "nothing to commit" not in committed.stdout:
        # A dirty dev tree always has staged changes; a clean bound checkout
        # may already carry identical content, in which case HEAD IS the
        # materialization and there is nothing to commit.
        raise RuntimeError(
            "materialized-checkout commit failed: "
            f"{committed.stdout} {committed.stderr}"
        )
    return destination


def checkout_head_commit(checkout: Path) -> str:
    """Independently resolve the checkout's HEAD commit (the test oracle)."""

    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_in_checkout(
    checkout: Path,
    code: str,
    *,
    env: dict[str, str] | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """Execute python *code* in a subprocess with *checkout* as CWD and import root (r3 R3-01)."""

    import sys

    sub_env = dict(os.environ)
    sub_env["PYTHONPATH"] = str(checkout)
    if env:
        sub_env.update(env)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=checkout,
        env=sub_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


__all__ = [
    "checkout_head_commit",
    "materialize_clean_checkout",
    "repository_root",
    "run_in_checkout",
]
