"""F11 (r2 / r3): governed implementation identity — resolver + equality gates.

Two tiers, matching the r2 prescription (F11.1-F11.4) and r3 prescriptions (R3-01, R3-02):

1. Resolver battery: fail CLOSED on a dirty tracked tree, untracked
   executable source under a runtime package root, a missing git executable,
   a malformed / all-zero / nonexistent commit identity, an unrelated repository
   root, and a process running from a wrong CWD; and no environment-variable
   override and no fallback constant may influence the resolved identity.
   Ordinary, staged, assume-unchanged, skip-worktree, and git-config env
   overrides are tested using actual imports in each copy, failing closed
   with typed ViewerOpenError(SESSION_OPEN_FAILED) before any open state or
   provenance is minted.

2. Clean-checkout equality gate: the exact source bytes under test are
   materialized into a clean temp git checkout (clone at HEAD + copy the working
   tree's ``apps/`` + ``tests/`` + commit there), executed via subprocess with
   that checkout as CWD and import root, and every newly minted provenance SHA
   (session open, frame renders, HUMAN aggregate) is compared against
   ``git rev-parse HEAD`` executed INDEPENDENTLY in that temp checkout.
   The expected value is whatever that commit IS — computed by the test's own
   subprocess call, never hard-coded.

Also covered: standalone mint rejection of wrong existing SHA or dirty checkout,
unrelated repo root rejection across all viewer entrypoints, zero-sentinel rejection,
and reopen no-relabeling guarantee.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from apps.viewer.build_identity import (
    BuildIdentityError,
    repo_root_from_file,
    resolve_build_identity,
    validate_commit_sha,
    verify_commit_exists,
)
from tests.apps.clean_checkout import (
    checkout_head_commit,
    materialize_clean_checkout,
    repository_root,
    run_in_checkout,
)

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _deterministic_ulids():
    class _Factory:
        def __init__(self) -> None:
            self._n = 0

        def __call__(self, at_ms: int | None = None) -> str:
            del at_ms
            self._n += 1
            value = (0x01A9 << 80) | self._n
            return "".join(
                _ULID_ALPHABET[(value >> (shift * 5)) & 0x1F]
                for shift in range(25, -1, -1)
            )

    return _Factory()


def _is_independently_dirty(root: Path) -> bool:
    """Independent oracle for whether checkout at *root* is dirty (R3-02 / R5-01 / R5-02).

    Report-only artifacts are excluded. Untracked non-runtime files are excluded.
    Does NOT use the product resolver.
    """
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "status.showUntrackedFiles=all",
            "status",
            "--porcelain",
            "-z",
            "--untracked-files=all",
        ],
        capture_output=True,
        text=False,
        check=True,
    )
    entries = [
        e.decode("utf-8", "surrogateescape")
        for e in proc.stdout.split(b"\x00")
        if e
    ]
    runtime_prefixes = (
        "apps/",
        "core/",
        "models/",
        "validation/",
        "visualization/",
        "tests/",
    )
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        if len(entry) < 4:
            continue
        status_code = entry[:2]
        path = entry[3:]
        if status_code[0] in {"R", "C"} and i < len(entries):
            i += 1
        if path.startswith("reports/"):
            continue
        if status_code == "??":
            if any(path.startswith(prefix) for prefix in runtime_prefixes):
                if path.endswith("/") or (root / path).is_dir():
                    return True
        else:
            return True

    # Supplement with ignore-independent untracked enumeration (r4 R4-02 / r5 R5-01 / r5 R5-02)
    others_proc = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--others",
            "-z",
            "--",
            *runtime_prefixes,
        ],
        capture_output=True,
        text=False,
        check=True,
    )
    others_entries = [
        e.decode("utf-8", "surrogateescape")
        for e in others_proc.stdout.split(b"\x00")
        if e
    ]
    # Collect tracked files independently
    tracked_proc = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
        ],
        capture_output=True,
        text=False,
        check=True,
    )
    tracked_files = {
        e.decode("utf-8", "surrogateescape")
        for e in tracked_proc.stdout.split(b"\x00")
        if e
    }
    tool_cache_dirs = frozenset({".pytest_cache", ".mypy_cache", ".ruff_cache"})
    allowed_tool_exts = frozenset({"", ".tag", ".md", ".json", ".txt", ".lock", ".db"})
    forbidden_exts = frozenset({
        ".py",
        ".pyw",
        ".pyi",
        ".pyd",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".sh",
        ".bash",
        ".bin",
    })

    for entry in others_entries:
        if entry.startswith("reports/"):
            continue
        if not any(entry.startswith(prefix) for prefix in runtime_prefixes):
            continue
        # Directory summary refusal (R5-01)
        if entry.endswith("/") or (root / entry).is_dir():
            return True
        p = Path(entry)
        if p.suffix in forbidden_exts:
            return True
        full_p = root / entry
        if full_p.is_file() and os.access(full_p, os.X_OK):
            return True
        if p.parent.name == "__pycache__":
            if p.suffix not in {".pyc", ".pyo"}:
                return True
            module_stem = p.name.split(".")[0]
            cand = (p.parent.parent / f"{module_stem}.py").as_posix()
            if cand not in tracked_files:
                return True
            continue
        if "__pycache__" not in p.parts and any(d in p.parts for d in tool_cache_dirs):
            if p.suffix.lower() in allowed_tool_exts:
                continue
        return True
    return False


def _independent_head(root: Path) -> str:
    """The test's own ``git rev-parse HEAD`` oracle."""

    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Resolver battery (direct, fail-closed paths)
# ---------------------------------------------------------------------------


def test_resolver_matches_the_independently_determined_tree_state() -> None:
    """Dirty tracked tree -> refuse; clean -> exact independent HEAD.

    This holds in the dirty dev tree while the fix round is being written
    AND in a clean bound checkout (the r3 reviewer's equality context).
    Report-only artifacts are excluded (r3 R3-02).
    """

    root = repository_root()
    if _is_independently_dirty(root):
        with pytest.raises(BuildIdentityError) as excinfo:
            resolve_build_identity()
        assert "dirty source" in str(excinfo.value)
    else:
        assert resolve_build_identity() == _independent_head(root)


def test_resolver_fails_closed_on_untracked_runtime_source(tmp_path: Path) -> None:
    source = materialize_clean_checkout(tmp_path / "base")
    (source / "apps" / "viewer" / "untracked_evil.py").write_text("x = 1\n")
    code = (
        "from apps.viewer.build_identity import BuildIdentityError, resolve_build_identity\n"
        "try:\n"
        "    resolve_build_identity()\n"
        "    raise SystemExit(1)\n"
        "except BuildIdentityError as exc:\n"
        "    assert 'untracked executable source' in str(exc)\n"
    )
    res = run_in_checkout(source, code)
    assert res.returncode == 0, res.stderr


def test_resolver_fails_closed_on_missing_git(tmp_path: Path) -> None:
    source = materialize_clean_checkout(tmp_path / "base")
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    code = (
        "from apps.viewer.build_identity import BuildIdentityError, resolve_build_identity\n"
        "try:\n"
        "    resolve_build_identity()\n"
        "    raise SystemExit(1)\n"
        "except BuildIdentityError as exc:\n"
        "    assert 'git executable not found' in str(exc)\n"
    )
    res = run_in_checkout(source, code, env={"PATH": str(empty_bin)})
    assert res.returncode == 0, res.stderr


def test_resolver_rejects_malformed_and_zero_identity() -> None:
    for bad in ("not-a-sha", "ABCDEF", "0" * 40, "0x" + "a" * 38, 12345, None):
        with pytest.raises(BuildIdentityError):
            validate_commit_sha(bad)
    accepted = validate_commit_sha("a" * 39 + "1")
    assert accepted == "a" * 39 + "1"


def test_resolver_fails_closed_on_nonexistent_identity() -> None:
    with pytest.raises(BuildIdentityError):
        verify_commit_exists("b" * 39 + "2")


def test_resolver_fails_closed_on_wrong_cwd(tmp_path: Path) -> None:
    source = materialize_clean_checkout(tmp_path / "base")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    code = (
        f"import os\n"
        f"os.chdir({str(elsewhere)!r})\n"
        "from apps.viewer.build_identity import BuildIdentityError, resolve_build_identity\n"
        "try:\n"
        "    resolve_build_identity()\n"
        "    raise SystemExit(1)\n"
        "except BuildIdentityError as exc:\n"
        "    assert 'wrong CWD' in str(exc)\n"
    )
    res = run_in_checkout(source, code)
    assert res.returncode == 0, res.stderr


def test_resolver_happy_path_in_a_clean_checkout(tmp_path: Path) -> None:
    source = materialize_clean_checkout(tmp_path / "base")
    expected = checkout_head_commit(source)
    code = (
        "from apps.viewer.build_identity import resolve_build_identity\n"
        "sha = resolve_build_identity()\n"
        "assert len(sha) == 40 and sha == sha.lower()\n"
        "assert set(sha) <= set('0123456789abcdef')\n"
        "assert sha != '0' * 40\n"
        "print(sha)\n"
    )
    res = run_in_checkout(source, code)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == expected


def test_no_environment_variable_can_override_the_identity(tmp_path: Path) -> None:
    """A planted CCTA_* env var cannot influence resolution or validation."""

    source = materialize_clean_checkout(tmp_path / "base")
    expected = checkout_head_commit(source)
    planted = "f" * 39 + "3"
    code = (
        "from apps.viewer.build_identity import (\n"
        "    BuildIdentityError, resolve_build_identity, validate_commit_sha\n"
        ")\n"
        "sha = resolve_build_identity()\n"
        f"assert sha != {planted!r}\n"
        "try:\n"
        "    validate_commit_sha('0' * 40)\n"
        "    raise SystemExit(1)\n"
        "except BuildIdentityError:\n"
        "    pass\n"
        "print(sha)\n"
    )
    res = run_in_checkout(
        source, code, env={"CCTA_BUILD_SHA": planted, "CCTA_COMMIT": planted}
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == expected


def test_cleanliness_excludes_report_only_artifacts(tmp_path: Path) -> None:
    source = materialize_clean_checkout(tmp_path / "base")
    (source / "reports" / "untracked_note.md").write_text("note")
    code1 = (
        "from apps.viewer.build_identity import verify_source_cleanliness\n"
        "verify_source_cleanliness()\n"
    )
    res1 = run_in_checkout(source, code1)
    assert res1.returncode == 0, res1.stderr

    (source / "apps" / "viewer" / "tracked_note.py").write_text("x = 1\n")
    code2 = (
        "from apps.viewer.build_identity import BuildIdentityError, verify_source_cleanliness\n"
        "try:\n"
        "    verify_source_cleanliness()\n"
        "    raise SystemExit(1)\n"
        "except BuildIdentityError:\n"
        "    pass\n"
    )
    res2 = run_in_checkout(source, code2)
    assert res2.returncode == 0, res2.stderr


def test_repo_root_resolves_from_the_module_location() -> None:
    assert repo_root_from_file() == repository_root()


# ---------------------------------------------------------------------------
# R3-01 tests: producing-root truth across all entrypoints
# ---------------------------------------------------------------------------


def test_unrelated_repo_root_rejected(tmp_path: Path) -> None:
    """An unrelated repo_root is rejected fail-closed across all entrypoints (r3 R3-01)."""

    checkout = materialize_clean_checkout(tmp_path / "checkout")
    unrelated = tmp_path / "unrelated_repo"
    unrelated.mkdir()
    subprocess.run(["git", "init", str(unrelated)], check=True)
    (unrelated / "file.txt").write_text("hello")
    subprocess.run(["git", "-C", str(unrelated), "add", "."], check=True)
    subprocess.run(["git", "-C", str(unrelated), "commit", "-m", "init"], check=True)

    ws_dir = tmp_path / "ws"
    code = f'''
from pathlib import Path
from apps.viewer.build_identity import (
    BuildIdentityError, resolve_build_identity, verify_commit_exists, verify_source_cleanliness,
)
from apps.viewer.engine import ViewerEngine, ViewerOpenError
from apps.viewer.frames import mint_render_variant, GeometryFrame
from apps.viewer.grammar import RejectionCode
from apps.workstation.round_trip import RoundTripVerificationEngine, CHECK_VIEWER
from tests.apps.viewer_pipeline import build_viewer_session

unrelated = Path({str(unrelated)!r})

# 1. resolve_build_identity
try:
    resolve_build_identity(repo_root=unrelated)
    raise SystemExit("resolve_build_identity did not reject unrelated")
except BuildIdentityError:
    pass

# 2. verify_commit_exists
try:
    verify_commit_exists("a" * 40, repo_root=unrelated)
    raise SystemExit("verify_commit_exists did not reject unrelated")
except BuildIdentityError:
    pass

# 3. verify_source_cleanliness
try:
    verify_source_cleanliness(root=unrelated)
    raise SystemExit("verify_source_cleanliness did not reject unrelated")
except BuildIdentityError:
    pass

# 4. ViewerEngine.open
fixture = build_viewer_session(Path({str(ws_dir)!r}))
engine = ViewerEngine(fixture.session, repo_root=unrelated)
try:
    engine.open()
    raise SystemExit("ViewerEngine did not reject unrelated")
except ViewerOpenError as exc:
    assert exc.code is RejectionCode.SESSION_OPEN_FAILED
assert engine.new_provenance() == ()

# 5. mint_render_variant
mpr_art = fixture.mpr_artifacts[0]
payload = fixture.session.payload_documents[mpr_art.artifact_id]
digest = str(payload["pixel_blob_ref"]["sha256"])
pixel_blob = fixture.session.pixel_blobs[digest]
geometry = GeometryFrame(
    plane="AXIAL",
    frame_index=0,
    mpr_artifact_id=mpr_art.artifact_id,
    plane_coordinate_mm=0.0,
    source_volume_ref="vol-1",
)
try:
    mint_render_variant(
        session_id="test-session",
        geometry=geometry,
        wl_preset_id="CT_MEDIASTINUM",
        center=40.0,
        width=400.0,
        mpr_artifact=mpr_art,
        mpr_payload=payload,
        pixel_blob=pixel_blob,
        provenance_upstream=(mpr_art.provenance_ref,),
        code_commit_sha="a" * 40,
        repo_root=unrelated,
    )
    raise SystemExit("mint_render_variant did not reject unrelated")
except BuildIdentityError:
    pass

# 6. RoundTripVerificationEngine
engine_valid = ViewerEngine(fixture.session)
engine_valid.open()
engine_valid.persist(fixture.store)
reopened = fixture.store.reopen()
report = RoundTripVerificationEngine(fixture.workspace, viewer_repo_root=unrelated).verify(reopened)
assert report.verdict == "FAIL"
assert not report.checks[CHECK_VIEWER].passed
'''
    res = run_in_checkout(checkout, code)
    assert res.returncode == 0, res.stderr


def test_standalone_mint_rejects_wrong_existing_sha(tmp_path: Path) -> None:
    """Standalone mint rejects an existing prior commit (HEAD~1) and mints no provenance (R3-01)."""

    checkout = materialize_clean_checkout(tmp_path / "checkout")
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "--allow-empty", "-m", "prior commit"],
        check=True,
    )
    prior_sha = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD~1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    ws_dir = tmp_path / "ws"
    code = f'''
from pathlib import Path
from apps.viewer.build_identity import BuildIdentityError
from apps.viewer.frames import mint_render_variant, GeometryFrame
from tests.apps.viewer_pipeline import build_viewer_session

fixture = build_viewer_session(Path({str(ws_dir)!r}))
mpr_art = fixture.mpr_artifacts[0]
payload = fixture.session.payload_documents[mpr_art.artifact_id]
digest = str(payload["pixel_blob_ref"]["sha256"])
pixel_blob = fixture.session.pixel_blobs[digest]
geometry = GeometryFrame(
    plane="AXIAL",
    frame_index=0,
    mpr_artifact_id=mpr_art.artifact_id,
    plane_coordinate_mm=0.0,
    source_volume_ref="vol-1",
)

try:
    mint_render_variant(
        session_id="test-session",
        geometry=geometry,
        wl_preset_id="CT_MEDIASTINUM",
        center=40.0,
        width=400.0,
        mpr_artifact=mpr_art,
        mpr_payload=payload,
        pixel_blob=pixel_blob,
        provenance_upstream=(mpr_art.provenance_ref,),
        code_commit_sha={prior_sha!r},
    )
    raise SystemExit("mint_render_variant did not raise on prior sha")
except BuildIdentityError as exc:
    assert "does not match current producing checkout" in str(exc)
'''
    res = run_in_checkout(checkout, code)
    assert res.returncode == 0, res.stderr


def test_standalone_mint_rejects_dirty_checkout(tmp_path: Path) -> None:
    """Standalone mint rejects a dirty checkout and mints no provenance (r3 R3-01)."""

    checkout = materialize_clean_checkout(tmp_path / "checkout")
    head_sha = checkout_head_commit(checkout)
    (checkout / "apps" / "viewer" / "untracked_mint.py").write_text("x = 1\n")
    ws_dir = tmp_path / "ws"
    code = f'''
from pathlib import Path
from apps.viewer.build_identity import BuildIdentityError
from apps.viewer.frames import mint_render_variant, GeometryFrame
from tests.apps.viewer_pipeline import build_viewer_session

fixture = build_viewer_session(Path({str(ws_dir)!r}))
mpr_art = fixture.mpr_artifacts[0]
payload = fixture.session.payload_documents[mpr_art.artifact_id]
digest = str(payload["pixel_blob_ref"]["sha256"])
pixel_blob = fixture.session.pixel_blobs[digest]
geometry = GeometryFrame(
    plane="AXIAL",
    frame_index=0,
    mpr_artifact_id=mpr_art.artifact_id,
    plane_coordinate_mm=0.0,
    source_volume_ref="vol-1",
)

try:
    mint_render_variant(
        session_id="test-session",
        geometry=geometry,
        wl_preset_id="CT_MEDIASTINUM",
        center=40.0,
        width=400.0,
        mpr_artifact=mpr_art,
        mpr_payload=payload,
        pixel_blob=pixel_blob,
        provenance_upstream=(mpr_art.provenance_ref,),
        code_commit_sha={head_sha!r},
    )
    raise SystemExit("mint_render_variant did not raise on dirty checkout")
except BuildIdentityError as exc:
    assert "dirty source" in str(exc)
'''
    res = run_in_checkout(checkout, code)
    assert res.returncode == 0, res.stderr


# ---------------------------------------------------------------------------
# R3-02 tests: dirtiness fails closed with typed open failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dirtiness_kind",
    [
        "ordinary",
        "staged",
        "assume_unchanged",
        "skip_worktree",
        "git_config_override",
    ],
)
def test_dirtiness_fails_closed_typed_open_and_resolver(
    tmp_path: Path, dirtiness_kind: str
) -> None:
    """Ordinary/staged/assume-unchanged/skip-worktree/git-config override fail closed (r3 R3-02).

    Every failure produces typed ViewerOpenError(SESSION_OPEN_FAILED) before any
    open state or provenance is minted.
    """

    checkout = materialize_clean_checkout(tmp_path / f"chk_{dirtiness_kind}")
    engine_file = checkout / "apps" / "viewer" / "engine.py"
    env_vars: dict[str, str] = {}

    if dirtiness_kind == "ordinary":
        engine_file.write_text(engine_file.read_text() + "\n# dirty ordinary\n")
    elif dirtiness_kind == "staged":
        engine_file.write_text(engine_file.read_text() + "\n# dirty staged\n")
        subprocess.run(
            ["git", "-C", str(checkout), "add", "apps/viewer/engine.py"], check=True
        )
    elif dirtiness_kind == "assume_unchanged":
        engine_file.write_text(engine_file.read_text() + "\n# dirty assume unchanged\n")
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "update-index",
                "--assume-unchanged",
                "apps/viewer/engine.py",
            ],
            check=True,
        )
    elif dirtiness_kind == "skip_worktree":
        engine_file.write_text(engine_file.read_text() + "\n# dirty skip worktree\n")
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "update-index",
                "--skip-worktree",
                "apps/viewer/engine.py",
            ],
            check=True,
        )
    elif dirtiness_kind == "git_config_override":
        (checkout / "apps" / "viewer" / "injected_runtime.py").write_text("x = 1\n")
        env_vars = {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "status.showUntrackedFiles",
            "GIT_CONFIG_VALUE_0": "no",
        }

    ws_dir = tmp_path / f"ws_{dirtiness_kind}"
    code = f'''
from pathlib import Path
from apps.viewer.build_identity import BuildIdentityError, resolve_build_identity
from apps.viewer.engine import ViewerEngine, ViewerOpenError
from apps.viewer.grammar import RejectionCode
from tests.apps.viewer_pipeline import build_viewer_session

try:
    resolve_build_identity()
    raise SystemExit("resolve_build_identity did not raise on dirty tree")
except BuildIdentityError:
    pass

fixture = build_viewer_session(Path({str(ws_dir)!r}))
engine = ViewerEngine(fixture.session)
try:
    engine.open()
    raise SystemExit("engine.open did not raise on dirty tree")
except ViewerOpenError as exc:
    assert exc.code is RejectionCode.SESSION_OPEN_FAILED
assert engine.new_provenance() == ()
'''
    res = run_in_checkout(checkout, code, env=env_vars if env_vars else None)
    assert res.returncode == 0, res.stderr


# ---------------------------------------------------------------------------
# R4-02 tests: repository-local git exclude cannot suppress untracked source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule_kind", "rel_path", "exclude_rule", "import_mod"),
    [
        (
            "exact_file",
            "apps/viewer/injected_runtime.py",
            "apps/viewer/injected_runtime.py\n",
            "apps.viewer.injected_runtime",
        ),
        (
            "ignored_directory",
            "apps/viewer/injected_dir/injected_module.py",
            "apps/viewer/injected_dir/\n",
            "apps.viewer.injected_dir.injected_module",
        ),
    ],
)
def test_git_exclude_suppression_fails_closed(
    tmp_path: Path,
    rule_kind: str,
    rel_path: str,
    exclude_rule: str,
    import_mod: str,
) -> None:
    """A repository-local git exclude rule cannot suppress untracked producing source (r4 R4-02).

    Exercises both an exact-file .git/info/exclude rule and an ignored directory containing
    the runtime module. Uses a disposable checkout with subprocess CWD and actual imports
    rooted there. Asserts:
    - Injected file exists, is absent from committed tree, imports, and is invisible to status.
    - Expected SHA is independently obtained via git rev-parse HEAD.
    - Resolver raises BuildIdentityError.
    - Fresh engine open raises typed ViewerOpenError(SESSION_OPEN_FAILED) BEFORE any state
      or provenance is minted.
    - Standalone mint with independently observed HEAD raises BuildIdentityError producing no
      render or provenance result.
    - Session provenance matches the pre-attempt snapshot.
    """
    checkout = materialize_clean_checkout(tmp_path / f"chk_{rule_kind}")
    expected_sha = checkout_head_commit(checkout)

    injected_file = checkout / rel_path
    injected_file.parent.mkdir(parents=True, exist_ok=True)
    injected_file.write_text("INJECTED_SENTINEL = 'untracked_exploit'\n")

    # Add rule to .git/info/exclude
    exclude_file = checkout / ".git" / "info" / "exclude"
    exclude_file.parent.mkdir(parents=True, exist_ok=True)
    with exclude_file.open("a", encoding="utf-8") as f:
        f.write(exclude_rule)

    ws_dir = tmp_path / f"ws_{rule_kind}"

    code = f'''
import importlib
import subprocess
from pathlib import Path
from apps.viewer.build_identity import BuildIdentityError, resolve_build_identity
from apps.viewer.engine import ViewerEngine, ViewerOpenError
from apps.viewer.frames import mint_render_variant, GeometryFrame
from apps.viewer.grammar import RejectionCode
from tests.apps.viewer_pipeline import build_viewer_session

checkout_root = Path.cwd()
injected_rel = {rel_path!r}
injected_full = checkout_root / injected_rel
expected_sha = {expected_sha!r}

# 1. Establish the injected file exists on disk
assert injected_full.is_file(), "injected file does not exist on disk"

# 2. Establish the injected file is absent from the committed tree
ls_tree = subprocess.run(
    ["git", "ls-tree", "HEAD", "--", injected_rel],
    capture_output=True,
    text=True,
    check=True,
)
assert ls_tree.stdout.strip() == "", "injected file must be absent from committed tree"

# 3. Establish the injected file imports from this checkout
mod = importlib.import_module({import_mod!r})
assert getattr(mod, "INJECTED_SENTINEL", None) == "untracked_exploit"

# 4. Establish the injected file is INVISIBLE to the pinned git status command
status_res = subprocess.run(
    [
        "git",
        "-c",
        "status.showUntrackedFiles=all",
        "status",
        "--porcelain",
        "-z",
        "--untracked-files=all",
    ],
    capture_output=True,
    text=False,
    check=True,
)
entries = [
    e.decode("utf-8", "surrogateescape")
    for e in status_res.stdout.split(b"\\x00")
    if e
]
status_paths = [e[3:] for e in entries if len(e) >= 4]
assert injected_rel not in status_paths, (
    f"injected file {{injected_rel}} must be invisible to git status"
)

# 5. Pre-attempt snapshot of fixture session provenance
fixture = build_viewer_session(Path({str(ws_dir)!r}))
initial_provenance = tuple(fixture.session.provenance)

# 6. Resolver raises BuildIdentityError
try:
    resolve_build_identity()
    raise SystemExit("resolve_build_identity did not raise BuildIdentityError")
except BuildIdentityError as exc:
    assert "dirty source" in str(exc)

# 7. Fresh engine open raises typed SESSION_OPEN_FAILED before open state or provenance minted
engine = ViewerEngine(fixture.session)
try:
    engine.open()
    raise SystemExit("engine.open did not raise ViewerOpenError")
except ViewerOpenError as exc:
    assert exc.code is RejectionCode.SESSION_OPEN_FAILED
assert engine.new_provenance() == ()

# 8. Standalone mint with independently observed HEAD raises BuildIdentityError producing no result
mpr_art = fixture.mpr_artifacts[0]
payload = fixture.session.payload_documents[mpr_art.artifact_id]
digest = str(payload["pixel_blob_ref"]["sha256"])
pixel_blob = fixture.session.pixel_blobs[digest]
geometry = GeometryFrame(
    plane="AXIAL",
    frame_index=0,
    mpr_artifact_id=mpr_art.artifact_id,
    plane_coordinate_mm=0.0,
    source_volume_ref="vol-1",
)
try:
    mint_render_variant(
        session_id="test-session",
        geometry=geometry,
        wl_preset_id="CT_MEDIASTINUM",
        center=40.0,
        width=400.0,
        mpr_artifact=mpr_art,
        mpr_payload=payload,
        pixel_blob=pixel_blob,
        provenance_upstream=(mpr_art.provenance_ref,),
        code_commit_sha=expected_sha,
    )
    raise SystemExit("mint_render_variant did not raise BuildIdentityError")
except BuildIdentityError as exc:
    assert "dirty source" in str(exc)

# 9. Provenance compared to the fixture's pre-attempt snapshot
assert tuple(fixture.session.provenance) == initial_provenance
'''
    res = run_in_checkout(checkout, code)
    assert res.returncode == 0, res.stderr


def test_cleanliness_allows_ordinary_caches_and_ignored_reports(tmp_path: Path) -> None:
    """Ordinary generated caches and ignored reports allowed; smuggled .py fails (r4 R4-02)."""
    checkout = materialize_clean_checkout(tmp_path / "chk_caches")
    expected_sha = checkout_head_commit(checkout)

    # 1. Ignored report in reports/
    (checkout / "reports" / "ignored_review.md").write_text("# Review note\n")
    exclude_file = checkout / ".git" / "info" / "exclude"
    exclude_file.parent.mkdir(parents=True, exist_ok=True)
    with exclude_file.open("a", encoding="utf-8") as f:
        f.write("reports/ignored_review.md\n")

    # 2. Ordinary bytecode cache file for a real tracked source (apps/viewer/engine.py)
    cache_dir = checkout / "apps" / "viewer" / "__pycache__"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "engine.cpython-312.pyc").write_bytes(b"\x00\x00\x00\x00")

    # 3. Tool cache file (.pytest_cache)
    pytest_cache_dir = checkout / "tests" / ".pytest_cache" / "v" / "cache"
    pytest_cache_dir.mkdir(parents=True, exist_ok=True)
    (pytest_cache_dir / "nodeids").write_text("[]\n")

    code = f'''
from pathlib import Path
from apps.viewer.build_identity import resolve_build_identity, verify_source_cleanliness

verify_source_cleanliness()
sha = resolve_build_identity()
assert sha == {expected_sha!r}
'''
    res = run_in_checkout(checkout, code)
    assert res.returncode == 0, res.stderr

    # 4. Now smuggle a .py source file inside __pycache__ -> must FAIL closed
    smuggled = checkout / "apps" / "viewer" / "__pycache__" / "smuggled_module.py"
    smuggled.write_text("x = 1\n")
    code_smuggled = '''
from apps.viewer.build_identity import BuildIdentityError, verify_source_cleanliness
try:
    verify_source_cleanliness()
    raise SystemExit("verify_source_cleanliness did not fail on smuggled .py in __pycache__")
except BuildIdentityError as exc:
    assert "dirty source" in str(exc)
'''
    res2 = run_in_checkout(checkout, code_smuggled)
    assert res2.returncode == 0, res2.stderr
    smuggled.unlink()

    # 5. Sourceless bytecode inside __pycache__ must FAIL closed (r5 R5-02)
    sourceless = cache_dir / "sourceless_mod.cpython-312.pyc"
    sourceless.write_bytes(b"\x00\x00\x00\x00")
    code_sourceless = '''
from apps.viewer.build_identity import BuildIdentityError, verify_source_cleanliness
try:
    verify_source_cleanliness()
    raise SystemExit("verify_source_cleanliness did not fail on sourceless bytecode in __pycache__")
except BuildIdentityError as exc:
    assert "dirty source" in str(exc)
'''
    res3 = run_in_checkout(checkout, code_sourceless)
    assert res3.returncode == 0, res3.stderr


# ---------------------------------------------------------------------------
# Engine-open fail-closed paths (typed error, zero provenance minted)
# ---------------------------------------------------------------------------


def test_open_fails_closed_on_missing_git_with_typed_error(tmp_path: Path) -> None:
    checkout = materialize_clean_checkout(tmp_path / "checkout")
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    ws_dir = tmp_path / "ws"
    code = f'''
from pathlib import Path
from apps.viewer.engine import ViewerEngine, ViewerOpenError
from apps.viewer.grammar import RejectionCode
from tests.apps.viewer_pipeline import build_viewer_session

fixture = build_viewer_session(Path({str(ws_dir)!r}))
engine = ViewerEngine(fixture.session)
try:
    engine.open()
    raise SystemExit(1)
except ViewerOpenError as exc:
    assert exc.code is RejectionCode.SESSION_OPEN_FAILED
    assert "git executable not found" in "; ".join(exc.problems)
assert engine.new_provenance() == ()
'''
    res = run_in_checkout(checkout, code, env={"PATH": str(empty_bin)})
    assert res.returncode == 0, res.stderr


def test_open_fails_closed_on_wrong_cwd_with_typed_error(tmp_path: Path) -> None:
    checkout = materialize_clean_checkout(tmp_path / "checkout")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    ws_dir = tmp_path / "ws"
    code = f'''
import os
from pathlib import Path
os.chdir({str(elsewhere)!r})
from apps.viewer.engine import ViewerEngine, ViewerOpenError
from apps.viewer.grammar import RejectionCode
from tests.apps.viewer_pipeline import build_viewer_session

fixture = build_viewer_session(Path({str(ws_dir)!r}))
engine = ViewerEngine(fixture.session)
try:
    engine.open()
    raise SystemExit(1)
except ViewerOpenError as exc:
    assert exc.code is RejectionCode.SESSION_OPEN_FAILED
    assert "wrong CWD" in "; ".join(exc.problems)
assert engine.new_provenance() == ()
'''
    res = run_in_checkout(checkout, code)
    assert res.returncode == 0, res.stderr


# ---------------------------------------------------------------------------
# The clean-checkout equality gate (all three call sites + HUMAN aggregate)
# ---------------------------------------------------------------------------


def test_minted_provenance_shas_equal_the_independent_head(tmp_path: Path) -> None:
    """F11.4 equality gate: session, frame, and HUMAN mint sites + zlib pin (r3 R3-01)."""

    checkout = materialize_clean_checkout(tmp_path / "checkout")
    expected_sha = checkout_head_commit(checkout)
    ws_dir = tmp_path / "ws"
    code = f'''
import json
from pathlib import Path
from apps.viewer.engine import ViewerEngine
from apps.viewer.png_encoder import ZLIB_VERSION_PIN
from core.provenance import NotApplicable
from tests.apps.test_viewer_build_identity import _deterministic_ulids
from tests.apps.viewer_pipeline import build_viewer_session

fixture = build_viewer_session(Path({str(ws_dir)!r}))
engine = ViewerEngine(
    fixture.session,
    id_factory=_deterministic_ulids(),
    now=lambda: "2026-09-07T00:00:00Z",
)
engine.open()
session_id = engine.session_id()

def minted_shas() -> set[str]:
    return {{
        record.environment.code_commit_sha
        for record in engine.new_provenance()
    }}

# Site 1 (session open) + site 2 (raw frame renders, all planes):
assert minted_shas() == {{{expected_sha!r}}}

# Site 2 again: a WL re-mint carries the same governed identity.
body = json.dumps(
    {{
        "action": "SET_WL_PRESET",
        "params": {{"preset_id": "CT_MEDIASTINUM"}},
        "client_nonce": "wl-identity-000000001",
        "view": {{"plane": "AXIAL", "frame_index": 0}},
        "lineage": {{"session_id": session_id, "parent_action_nonce": None}},
    }},
    sort_keys=True,
    separators=(",", ":"),
).encode()
engine.apply_action_bytes(body, path_session_id=session_id)
assert minted_shas() == {{{expected_sha!r}}}

# Site 3: the HUMAN seam aggregate environment SHA.
body_corr = json.dumps(
    {{
        "action": "REQUEST_CORRECTION",
        "params": {{
            "subject_ref": fixture.study_id,
            "proposed_grade": "GOOD",
            "justification_code": "HUMAN_EVIDENCE",
        }},
        "client_nonce": "correction-identity-00001",
        "view": {{"plane": "AXIAL", "frame_index": 0}},
        "lineage": {{"session_id": session_id, "parent_action_nonce": "wl-identity-000000001"}},
    }},
    sort_keys=True,
    separators=(",", ":"),
).encode()
engine.apply_action_bytes(body_corr, path_session_id=session_id)
human_records = [
    record
    for record in engine.new_provenance()
    if record.actor.actor_type.value == "HUMAN"
]
assert human_records, "the seam route minted no HUMAN aggregate record"
assert minted_shas() == {{{expected_sha!r}}}

# the zlib pin and the producing SHA are recorded together per record
frame_record = next(
    record
    for record in engine.new_provenance()
    if record.subject_ref.startswith("viewer_frame:")
)
parameters = frame_record.parameters_ref
assert not isinstance(parameters, NotApplicable)
assert ZLIB_VERSION_PIN in parameters.blob_ref
assert frame_record.environment.code_commit_sha == {expected_sha!r}
'''
    res = run_in_checkout(checkout, code)
    assert res.returncode == 0, res.stderr


def test_reopen_preserves_original_provenance_and_rejects_zero_sentinel(tmp_path: Path) -> None:
    """Persist/reopen byte-for-byte; a zero sentinel must FAIL verification (r3 R3-01)."""

    checkout = materialize_clean_checkout(tmp_path / "checkout")
    expected_sha = checkout_head_commit(checkout)
    ws_dir = tmp_path / "ws"
    code = f'''
from dataclasses import replace as dataclass_replace
from pathlib import Path
from apps.viewer.engine import ViewerEngine, ViewerOpenError
from apps.workstation.round_trip import RoundTripVerificationEngine
from apps.workstation.session_store import RestoredSession
from tests.apps.test_viewer_build_identity import _deterministic_ulids
from tests.apps.viewer_pipeline import build_viewer_session

fixture = build_viewer_session(Path({str(ws_dir)!r}))
engine = ViewerEngine(
    fixture.session,
    id_factory=_deterministic_ulids(),
    now=lambda: "2026-09-07T00:00:00Z",
)
engine.open()
before = {{
    record.provenance_id: record.environment.code_commit_sha
    for record in engine.new_provenance()
}}
assert before and set(before.values()) == {{{expected_sha!r}}}
engine.persist(fixture.store)
reopened = fixture.store.reopen()
report = RoundTripVerificationEngine(fixture.workspace).verify(reopened)
assert report.verdict == "PASS", report.to_dict()

# Reopen NEVER relabels old records with the current build SHA
reopened_engine = ViewerEngine(reopened)
reopened_engine.open()
after = {{
    record.provenance_id: record.environment.code_commit_sha
    for record in reopened.provenance
    if record.provenance_id in before
}}
assert after == before

# A zero sentinel injected into a viewer-produced persisted record must
# FAIL verification (fail-closed, typed open failure, no fabrication).
victim = next(
    record
    for record in reopened.provenance
    if record.subject_ref.startswith("viewer_")
)
zeroed = dataclass_replace(
    victim,
    environment=dataclass_replace(
        victim.environment, code_commit_sha="0" * 40
    ),
)
tampered = RestoredSession(
    paths=reopened.paths,
    study=reopened.study,
    series=reopened.series,
    volumes=reopened.volumes,
    artifacts=reopened.artifacts,
    payload_documents=dict(reopened.payload_documents),
    provenance=tuple(
        zeroed if record.provenance_id == victim.provenance_id else record
        for record in reopened.provenance
    ),
    log=reopened.log,
    skipped_instances=reopened.skipped_instances,
    projections=reopened.projections,
    session_digest=reopened.session_digest,
    pixel_blobs=dict(reopened.pixel_blobs),
)
tampered_engine = ViewerEngine(tampered)
try:
    tampered_engine.open()
    raise SystemExit("tampered engine did not fail on zero sentinel")
except ViewerOpenError as exc:
    assert "forty zeros" in "; ".join(exc.problems)
'''
    res = run_in_checkout(checkout, code)
    assert res.returncode == 0, res.stderr


# ---------------------------------------------------------------------------
# R5-01 tests: nested Git repository directory summary fails closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leaf_ext", [".py", ".pyw", ".pyi"])
def test_nested_git_repository_directory_summary_fails_closed(
    tmp_path: Path, leaf_ext: str
) -> None:
    """A nested git repository returned as a directory summary fails closed (r5 R5-01).

    git ls-files --others -z returns a nested repository as a directory summary
    ('apps/viewer/__pycache__/nested/'). Under R5-01, directory-summary entries under
    runtime package roots are refused BEFORE any cache exclusion.
    Asserts:
    - Inner git repo created at apps/viewer/__pycache__/nested with committed sentinel.
    - Outer HEAD, tree absence, status directory summary ('?? apps/viewer/__pycache__/nested/'),
      and ls-files directory summary ('apps/viewer/__pycache__/nested/') independently established.
    - Sentinel is imported normally from checkout/CWD (for .py).
    - Resolver raises BuildIdentityError.
    - Fresh engine open raises typed ViewerOpenError(SESSION_OPEN_FAILED) with empty new_provenance.
    - All public state accessors are refused typed.
    - Standalone mint with independently observed HEAD raises BuildIdentityError with no result.
    - Input provenance matches the pre-attempt snapshot.
    - Expected values NEVER via the product resolver nor by duplicating its cache predicate.
    """
    clean_ext = leaf_ext.lstrip(".")
    checkout = materialize_clean_checkout(tmp_path / f"chk_nested_{clean_ext}")
    expected_sha = checkout_head_commit(checkout)

    nested_dir = checkout / "apps" / "viewer" / "__pycache__" / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(nested_dir)], check=True)
    sentinel_name = f"injected_runtime{leaf_ext}"
    sentinel_file = nested_dir / sentinel_name
    sentinel_file.write_text("INJECTED_SENTINEL = 'nested_exploit'\n")
    subprocess.run(["git", "-C", str(nested_dir), "add", sentinel_name], check=True)
    subprocess.run(
        ["git", "-C", str(nested_dir), "commit", "-m", "nested commit"],
        check=True,
    )

    ws_dir = tmp_path / f"ws_nested_{clean_ext}"

    code = f'''
import importlib
import subprocess
from pathlib import Path
from apps.viewer.build_identity import BuildIdentityError, resolve_build_identity
from apps.viewer.engine import ViewerEngine, ViewerOpenError
from apps.viewer.frames import mint_render_variant, GeometryFrame
from apps.viewer.grammar import RejectionCode
from tests.apps.viewer_pipeline import build_viewer_session

checkout_root = Path.cwd()
nested_rel = "apps/viewer/__pycache__/nested"
expected_sha = {expected_sha!r}
leaf_ext = {leaf_ext!r}

# 1. Independently prove outer tree: sentinel is absent from outer committed tree
ls_tree = subprocess.run(
    ["git", "ls-tree", "HEAD", "--", nested_rel],
    capture_output=True,
    text=True,
    check=True,
)
assert ls_tree.stdout.strip() == "", "nested repo must be absent from outer committed tree"

# 2. Independently prove outer status contains the directory summary
status_res = subprocess.run(
    [
        "git",
        "-c",
        "status.showUntrackedFiles=all",
        "status",
        "--porcelain",
        "-z",
        "--untracked-files=all",
    ],
    capture_output=True,
    text=False,
    check=True,
)
entries = [
    e.decode("utf-8", "surrogateescape")
    for e in status_res.stdout.split(b"\\x00")
    if e
]
status_paths = [e[3:] for e in entries if len(e) >= 4]
assert not any(p.startswith("apps/viewer/__pycache__/nested") for p in status_paths), (
    f"nested repo must be hidden from pinned status, got: {{status_paths}}"
)

# 3. Independently record the directory summary from ignore-independent enumeration
others_res = subprocess.run(
    [
        "git",
        "ls-files",
        "--others",
        "-z",
        "--",
        "apps/",
    ],
    capture_output=True,
    text=False,
    check=True,
)
others_entries = [
    e.decode("utf-8", "surrogateescape")
    for e in others_res.stdout.split(b"\\x00")
    if e
]
assert any(e.startswith("apps/viewer/__pycache__/nested") for e in others_entries), (
    f"directory summary must be returned by git ls-files --others, got: {{others_entries}}"
)

# 4. If .py, verify normal import from checkout/CWD
if leaf_ext == ".py":
    mod = importlib.import_module("apps.viewer.__pycache__.nested.injected_runtime")
    assert getattr(mod, "INJECTED_SENTINEL", None) == "nested_exploit"

# 5. Pre-attempt snapshot of fixture session provenance
fixture = build_viewer_session(Path({str(ws_dir)!r}))
initial_provenance = tuple(fixture.session.provenance)

# 6. Resolver raises BuildIdentityError
try:
    resolve_build_identity()
    raise SystemExit("resolve_build_identity did not raise BuildIdentityError")
except BuildIdentityError as exc:
    assert "dirty source" in str(exc) or "untracked directory summary" in str(exc)

# 7. Fresh engine open raises typed SESSION_OPEN_FAILED, zero new provenance,
# all public state accessors refused typed
engine = ViewerEngine(fixture.session)
try:
    engine.open()
    raise SystemExit("engine.open did not raise ViewerOpenError")
except ViewerOpenError as exc:
    assert exc.code is RejectionCode.SESSION_OPEN_FAILED
assert engine.new_provenance() == ()

for accessor, call in [
    ("session_id", lambda: engine.session_id()),
    ("session_document", lambda: engine.session_document()),
    ("state_document", lambda: engine.state_document()),
    ("frames_document", lambda: engine.frames_document("AXIAL")),
    ("qc_panel_document", lambda: engine.qc_panel_document()),
    ("png_for_frame", lambda: engine.png_for_frame("dummy-frame-id")),
    ("apply_action_bytes", lambda: engine.apply_action_bytes(b"{{}}", path_session_id="dummy")),
]:
    try:
        call()
        raise SystemExit(f"accessor {{accessor}} did not raise ViewerOpenError")
    except ViewerOpenError as exc:
        assert exc.code is RejectionCode.SESSION_OPEN_FAILED

# 8. Standalone mint with independently observed HEAD raises BuildIdentityError with no result
mpr_art = fixture.mpr_artifacts[0]
payload = fixture.session.payload_documents[mpr_art.artifact_id]
digest = str(payload["pixel_blob_ref"]["sha256"])
pixel_blob = fixture.session.pixel_blobs[digest]
geometry = GeometryFrame(
    plane="AXIAL",
    frame_index=0,
    mpr_artifact_id=mpr_art.artifact_id,
    plane_coordinate_mm=0.0,
    source_volume_ref="vol-1",
)
try:
    mint_render_variant(
        session_id="test-session",
        geometry=geometry,
        wl_preset_id="CT_MEDIASTINUM",
        center=40.0,
        width=400.0,
        mpr_artifact=mpr_art,
        mpr_payload=payload,
        pixel_blob=pixel_blob,
        provenance_upstream=(mpr_art.provenance_ref,),
        code_commit_sha=expected_sha,
    )
    raise SystemExit("mint_render_variant did not raise BuildIdentityError")
except BuildIdentityError as exc:
    assert "dirty source" in str(exc)

# 9. Input provenance unchanged vs pre-attempt snapshot
assert tuple(fixture.session.provenance) == initial_provenance
'''
    res = run_in_checkout(checkout, code)
    assert res.returncode == 0, res.stderr


# ---------------------------------------------------------------------------
# R5-02 tests: narrow file-only cache exemptions (no blanket suffix exemptions)
# ---------------------------------------------------------------------------


def test_sourceless_bytecode_in_runtime_root_fails_closed(tmp_path: Path) -> None:
    """Sourceless bytecode in runtime package root fails closed (r5 R5-02).

    A top-level package injected_runtime.pyc (compiled via stdlib py_compile with
    source deleted) imports normally and is hidden from pinned status by Git's
    bytecode ignore. Under R5-02, blanket bytecode exemptions are replaced with narrow
    file-only patterns requiring conventional __pycache__ location and an identifiable
    corresponding tracked source in git.
    Asserts:
    - apps/viewer/injected_runtime.pyc created via py_compile and source deleted.
    - Imported normally from checkout/CWD.
    - Absent committed-tree membership and hidden pinned status independently established.
    - Resolver raises BuildIdentityError.
    - Fresh engine open raises typed ViewerOpenError(SESSION_OPEN_FAILED) with zero new provenance.
    - All public state accessors are refused typed.
    - Standalone mint raises BuildIdentityError with no result.
    - Input provenance matches pre-attempt snapshot.
    - Does NOT use _is_independently_dirty as expected oracle.
    """
    import py_compile

    checkout = materialize_clean_checkout(tmp_path / "chk_sourceless_pyc")
    expected_sha = checkout_head_commit(checkout)

    # 1. Create temporary source, compile to apps/viewer/injected_runtime.pyc, remove source
    temp_src = checkout / "apps" / "viewer" / "_temp_injected_source.py"
    temp_src.write_text("INJECTED_SENTINEL = 'sourceless_exploit'\n")
    pyc_dest = checkout / "apps" / "viewer" / "injected_runtime.pyc"
    py_compile.compile(str(temp_src), cfile=str(pyc_dest), doraise=True)
    temp_src.unlink()
    assert pyc_dest.is_file(), "compiled pyc must exist"
    assert not temp_src.exists(), "temp source must be deleted"

    ws_dir = tmp_path / "ws_sourceless_pyc"

    code = f'''
import importlib
import subprocess
from pathlib import Path
from apps.viewer.build_identity import BuildIdentityError, resolve_build_identity
from apps.viewer.engine import ViewerEngine, ViewerOpenError
from apps.viewer.frames import mint_render_variant, GeometryFrame
from apps.viewer.grammar import RejectionCode
from tests.apps.viewer_pipeline import build_viewer_session

checkout_root = Path.cwd()
pyc_rel = "apps/viewer/injected_runtime.pyc"
expected_sha = {expected_sha!r}

# 1. Establish normal import from checkout/CWD
mod = importlib.import_module("apps.viewer.injected_runtime")
assert getattr(mod, "INJECTED_SENTINEL", None) == "sourceless_exploit"

# 2. Independently establish absent committed-tree membership
ls_tree = subprocess.run(
    ["git", "ls-tree", "HEAD", "--", pyc_rel],
    capture_output=True,
    text=True,
    check=True,
)
assert ls_tree.stdout.strip() == "", "injected pyc must be absent from committed tree"

# 3. Independently establish hidden pinned status (hidden by *.py[cod] in .gitignore)
status_res = subprocess.run(
    [
        "git",
        "-c",
        "status.showUntrackedFiles=all",
        "status",
        "--porcelain",
        "-z",
        "--untracked-files=all",
    ],
    capture_output=True,
    text=False,
    check=True,
)
entries = [
    e.decode("utf-8", "surrogateescape")
    for e in status_res.stdout.split(b"\\x00")
    if e
]
status_paths = [e[3:] for e in entries if len(e) >= 4]
assert pyc_rel not in status_paths, f"injected pyc must be hidden from status: {{status_paths}}"

# 4. Pre-attempt snapshot of fixture session provenance
fixture = build_viewer_session(Path({str(ws_dir)!r}))
initial_provenance = tuple(fixture.session.provenance)

# 5. Resolver raises BuildIdentityError
try:
    resolve_build_identity()
    raise SystemExit("resolve_build_identity did not raise on sourceless bytecode")
except BuildIdentityError as exc:
    assert "dirty source" in str(exc)

# 6. Fresh engine open raises typed SESSION_OPEN_FAILED, zero new provenance,
# all public state accessors refused typed
engine = ViewerEngine(fixture.session)
try:
    engine.open()
    raise SystemExit("engine.open did not raise ViewerOpenError")
except ViewerOpenError as exc:
    assert exc.code is RejectionCode.SESSION_OPEN_FAILED
assert engine.new_provenance() == ()

for accessor, call in [
    ("session_id", lambda: engine.session_id()),
    ("session_document", lambda: engine.session_document()),
    ("state_document", lambda: engine.state_document()),
    ("frames_document", lambda: engine.frames_document("AXIAL")),
    ("qc_panel_document", lambda: engine.qc_panel_document()),
    ("png_for_frame", lambda: engine.png_for_frame("dummy-frame-id")),
    ("apply_action_bytes", lambda: engine.apply_action_bytes(b"{{}}", path_session_id="dummy")),
]:
    try:
        call()
        raise SystemExit(f"accessor {{accessor}} did not raise ViewerOpenError")
    except ViewerOpenError as exc:
        assert exc.code is RejectionCode.SESSION_OPEN_FAILED

# 7. Standalone mint with independently observed HEAD raises BuildIdentityError with no result
mpr_art = fixture.mpr_artifacts[0]
payload = fixture.session.payload_documents[mpr_art.artifact_id]
digest = str(payload["pixel_blob_ref"]["sha256"])
pixel_blob = fixture.session.pixel_blobs[digest]
geometry = GeometryFrame(
    plane="AXIAL",
    frame_index=0,
    mpr_artifact_id=mpr_art.artifact_id,
    plane_coordinate_mm=0.0,
    source_volume_ref="vol-1",
)
try:
    mint_render_variant(
        session_id="test-session",
        geometry=geometry,
        wl_preset_id="CT_MEDIASTINUM",
        center=40.0,
        width=400.0,
        mpr_artifact=mpr_art,
        mpr_payload=payload,
        pixel_blob=pixel_blob,
        provenance_upstream=(mpr_art.provenance_ref,),
        code_commit_sha=expected_sha,
    )
    raise SystemExit("mint_render_variant did not raise BuildIdentityError")
except BuildIdentityError as exc:
    assert "dirty source" in str(exc)

# 8. Input provenance unchanged vs pre-attempt snapshot
assert tuple(fixture.session.provenance) == initial_provenance
'''
    res = run_in_checkout(checkout, code)
    assert res.returncode == 0, res.stderr


def test_direct_cache_classification_controls(tmp_path: Path) -> None:
    """Direct negative and positive cache-classification controls (r5 R5-02).

    Negative controls:
    - Native extensions (.pyd, .so, .dylib, .dll) rejected anywhere.
    - Executable leaves masquerading as cache content (.sh, .exe, .bin, .py, .pyw, .pyi) rejected.
    - Sourceless bytecode in __pycache__ (no tracked source) rejected.
    - Bytecode outside __pycache__ rejected.
    - Directory summaries rejected.

    Positive controls:
    - Real interpreter cache files in __pycache__/ with tracked source accepted.
    - Real tool cache artifacts in designated tool cache directories accepted.
    """
    from apps.viewer.build_identity import _is_allowed_cache_artifact

    checkout = materialize_clean_checkout(tmp_path / "chk_direct_cache")
    tracked_source = "apps/viewer/engine.py"
    assert (checkout / tracked_source).is_file()

    # Tracked files set from checkout
    tracked_files = frozenset({
        "apps/viewer/engine.py",
        "apps/viewer/build_identity.py",
        "core/enums.py",
    })

    # Negative controls — native extensions (no native-code execution claimed)
    for ext in (".pyd", ".so", ".dylib", ".dll"):
        assert not _is_allowed_cache_artifact(
            f"apps/viewer/extension{ext}", root=checkout, tracked_files=tracked_files
        )
        assert not _is_allowed_cache_artifact(
            f"apps/viewer/__pycache__/extension{ext}", root=checkout, tracked_files=tracked_files
        )
        assert not _is_allowed_cache_artifact(
            f"tests/.pytest_cache/extension{ext}", root=checkout, tracked_files=tracked_files
        )

    # Negative controls — executable leaves masquerading as cache content
    for exec_leaf in (
        "apps/viewer/__pycache__/script.sh",
        "tests/.pytest_cache/script.sh",
        "tests/.pytest_cache/run.exe",
        "tests/.pytest_cache/binary.bin",
        "apps/viewer/__pycache__/script.py",
        "apps/viewer/__pycache__/script.pyw",
        "apps/viewer/__pycache__/script.pyi",
        "tests/.pytest_cache/test.py",
        "tests/.pytest_cache/test.pyw",
        "tests/.pytest_cache/test.pyi",
    ):
        assert not _is_allowed_cache_artifact(
            exec_leaf, root=checkout, tracked_files=tracked_files
        )

    # Negative control — sourceless bytecode in __pycache__ (no tracked source)
    assert not _is_allowed_cache_artifact(
        "apps/viewer/__pycache__/sourceless_mod.cpython-312.pyc",
        root=checkout,
        tracked_files=tracked_files,
    )

    # Negative control — bytecode in runtime directory outside __pycache__
    assert not _is_allowed_cache_artifact(
        "apps/viewer/engine.pyc", root=checkout, tracked_files=tracked_files
    )

    # Negative control — directory summaries (R5-01)
    nested_dir = checkout / "apps" / "viewer" / "__pycache__" / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    assert not _is_allowed_cache_artifact(
        "apps/viewer/__pycache__/nested/", root=checkout, tracked_files=tracked_files
    )
    assert not _is_allowed_cache_artifact(
        "apps/viewer/__pycache__/nested", root=checkout, tracked_files=tracked_files
    )

    # Positive controls — real interpreter cache in conventional location with tracked source
    assert _is_allowed_cache_artifact(
        "apps/viewer/__pycache__/engine.cpython-312.pyc",
        root=checkout,
        tracked_files=tracked_files,
    )
    assert _is_allowed_cache_artifact(
        "apps/viewer/__pycache__/engine.cpython-312.opt-1.pyc",
        root=checkout,
        tracked_files=tracked_files,
    )
    assert _is_allowed_cache_artifact(
        "core/__pycache__/enums.cpython-312.pyc",
        root=checkout,
        tracked_files=tracked_files,
    )

    # Positive controls — real tool cache artifacts
    for tool_path in (
        "tests/.pytest_cache/v/cache/nodeids",
        "tests/.pytest_cache/v/cache/lastfailed",
        "tests/.pytest_cache/CACHEDIR.TAG",
        "tests/.pytest_cache/README.md",
        ".mypy_cache/3.12/cache.db",
        ".mypy_cache/3.12/item.meta.json",
        ".ruff_cache/0.16.5/14844735119671959515",
    ):
        assert _is_allowed_cache_artifact(
            tool_path, root=checkout, tracked_files=tracked_files
        )
