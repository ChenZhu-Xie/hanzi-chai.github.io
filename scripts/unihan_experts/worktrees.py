"""Safe detached worktree provisioning for pinned historical experts."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .models import ExpertManifest


def _git(
    repo: Path,
    *args: str,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=text,
    )


def resolve_commit(repo: Path, revision: str) -> str:
    result = _git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}", check=False)
    if result.returncode != 0:
        raise ValueError(f"cannot resolve commit {revision!r}: {result.stderr.strip()}")
    resolved = result.stdout.strip().lower()
    if len(resolved) != 40:
        raise ValueError(f"Git returned a non-full commit ID for {revision!r}")
    return resolved


def source_tree_hash(repo: Path, commit: str) -> str:
    resolved = resolve_commit(repo, commit)
    result = _git(
        repo,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        resolved,
        text=False,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _registered_worktrees(repo: Path) -> dict[str, dict[str, str]]:
    output = _git(repo, "worktree", "list", "--porcelain").stdout
    records: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}
    for line in [*output.splitlines(), ""]:
        if not line:
            if current:
                path = str(Path(current["worktree"]).resolve()).casefold()
                records[path] = current
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    return records


def _assert_clean_detached(path: Path, expected_commit: str) -> None:
    actual = resolve_commit(path, "HEAD")
    if actual != expected_commit:
        raise ValueError(
            f"historical expert worktree is at wrong commit: {actual} != {expected_commit}"
        )
    branch = _git(path, "symbolic-ref", "-q", "HEAD", check=False)
    if branch.returncode == 0:
        raise ValueError(
            f"historical expert worktree must be detached, found {branch.stdout.strip()}"
        )
    dirty = _git(path, "status", "--porcelain").stdout
    if dirty:
        raise ValueError(f"historical expert worktree is dirty: {path}")


def ensure_expert_worktree(
    repo: Path, root: Path, manifest: ExpertManifest
) -> Path:
    repo = repo.resolve()
    root = root.resolve()
    expected = resolve_commit(repo, manifest.commit)
    target = root / manifest.expert_id
    target_key = str(target.resolve()).casefold()
    registered = _registered_worktrees(repo)

    if target_key in registered:
        _assert_clean_detached(target, expected)
        return target
    if target.exists():
        raise ValueError(
            f"expert path exists but is not a registered Git worktree: {target}"
        )

    root.mkdir(parents=True, exist_ok=True)
    result = _git(
        repo,
        "worktree",
        "add",
        "--detach",
        str(target),
        expected,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot provision expert worktree {target}: {result.stderr.strip()}"
        )
    _assert_clean_detached(target, expected)
    return target
