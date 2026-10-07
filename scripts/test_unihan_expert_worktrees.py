import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from unihan_experts.models import ExpertManifest
from unihan_experts.process import run_command
from unihan_experts.worktrees import (
    ensure_expert_worktree,
    resolve_commit,
    source_tree_hash,
)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def manifest(commit):
    return ExpertManifest(
        expert_id="test-expert",
        schema_version=1,
        commit=commit,
        adapter="classic-v1",
        decoder="legacy",
        config={},
        rule_artifacts=(),
        parents=(),
        declared_capabilities=(),
        resource_class="small",
    )


class TemporaryRepository:
    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        git(self.repo, "config", "user.name", "Test User")
        git(self.repo, "config", "user.email", "test@example.invalid")
        (self.repo / "model.txt").write_text("one\n", encoding="utf-8")
        git(self.repo, "add", "model.txt")
        git(self.repo, "commit", "-q", "-m", "one")
        self.first = git(self.repo, "rev-parse", "HEAD")
        (self.repo / "model.txt").write_text("two\n", encoding="utf-8")
        git(self.repo, "commit", "-q", "-am", "two")
        self.second = git(self.repo, "rev-parse", "HEAD")
        return self

    def __exit__(self, exc_type, exc, traceback):
        subprocess.run(
            ["git", "-C", str(self.repo), "worktree", "prune"],
            check=False,
            capture_output=True,
        )
        self.temp.cleanup()


class WorktreeTests(unittest.TestCase):
    def test_resolve_commit_returns_full_sha(self):
        with TemporaryRepository() as fixture:
            self.assertEqual(resolve_commit(fixture.repo, fixture.first[:8]), fixture.first)

    def test_source_tree_hash_changes_with_tracked_source(self):
        with TemporaryRepository() as fixture:
            self.assertNotEqual(
                source_tree_hash(fixture.repo, fixture.first),
                source_tree_hash(fixture.repo, fixture.second),
            )

    def test_provisions_detached_worktree_at_exact_commit_and_is_idempotent(self):
        with TemporaryRepository() as fixture:
            root = fixture.root / "experts"
            path = ensure_expert_worktree(
                fixture.repo, root, manifest(fixture.first[:8])
            )
            self.assertEqual(git(path, "rev-parse", "HEAD"), fixture.first)
            self.assertEqual(git(path, "rev-parse", "--abbrev-ref", "HEAD"), "HEAD")
            self.assertEqual(
                ensure_expert_worktree(
                    fixture.repo, root, manifest(fixture.first[:8])
                ),
                path,
            )

    def test_wrong_registered_commit_is_rejected_instead_of_reset(self):
        with TemporaryRepository() as fixture:
            root = fixture.root / "experts"
            root.mkdir()
            path = root / "test-expert"
            git(fixture.repo, "worktree", "add", "--detach", str(path), fixture.first)
            with self.assertRaisesRegex(ValueError, "wrong commit"):
                ensure_expert_worktree(
                    fixture.repo, root, manifest(fixture.second)
                )
            self.assertEqual(git(path, "rev-parse", "HEAD"), fixture.first)

    def test_dirty_historical_worktree_is_rejected(self):
        with TemporaryRepository() as fixture:
            root = fixture.root / "experts"
            path = ensure_expert_worktree(
                fixture.repo, root, manifest(fixture.first)
            )
            (path / "model.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dirty"):
                ensure_expert_worktree(
                    fixture.repo, root, manifest(fixture.first)
                )

    def test_unrelated_worktree_is_never_removed(self):
        with TemporaryRepository() as fixture:
            unrelated = fixture.root / "unrelated"
            git(
                fixture.repo,
                "worktree",
                "add",
                "--detach",
                str(unrelated),
                fixture.second,
            )
            ensure_expert_worktree(
                fixture.repo, fixture.root / "experts", manifest(fixture.first)
            )
            self.assertTrue(unrelated.is_dir())
            self.assertEqual(git(unrelated, "rev-parse", "HEAD"), fixture.second)


class CommandTests(unittest.TestCase):
    def test_nonzero_command_retains_stdout_stderr(self):
        result = run_command(
            [
                sys.executable,
                "-c",
                "import sys; print('out'); print('err', file=sys.stderr); raise SystemExit(7)",
            ],
            timeout_seconds=5,
        )
        self.assertEqual(result.exit_code, 7)
        self.assertIn("out", result.stdout)
        self.assertIn("err", result.stderr)
        self.assertFalse(result.timed_out)

    def test_timeout_is_explicit(self):
        result = run_command(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout_seconds=0.1,
        )
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)
        self.assertGreaterEqual(result.wall_seconds, 0.09)

    def test_resource_fields_are_observed_or_explained(self):
        result = run_command(
            [sys.executable, "-c", "print('ok')"], timeout_seconds=5
        )
        self.assertTrue(
            result.peak_memory_bytes is not None
            or result.resource_unavailable_reason is not None
        )
        self.assertTrue(
            result.cpu_seconds is not None
            or result.resource_unavailable_reason is not None
        )


if __name__ == "__main__":
    unittest.main()
