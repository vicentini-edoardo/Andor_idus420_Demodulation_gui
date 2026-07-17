from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from idus420_gui.gui.main_window import _run_git_update


def test_git_update_creates_tracking_branch_and_returns_changelog() -> None:
    responses = {
        ("git", "rev-parse", "HEAD"): [
            SimpleNamespace(stdout="old-head\n", stderr="", returncode=0),
            SimpleNamespace(stdout="new-head\n", stderr="", returncode=0),
        ],
        ("git", "fetch", "--prune", "origin"): [
            SimpleNamespace(stdout="", stderr="", returncode=0),
        ],
        ("git", "branch", "--show-current"): [
            SimpleNamespace(stdout="feature\n", stderr="", returncode=0),
        ],
        ("git", "checkout", "main"): [
            SimpleNamespace(stdout="", stderr="missing branch", returncode=1),
        ],
        ("git", "checkout", "-b", "main", "--track", "origin/main"): [
            SimpleNamespace(stdout="Switched to a new branch 'main'\n", stderr="", returncode=0),
        ],
        ("git", "branch", "--set-upstream-to", "origin/main", "main"): [
            SimpleNamespace(stdout="", stderr="", returncode=0),
        ],
        ("git", "pull", "--ff-only"): [
            SimpleNamespace(stdout="Updating old-head..new-head\n", stderr="", returncode=0),
        ],
        ("git", "log", "old-head..new-head", "--pretty=format:• %s", "--no-merges"): [
            SimpleNamespace(stdout="• Fix updater\n", stderr="", returncode=0),
        ],
    }
    calls: list[tuple[str, ...]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        key = tuple(cmd)
        calls.append(key)
        return responses[key].pop(0)

    changelog, changed = _run_git_update(Path("/repo"), "main", run=fake_run)

    assert changed is True
    assert changelog == "• Fix updater"
    assert ("git", "checkout", "-b", "main", "--track", "origin/main") in calls
    assert ("git", "pull", "--ff-only") in calls


def test_git_update_reports_no_change() -> None:
    responses = {
        ("git", "rev-parse", "HEAD"): [
            SimpleNamespace(stdout="same-head\n", stderr="", returncode=0),
            SimpleNamespace(stdout="same-head\n", stderr="", returncode=0),
        ],
        ("git", "fetch", "--prune", "origin"): [
            SimpleNamespace(stdout="", stderr="", returncode=0),
        ],
        ("git", "branch", "--show-current"): [
            SimpleNamespace(stdout="main\n", stderr="", returncode=0),
        ],
        ("git", "branch", "--set-upstream-to", "origin/main", "main"): [
            SimpleNamespace(stdout="", stderr="", returncode=0),
        ],
        ("git", "pull", "--ff-only"): [
            SimpleNamespace(stdout="Already up to date.\n", stderr="", returncode=0),
        ],
    }

    def fake_run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        return responses[tuple(cmd)].pop(0)

    changelog, changed = _run_git_update(Path("/repo"), "main", run=fake_run)

    assert changed is False
    assert changelog == "Already up to date."
