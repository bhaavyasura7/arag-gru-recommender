"""Guard the repo-integrity invariants from the spec.

1. Every Python module under ``src/`` must be tracked by git (so nothing silently
   ships untracked, e.g. after a bad ``.gitignore`` rule).
2. The data package is named ``dataprep``, not ``data`` (a bare ``data/`` ignore
   rule would otherwise swallow it).
"""
import subprocess
from pathlib import Path

from arag.config import REPO_ROOT


def _git_tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return set(out.stdout.splitlines())


def test_every_src_module_is_tracked_by_git():
    tracked = _git_tracked_files()
    src = REPO_ROOT / "src"
    untracked = [
        str(p.relative_to(REPO_ROOT))
        for p in src.rglob("*.py")
        if str(p.relative_to(REPO_ROOT)) not in tracked
    ]
    assert not untracked, f"untracked modules under src/: {untracked}"


def test_dataprep_package_exists():
    assert (REPO_ROOT / "src" / "arag" / "dataprep").is_dir()
    assert not (REPO_ROOT / "src" / "arag" / "data").exists()
