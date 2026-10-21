"""Git-blob verification is a review path, not permission to collect dirty runs."""

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/phase3"))
from validate import committed_provenance


def test_committed_review_ignores_later_working_files_but_hashes_git_blobs(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path).decode().strip()

    git("init", "-q")
    git("config", "user.name", "Validation unit test")
    git("config", "user.email", "test@example.invalid")
    (tmp_path / "src").mkdir()
    original = b"# original\n"
    (tmp_path / "src/example.py").write_bytes(original)
    (tmp_path / "pyproject.toml").write_text("# fixture\n")
    git("add", "src", "pyproject.toml")
    git("commit", "-qm", "Unit fixture only")
    commit = git("rev-parse", "HEAD")
    (tmp_path / "src/example.py").write_text("# uncommitted replacement\n")
    result = committed_provenance(commit, tmp_path)
    assert result["source_commit"] == commit
    assert result["source_file_sha256"]["src/example.py"] == hashlib.sha256(original).hexdigest()
    assert set(result["source_file_sha256"]) == {"src/example.py", "pyproject.toml"}


@pytest.mark.parametrize("commit", ["HEAD", "main", "--help", "a" * 39, "A" * 40])
def test_committed_review_rejects_ambiguous_or_option_like_refs(commit):
    with pytest.raises(ValueError, match="full measured"):
        committed_provenance(commit)
