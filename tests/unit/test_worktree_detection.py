"""Unit tests for Git worktree detection utilities."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import subprocess

from chunkhound.utils.worktree_detection import (
    WorktreeInfo,
    detect_worktree_info,
    compute_worktree_id,
    get_git_head_ref,
    compute_changed_files_via_git,
    fallback_compute_changed_files,
    _matches_patterns
)


class TestDetectWorktreeInfo:
    """Tests for detect_worktree_info function."""

    def test_detect_main_worktree(self, tmp_path):
        """Test detection of main Git worktree (.git is directory)."""
        # Create main worktree structure
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        info = detect_worktree_info(tmp_path)

        assert info.is_main is True
        assert info.is_linked is False
        assert info.main_worktree_path is None
        assert info.git_dir == git_dir
        assert len(info.worktree_id) == 16
        assert info.git_repo_path == git_dir

    def test_detect_linked_worktree(self, tmp_path):
        """Test detection of linked Git worktree (.git is file)."""
        # Create main worktree
        main_worktree = tmp_path / "main"
        main_worktree.mkdir()
        main_git_dir = main_worktree / ".git"
        main_git_dir.mkdir()
        (main_git_dir / "worktrees").mkdir()

        # Create linked worktree
        linked_worktree = tmp_path / "feature"
        linked_worktree.mkdir()
        linked_git_dir = main_git_dir / "worktrees" / "feature"
        linked_git_dir.mkdir()

        # Write .git file pointing to linked git dir
        git_file = linked_worktree / ".git"
        git_file.write_text(f"gitdir: {linked_git_dir}")

        info = detect_worktree_info(linked_worktree)

        assert info.is_main is False
        assert info.is_linked is True
        assert info.main_worktree_path == main_worktree
        assert info.git_dir == linked_git_dir
        assert len(info.worktree_id) == 16
        assert info.git_repo_path == main_git_dir

    def test_detect_non_git_directory(self, tmp_path):
        """Test detection of non-Git directory."""
        info = detect_worktree_info(tmp_path)

        assert info.is_main is False
        assert info.is_linked is False
        assert info.main_worktree_path is None
        assert info.git_dir is None
        assert len(info.worktree_id) == 16
        assert info.git_repo_path is None

    def test_detect_malformed_git_file(self, tmp_path):
        """Test handling of malformed .git file."""
        # Create .git file with invalid content
        git_file = tmp_path / ".git"
        git_file.write_text("invalid content")

        info = detect_worktree_info(tmp_path)

        # Should treat as non-Git directory
        assert info.is_main is False
        assert info.is_linked is False


class TestComputeWorktreeId:
    """Tests for compute_worktree_id function."""

    def test_consistent_id_for_same_path(self, tmp_path):
        """Test that same path produces same ID."""
        id1 = compute_worktree_id(tmp_path)
        id2 = compute_worktree_id(tmp_path)

        assert id1 == id2
        assert len(id1) == 16

    def test_different_ids_for_different_paths(self, tmp_path):
        """Test that different paths produce different IDs."""
        path1 = tmp_path / "dir1"
        path2 = tmp_path / "dir2"
        path1.mkdir()
        path2.mkdir()

        id1 = compute_worktree_id(path1)
        id2 = compute_worktree_id(path2)

        assert id1 != id2


class TestGetGitHeadRef:
    """Tests for get_git_head_ref function."""

    @patch('subprocess.run')
    def test_get_head_ref_success(self, mock_run, tmp_path):
        """Test successful retrieval of HEAD ref."""
        mock_run.return_value = Mock(
            stdout="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0\n",
            returncode=0
        )

        ref = get_git_head_ref(tmp_path)

        assert ref == "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0"
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ['git', 'rev-parse', 'HEAD']

    @patch('subprocess.run')
    def test_get_head_ref_failure(self, mock_run, tmp_path):
        """Test handling of Git command failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'git')

        ref = get_git_head_ref(tmp_path)

        assert ref is None

    @patch('subprocess.run')
    def test_get_head_ref_timeout(self, mock_run, tmp_path):
        """Test handling of Git command timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired('git', 5)

        ref = get_git_head_ref(tmp_path)

        assert ref is None


class TestComputeChangedFilesViaGit:
    """Tests for compute_changed_files_via_git function."""

    @patch('chunkhound.utils.worktree_detection.get_git_head_ref')
    @patch('subprocess.run')
    @patch('chunkhound.utils.file_discovery.discover_files')
    def test_compute_changes_success(
        self, mock_discover, mock_run, mock_get_ref, tmp_path
    ):
        """Test successful Git-based change detection."""
        main_wt = tmp_path / "main"
        feature_wt = tmp_path / "feature"
        main_wt.mkdir()
        feature_wt.mkdir()

        # Mock Git HEAD refs
        mock_get_ref.side_effect = ['abc123', 'def456']

        # Mock git diff-tree output
        mock_run.return_value = Mock(
            stdout="A\tfile1.py\nM\tfile2.py\nD\tfile3.py\n",
            returncode=0
        )

        # Mock file discovery
        mock_discover.return_value = [
            feature_wt / "file1.py",
            feature_wt / "file2.py",
            feature_wt / "file4.py"  # Unchanged file
        ]

        changes = compute_changed_files_via_git(
            main_wt, feature_wt, ["*.py"], []
        )

        assert len(changes['added']) == 1
        assert changes['added'][0] == feature_wt / "file1.py"

        assert len(changes['modified']) == 1
        assert changes['modified'][0] == feature_wt / "file2.py"

        assert len(changes['deleted']) == 1
        assert changes['deleted'][0] == main_wt / "file3.py"

        assert len(changes['unchanged']) == 1
        assert changes['unchanged'][0] == feature_wt / "file4.py"

    @patch('chunkhound.utils.worktree_detection.get_git_head_ref')
    def test_compute_changes_missing_ref(self, mock_get_ref, tmp_path):
        """Test handling when HEAD ref cannot be determined."""
        main_wt = tmp_path / "main"
        feature_wt = tmp_path / "feature"
        main_wt.mkdir()
        feature_wt.mkdir()

        # Mock missing HEAD ref
        mock_get_ref.return_value = None

        with pytest.raises(ValueError, match="Could not determine HEAD refs"):
            compute_changed_files_via_git(main_wt, feature_wt, [], [])


class TestFallbackComputeChangedFiles:
    """Tests for fallback_compute_changed_files function."""

    @patch('chunkhound.utils.hashing.compute_file_hash')
    @patch('chunkhound.utils.file_discovery.discover_files')
    def test_fallback_compute_changes(
        self, mock_discover, mock_hash, tmp_path
    ):
        """Test hash-based change detection fallback."""
        main_wt = tmp_path / "main"
        feature_wt = tmp_path / "feature"
        main_wt.mkdir()
        feature_wt.mkdir()

        # Main index files with hashes
        main_index_files = [
            {'path': 'file1.py', 'content_hash': 'hash1'},
            {'path': 'file2.py', 'content_hash': 'hash2'},
            {'path': 'file3.py', 'content_hash': 'hash3'}
        ]

        # Mock file discovery in feature worktree
        mock_discover.return_value = [
            feature_wt / "file1.py",  # Unchanged
            feature_wt / "file2.py",  # Modified
            feature_wt / "file4.py"   # Added
        ]

        # Mock hash computation
        def mock_hash_fn(path):
            if path.name == "file1.py":
                return "hash1"  # Same as main
            elif path.name == "file2.py":
                return "hash2_modified"  # Different from main
            elif path.name == "file4.py":
                return "hash4"  # New file
            return "unknown"

        mock_hash.side_effect = mock_hash_fn

        changes = fallback_compute_changed_files(
            main_wt, feature_wt, main_index_files, [], []
        )

        assert len(changes['added']) == 1
        assert changes['added'][0].name == "file4.py"

        assert len(changes['modified']) == 1
        assert changes['modified'][0].name == "file2.py"

        assert len(changes['unchanged']) == 1
        assert changes['unchanged'][0].name == "file1.py"

        assert len(changes['deleted']) == 1
        assert changes['deleted'][0].name == "file3.py"

    @patch('chunkhound.utils.hashing.compute_file_hash')
    @patch('chunkhound.utils.file_discovery.discover_files')
    def test_fallback_hash_failure(self, mock_discover, mock_hash, tmp_path):
        """Test handling of hash computation failure."""
        main_wt = tmp_path / "main"
        feature_wt = tmp_path / "feature"
        main_wt.mkdir()
        feature_wt.mkdir()

        main_index_files = [
            {'path': 'file1.py', 'content_hash': 'hash1'}
        ]

        mock_discover.return_value = [feature_wt / "file1.py"]

        # Hash computation fails
        mock_hash.side_effect = Exception("Hash failed")

        changes = fallback_compute_changed_files(
            main_wt, feature_wt, main_index_files, [], []
        )

        # Should treat as modified on failure
        assert len(changes['modified']) == 1
        assert changes['modified'][0].name == "file1.py"


class TestMatchesPatterns:
    """Tests for _matches_patterns function."""

    def test_no_patterns_includes_all(self):
        """Test that empty patterns include all files."""
        assert _matches_patterns("file.py", [], []) is True
        assert _matches_patterns("file.txt", [], []) is True

    def test_exclude_patterns(self):
        """Test exclusion patterns."""
        assert _matches_patterns("test_file.py", [], ["test_*"]) is False
        assert _matches_patterns("file.py", [], ["test_*"]) is True

    def test_include_patterns(self):
        """Test inclusion patterns."""
        assert _matches_patterns("file.py", ["*.py"], []) is True
        assert _matches_patterns("file.txt", ["*.py"], []) is False

    def test_include_and_exclude(self):
        """Test combination of include and exclude patterns."""
        # Exclude takes precedence
        assert _matches_patterns(
            "test_file.py", ["*.py"], ["test_*"]
        ) is False

        # Matches include but not exclude
        assert _matches_patterns(
            "file.py", ["*.py"], ["test_*"]
        ) is True

    def test_wildcard_patterns(self):
        """Test various wildcard patterns."""
        assert _matches_patterns("src/file.py", ["src/*.py"], []) is True
        # Note: fnmatch treats ** as a literal **, not recursive glob
        # This test was incorrect - fnmatch('src/sub/file.py', 'src/**/*.py') returns True
        assert _matches_patterns("src/sub/file.py", ["src/**/*.py"], []) is True
        assert _matches_patterns("file.pyc", ["*.py"], []) is False
