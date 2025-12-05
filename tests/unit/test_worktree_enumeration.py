"""
Test Git worktree enumeration functionality.

This module tests the enumeration of Git worktrees, including discovery of
linked worktrees, parsing git worktree list output, and filesystem-based
fallback mechanisms.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from chunkhound.utils.worktree_enumeration import (
    EnumeratedWorktree,
    _enumerate_via_filesystem,
    _enumerate_via_git_command,
    _get_current_branch,
    _parse_worktree_entry,
    enumerate_worktrees,
    get_main_worktree,
    get_worktree_by_id,
)
from chunkhound.utils.worktree_detection import WorktreeInfo


class TestEnumeratedWorktree:
    """Test EnumeratedWorktree dataclass properties."""

    def test_worktree_id_delegates_to_info(self):
        """Test worktree_id property delegates to info.worktree_id."""
        info = MagicMock(spec=WorktreeInfo)
        info.worktree_id = "abc123def456789"
        info.is_main = True

        wt = EnumeratedWorktree(
            info=info,
            path=Path("/repo"),
            branch="main",
            head_ref="abc123def456",
            is_bare=False,
            is_detached=False,
        )

        assert wt.worktree_id == "abc123def456789"

    def test_is_main_delegates_to_info(self):
        """Test is_main property delegates to info.is_main."""
        info = MagicMock(spec=WorktreeInfo)
        info.worktree_id = "abc123"
        info.is_main = True

        wt = EnumeratedWorktree(
            info=info,
            path=Path("/repo"),
            branch="main",
            head_ref="abc123",
            is_bare=False,
            is_detached=False,
        )

        assert wt.is_main is True

        # Test with False
        info.is_main = False
        wt2 = EnumeratedWorktree(
            info=info,
            path=Path("/repo-feature"),
            branch="feature/test",
            head_ref="def456",
            is_bare=False,
            is_detached=False,
        )

        assert wt2.is_main is False

    def test_display_name_returns_branch_when_available(self):
        """Test display_name returns branch name when available."""
        info = MagicMock(spec=WorktreeInfo)
        info.worktree_id = "abc123"
        info.is_main = True

        wt = EnumeratedWorktree(
            info=info,
            path=Path("/repo"),
            branch="main",
            head_ref="abc123def456",
            is_bare=False,
            is_detached=False,
        )

        assert wt.display_name == "main"

        # Test with feature branch
        wt2 = EnumeratedWorktree(
            info=info,
            path=Path("/repo-feature"),
            branch="feature/authentication",
            head_ref="def456",
            is_bare=False,
            is_detached=False,
        )

        assert wt2.display_name == "feature/authentication"

    def test_display_name_returns_detached_format_when_detached(self):
        """Test display_name returns detached format when detached."""
        info = MagicMock(spec=WorktreeInfo)
        info.worktree_id = "abc123"
        info.is_main = False

        wt = EnumeratedWorktree(
            info=info,
            path=Path("/repo-detached"),
            branch=None,
            head_ref="abc123def456789012345678901234567890",
            is_bare=False,
            is_detached=True,
        )

        # Should show first 8 characters of commit hash
        assert wt.display_name == "detached@abc123de"

    def test_display_name_returns_path_name_as_fallback(self):
        """Test display_name returns path name as fallback."""
        info = MagicMock(spec=WorktreeInfo)
        info.worktree_id = "abc123"
        info.is_main = False

        wt = EnumeratedWorktree(
            info=info,
            path=Path("/repo-unknown"),
            branch=None,
            head_ref=None,
            is_bare=False,
            is_detached=False,
        )

        assert wt.display_name == "repo-unknown"


class TestEnumerateWorktrees:
    """Test enumerate_worktrees function."""

    @patch('chunkhound.utils.worktree_enumeration.detect_worktree_info')
    def test_returns_empty_list_for_non_git_directory(self, mock_detect):
        """Test returns empty list for non-git directory."""
        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.git_repo_path = None
        mock_detect.return_value = mock_info

        result = enumerate_worktrees(Path("/not-a-repo"))

        assert result == []

    @patch('chunkhound.utils.worktree_enumeration._enumerate_via_git_command')
    @patch('chunkhound.utils.worktree_enumeration.detect_worktree_info')
    def test_returns_main_worktree_when_no_linked_worktrees(
        self, mock_detect, mock_git_enum
    ):
        """Test returns main worktree when no linked worktrees exist."""
        # Setup main worktree detection
        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.git_repo_path = Path("/repo/.git")
        mock_info.is_main = True
        mock_info.main_worktree_path = None
        mock_info.worktree_id = "abc123"
        mock_detect.return_value = mock_info

        # Git command returns single worktree
        main_wt = MagicMock(spec=EnumeratedWorktree)
        main_wt.is_main = True
        main_wt.branch = "main"
        mock_git_enum.return_value = [main_wt]

        result = enumerate_worktrees(Path("/repo"))

        assert len(result) == 1
        assert result[0].is_main is True

    @patch('chunkhound.utils.worktree_enumeration._enumerate_via_git_command')
    @patch('chunkhound.utils.worktree_enumeration.detect_worktree_info')
    def test_discovers_linked_worktrees_via_git_command(
        self, mock_detect, mock_git_enum
    ):
        """Test discovers linked worktrees via git command."""
        # Setup worktree detection
        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.git_repo_path = Path("/repo/.git")
        mock_info.is_main = True
        mock_info.main_worktree_path = None
        mock_detect.return_value = mock_info

        # Git command returns main + linked worktrees
        main_wt = MagicMock(spec=EnumeratedWorktree)
        main_wt.is_main = True
        main_wt.branch = "main"
        main_wt.path = Path("/repo")

        linked_wt = MagicMock(spec=EnumeratedWorktree)
        linked_wt.is_main = False
        linked_wt.branch = "feature/auth"
        linked_wt.path = Path("/repo-feature-auth")

        mock_git_enum.return_value = [main_wt, linked_wt]

        result = enumerate_worktrees(Path("/repo"))

        assert len(result) == 2
        assert result[0].is_main is True
        assert result[1].is_main is False

    @patch('chunkhound.utils.worktree_enumeration._enumerate_via_filesystem')
    @patch('chunkhound.utils.worktree_enumeration._enumerate_via_git_command')
    @patch('chunkhound.utils.worktree_enumeration.detect_worktree_info')
    def test_fallback_to_filesystem_when_git_command_fails(
        self, mock_detect, mock_git_enum, mock_fs_enum
    ):
        """Test fallback to filesystem when git command fails."""
        # Setup worktree detection
        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.git_repo_path = Path("/repo/.git")
        mock_info.is_main = True
        mock_info.main_worktree_path = None
        mock_detect.return_value = mock_info

        # Git command fails (returns empty list)
        mock_git_enum.return_value = []

        # Filesystem fallback returns worktrees
        main_wt = MagicMock(spec=EnumeratedWorktree)
        main_wt.is_main = True
        main_wt.branch = "main"
        mock_fs_enum.return_value = [main_wt]

        result = enumerate_worktrees(Path("/repo"))

        assert len(result) == 1
        assert mock_fs_enum.called

    @patch('chunkhound.utils.worktree_enumeration._enumerate_via_git_command')
    @patch('chunkhound.utils.worktree_enumeration.detect_worktree_info')
    def test_sorting_main_first_then_by_branch(self, mock_detect, mock_git_enum):
        """Test sorting places main worktree first, then by branch name."""
        # Setup worktree detection
        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.git_repo_path = Path("/repo/.git")
        mock_info.is_main = True
        mock_info.main_worktree_path = None
        mock_detect.return_value = mock_info

        # Return worktrees in non-sorted order
        wt_feature_b = MagicMock(spec=EnumeratedWorktree)
        wt_feature_b.is_main = False
        wt_feature_b.branch = "feature/b"
        wt_feature_b.path = Mock()
        wt_feature_b.path.name = "b"

        wt_main = MagicMock(spec=EnumeratedWorktree)
        wt_main.is_main = True
        wt_main.branch = "main"
        wt_main.path = Mock()
        wt_main.path.name = "main"

        wt_feature_a = MagicMock(spec=EnumeratedWorktree)
        wt_feature_a.is_main = False
        wt_feature_a.branch = "feature/a"
        wt_feature_a.path = Mock()
        wt_feature_a.path.name = "a"

        # Return in unsorted order
        mock_git_enum.return_value = [wt_feature_b, wt_main, wt_feature_a]

        result = enumerate_worktrees(Path("/repo"))

        # Should be sorted: main first, then alphabetically by branch
        assert len(result) == 3
        assert result[0].is_main is True
        assert result[1].branch == "feature/a"
        assert result[2].branch == "feature/b"


class TestEnumerateViaGitCommand:
    """Test _enumerate_via_git_command function."""

    @patch('subprocess.run')
    def test_parses_porcelain_output_with_main_worktree(self, mock_run):
        """Test parsing porcelain output with main worktree."""
        mock_run.return_value = Mock(
            stdout="""worktree /repo
HEAD abc123
branch refs/heads/main

"""
        )

        with patch('chunkhound.utils.worktree_enumeration._parse_worktree_entry') as mock_parse:
            mock_wt = MagicMock(spec=EnumeratedWorktree)
            mock_parse.return_value = mock_wt

            result = _enumerate_via_git_command(Path("/repo"))

            assert len(result) == 1
            mock_parse.assert_called_once()
            call_args = mock_parse.call_args[0][0]
            assert call_args['worktree'] == '/repo'
            assert call_args['head'] == 'abc123'
            assert call_args['branch'] == 'refs/heads/main'

    @patch('subprocess.run')
    def test_parses_multiple_worktrees(self, mock_run):
        """Test parsing with multiple worktrees."""
        mock_run.return_value = Mock(
            stdout="""worktree /repo
HEAD abc123
branch refs/heads/main

worktree /repo-feature
HEAD def456
branch refs/heads/feature/test

"""
        )

        with patch('chunkhound.utils.worktree_enumeration._parse_worktree_entry') as mock_parse:
            mock_wt1 = MagicMock(spec=EnumeratedWorktree)
            mock_wt2 = MagicMock(spec=EnumeratedWorktree)
            mock_parse.side_effect = [mock_wt1, mock_wt2]

            result = _enumerate_via_git_command(Path("/repo"))

            assert len(result) == 2
            assert mock_parse.call_count == 2

    @patch('subprocess.run')
    def test_handles_bare_worktrees(self, mock_run):
        """Test handling bare worktrees."""
        mock_run.return_value = Mock(
            stdout="""worktree /repo.git
bare

"""
        )

        with patch('chunkhound.utils.worktree_enumeration._parse_worktree_entry') as mock_parse:
            mock_wt = MagicMock(spec=EnumeratedWorktree)
            mock_parse.return_value = mock_wt

            result = _enumerate_via_git_command(Path("/repo.git"))

            assert len(result) == 1
            call_args = mock_parse.call_args[0][0]
            assert call_args['bare'] is True

    @patch('subprocess.run')
    def test_handles_detached_head(self, mock_run):
        """Test handling detached HEAD."""
        mock_run.return_value = Mock(
            stdout="""worktree /repo
HEAD abc123
detached

"""
        )

        with patch('chunkhound.utils.worktree_enumeration._parse_worktree_entry') as mock_parse:
            mock_wt = MagicMock(spec=EnumeratedWorktree)
            mock_parse.return_value = mock_wt

            result = _enumerate_via_git_command(Path("/repo"))

            assert len(result) == 1
            call_args = mock_parse.call_args[0][0]
            assert call_args['detached'] is True

    @patch('subprocess.run')
    def test_handles_subprocess_errors_gracefully(self, mock_run):
        """Test handles subprocess errors gracefully."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'git')

        result = _enumerate_via_git_command(Path("/repo"))

        assert result == []

    @patch('subprocess.run')
    def test_handles_timeout_gracefully(self, mock_run):
        """Test handles timeout gracefully."""
        mock_run.side_effect = subprocess.TimeoutExpired('git', 10)

        result = _enumerate_via_git_command(Path("/repo"))

        assert result == []

    @patch('subprocess.run')
    def test_handles_file_not_found_gracefully(self, mock_run):
        """Test handles git not found gracefully."""
        mock_run.side_effect = FileNotFoundError("git not found")

        result = _enumerate_via_git_command(Path("/repo"))

        assert result == []


class TestParseWorktreeEntry:
    """Test _parse_worktree_entry function."""

    @patch('chunkhound.utils.worktree_enumeration.detect_worktree_info')
    def test_parses_complete_entry(self, mock_detect):
        """Test parsing complete entry (worktree, HEAD, branch)."""
        entry = {
            'worktree': '/repo',
            'head': 'abc123',
            'branch': 'refs/heads/main',
        }

        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.worktree_id = "abc123"
        mock_detect.return_value = mock_info

        with patch('pathlib.Path.exists', return_value=True):
            result = _parse_worktree_entry(entry)

        assert result is not None
        assert result.path == Path('/repo').resolve()
        assert result.branch == "main"
        assert result.head_ref == "abc123"
        assert result.is_bare is False
        assert result.is_detached is False

    @patch('chunkhound.utils.worktree_enumeration.detect_worktree_info')
    def test_parses_bare_worktree_entry(self, mock_detect):
        """Test parsing bare worktree entry."""
        entry = {
            'worktree': '/repo.git',
            'bare': True,
        }

        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.worktree_id = "abc123"
        mock_detect.return_value = mock_info

        with patch('pathlib.Path.exists', return_value=True):
            result = _parse_worktree_entry(entry)

        assert result is not None
        assert result.is_bare is True

    @patch('chunkhound.utils.worktree_enumeration.detect_worktree_info')
    def test_parses_detached_head_entry(self, mock_detect):
        """Test parsing detached HEAD entry."""
        entry = {
            'worktree': '/repo',
            'head': 'abc123',
            'detached': True,
        }

        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.worktree_id = "abc123"
        mock_detect.return_value = mock_info

        with patch('pathlib.Path.exists', return_value=True):
            result = _parse_worktree_entry(entry)

        assert result is not None
        assert result.is_detached is True
        assert result.branch is None

    def test_returns_none_for_nonexistent_path(self):
        """Test returns None for non-existent path."""
        entry = {
            'worktree': '/nonexistent/path',
            'head': 'abc123',
            'branch': 'refs/heads/main',
        }

        with patch('pathlib.Path.exists', return_value=False):
            result = _parse_worktree_entry(entry)

        assert result is None

    @patch('chunkhound.utils.worktree_enumeration.detect_worktree_info')
    def test_branch_ref_parsing_removes_refs_heads_prefix(self, mock_detect):
        """Test branch ref parsing (refs/heads/main -> main)."""
        entry = {
            'worktree': '/repo',
            'head': 'abc123',
            'branch': 'refs/heads/feature/authentication',
        }

        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.worktree_id = "abc123"
        mock_detect.return_value = mock_info

        with patch('pathlib.Path.exists', return_value=True):
            result = _parse_worktree_entry(entry)

        assert result is not None
        assert result.branch == "feature/authentication"

    @patch('chunkhound.utils.worktree_enumeration.detect_worktree_info')
    def test_branch_ref_parsing_handles_non_standard_refs(self, mock_detect):
        """Test branch ref parsing handles non-standard refs."""
        entry = {
            'worktree': '/repo',
            'head': 'abc123',
            'branch': 'refs/remotes/origin/main',
        }

        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.worktree_id = "abc123"
        mock_detect.return_value = mock_info

        with patch('pathlib.Path.exists', return_value=True):
            result = _parse_worktree_entry(entry)

        assert result is not None
        # Non-standard refs should be kept as-is
        assert result.branch == "refs/remotes/origin/main"


class TestEnumerateViaFilesystem:
    """Test _enumerate_via_filesystem fallback function."""

    @patch('chunkhound.utils.worktree_enumeration._get_current_branch')
    @patch('chunkhound.utils.worktree_enumeration.get_git_head_ref')
    @patch('chunkhound.utils.worktree_enumeration.detect_worktree_info')
    def test_discovers_main_worktree(self, mock_detect, mock_head, mock_branch):
        """Test discovers main worktree."""
        main_path = Path("/repo")
        git_repo_path = Path("/repo/.git")

        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.worktree_id = "abc123"
        mock_detect.return_value = mock_info

        mock_head.return_value = "abc123def456"
        mock_branch.return_value = "main"

        with patch('pathlib.Path.is_dir', return_value=False):
            result = _enumerate_via_filesystem(main_path, git_repo_path)

        assert len(result) == 1
        assert result[0].branch == "main"

    @patch('chunkhound.utils.worktree_enumeration._get_current_branch')
    @patch('chunkhound.utils.worktree_enumeration.get_git_head_ref')
    @patch('chunkhound.utils.worktree_enumeration.detect_worktree_info')
    def test_discovers_linked_worktrees_from_git_worktrees(
        self, mock_detect, mock_head, mock_branch
    ):
        """Test discovers linked worktrees from .git/worktrees/."""
        main_path = Path("/repo")
        git_repo_path = Path("/repo/.git")

        # Mock main worktree info
        main_info = MagicMock(spec=WorktreeInfo)
        main_info.worktree_id = "main123"

        # Mock linked worktree info
        linked_info = MagicMock(spec=WorktreeInfo)
        linked_info.worktree_id = "linked123"

        mock_detect.side_effect = [main_info, linked_info]
        mock_head.side_effect = ["abc123", "def456"]
        mock_branch.side_effect = ["main", "feature/test"]

        # Mock filesystem structure
        worktrees_dir = git_repo_path / "worktrees"
        feature_dir = worktrees_dir / "feature-test"
        gitdir_file = feature_dir / "gitdir"

        def mock_is_dir(self):
            return self in (worktrees_dir, feature_dir)

        def mock_is_file(self):
            return self == gitdir_file

        def mock_iterdir(self):
            if self == worktrees_dir:
                return [feature_dir]
            return []

        def mock_read_text(self):
            if self == gitdir_file:
                return "/repo-feature/.git"
            raise FileNotFoundError()

        with patch('pathlib.Path.is_dir', mock_is_dir), \
             patch('pathlib.Path.is_file', mock_is_file), \
             patch('pathlib.Path.iterdir', mock_iterdir), \
             patch('pathlib.Path.read_text', mock_read_text), \
             patch('pathlib.Path.exists', return_value=True):

            result = _enumerate_via_filesystem(main_path, git_repo_path)

        assert len(result) == 2
        assert result[0].branch == "main"
        assert result[1].branch == "feature/test"

    @patch('chunkhound.utils.worktree_enumeration._get_current_branch')
    @patch('chunkhound.utils.worktree_enumeration.get_git_head_ref')
    @patch('chunkhound.utils.worktree_enumeration.detect_worktree_info')
    def test_handles_missing_gitdir_files_gracefully(
        self, mock_detect, mock_head, mock_branch
    ):
        """Test handles missing gitdir files gracefully."""
        main_path = Path("/repo")
        git_repo_path = Path("/repo/.git")

        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.worktree_id = "abc123"
        mock_detect.return_value = mock_info

        mock_head.return_value = "abc123"
        mock_branch.return_value = "main"

        # Mock worktrees directory exists but no gitdir files
        worktrees_dir = git_repo_path / "worktrees"
        feature_dir = worktrees_dir / "feature-test"

        def mock_is_dir(self):
            return self in (worktrees_dir, feature_dir)

        def mock_is_file(self):
            return False  # No gitdir file

        def mock_iterdir(self):
            if self == worktrees_dir:
                return [feature_dir]
            return []

        with patch('pathlib.Path.is_dir', mock_is_dir), \
             patch('pathlib.Path.is_file', mock_is_file), \
             patch('pathlib.Path.iterdir', mock_iterdir):

            result = _enumerate_via_filesystem(main_path, git_repo_path)

        # Should only have main worktree
        assert len(result) == 1
        assert result[0].branch == "main"

    @patch('chunkhound.utils.worktree_enumeration._get_current_branch')
    @patch('chunkhound.utils.worktree_enumeration.get_git_head_ref')
    @patch('chunkhound.utils.worktree_enumeration.detect_worktree_info')
    def test_handles_broken_symlinks_gracefully(
        self, mock_detect, mock_head, mock_branch
    ):
        """Test handles broken symlinks gracefully."""
        main_path = Path("/repo")
        git_repo_path = Path("/repo/.git")

        main_info = MagicMock(spec=WorktreeInfo)
        main_info.worktree_id = "main123"
        mock_detect.return_value = main_info

        mock_head.return_value = "abc123"
        mock_branch.return_value = "main"

        # Mock worktrees directory with broken symlink
        worktrees_dir = git_repo_path / "worktrees"
        feature_dir = worktrees_dir / "feature-test"
        gitdir_file = feature_dir / "gitdir"

        def mock_is_dir(self):
            return self in (worktrees_dir, feature_dir)

        def mock_is_file(self):
            return self == gitdir_file

        def mock_iterdir(self):
            if self == worktrees_dir:
                return [feature_dir]
            return []

        def mock_read_text(self):
            if self == gitdir_file:
                return "/nonexistent/worktree/.git"
            raise FileNotFoundError()

        def mock_exists(self):
            # Worktree path doesn't exist (broken)
            if str(self) == "/nonexistent/worktree":
                return False
            return True

        with patch('pathlib.Path.is_dir', mock_is_dir), \
             patch('pathlib.Path.is_file', mock_is_file), \
             patch('pathlib.Path.iterdir', mock_iterdir), \
             patch('pathlib.Path.read_text', mock_read_text), \
             patch('pathlib.Path.exists', mock_exists):

            result = _enumerate_via_filesystem(main_path, git_repo_path)

        # Should only have main worktree (linked worktree skipped)
        assert len(result) == 1
        assert result[0].branch == "main"


class TestGetCurrentBranch:
    """Test _get_current_branch function."""

    @patch('subprocess.run')
    def test_returns_branch_name(self, mock_run):
        """Test returns branch name."""
        mock_run.return_value = Mock(stdout="main\n")

        result = _get_current_branch(Path("/repo"))

        assert result == "main"

    @patch('subprocess.run')
    def test_returns_none_for_detached_head(self, mock_run):
        """Test returns None for detached HEAD."""
        mock_run.return_value = Mock(stdout="HEAD\n")

        result = _get_current_branch(Path("/repo"))

        assert result is None

    @patch('subprocess.run')
    def test_returns_none_on_error(self, mock_run):
        """Test returns None on error."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'git')

        result = _get_current_branch(Path("/repo"))

        assert result is None

    @patch('subprocess.run')
    def test_returns_none_on_timeout(self, mock_run):
        """Test returns None on timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired('git', 5)

        result = _get_current_branch(Path("/repo"))

        assert result is None


class TestGetWorktreeById:
    """Test get_worktree_by_id function."""

    @patch('chunkhound.utils.worktree_enumeration.enumerate_worktrees')
    def test_finds_worktree_by_id(self, mock_enumerate):
        """Test finds worktree by ID."""
        wt1 = MagicMock(spec=EnumeratedWorktree)
        wt1.worktree_id = "abc123"

        wt2 = MagicMock(spec=EnumeratedWorktree)
        wt2.worktree_id = "def456"

        mock_enumerate.return_value = [wt1, wt2]

        result = get_worktree_by_id(Path("/repo"), "def456")

        assert result == wt2

    @patch('chunkhound.utils.worktree_enumeration.enumerate_worktrees')
    def test_returns_none_when_id_not_found(self, mock_enumerate):
        """Test returns None when ID not found."""
        wt1 = MagicMock(spec=EnumeratedWorktree)
        wt1.worktree_id = "abc123"

        mock_enumerate.return_value = [wt1]

        result = get_worktree_by_id(Path("/repo"), "notfound")

        assert result is None


class TestGetMainWorktree:
    """Test get_main_worktree function."""

    @patch('chunkhound.utils.worktree_enumeration.enumerate_worktrees')
    def test_finds_main_worktree(self, mock_enumerate):
        """Test finds main worktree."""
        main_wt = MagicMock(spec=EnumeratedWorktree)
        main_wt.is_main = True

        linked_wt = MagicMock(spec=EnumeratedWorktree)
        linked_wt.is_main = False

        mock_enumerate.return_value = [main_wt, linked_wt]

        result = get_main_worktree(Path("/repo"))

        assert result == main_wt

    @patch('chunkhound.utils.worktree_enumeration.enumerate_worktrees')
    def test_returns_none_for_non_git_directory(self, mock_enumerate):
        """Test returns None for non-git directory."""
        mock_enumerate.return_value = []

        result = get_main_worktree(Path("/not-a-repo"))

        assert result is None
