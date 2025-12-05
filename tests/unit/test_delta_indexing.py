"""Unit tests for Phase 3 delta indexing in IndexingCoordinator.

Phase 3 implements delta indexing for linked worktrees:
- Detects linked worktrees that have indexed main worktrees
- Uses git diff-tree to compute changed files
- Only processes added/modified files
- Creates inheritance records for unchanged files
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Any

from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.utils.worktree_detection import WorktreeInfo
from chunkhound.core.models import File, Chunk


class TestDeltaModeActivation:
    """Tests for delta indexing mode activation logic."""

    @pytest.fixture
    def mock_db_provider(self):
        """Create mock database provider with worktree methods."""
        db = Mock()
        db.get_worktree = Mock(return_value={
            'worktree_id': 'main_wt_id',
            'indexed_at': '2025-01-01T00:00:00',
            'is_main': True
        })
        db.get_worktree_file_count = Mock(return_value={'owned': 100})
        db.upsert_worktree = Mock()
        db.update_worktree_indexed_at = Mock()
        db.create_file_inheritances_batch = Mock(return_value=50)
        db.get_file_by_path = Mock(return_value={'id': 1, 'path': 'test.py'})
        return db

    @pytest.fixture
    def coordinator(self, mock_db_provider):
        """Create IndexingCoordinator with mocked dependencies."""
        return IndexingCoordinator(
            database_provider=mock_db_provider,
            base_directory=Path("/test/feature"),
            embedding_provider=None,
            progress=None,
            config=None
        )

    def test_delta_mode_activates_for_linked_worktree_with_indexed_main(
        self, coordinator, mock_db_provider, tmp_path
    ):
        """Test delta mode activates when linked worktree has indexed main."""
        # Create worktree structure
        main_wt = tmp_path / "main"
        feature_wt = tmp_path / "feature"
        main_wt.mkdir()
        feature_wt.mkdir()

        # Set up linked worktree info
        worktree_info = WorktreeInfo(
            is_main=False,
            is_linked=True,
            main_worktree_path=main_wt,
            git_dir=main_wt / ".git" / "worktrees" / "feature",
            worktree_id="feature_wt_id",
            git_repo_path=main_wt / ".git"
        )

        # Test activation logic
        coordinator._current_worktree_info = worktree_info
        is_indexed = coordinator._check_main_worktree_indexed(worktree_info)

        assert is_indexed is True
        mock_db_provider.get_worktree.assert_called_once()
        mock_db_provider.get_worktree_file_count.assert_called_once()

    def test_delta_mode_skips_main_worktree(self, coordinator, mock_db_provider):
        """Test delta mode does not activate for main worktree."""
        worktree_info = WorktreeInfo(
            is_main=True,
            is_linked=False,
            main_worktree_path=None,
            git_dir=Path("/test/main/.git"),
            worktree_id="main_wt_id",
            git_repo_path=Path("/test/main/.git")
        )

        is_indexed = coordinator._check_main_worktree_indexed(worktree_info)

        assert is_indexed is False
        mock_db_provider.get_worktree.assert_not_called()

    def test_delta_mode_skips_when_main_not_indexed(
        self, coordinator, mock_db_provider
    ):
        """Test delta mode does not activate when main worktree not indexed."""
        worktree_info = WorktreeInfo(
            is_main=False,
            is_linked=True,
            main_worktree_path=Path("/test/main"),
            git_dir=Path("/test/feature/.git"),
            worktree_id="feature_wt_id",
            git_repo_path=Path("/test/main/.git")
        )

        # Mock main worktree not indexed
        mock_db_provider.get_worktree.return_value = {
            'worktree_id': 'main_wt_id',
            'indexed_at': None,  # Not indexed
            'is_main': True
        }

        is_indexed = coordinator._check_main_worktree_indexed(worktree_info)

        assert is_indexed is False

    def test_delta_mode_skips_when_main_has_no_files(
        self, coordinator, mock_db_provider
    ):
        """Test delta mode does not activate when main worktree has no files."""
        worktree_info = WorktreeInfo(
            is_main=False,
            is_linked=True,
            main_worktree_path=Path("/test/main"),
            git_dir=Path("/test/feature/.git"),
            worktree_id="feature_wt_id",
            git_repo_path=Path("/test/main/.git")
        )

        # Mock main worktree indexed but no files
        mock_db_provider.get_worktree.return_value = {
            'worktree_id': 'main_wt_id',
            'indexed_at': '2025-01-01T00:00:00',
            'is_main': True
        }
        mock_db_provider.get_worktree_file_count.return_value = {'owned': 0}

        is_indexed = coordinator._check_main_worktree_indexed(worktree_info)

        assert is_indexed is False


class TestChangedFilesDetection:
    """Tests for Git-based changed files detection."""

    @pytest.mark.parametrize("git_output,expected_counts", [
        # Standard git diff output
        ("A\tfile1.py\nM\tfile2.py\nD\tfile3.py\n", {
            'added': 1, 'modified': 1, 'deleted': 1
        }),
        # Only additions
        ("A\tnew1.py\nA\tnew2.py\n", {
            'added': 2, 'modified': 0, 'deleted': 0
        }),
        # Only modifications
        ("M\tchanged1.py\nM\tchanged2.py\nM\tchanged3.py\n", {
            'added': 0, 'modified': 3, 'deleted': 0
        }),
        # Mixed with pattern filtering
        ("A\tfile.py\nA\tfile.txt\nM\ttest.py\n", {
            'added': 1, 'modified': 1, 'deleted': 0  # .txt filtered out
        }),
        # Empty diff (no changes)
        ("", {
            'added': 0, 'modified': 0, 'deleted': 0
        }),
    ])
    @patch('chunkhound.utils.file_patterns.discover_files')
    @patch('subprocess.run')
    @patch('chunkhound.utils.worktree_detection.get_git_head_ref')
    def test_git_diff_parsing(
        self, mock_get_ref, mock_run, mock_discover,
        git_output, expected_counts, tmp_path
    ):
        """Test parsing of git diff-tree output."""
        from chunkhound.utils.worktree_detection import compute_changed_files_via_git

        main_wt = tmp_path / "main"
        feature_wt = tmp_path / "feature"
        main_wt.mkdir()
        feature_wt.mkdir()

        # Mock Git operations
        mock_get_ref.side_effect = ['abc123', 'def456']
        mock_run.return_value = Mock(stdout=git_output, returncode=0)

        # Mock discover_files to return all feature files
        all_feature_files = []
        for line in git_output.strip().split('\n'):
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                status, path = parts
                if status in ('A', 'M') and path.endswith('.py'):
                    all_feature_files.append(feature_wt / path)

        # Add an unchanged file
        all_feature_files.append(feature_wt / "unchanged.py")
        mock_discover.return_value = all_feature_files

        # Run detection
        changes = compute_changed_files_via_git(
            main_wt, feature_wt, ["*.py"], []
        )

        assert len(changes['added']) == expected_counts['added']
        assert len(changes['modified']) == expected_counts['modified']
        assert len(changes['deleted']) == expected_counts['deleted']

    @patch('chunkhound.utils.worktree_detection.get_git_head_ref')
    def test_changed_files_detection_error_handling(
        self, mock_get_ref, tmp_path
    ):
        """Test error handling when git operations fail."""
        from chunkhound.utils.worktree_detection import compute_changed_files_via_git

        main_wt = tmp_path / "main"
        feature_wt = tmp_path / "feature"
        main_wt.mkdir()
        feature_wt.mkdir()

        # Mock missing HEAD ref
        mock_get_ref.return_value = None

        with pytest.raises(ValueError, match="Could not determine HEAD refs"):
            compute_changed_files_via_git(main_wt, feature_wt, [], [])


class TestInheritanceRecordCreation:
    """Tests for file inheritance record creation."""

    @pytest.fixture
    def mock_db_provider(self):
        """Create mock database provider with inheritance methods."""
        db = Mock()
        db.create_file_inheritances_batch = Mock(return_value=50)
        db.create_file_inheritance = Mock()
        db.get_file_by_path = Mock(return_value={'id': 1, 'path': 'test.py'})
        return db

    @pytest.fixture
    def coordinator(self, mock_db_provider, tmp_path):
        """Create IndexingCoordinator with mocked dependencies."""
        return IndexingCoordinator(
            database_provider=mock_db_provider,
            base_directory=tmp_path / "feature",
            embedding_provider=None,
            progress=None,
            config=None
        )

    @pytest.mark.asyncio
    async def test_create_inheritance_records_batch(
        self, coordinator, mock_db_provider, tmp_path
    ):
        """Test batch creation of inheritance records."""
        main_wt = tmp_path / "main"
        feature_wt = tmp_path / "feature"
        main_wt.mkdir()
        feature_wt.mkdir()

        # Create test files
        unchanged_files = []
        for i in range(5):
            test_file = feature_wt / f"file{i}.py"
            test_file.write_text(f"content {i}")
            unchanged_files.append(test_file)

        worktree_info = WorktreeInfo(
            is_main=False,
            is_linked=True,
            main_worktree_path=main_wt,
            git_dir=feature_wt / ".git",
            worktree_id="feature_wt_id",
            git_repo_path=main_wt / ".git"
        )

        # Create inheritance records
        count = await coordinator._create_inheritance_records(
            worktree_info=worktree_info,
            unchanged_files=unchanged_files,
            head_ref="abc123"
        )

        assert count == 50  # Mock return value
        mock_db_provider.create_file_inheritances_batch.assert_called_once()

        # Verify batch structure
        call_args = mock_db_provider.create_file_inheritances_batch.call_args
        inheritances = call_args[0][0]

        assert isinstance(inheritances, list)
        assert len(inheritances) == 5

        for inheritance in inheritances:
            assert 'worktree_id' in inheritance
            assert 'file_id' in inheritance
            assert 'source_worktree_id' in inheritance
            assert 'source_head_ref' in inheritance
            assert inheritance['worktree_id'] == "feature_wt_id"
            assert inheritance['source_head_ref'] == "abc123"

    @pytest.mark.asyncio
    async def test_create_inheritance_records_empty_list(
        self, coordinator, mock_db_provider, tmp_path
    ):
        """Test handling of empty unchanged files list."""
        worktree_info = WorktreeInfo(
            is_main=False,
            is_linked=True,
            main_worktree_path=tmp_path / "main",
            git_dir=tmp_path / "feature" / ".git",
            worktree_id="feature_wt_id",
            git_repo_path=tmp_path / "main" / ".git"
        )

        count = await coordinator._create_inheritance_records(
            worktree_info=worktree_info,
            unchanged_files=[],
            head_ref="abc123"
        )

        assert count == 0
        mock_db_provider.create_file_inheritances_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_inheritance_records_no_main_worktree(
        self, coordinator, mock_db_provider, tmp_path
    ):
        """Test handling when main worktree path is missing."""
        worktree_info = WorktreeInfo(
            is_main=False,
            is_linked=True,
            main_worktree_path=None,  # Missing main worktree
            git_dir=tmp_path / "feature" / ".git",
            worktree_id="feature_wt_id",
            git_repo_path=tmp_path / "main" / ".git"
        )

        unchanged_files = [tmp_path / "feature" / "test.py"]

        count = await coordinator._create_inheritance_records(
            worktree_info=worktree_info,
            unchanged_files=unchanged_files,
            head_ref="abc123"
        )

        assert count == 0
        mock_db_provider.create_file_inheritances_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_inheritance_records_fallback_to_individual(
        self, coordinator, mock_db_provider, tmp_path
    ):
        """Test fallback to individual inserts when batch fails."""
        main_wt = tmp_path / "main"
        feature_wt = tmp_path / "feature"
        main_wt.mkdir()
        feature_wt.mkdir()

        test_file = feature_wt / "test.py"
        test_file.write_text("content")

        worktree_info = WorktreeInfo(
            is_main=False,
            is_linked=True,
            main_worktree_path=main_wt,
            git_dir=feature_wt / ".git",
            worktree_id="feature_wt_id",
            git_repo_path=main_wt / ".git"
        )

        # Mock batch creation failure
        mock_db_provider.create_file_inheritances_batch.side_effect = Exception(
            "Batch insert failed"
        )
        mock_db_provider.create_file_inheritance = Mock()

        count = await coordinator._create_inheritance_records(
            worktree_info=worktree_info,
            unchanged_files=[test_file],
            head_ref="abc123"
        )

        assert count == 1
        mock_db_provider.create_file_inheritance.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_inheritance_records_file_outside_base(
        self, coordinator, mock_db_provider, tmp_path
    ):
        """Test handling of files outside base directory."""
        main_wt = tmp_path / "main"
        feature_wt = tmp_path / "feature"
        outside_wt = tmp_path / "outside"
        main_wt.mkdir()
        feature_wt.mkdir()
        outside_wt.mkdir()

        # File outside the base directory
        outside_file = outside_wt / "test.py"
        outside_file.write_text("content")

        worktree_info = WorktreeInfo(
            is_main=False,
            is_linked=True,
            main_worktree_path=main_wt,
            git_dir=feature_wt / ".git",
            worktree_id="feature_wt_id",
            git_repo_path=main_wt / ".git"
        )

        # This should skip the file
        count = await coordinator._create_inheritance_records(
            worktree_info=worktree_info,
            unchanged_files=[outside_file],
            head_ref="abc123"
        )

        # Should not create any records for files outside base
        assert count == 0


class TestWorktreeRegistration:
    """Tests for worktree registration in database."""

    @pytest.fixture
    def mock_db_provider(self):
        """Create mock database provider with worktree methods."""
        db = Mock()
        db.upsert_worktree = Mock()
        return db

    @pytest.fixture
    def coordinator(self, mock_db_provider, tmp_path):
        """Create IndexingCoordinator with mocked dependencies."""
        return IndexingCoordinator(
            database_provider=mock_db_provider,
            base_directory=tmp_path,
            embedding_provider=None,
            progress=None,
            config=None
        )

    @patch('chunkhound.services.indexing_coordinator.get_git_head_ref')
    @patch('chunkhound.services.indexing_coordinator.detect_worktree_info')
    def test_register_main_worktree(
        self, mock_detect, mock_get_ref, coordinator, mock_db_provider, tmp_path
    ):
        """Test registration of main worktree."""
        mock_get_ref.return_value = "abc123"
        mock_detect.return_value = WorktreeInfo(
            is_main=True,
            is_linked=False,
            main_worktree_path=None,
            git_dir=tmp_path / ".git",
            worktree_id="main_wt_id",
            git_repo_path=tmp_path / ".git"
        )

        worktree_info = coordinator._register_worktree(tmp_path)

        assert worktree_info.is_main is True
        mock_db_provider.upsert_worktree.assert_called_once()

        call_args = mock_db_provider.upsert_worktree.call_args
        assert call_args[1]['worktree_id'] == "main_wt_id"
        assert call_args[1]['is_main'] is True
        assert call_args[1]['main_worktree_id'] is None
        assert call_args[1]['head_ref'] == "abc123"

    @patch('chunkhound.services.indexing_coordinator.compute_worktree_id')
    @patch('chunkhound.services.indexing_coordinator.get_git_head_ref')
    @patch('chunkhound.services.indexing_coordinator.detect_worktree_info')
    def test_register_linked_worktree(
        self, mock_detect, mock_get_ref, mock_compute_id,
        coordinator, mock_db_provider, tmp_path
    ):
        """Test registration of linked worktree with main reference."""
        main_wt = tmp_path / "main"
        feature_wt = tmp_path / "feature"
        main_wt.mkdir()
        feature_wt.mkdir()

        mock_get_ref.return_value = "def456"
        mock_compute_id.return_value = "main_wt_id"
        mock_detect.return_value = WorktreeInfo(
            is_main=False,
            is_linked=True,
            main_worktree_path=main_wt,
            git_dir=feature_wt / ".git",
            worktree_id="feature_wt_id",
            git_repo_path=main_wt / ".git"
        )

        worktree_info = coordinator._register_worktree(feature_wt)

        assert worktree_info.is_linked is True
        mock_db_provider.upsert_worktree.assert_called_once()

        call_args = mock_db_provider.upsert_worktree.call_args
        assert call_args[1]['worktree_id'] == "feature_wt_id"
        assert call_args[1]['is_main'] is False
        assert call_args[1]['main_worktree_id'] == "main_wt_id"
        assert call_args[1]['head_ref'] == "def456"
