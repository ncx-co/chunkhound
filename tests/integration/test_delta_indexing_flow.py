"""Integration tests for Phase 3 delta indexing flow.

Tests the complete delta indexing workflow:
1. Index main worktree with files
2. Create linked worktree with some changes
3. Index linked worktree using delta mode
4. Verify only changed files were processed
5. Verify inheritance records created for unchanged files
"""

import pytest
import subprocess
from pathlib import Path
from typing import Any

from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from tests.fixtures.fake_providers import FakeEmbeddingProvider
from chunkhound.parsers.universal_parser import UniversalParser
from chunkhound.core.types.common import Language


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary Git repository for testing."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(
        ['git', 'init'],
        cwd=repo_path,
        capture_output=True,
        check=True
    )
    subprocess.run(
        ['git', 'config', 'user.email', 'test@example.com'],
        cwd=repo_path,
        capture_output=True,
        check=True
    )
    subprocess.run(
        ['git', 'config', 'user.name', 'Test User'],
        cwd=repo_path,
        capture_output=True,
        check=True
    )

    return repo_path


@pytest.fixture
def main_worktree_with_files(temp_git_repo):
    """Create main worktree with some Python files and commit them."""
    # Create test files
    files = {
        'file1.py': 'def func1():\n    return "original"\n',
        'file2.py': 'def func2():\n    return "original"\n',
        'file3.py': 'def func3():\n    return "original"\n',
        'unchanged.py': 'def unchanged():\n    return "stays same"\n'
    }

    for filename, content in files.items():
        file_path = temp_git_repo / filename
        file_path.write_text(content)

    # Commit files
    subprocess.run(
        ['git', 'add', '.'],
        cwd=temp_git_repo,
        capture_output=True,
        check=True
    )
    subprocess.run(
        ['git', 'commit', '-m', 'Initial commit'],
        cwd=temp_git_repo,
        capture_output=True,
        check=True
    )

    return temp_git_repo


@pytest.fixture
def feature_worktree(main_worktree_with_files, tmp_path):
    """Create a linked feature worktree with some changes."""
    main_wt = main_worktree_with_files
    feature_wt = tmp_path / "feature-worktree"

    # Create linked worktree
    subprocess.run(
        ['git', 'worktree', 'add', str(feature_wt), '-b', 'feature'],
        cwd=main_wt,
        capture_output=True,
        check=True
    )

    # Make changes in feature worktree
    # 1. Add a new file
    new_file = feature_wt / "new_file.py"
    new_file.write_text('def new_func():\n    return "added"\n')

    # 2. Modify an existing file
    modified_file = feature_wt / "file1.py"
    modified_file.write_text('def func1():\n    return "modified"\n')

    # 3. Delete a file
    deleted_file = feature_wt / "file3.py"
    deleted_file.unlink()

    # 4. Leave unchanged.py as is (no changes)

    # Commit changes
    subprocess.run(
        ['git', 'add', '.'],
        cwd=feature_wt,
        capture_output=True,
        check=True
    )
    subprocess.run(
        ['git', 'commit', '-m', 'Feature changes'],
        cwd=feature_wt,
        capture_output=True,
        check=True
    )

    return {
        'main': main_wt,
        'feature': feature_wt,
        'added': ['new_file.py'],
        'modified': ['file1.py'],
        'deleted': ['file3.py'],
        'unchanged': ['file2.py', 'unchanged.py']
    }


@pytest.fixture
def db_provider(tmp_path):
    """Create DuckDB provider for testing."""
    db_path = tmp_path / "test.db"
    provider = DuckDBProvider(str(db_path), base_directory=tmp_path)
    yield provider
    provider.close()


@pytest.fixture
def embedding_provider():
    """Create fake embedding provider."""
    return FakeEmbeddingProvider(dimensions=384)


@pytest.fixture
def python_parser():
    """Create Python parser."""
    parser = UniversalParser(Language.PYTHON)
    parser.setup()
    return parser


@pytest.mark.integration
class TestDeltaIndexingFlow:
    """Integration tests for complete delta indexing workflow."""

    @pytest.mark.asyncio
    async def test_full_delta_indexing_workflow(
        self,
        feature_worktree,
        db_provider,
        embedding_provider,
        python_parser
    ):
        """Test complete delta indexing: index main, then delta index feature."""
        main_wt = feature_worktree['main']
        feature_wt = feature_worktree['feature']

        # Phase 1: Index main worktree
        main_coordinator = IndexingCoordinator(
            database_provider=db_provider,
            base_directory=main_wt,
            embedding_provider=embedding_provider,
            language_parsers={Language.PYTHON: python_parser},
            progress=None,
            config=None
        )

        main_result = await main_coordinator.process_directory(
            directory=main_wt,
            patterns=["*.py"],
            exclude_patterns=[]
        )

        # Verify main indexing succeeded
        assert main_result['status'] == 'success'
        assert main_result['files_processed'] == 4  # file1, file2, file3, unchanged
        assert main_result['total_chunks'] > 0

        # Get main worktree file count
        main_files = db_provider.get_files()
        assert len(main_files) == 4

        # Phase 2: Delta index feature worktree
        feature_coordinator = IndexingCoordinator(
            database_provider=db_provider,
            base_directory=feature_wt,
            embedding_provider=embedding_provider,
            language_parsers={Language.PYTHON: python_parser},
            progress=None,
            config=None
        )

        feature_result = await feature_coordinator.process_directory(
            directory=feature_wt,
            patterns=["*.py"],
            exclude_patterns=[]
        )

        # Verify delta indexing succeeded
        assert feature_result['status'] == 'success'
        assert feature_result.get('delta_mode') is True

        # Should only process changed files (added + modified)
        # new_file.py (added) + file1.py (modified) = 2 files
        assert feature_result['files_processed'] == 2

        # Verify inheritance records created for unchanged files
        assert 'inherited_files' in feature_result
        # file2.py and unchanged.py should be inherited
        assert feature_result['inherited_files'] == 2

        # Verify database state
        all_files = db_provider.get_files()
        # Main: 4 files, Feature: 2 new/modified files = 6 total
        assert len(all_files) >= 6

        # Verify inheritance records exist
        if hasattr(db_provider, 'get_file_inheritances'):
            inheritances = db_provider.get_file_inheritances(
                worktree_id=feature_coordinator._current_worktree_id
            )
            assert len(inheritances) == 2  # file2.py, unchanged.py

    @pytest.mark.asyncio
    async def test_delta_indexing_with_no_changes(
        self,
        main_worktree_with_files,
        tmp_path,
        db_provider,
        embedding_provider,
        python_parser
    ):
        """Test delta indexing when no files changed (all inherited)."""
        main_wt = main_worktree_with_files

        # Index main worktree
        main_coordinator = IndexingCoordinator(
            database_provider=db_provider,
            base_directory=main_wt,
            embedding_provider=embedding_provider,
            language_parsers={Language.PYTHON: python_parser},
            progress=None,
            config=None
        )

        await main_coordinator.process_directory(
            directory=main_wt,
            patterns=["*.py"],
            exclude_patterns=[]
        )

        # Create feature worktree with NO changes
        feature_wt = tmp_path / "no-changes-worktree"
        subprocess.run(
            ['git', 'worktree', 'add', str(feature_wt), '-b', 'no-changes'],
            cwd=main_wt,
            capture_output=True,
            check=True
        )

        # Index feature worktree (should use delta mode)
        feature_coordinator = IndexingCoordinator(
            database_provider=db_provider,
            base_directory=feature_wt,
            embedding_provider=embedding_provider,
            language_parsers={Language.PYTHON: python_parser},
            progress=None,
            config=None
        )

        feature_result = await feature_coordinator.process_directory(
            directory=feature_wt,
            patterns=["*.py"],
            exclude_patterns=[]
        )

        # Verify delta mode with no changes
        assert feature_result['status'] == 'delta_no_changes'
        assert feature_result['files_processed'] == 0
        assert feature_result['delta_mode'] is True
        assert feature_result['inherited_files'] == 4  # All files inherited

    @pytest.mark.asyncio
    async def test_delta_indexing_fallback_on_git_failure(
        self,
        main_worktree_with_files,
        tmp_path,
        db_provider,
        embedding_provider,
        python_parser
    ):
        """Test fallback to full indexing when Git operations fail."""
        main_wt = main_worktree_with_files

        # Index main worktree
        main_coordinator = IndexingCoordinator(
            database_provider=db_provider,
            base_directory=main_wt,
            embedding_provider=embedding_provider,
            language_parsers={Language.PYTHON: python_parser},
            progress=None,
            config=None
        )

        await main_coordinator.process_directory(
            directory=main_wt,
            patterns=["*.py"],
            exclude_patterns=[]
        )

        # Create feature worktree
        feature_wt = tmp_path / "fallback-worktree"
        subprocess.run(
            ['git', 'worktree', 'add', str(feature_wt), '-b', 'fallback'],
            cwd=main_wt,
            capture_output=True,
            check=True
        )

        # Corrupt the git worktree to trigger fallback
        # (Remove the .git file to make git commands fail)
        git_file = feature_wt / ".git"
        if git_file.exists():
            git_file.unlink()

        # Index feature worktree (should fallback to full indexing)
        feature_coordinator = IndexingCoordinator(
            database_provider=db_provider,
            base_directory=feature_wt,
            embedding_provider=embedding_provider,
            language_parsers={Language.PYTHON: python_parser},
            progress=None,
            config=None
        )

        # This should succeed with full indexing fallback
        feature_result = await feature_coordinator.process_directory(
            directory=feature_wt,
            patterns=["*.py"],
            exclude_patterns=[]
        )

        # Should succeed but NOT use delta mode
        assert feature_result['status'] == 'success'
        assert feature_result.get('delta_mode') is not True
        assert feature_result['files_processed'] == 4  # Full indexing

    @pytest.mark.asyncio
    async def test_delta_indexing_respects_patterns(
        self,
        feature_worktree,
        db_provider,
        embedding_provider,
        python_parser
    ):
        """Test that delta indexing respects file patterns."""
        main_wt = feature_worktree['main']
        feature_wt = feature_worktree['feature']

        # Add a non-Python file to main worktree
        txt_file = main_wt / "readme.txt"
        txt_file.write_text("This is a text file")
        subprocess.run(
            ['git', 'add', 'readme.txt'],
            cwd=main_wt,
            capture_output=True,
            check=True
        )
        subprocess.run(
            ['git', 'commit', '-m', 'Add readme'],
            cwd=main_wt,
            capture_output=True,
            check=True
        )

        # Index main worktree (only .py files)
        main_coordinator = IndexingCoordinator(
            database_provider=db_provider,
            base_directory=main_wt,
            embedding_provider=embedding_provider,
            language_parsers={Language.PYTHON: python_parser},
            progress=None,
            config=None
        )

        await main_coordinator.process_directory(
            directory=main_wt,
            patterns=["*.py"],  # Only Python files
            exclude_patterns=[]
        )

        # Delta index feature worktree (only .py files)
        feature_coordinator = IndexingCoordinator(
            database_provider=db_provider,
            base_directory=feature_wt,
            embedding_provider=embedding_provider,
            language_parsers={Language.PYTHON: python_parser},
            progress=None,
            config=None
        )

        feature_result = await feature_coordinator.process_directory(
            directory=feature_wt,
            patterns=["*.py"],  # Only Python files
            exclude_patterns=[]
        )

        # Should still use delta mode
        assert feature_result.get('delta_mode') is True

        # txt file should not affect counts
        assert feature_result['files_processed'] == 2  # new_file.py, file1.py (modified)
        assert feature_result['inherited_files'] == 2  # file2.py, unchanged.py


@pytest.mark.integration
class TestDeltaIndexingEdgeCases:
    """Integration tests for edge cases in delta indexing."""

    @pytest.mark.asyncio
    async def test_delta_indexing_with_only_deletions(
        self,
        main_worktree_with_files,
        tmp_path,
        db_provider,
        embedding_provider,
        python_parser
    ):
        """Test delta indexing when only deletions occurred."""
        main_wt = main_worktree_with_files

        # Index main worktree
        main_coordinator = IndexingCoordinator(
            database_provider=db_provider,
            base_directory=main_wt,
            embedding_provider=embedding_provider,
            language_parsers={Language.PYTHON: python_parser},
            progress=None,
            config=None
        )

        await main_coordinator.process_directory(
            directory=main_wt,
            patterns=["*.py"],
            exclude_patterns=[]
        )

        # Create feature worktree and delete files
        feature_wt = tmp_path / "deletions-worktree"
        subprocess.run(
            ['git', 'worktree', 'add', str(feature_wt), '-b', 'deletions'],
            cwd=main_wt,
            capture_output=True,
            check=True
        )

        # Delete some files
        (feature_wt / "file1.py").unlink()
        (feature_wt / "file2.py").unlink()

        subprocess.run(
            ['git', 'add', '.'],
            cwd=feature_wt,
            capture_output=True,
            check=True
        )
        subprocess.run(
            ['git', 'commit', '-m', 'Delete files'],
            cwd=feature_wt,
            capture_output=True,
            check=True
        )

        # Index feature worktree
        feature_coordinator = IndexingCoordinator(
            database_provider=db_provider,
            base_directory=feature_wt,
            embedding_provider=embedding_provider,
            language_parsers={Language.PYTHON: python_parser},
            progress=None,
            config=None
        )

        feature_result = await feature_coordinator.process_directory(
            directory=feature_wt,
            patterns=["*.py"],
            exclude_patterns=[]
        )

        # Should use delta mode
        assert feature_result.get('delta_mode') is True

        # No added or modified files
        assert feature_result['files_processed'] == 0

        # Remaining files should be inherited
        # file3.py and unchanged.py remain
        assert feature_result['inherited_files'] == 2

    @pytest.mark.asyncio
    async def test_delta_indexing_with_all_files_changed(
        self,
        main_worktree_with_files,
        tmp_path,
        db_provider,
        embedding_provider,
        python_parser
    ):
        """Test delta indexing when all files are modified."""
        main_wt = main_worktree_with_files

        # Index main worktree
        main_coordinator = IndexingCoordinator(
            database_provider=db_provider,
            base_directory=main_wt,
            embedding_provider=embedding_provider,
            language_parsers={Language.PYTHON: python_parser},
            progress=None,
            config=None
        )

        await main_coordinator.process_directory(
            directory=main_wt,
            patterns=["*.py"],
            exclude_patterns=[]
        )

        # Create feature worktree and modify all files
        feature_wt = tmp_path / "all-changed-worktree"
        subprocess.run(
            ['git', 'worktree', 'add', str(feature_wt), '-b', 'all-changed'],
            cwd=main_wt,
            capture_output=True,
            check=True
        )

        # Modify all files
        for py_file in feature_wt.glob("*.py"):
            py_file.write_text(f'# Modified\n{py_file.read_text()}')

        subprocess.run(
            ['git', 'add', '.'],
            cwd=feature_wt,
            capture_output=True,
            check=True
        )
        subprocess.run(
            ['git', 'commit', '-m', 'Modify all files'],
            cwd=feature_wt,
            capture_output=True,
            check=True
        )

        # Index feature worktree
        feature_coordinator = IndexingCoordinator(
            database_provider=db_provider,
            base_directory=feature_wt,
            embedding_provider=embedding_provider,
            language_parsers={Language.PYTHON: python_parser},
            progress=None,
            config=None
        )

        feature_result = await feature_coordinator.process_directory(
            directory=feature_wt,
            patterns=["*.py"],
            exclude_patterns=[]
        )

        # Should still use delta mode
        assert feature_result.get('delta_mode') is True

        # All files modified = all files processed
        assert feature_result['files_processed'] == 4

        # No unchanged files to inherit
        assert feature_result['inherited_files'] == 0
