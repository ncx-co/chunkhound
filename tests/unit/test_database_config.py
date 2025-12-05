"""
Test database configuration functionality.

This module tests database configuration including worktree-aware path
resolution, provider selection, and validation.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.utils.worktree_detection import WorktreeInfo


class TestDatabaseConfigBasics:
    """Test basic database configuration functionality."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DatabaseConfig()

        assert config.path is None
        assert config.provider == "duckdb"
        assert config.max_disk_usage_mb is None
        assert config.lancedb_index_type is None
        assert config.lancedb_optimize_fragment_threshold == 100

    def test_custom_values(self):
        """Test setting custom configuration values."""
        config = DatabaseConfig(
            path=Path("/custom/db"),
            provider="lancedb",
            max_disk_usage_mb=1024.0,
            lancedb_index_type="ivf_hnsw_sq",
            lancedb_optimize_fragment_threshold=50,
        )

        assert config.path == Path("/custom/db")
        assert config.provider == "lancedb"
        assert config.max_disk_usage_mb == 1024.0
        assert config.lancedb_index_type == "ivf_hnsw_sq"
        assert config.lancedb_optimize_fragment_threshold == 50

    def test_path_validator_converts_string_to_path(self):
        """Test path validator converts string to Path object."""
        config = DatabaseConfig(path="/string/path")

        assert isinstance(config.path, Path)
        assert config.path == Path("/string/path")

    def test_provider_validator_accepts_valid_providers(self):
        """Test provider validator accepts valid providers."""
        config1 = DatabaseConfig(provider="duckdb")
        assert config1.provider == "duckdb"

        config2 = DatabaseConfig(provider="lancedb")
        assert config2.provider == "lancedb"

    def test_provider_validator_rejects_invalid_providers(self):
        """Test provider validator rejects invalid providers."""
        with pytest.raises(ValueError, match="Input should be"):
            DatabaseConfig(provider="invalid")

    def test_is_configured_returns_false_when_path_none(self):
        """Test is_configured returns False when path is None."""
        config = DatabaseConfig()
        assert config.is_configured() is False

    def test_is_configured_returns_true_when_path_set(self):
        """Test is_configured returns True when path is set."""
        config = DatabaseConfig(path=Path("/db"))
        assert config.is_configured() is True


class TestDatabaseConfigGetDbPath:
    """Test get_db_path method."""

    def test_raises_error_when_path_not_configured(self):
        """Test raises error when path not configured."""
        config = DatabaseConfig()

        with pytest.raises(ValueError, match="Database path not configured"):
            config.get_db_path()

    def test_returns_chunks_db_for_duckdb_provider(self, tmp_path):
        """Test returns chunks.db for DuckDB provider."""
        config = DatabaseConfig(path=tmp_path / "db", provider="duckdb")

        result = config.get_db_path()

        assert result == tmp_path / "db" / "chunks.db"
        assert (tmp_path / "db").is_dir()

    def test_returns_lancedb_directory_for_lancedb_provider(self, tmp_path):
        """Test returns .lancedb directory for LanceDB provider."""
        config = DatabaseConfig(path=tmp_path / "db", provider="lancedb")

        result = config.get_db_path()

        assert result == tmp_path / "db" / "lancedb.lancedb"
        assert (tmp_path / "db").is_dir()

    def test_handles_memory_database_without_creating_directory(self):
        """Test handles :memory: database without creating directory."""
        config = DatabaseConfig(path=Path(":memory:"), provider="duckdb")

        result = config.get_db_path()

        assert result == Path(":memory:")

    def test_raises_error_for_unknown_provider(self, tmp_path):
        """Test raises error for unknown provider."""
        # Bypass validation by setting directly
        config = DatabaseConfig(path=tmp_path / "db")
        config.provider = "unknown"  # type: ignore

        with pytest.raises(ValueError, match="Unknown database provider"):
            config.get_db_path()


class TestDatabaseConfigResolveWorktreeDbPath:
    """Test resolve_worktree_db_path method."""

    def test_returns_explicit_path_if_set(self):
        """Test returns explicit path if self.path is set."""
        explicit_path = Path("/explicit/db")
        config = DatabaseConfig(path=explicit_path)
        base_directory = Path("/repo")

        result = config.resolve_worktree_db_path(base_directory)

        assert result == explicit_path

    @patch('chunkhound.utils.worktree_detection.detect_worktree_info')
    def test_returns_main_worktree_db_path_for_linked_worktree(self, mock_detect):
        """Test returns main worktree's db path for linked worktree."""
        config = DatabaseConfig()  # path is None
        base_directory = Path("/repo-feature")
        main_worktree = Path("/repo")

        # Mock linked worktree detection
        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.is_linked = True
        mock_info.main_worktree_path = main_worktree
        mock_detect.return_value = mock_info

        result = config.resolve_worktree_db_path(base_directory)

        assert result == main_worktree / ".chunkhound" / "db"

    @patch('chunkhound.utils.worktree_detection.detect_worktree_info')
    def test_returns_local_db_path_for_main_worktree(self, mock_detect):
        """Test returns local db path for main worktree."""
        config = DatabaseConfig()  # path is None
        base_directory = Path("/repo")

        # Mock main worktree detection
        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.is_linked = False
        mock_info.main_worktree_path = None
        mock_detect.return_value = mock_info

        result = config.resolve_worktree_db_path(base_directory)

        assert result == base_directory / ".chunkhound" / "db"

    @patch('chunkhound.utils.worktree_detection.detect_worktree_info')
    def test_returns_local_db_path_for_non_git_directory(self, mock_detect):
        """Test returns local db path for non-git directory."""
        config = DatabaseConfig()  # path is None
        base_directory = Path("/not-a-repo")

        # Mock non-git directory detection
        mock_info = MagicMock(spec=WorktreeInfo)
        mock_info.is_linked = False
        mock_info.main_worktree_path = None
        mock_detect.return_value = mock_info

        result = config.resolve_worktree_db_path(base_directory)

        assert result == base_directory / ".chunkhound" / "db"

    def test_explicit_path_takes_precedence_over_worktree_detection(self):
        """Test explicit path takes precedence over worktree detection."""
        explicit_path = Path("/explicit/db")
        config = DatabaseConfig(path=explicit_path)
        base_directory = Path("/repo-feature")

        result = config.resolve_worktree_db_path(base_directory)

        # Should return explicit path, not main worktree path
        assert result == explicit_path


class TestDatabaseConfigLoadFromEnv:
    """Test load_from_env class method."""

    def test_loads_path_from_new_env_var(self, monkeypatch):
        """Test loads path from new environment variable."""
        monkeypatch.setenv("CHUNKHOUND_DATABASE__PATH", "/env/db")

        config_dict = DatabaseConfig.load_from_env()

        assert config_dict["path"] == Path("/env/db")

    def test_loads_path_from_legacy_env_var(self, monkeypatch):
        """Test loads path from legacy environment variable."""
        monkeypatch.setenv("CHUNKHOUND_DB_PATH", "/legacy/db")

        config_dict = DatabaseConfig.load_from_env()

        assert config_dict["path"] == Path("/legacy/db")

    def test_new_env_var_takes_precedence_over_legacy(self, monkeypatch):
        """Test new env var takes precedence over legacy."""
        monkeypatch.setenv("CHUNKHOUND_DATABASE__PATH", "/new/db")
        monkeypatch.setenv("CHUNKHOUND_DB_PATH", "/legacy/db")

        config_dict = DatabaseConfig.load_from_env()

        assert config_dict["path"] == Path("/new/db")

    def test_loads_provider_from_env(self, monkeypatch):
        """Test loads provider from environment."""
        monkeypatch.setenv("CHUNKHOUND_DATABASE__PROVIDER", "lancedb")

        config_dict = DatabaseConfig.load_from_env()

        assert config_dict["provider"] == "lancedb"

    def test_loads_lancedb_index_type_from_env(self, monkeypatch):
        """Test loads lancedb_index_type from environment."""
        monkeypatch.setenv("CHUNKHOUND_DATABASE__LANCEDB_INDEX_TYPE", "ivf_hnsw_sq")

        config_dict = DatabaseConfig.load_from_env()

        assert config_dict["lancedb_index_type"] == "ivf_hnsw_sq"

    def test_loads_lancedb_optimize_threshold_from_env(self, monkeypatch):
        """Test loads lancedb_optimize_fragment_threshold from environment."""
        monkeypatch.setenv("CHUNKHOUND_DATABASE__LANCEDB_OPTIMIZE_FRAGMENT_THRESHOLD", "50")

        config_dict = DatabaseConfig.load_from_env()

        assert config_dict["lancedb_optimize_fragment_threshold"] == 50

    def test_loads_max_disk_usage_from_env(self, monkeypatch):
        """Test loads max_disk_usage_mb from environment (converted from GB)."""
        monkeypatch.setenv("CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB", "2.5")

        config_dict = DatabaseConfig.load_from_env()

        assert config_dict["max_disk_usage_mb"] == 2560.0

    def test_invalid_max_disk_usage_ignored(self, monkeypatch):
        """Test invalid max_disk_usage_mb env var is silently ignored."""
        monkeypatch.setenv("CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB", "not-a-number")

        config_dict = DatabaseConfig.load_from_env()

        assert "max_disk_usage_mb" not in config_dict

    def test_returns_empty_dict_when_no_env_vars_set(self):
        """Test returns empty dict when no environment variables set."""
        # Clear any existing env vars
        for key in list(os.environ.keys()):
            if key.startswith("CHUNKHOUND_DATABASE"):
                del os.environ[key]

        config_dict = DatabaseConfig.load_from_env()

        assert config_dict == {}


class TestDatabaseConfigExtractCliOverrides:
    """Test extract_cli_overrides class method."""

    def test_extracts_db_argument(self):
        """Test extracts --db argument."""
        args = MagicMock()
        args.db = Path("/cli/db")
        args.database_path = None

        overrides = DatabaseConfig.extract_cli_overrides(args)

        assert overrides["path"] == Path("/cli/db")

    def test_extracts_database_path_argument(self):
        """Test extracts --database-path argument."""
        args = MagicMock()
        args.db = None
        args.database_path = Path("/cli/db")

        overrides = DatabaseConfig.extract_cli_overrides(args)

        assert overrides["path"] == Path("/cli/db")

    def test_database_path_used_when_db_is_none(self):
        """Test --database-path is used when --db is None."""
        args = MagicMock()
        args.db = None
        args.database_path = Path("/database-path")

        overrides = DatabaseConfig.extract_cli_overrides(args)

        assert overrides["path"] == Path("/database-path")

    def test_extracts_database_provider_argument(self):
        """Test extracts --database-provider argument."""
        args = MagicMock()
        args.database_provider = "lancedb"

        overrides = DatabaseConfig.extract_cli_overrides(args)

        assert overrides["provider"] == "lancedb"

    def test_extracts_max_disk_usage_argument(self):
        """Test extracts --max-disk-usage-gb argument (converted to MB)."""
        args = MagicMock()
        args.max_disk_usage_gb = 3.0

        overrides = DatabaseConfig.extract_cli_overrides(args)

        assert overrides["max_disk_usage_mb"] == 3072.0

    def test_handles_zero_max_disk_usage(self):
        """Test handles zero max_disk_usage_gb (valid edge case)."""
        args = MagicMock()
        args.max_disk_usage_gb = 0.0

        overrides = DatabaseConfig.extract_cli_overrides(args)

        assert overrides["max_disk_usage_mb"] == 0.0

    def test_ignores_none_max_disk_usage(self):
        """Test ignores None max_disk_usage_gb."""
        args = MagicMock()
        args.max_disk_usage_gb = None

        overrides = DatabaseConfig.extract_cli_overrides(args)

        assert "max_disk_usage_mb" not in overrides

    def test_returns_empty_dict_when_no_arguments_present(self):
        """Test returns empty dict when no arguments present."""
        args = MagicMock()
        # Simulate missing attributes
        del args.db
        del args.database_path
        del args.database_provider
        del args.max_disk_usage_gb

        overrides = DatabaseConfig.extract_cli_overrides(args)

        assert overrides == {}


class TestDatabaseConfigRepr:
    """Test __repr__ method."""

    def test_repr_includes_provider_and_path(self):
        """Test repr includes provider and path."""
        config = DatabaseConfig(path=Path("/db"), provider="duckdb")

        repr_str = repr(config)

        assert "provider=duckdb" in repr_str
        assert "path=" in repr_str

    def test_repr_includes_max_disk_usage_when_set(self):
        """Test repr includes max_disk_usage_mb when set."""
        config = DatabaseConfig(
            path=Path("/db"), provider="duckdb", max_disk_usage_mb=4300.8
        )

        repr_str = repr(config)

        assert "max_disk_usage_mb=4300.8" in repr_str

    def test_repr_excludes_max_disk_usage_when_none(self):
        """Test repr excludes max_disk_usage_mb when None."""
        config = DatabaseConfig(path=Path("/db"), provider="duckdb")

        repr_str = repr(config)

        assert "max_disk_usage_mb" not in repr_str
