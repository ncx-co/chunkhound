"""Tests for worktree_support_enabled configuration toggle.

This module tests:
1. Configuration option loading from env/CLI/file
2. Disabled mode behavior (worktree features off)
3. Enabled mode behavior (worktree features on)
4. MCP tool visibility based on config
5. MCP tool graceful handling when disabled (defense-in-depth)
6. Database provider worktree_enabled property
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.core.config.config import Config
from chunkhound.mcp_server.tools import (
    TOOL_REGISTRY,
    list_worktrees_impl,
    search_regex_impl,
    search_semantic_impl,
)


class TestIndexingConfigWorktreeSupport:
    """Tests for worktree_support_enabled config option."""

    def test_worktree_support_defaults_false(self):
        """Verify worktree_support_enabled defaults to False."""
        config = IndexingConfig()
        assert config.worktree_support_enabled is False

    def test_worktree_support_can_be_enabled(self):
        """Verify worktree_support_enabled can be set to True."""
        config = IndexingConfig(worktree_support_enabled=True)
        assert config.worktree_support_enabled is True

    def test_worktree_support_from_dict(self):
        """Verify worktree_support_enabled loads from dict config."""
        config = IndexingConfig(**{"worktree_support_enabled": True})
        assert config.worktree_support_enabled is True

    def test_worktree_support_from_env_true(self):
        """Verify worktree_support_enabled loads from environment variable (true)."""
        with patch.dict(
            "os.environ", {"CHUNKHOUND_INDEXING__WORKTREE_SUPPORT_ENABLED": "true"}
        ):
            config_dict = IndexingConfig.load_from_env()
            config = IndexingConfig(**config_dict)
            assert config.worktree_support_enabled is True

    def test_worktree_support_from_env_false(self):
        """Verify worktree_support_enabled loads from environment variable (false)."""
        with patch.dict(
            "os.environ", {"CHUNKHOUND_INDEXING__WORKTREE_SUPPORT_ENABLED": "false"}
        ):
            config_dict = IndexingConfig.load_from_env()
            # When env var is "false", worktree_support_enabled won't be in dict
            # because the condition `worktree_support.lower() in ("true", "1", "yes")`
            # is False, so the key won't be set
            config = IndexingConfig(**config_dict)
            assert config.worktree_support_enabled is False

    def test_worktree_support_from_env_1(self):
        """Verify worktree_support_enabled accepts '1' as true."""
        with patch.dict(
            "os.environ", {"CHUNKHOUND_INDEXING__WORKTREE_SUPPORT_ENABLED": "1"}
        ):
            config_dict = IndexingConfig.load_from_env()
            config = IndexingConfig(**config_dict)
            assert config.worktree_support_enabled is True

    def test_worktree_support_from_env_yes(self):
        """Verify worktree_support_enabled accepts 'yes' as true."""
        with patch.dict(
            "os.environ", {"CHUNKHOUND_INDEXING__WORKTREE_SUPPORT_ENABLED": "yes"}
        ):
            config_dict = IndexingConfig.load_from_env()
            config = IndexingConfig(**config_dict)
            assert config.worktree_support_enabled is True


class TestMCPToolsDisabledMode:
    """Tests for MCP tools graceful handling when disabled (defense-in-depth).

    Even though list_worktrees is hidden when disabled, if someone calls it
    directly via the API, it should return a helpful message instead of an error.
    """

    @pytest.mark.asyncio
    async def test_list_worktrees_returns_message_when_called_directly(self):
        """Test list_worktrees returns informative message if called when disabled."""
        mock_provider = Mock()
        mock_provider.is_connected = True
        mock_provider.worktree_enabled = False

        mock_services = Mock()
        mock_services.provider = mock_provider

        result = await list_worktrees_impl(services=mock_services)

        # Verify defense-in-depth response
        assert result["worktrees"] == []
        assert result["current_worktree_id"] is None
        assert "message" in result
        assert "disabled" in result["message"].lower()
        assert "worktree_support_enabled" in result["message"].lower()

        # Verify provider methods NOT called
        assert not mock_provider.list_worktrees.called
        assert not mock_provider.get_worktree_file_count.called

    @pytest.mark.asyncio
    async def test_search_regex_ignores_worktree_scope_when_disabled(self):
        """Test search_regex ignores worktree_scope parameter when disabled."""
        mock_provider = Mock()
        mock_provider.is_connected = True
        mock_provider.worktree_enabled = False

        mock_search_service = Mock()
        mock_search_service.search_regex.return_value = ([], {"offset": 0, "page_size": 10, "has_more": False})

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        # Call with worktree_scope parameter
        result = await search_regex_impl(
            services=mock_services,
            pattern="test",
            worktree_scope="all",
        )

        # Verify search was called with worktree_ids=None (not ["all"])
        mock_search_service.search_regex.assert_called_once()
        call_kwargs = mock_search_service.search_regex.call_args.kwargs
        assert call_kwargs.get("worktree_ids") is None

    @pytest.mark.asyncio
    async def test_search_semantic_ignores_worktree_scope_when_disabled(self):
        """Test search_semantic ignores worktree_scope parameter when disabled."""
        mock_provider = Mock()
        mock_provider.is_connected = True
        mock_provider.worktree_enabled = False

        mock_search_service = AsyncMock()
        mock_search_service.search_semantic.return_value = ([], {"offset": 0, "page_size": 10, "has_more": False})

        mock_embedding_manager = Mock()
        mock_embedding_manager.list_providers.return_value = ["openai"]
        mock_embedding_provider = Mock()
        mock_embedding_provider.name = "openai"
        mock_embedding_provider.model = "text-embedding-3-small"
        mock_embedding_manager.get_provider.return_value = mock_embedding_provider

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        # Call with worktree_scope parameter
        result = await search_semantic_impl(
            services=mock_services,
            embedding_manager=mock_embedding_manager,
            query="test",
            worktree_scope="all",
        )

        # Verify search was called with worktree_ids=None (not ["all"])
        mock_search_service.search_semantic.assert_called_once()
        call_kwargs = mock_search_service.search_semantic.call_args.kwargs
        assert call_kwargs.get("worktree_ids") is None


class TestMCPToolsEnabledMode:
    """Tests for MCP tools when worktree support is enabled."""

    @pytest.mark.asyncio
    async def test_list_worktrees_returns_worktrees_when_enabled(self):
        """Test list_worktrees returns worktree data when enabled."""
        mock_provider = Mock()
        mock_provider.is_connected = True
        mock_provider.worktree_enabled = True
        mock_provider.list_worktrees.return_value = [
            {"id": "wt1", "path": "/repo", "is_main": True}
        ]
        mock_provider.get_worktree_file_count.return_value = 10

        mock_services = Mock()
        mock_services.provider = mock_provider

        result = await list_worktrees_impl(services=mock_services)

        # Verify enabled mode response
        assert len(result["worktrees"]) == 1
        assert result["worktrees"][0]["worktree_id"] == "wt1"
        assert "message" not in result or "disabled" not in result.get("message", "").lower()

        # Verify provider methods were called
        mock_provider.list_worktrees.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_regex_uses_worktree_scope_when_enabled(self):
        """Test search_regex uses worktree_scope parameter when enabled."""
        mock_provider = Mock()
        mock_provider.is_connected = True
        mock_provider.worktree_enabled = True

        mock_search_service = Mock()
        mock_search_service.search_regex.return_value = ([], {"offset": 0, "page_size": 10, "has_more": False})

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        # Call with worktree_scope parameter
        result = await search_regex_impl(
            services=mock_services,
            pattern="test",
            worktree_scope="all",
        )

        # Verify search was called with worktree_ids=["all"]
        mock_search_service.search_regex.assert_called_once()
        call_kwargs = mock_search_service.search_regex.call_args.kwargs
        assert call_kwargs.get("worktree_ids") == ["all"]

    @pytest.mark.asyncio
    async def test_search_regex_parses_comma_separated_worktree_ids(self):
        """Test search_regex parses comma-separated worktree IDs when enabled."""
        mock_provider = Mock()
        mock_provider.is_connected = True
        mock_provider.worktree_enabled = True

        mock_search_service = Mock()
        mock_search_service.search_regex.return_value = ([], {"offset": 0, "page_size": 10, "has_more": False})

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        # Call with comma-separated worktree IDs
        result = await search_regex_impl(
            services=mock_services,
            pattern="test",
            worktree_scope="wt1, wt2, wt3",
        )

        # Verify search was called with parsed worktree IDs
        call_kwargs = mock_search_service.search_regex.call_args.kwargs
        assert call_kwargs.get("worktree_ids") == ["wt1", "wt2", "wt3"]

    @pytest.mark.asyncio
    async def test_search_regex_current_scope_means_no_filter(self):
        """Test search_regex 'current' scope means no worktree filter."""
        mock_provider = Mock()
        mock_provider.is_connected = True
        mock_provider.worktree_enabled = True

        mock_search_service = Mock()
        mock_search_service.search_regex.return_value = ([], {"offset": 0, "page_size": 10, "has_more": False})

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        # Call with 'current' scope
        result = await search_regex_impl(
            services=mock_services,
            pattern="test",
            worktree_scope="current",
        )

        # Verify search was called with worktree_ids=None (current = no filter)
        call_kwargs = mock_search_service.search_regex.call_args.kwargs
        assert call_kwargs.get("worktree_ids") is None


class TestDatabaseProviderWorktreeEnabled:
    """Tests for database provider worktree_enabled property."""

    def test_duckdb_provider_worktree_enabled_defaults_false(self):
        """Test DuckDBProvider worktree_enabled defaults to False."""
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = DuckDBProvider(
                db_path=Path(tmpdir) / "test.db",
                base_directory=Path(tmpdir),
            )
            assert provider.worktree_enabled is False

    def test_duckdb_provider_worktree_enabled_can_be_true(self):
        """Test DuckDBProvider worktree_enabled can be set to True."""
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = DuckDBProvider(
                db_path=Path(tmpdir) / "test.db",
                base_directory=Path(tmpdir),
                worktree_enabled=True,
            )
            assert provider.worktree_enabled is True

    def test_lancedb_provider_worktree_enabled_defaults_false(self):
        """Test LanceDBProvider worktree_enabled defaults to False."""
        from chunkhound.providers.database.lancedb_provider import LanceDBProvider
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LanceDBProvider(
                db_path=Path(tmpdir) / "test.lance",
                base_directory=Path(tmpdir),
            )
            assert provider.worktree_enabled is False

    def test_lancedb_provider_worktree_enabled_can_be_true(self):
        """Test LanceDBProvider worktree_enabled can be set to True."""
        from chunkhound.providers.database.lancedb_provider import LanceDBProvider
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LanceDBProvider(
                db_path=Path(tmpdir) / "test.lance",
                base_directory=Path(tmpdir),
                worktree_enabled=True,
            )
            assert provider.worktree_enabled is True
