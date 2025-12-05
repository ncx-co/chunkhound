"""Tests for Phase 5 MCP worktree integration.

This module tests:
1. list_worktrees tool registration and schema
2. worktree_scope parameter in search tools
3. Integration between MCP server and worktree functionality
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from chunkhound.mcp_server.tools import (
    TOOL_REGISTRY,
    list_worktrees_impl,
    search_regex_impl,
    search_semantic_impl,
)


class TestListWorktreesTool:
    """Tests for list_worktrees tool registration and schema."""

    def test_list_worktrees_in_registry(self):
        """Verify list_worktrees is registered in TOOL_REGISTRY."""
        assert "list_worktrees" in TOOL_REGISTRY
        tool = TOOL_REGISTRY["list_worktrees"]
        assert tool.name == "list_worktrees"

    def test_list_worktrees_description(self):
        """Verify list_worktrees has comprehensive description."""
        tool = TOOL_REGISTRY["list_worktrees"]
        assert len(tool.description) > 50
        assert "worktree" in tool.description.lower()
        assert "indexed" in tool.description.lower()

    def test_list_worktrees_no_parameters_required(self):
        """Verify list_worktrees takes no user parameters."""
        tool = TOOL_REGISTRY["list_worktrees"]
        required = tool.parameters.get("required", [])

        # Should have no required parameters (services filtered)
        assert len(required) == 0

    def test_list_worktrees_requires_no_embeddings(self):
        """Verify list_worktrees doesn't require embeddings."""
        tool = TOOL_REGISTRY["list_worktrees"]
        assert not tool.requires_embeddings

    @pytest.mark.asyncio
    async def test_list_worktrees_implementation_calls_provider(self):
        """Test list_worktrees implementation calls provider.list_worktrees()."""
        # Mock database services
        mock_provider = Mock()
        mock_provider.is_connected = True
        mock_provider.list_worktrees.return_value = [
            {
                "id": "main123",
                "path": "/repo/main",
                "is_main": True,
                "main_worktree_id": None,
                "head_ref": "abc123",
                "indexed_at": "2025-01-01T00:00:00",
            },
            {
                "id": "feature123",
                "path": "/repo-feature",
                "is_main": False,
                "main_worktree_id": "main123",
                "head_ref": "def456",
                "indexed_at": "2025-01-02T00:00:00",
            }
        ]
        mock_provider.get_worktree_file_count.side_effect = [
            {"owned": 100, "inherited": 0, "total": 100},
            {"owned": 5, "inherited": 95, "total": 100},
        ]

        mock_services = Mock()
        mock_services.provider = mock_provider

        # Call implementation
        result = await list_worktrees_impl(services=mock_services)

        # Verify provider was called
        mock_provider.list_worktrees.assert_called_once()
        assert mock_provider.get_worktree_file_count.call_count == 2

        # Verify response structure
        assert "worktrees" in result
        assert "current_worktree_id" in result
        # current_worktree_id may be None or a string

        # Verify worktree data enrichment
        worktrees = result["worktrees"]
        assert len(worktrees) == 2

        # Main worktree
        assert worktrees[0]["worktree_id"] == "main123"
        assert worktrees[0]["is_main"] is True
        assert worktrees[0]["file_count"]["owned"] == 100
        assert worktrees[0]["file_count"]["total"] == 100

        # Linked worktree
        assert worktrees[1]["worktree_id"] == "feature123"
        assert worktrees[1]["is_main"] is False
        assert worktrees[1]["file_count"]["owned"] == 5
        assert worktrees[1]["file_count"]["inherited"] == 95

    @pytest.mark.asyncio
    async def test_list_worktrees_connects_database(self):
        """Test list_worktrees connects database if not connected."""
        mock_provider = Mock()
        mock_provider.is_connected = False
        mock_provider.connect = Mock()
        mock_provider.list_worktrees.return_value = []
        mock_provider.get_worktree_file_count.return_value = {"owned": 0, "inherited": 0, "total": 0}

        mock_services = Mock()
        mock_services.provider = mock_provider

        await list_worktrees_impl(services=mock_services)

        # Verify connection was established
        mock_provider.connect.assert_called_once()


class TestWorktreeScopeParameter:
    """Tests for worktree_scope parameter in search tools."""

    def test_search_regex_has_worktree_scope_parameter(self):
        """Verify search_regex has worktree_scope parameter."""
        tool = TOOL_REGISTRY["search_regex"]
        props = tool.parameters["properties"]

        assert "worktree_scope" in props
        assert props["worktree_scope"]["type"] == "string"

        # Should not be required
        required = tool.parameters.get("required", [])
        assert "worktree_scope" not in required

    def test_search_semantic_has_worktree_scope_parameter(self):
        """Verify search_semantic has worktree_scope parameter."""
        tool = TOOL_REGISTRY["search_semantic"]
        props = tool.parameters["properties"]

        assert "worktree_scope" in props
        assert props["worktree_scope"]["type"] == "string"

        # Should not be required
        required = tool.parameters.get("required", [])
        assert "worktree_scope" not in required

    def test_worktree_scope_description(self):
        """Verify worktree_scope has description in docstring."""
        tool = TOOL_REGISTRY["search_regex"]
        props = tool.parameters["properties"]

        # Description should be extracted from docstring
        assert "description" in props["worktree_scope"]
        desc = props["worktree_scope"]["description"].lower()
        assert "worktree" in desc

    @pytest.mark.asyncio
    async def test_search_regex_default_worktree_scope(self):
        """Test search_regex with default (None) worktree_scope."""
        mock_provider = Mock()
        mock_provider.is_connected = True

        mock_search_service = Mock()
        mock_search_service.search_regex = Mock(return_value=(
            [{"file_path": "test.py", "content": "test"}],
            {"offset": 0, "page_size": 10, "has_more": False}
        ))

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        await search_regex_impl(
            services=mock_services,
            pattern="test",
            page_size=10,
            offset=0,
            worktree_scope=None  # Default
        )

        # Verify search_regex was called with worktree_ids=None
        mock_search_service.search_regex.assert_called_once()
        call_kwargs = mock_search_service.search_regex.call_args[1]
        assert call_kwargs["worktree_ids"] is None

    @pytest.mark.asyncio
    async def test_search_regex_worktree_scope_current(self):
        """Test search_regex with worktree_scope='current'."""
        mock_provider = Mock()
        mock_provider.is_connected = True

        mock_search_service = Mock()
        mock_search_service.search_regex = Mock(return_value=(
            [{"file_path": "test.py", "content": "test"}],
            {"offset": 0, "page_size": 10, "has_more": False}
        ))

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        await search_regex_impl(
            services=mock_services,
            pattern="test",
            worktree_scope="current"  # Explicit current
        )

        # 'current' should result in worktree_ids=None
        call_kwargs = mock_search_service.search_regex.call_args[1]
        assert call_kwargs["worktree_ids"] is None

    @pytest.mark.asyncio
    async def test_search_regex_worktree_scope_all(self):
        """Test search_regex with worktree_scope='all'."""
        mock_provider = Mock()
        mock_provider.is_connected = True

        mock_search_service = Mock()
        mock_search_service.search_regex = Mock(return_value=(
            [{"file_path": "test.py", "content": "test"}],
            {"offset": 0, "page_size": 10, "has_more": False}
        ))

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        await search_regex_impl(
            services=mock_services,
            pattern="test",
            worktree_scope="all"  # Search all worktrees
        )

        # 'all' should result in worktree_ids=["all"]
        call_kwargs = mock_search_service.search_regex.call_args[1]
        assert call_kwargs["worktree_ids"] == ["all"]

    @pytest.mark.asyncio
    async def test_search_regex_worktree_scope_specific_ids(self):
        """Test search_regex with comma-separated worktree IDs."""
        mock_provider = Mock()
        mock_provider.is_connected = True

        mock_search_service = Mock()
        mock_search_service.search_regex = Mock(return_value=(
            [{"file_path": "test.py", "content": "test"}],
            {"offset": 0, "page_size": 10, "has_more": False}
        ))

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        await search_regex_impl(
            services=mock_services,
            pattern="test",
            worktree_scope="abc123,def456"  # Specific IDs
        )

        # Should parse comma-separated IDs
        call_kwargs = mock_search_service.search_regex.call_args[1]
        assert call_kwargs["worktree_ids"] == ["abc123", "def456"]

    @pytest.mark.asyncio
    async def test_search_regex_worktree_scope_with_spaces(self):
        """Test search_regex handles worktree IDs with spaces."""
        mock_provider = Mock()
        mock_provider.is_connected = True

        mock_search_service = Mock()
        mock_search_service.search_regex = Mock(return_value=(
            [{"file_path": "test.py", "content": "test"}],
            {"offset": 0, "page_size": 10, "has_more": False}
        ))

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        await search_regex_impl(
            services=mock_services,
            pattern="test",
            worktree_scope=" abc123 , def456 "  # IDs with whitespace
        )

        # Should strip whitespace
        call_kwargs = mock_search_service.search_regex.call_args[1]
        assert call_kwargs["worktree_ids"] == ["abc123", "def456"]

    @pytest.mark.asyncio
    async def test_search_semantic_worktree_scope_all(self):
        """Test search_semantic with worktree_scope='all'."""
        mock_provider = Mock()
        mock_provider.is_connected = True

        mock_embedding_provider = Mock()
        mock_embedding_provider.name = "openai"
        mock_embedding_provider.model = "text-embedding-3-small"

        mock_embedding_manager = Mock()
        mock_embedding_manager.list_providers.return_value = ["openai"]
        mock_embedding_manager.get_provider.return_value = mock_embedding_provider

        mock_search_service = AsyncMock()
        mock_search_service.search_semantic = AsyncMock(return_value=(
            [{"file_path": "test.py", "content": "test"}],
            {"offset": 0, "page_size": 10, "has_more": False}
        ))

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        await search_semantic_impl(
            services=mock_services,
            embedding_manager=mock_embedding_manager,
            query="test query",
            worktree_scope="all"
        )

        # Verify search_semantic was called with worktree_ids=["all"]
        call_kwargs = mock_search_service.search_semantic.call_args[1]
        assert call_kwargs["worktree_ids"] == ["all"]

    @pytest.mark.asyncio
    async def test_search_semantic_worktree_scope_specific_ids(self):
        """Test search_semantic with specific worktree IDs."""
        mock_provider = Mock()
        mock_provider.is_connected = True

        mock_embedding_provider = Mock()
        mock_embedding_provider.name = "openai"
        mock_embedding_provider.model = "text-embedding-3-small"

        mock_embedding_manager = Mock()
        mock_embedding_manager.list_providers.return_value = ["openai"]
        mock_embedding_manager.get_provider.return_value = mock_embedding_provider

        mock_search_service = AsyncMock()
        mock_search_service.search_semantic = AsyncMock(return_value=(
            [{"file_path": "test.py", "content": "test"}],
            {"offset": 0, "page_size": 10, "has_more": False}
        ))

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        await search_semantic_impl(
            services=mock_services,
            embedding_manager=mock_embedding_manager,
            query="test query",
            worktree_scope="main123,feature456"
        )

        # Verify specific IDs were parsed
        call_kwargs = mock_search_service.search_semantic.call_args[1]
        assert call_kwargs["worktree_ids"] == ["main123", "feature456"]


class TestWorktreeScopeEdgeCases:
    """Tests for edge cases in worktree_scope parameter handling."""

    @pytest.mark.asyncio
    async def test_search_regex_empty_worktree_scope(self):
        """Test search_regex with empty string worktree_scope."""
        mock_provider = Mock()
        mock_provider.is_connected = True

        mock_search_service = Mock()
        mock_search_service.search_regex = Mock(return_value=(
            [], {"offset": 0, "page_size": 10, "has_more": False}
        ))

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        await search_regex_impl(
            services=mock_services,
            pattern="test",
            worktree_scope=""  # Empty string
        )

        # Empty string should be treated like None
        call_kwargs = mock_search_service.search_regex.call_args[1]
        assert call_kwargs["worktree_ids"] is None

    @pytest.mark.asyncio
    async def test_search_regex_case_insensitive_all(self):
        """Test search_regex handles 'ALL' case-insensitively."""
        mock_provider = Mock()
        mock_provider.is_connected = True

        mock_search_service = Mock()
        mock_search_service.search_regex = Mock(return_value=(
            [], {"offset": 0, "page_size": 10, "has_more": False}
        ))

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        await search_regex_impl(
            services=mock_services,
            pattern="test",
            worktree_scope="ALL"  # Uppercase
        )

        # Should normalize to lowercase
        call_kwargs = mock_search_service.search_regex.call_args[1]
        assert call_kwargs["worktree_ids"] == ["all"]

    @pytest.mark.asyncio
    async def test_search_regex_case_insensitive_current(self):
        """Test search_regex handles 'CURRENT' case-insensitively."""
        mock_provider = Mock()
        mock_provider.is_connected = True

        mock_search_service = Mock()
        mock_search_service.search_regex = Mock(return_value=(
            [], {"offset": 0, "page_size": 10, "has_more": False}
        ))

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        await search_regex_impl(
            services=mock_services,
            pattern="test",
            worktree_scope="CURRENT"  # Uppercase
        )

        # Should normalize and treat as None
        call_kwargs = mock_search_service.search_regex.call_args[1]
        assert call_kwargs["worktree_ids"] is None

    @pytest.mark.asyncio
    async def test_search_regex_filters_empty_ids(self):
        """Test search_regex filters out empty IDs from comma list."""
        mock_provider = Mock()
        mock_provider.is_connected = True

        mock_search_service = Mock()
        mock_search_service.search_regex = Mock(return_value=(
            [], {"offset": 0, "page_size": 10, "has_more": False}
        ))

        mock_services = Mock()
        mock_services.provider = mock_provider
        mock_services.search_service = mock_search_service

        await search_regex_impl(
            services=mock_services,
            pattern="test",
            worktree_scope="abc123,,,def456,,"  # Empty elements
        )

        # Should filter out empty strings
        call_kwargs = mock_search_service.search_regex.call_args[1]
        assert call_kwargs["worktree_ids"] == ["abc123", "def456"]


class TestWorktreeScopeDocumentation:
    """Tests for worktree_scope parameter documentation."""

    def test_search_regex_worktree_scope_description_complete(self):
        """Verify search_regex worktree_scope description is comprehensive."""
        tool = TOOL_REGISTRY["search_regex"]
        props = tool.parameters["properties"]
        desc = props["worktree_scope"]["description"]

        # Should mention key concepts
        assert "worktree" in desc.lower()
        assert "current" in desc.lower() or "default" in desc.lower()
        assert "all" in desc.lower()

    def test_search_semantic_worktree_scope_description_complete(self):
        """Verify search_semantic worktree_scope description is comprehensive."""
        tool = TOOL_REGISTRY["search_semantic"]
        props = tool.parameters["properties"]
        desc = props["worktree_scope"]["description"]

        # Should mention key concepts
        assert "worktree" in desc.lower()
        assert "current" in desc.lower() or "default" in desc.lower()
        assert "all" in desc.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
