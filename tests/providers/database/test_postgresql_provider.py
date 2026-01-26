"""Tests for PostgreSQL provider implementation.

NOTE: These tests require a running PostgreSQL instance with pgvector extension.
Set environment variables to configure the test database:
    CHUNKHOUND_TEST_PG_HOST=localhost
    CHUNKHOUND_TEST_PG_PORT=5432
    CHUNKHOUND_TEST_PG_DATABASE=chunkhound_test
    CHUNKHOUND_TEST_PG_USER=postgres
    CHUNKHOUND_TEST_PG_PASSWORD=postgres

Or use a connection string:
    CHUNKHOUND_TEST_PG_CONNECTION_STRING=postgresql://user:pass@localhost/dbname
"""

import os
from pathlib import Path

import pytest

from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.providers.database.postgresql_provider import PostgreSQLProvider


def get_test_connection_string() -> str | None:
    """Get PostgreSQL connection string from environment variables."""
    # Try connection string first
    conn_str = os.getenv("CHUNKHOUND_TEST_PG_CONNECTION_STRING")
    if conn_str:
        return conn_str

    # Build from individual components
    host = os.getenv("CHUNKHOUND_TEST_PG_HOST", "localhost")
    port = os.getenv("CHUNKHOUND_TEST_PG_PORT", "5432")
    database = os.getenv("CHUNKHOUND_TEST_PG_DATABASE")
    user = os.getenv("CHUNKHOUND_TEST_PG_USER")
    password = os.getenv("CHUNKHOUND_TEST_PG_PASSWORD")

    if not database or not user:
        return None

    conn_str = f"postgresql://{user}"
    if password:
        conn_str += f":{password}"
    conn_str += f"@{host}:{port}/{database}"

    return conn_str


@pytest.fixture
def connection_string():
    """Get test database connection string."""
    conn_str = get_test_connection_string()
    if not conn_str:
        pytest.skip(
            "PostgreSQL test database not configured. "
            "Set CHUNKHOUND_TEST_PG_* environment variables."
        )
    return conn_str


@pytest.fixture
def provider(connection_string, tmp_path):
    """Create a PostgreSQL provider instance for testing."""
    provider = PostgreSQLProvider(
        connection_string=connection_string,
        base_directory=tmp_path,
    )
    provider.connect()

    yield provider

    # Cleanup: drop all tables
    try:
        # Get all tables
        tables = provider._connection_manager.fetch_sync(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            """
        )

        # Drop all tables
        for row in tables:
            table_name = row["table_name"]
            provider._connection_manager.execute_sync(
                f"DROP TABLE IF EXISTS {table_name} CASCADE"
            )

    finally:
        provider.disconnect()


def test_provider_initialization(provider):
    """Test that provider initializes correctly."""
    assert provider.provider_type == "postgresql"
    assert provider.is_connected


def test_health_check(provider):
    """Test provider health check."""
    health = provider.health_check()

    assert health["provider"] == "postgresql"
    assert health["connected"] is True
    assert "version" in health
    assert "tables" in health


def test_schema_creation(provider):
    """Test that schema is created properly."""
    # Check that required tables exist
    required_tables = ["files", "chunks", "embeddings_1536", "schema_version"]

    for table_name in required_tables:
        assert provider._table_exists(table_name), f"Table {table_name} should exist"


def test_file_operations(provider, tmp_path):
    """Test file CRUD operations."""
    # Create a test file
    test_file = File(
        path=str(tmp_path / "test.py"),
        mtime=1234567890.0,
        size_bytes=100,
        language=Language.PYTHON,
    )

    # Insert file
    file_id = provider.insert_file(test_file)
    assert file_id > 0

    # Get file by path
    retrieved = provider.get_file_by_path(str(tmp_path / "test.py"))
    assert retrieved is not None
    assert retrieved["id"] == file_id
    assert retrieved["path"] == str(tmp_path / "test.py")
    assert retrieved["language"] == "python"

    # Get file by ID
    retrieved_by_id = provider.get_file_by_id(file_id)
    assert retrieved_by_id is not None
    assert retrieved_by_id["id"] == file_id


def test_chunk_operations(provider, tmp_path):
    """Test chunk CRUD operations."""
    # First create a file
    test_file = File(
        path=str(tmp_path / "test.py"),
        mtime=1234567890.0,
        size_bytes=100,
        language=Language.PYTHON,
    )
    file_id = provider.insert_file(test_file)

    # Create a test chunk
    test_chunk = Chunk(
        file_id=file_id,
        chunk_type=ChunkType.FUNCTION,
        symbol="test_function",
        code="def test_function():\n    pass",
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=30,
        language=Language.PYTHON,
    )

    # Insert chunk
    chunk_id = provider.insert_chunk(test_chunk)
    assert chunk_id > 0

    # Get chunk by ID
    retrieved = provider.get_chunk_by_id(chunk_id)
    assert retrieved is not None
    assert retrieved["id"] == chunk_id
    assert retrieved["file_id"] == file_id
    assert retrieved["symbol"] == "test_function"
    assert retrieved["chunk_type"] == "function"


def test_batch_chunk_operations(provider, tmp_path):
    """Test batch chunk insert."""
    # First create a file
    test_file = File(
        path=str(tmp_path / "test.py"),
        mtime=1234567890.0,
        size_bytes=100,
        language=Language.PYTHON,
    )
    file_id = provider.insert_file(test_file)

    # Create multiple chunks
    chunks = [
        Chunk(
            file_id=file_id,
            chunk_type=ChunkType.FUNCTION,
            symbol=f"test_function_{i}",
            code=f"def test_function_{i}():\n    pass",
            start_line=i * 3,
            end_line=i * 3 + 2,
            start_byte=i * 30,
            end_byte=i * 30 + 30,
            language=Language.PYTHON,
        )
        for i in range(5)
    ]

    # Insert chunks in batch
    chunk_ids = provider.insert_chunks_batch(chunks)
    assert len(chunk_ids) == 5
    assert all(chunk_id > 0 for chunk_id in chunk_ids)


def test_embedding_table_creation(provider):
    """Test that embedding tables are created dynamically."""
    # Ensure table for 768 dimensions
    table_name = provider._ensure_embedding_table_exists(768)
    assert table_name == "embeddings_768"

    # Verify table exists
    assert provider._table_exists("embeddings_768")

    # Ensure table for 384 dimensions
    table_name = provider._ensure_embedding_table_exists(384)
    assert table_name == "embeddings_384"

    # Verify table exists
    assert provider._table_exists("embeddings_384")


def test_get_all_embedding_tables(provider):
    """Test getting list of all embedding tables."""
    # Create a few embedding tables
    provider._ensure_embedding_table_exists(768)
    provider._ensure_embedding_table_exists(384)

    # Get all tables
    tables = provider._get_all_embedding_tables()

    assert "embeddings_1536" in tables  # Default table
    assert "embeddings_768" in tables
    assert "embeddings_384" in tables


def test_indexes_creation(provider):
    """Test that indexes are created."""
    # Indexes are created during connection, verify they exist
    # Query PostgreSQL system tables for indexes
    indexes = provider._connection_manager.fetch_sync(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'public'
        AND tablename IN ('files', 'chunks')
        """
    )

    index_names = [row["indexname"] for row in indexes]

    # Check for key indexes
    assert any("files_path" in name for name in index_names)
    assert any("chunks_file_id" in name for name in index_names)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
