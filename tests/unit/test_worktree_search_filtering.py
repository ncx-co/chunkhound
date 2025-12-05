"""Unit tests for Phase 4 worktree-scoped search functionality.

Tests the worktree_ids parameter for search_semantic() and search_regex() methods,
verifying that searches correctly filter to owned and inherited files.

Implementation details:
- DuckDB: Uses SQL filtering with f.worktree_id IN (...) OR f.id IN (SELECT file_id FROM file_worktree_inheritance WHERE worktree_id IN (...))
- LanceDB: Uses _get_worktree_scoped_file_ids() helper to build file ID set, then filters results
"""

import pytest
from pathlib import Path
from chunkhound.core.models import File, Chunk
from chunkhound.core.types.common import ChunkType, Language


# Helper function to set file worktree association (used in tests)
def _set_file_worktree(provider, file_id: int, worktree_id: str):
    """Set the worktree_id for a file via direct SQL/table update.

    This helper is needed because the File model doesn't include worktree_id
    in its constructor - that field is managed at the database level.
    """
    # For DuckDB, execute SQL directly
    if hasattr(provider, '_connection_manager'):
        conn = provider._connection_manager.connection
        conn.execute(
            "UPDATE files SET worktree_id = ? WHERE id = ?",
            [worktree_id, file_id]
        )
    # For LanceDB, update via table
    elif hasattr(provider, '_files_table') and provider._files_table:
        # Read current file record
        results = provider._files_table.search().where(f"id = {file_id}").to_list()
        if results:
            file_data = dict(results[0])
            file_data['worktree_id'] = worktree_id
            # Use merge_insert to update
            provider._files_table.merge_insert("id").when_matched_update_all().execute([file_data])


class TestWorktreeSearchParameterHandling:
    """Test parameter handling for worktree_ids in search methods."""

    def test_search_semantic_none_searches_all_files_duckdb(self, duckdb_provider, tmp_path):
        """Test worktree_ids=None searches all files without filtering (DuckDB)."""
        # Setup: Create worktrees
        duckdb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        duckdb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create files (without worktree_id in constructor - that field is not part of File model)
        file1 = File(path="main_file.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file2 = File(path="feature_file.py", mtime=456.0, language=Language.PYTHON, size_bytes=200)

        file1_id = duckdb_provider.insert_file(file1)
        file2_id = duckdb_provider.insert_file(file2)

        # Set worktree associations via direct SQL update
        _set_file_worktree(duckdb_provider, file1_id, "wt_main")
        _set_file_worktree(duckdb_provider, file2_id, "wt_feature")

        # Create chunks
        chunks = [
            Chunk(file_id=file1_id, code="def main(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="main"),
            Chunk(file_id=file2_id, code="def feature(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="feature"),
        ]
        chunk_ids = duckdb_provider.insert_chunks_batch(chunks)

        # Create embeddings
        embeddings = [
            {"chunk_id": chunk_ids[0], "provider": "test", "model": "test-model", "dims": 4, "embedding": [1.0, 0.0, 0.0, 0.0]},
            {"chunk_id": chunk_ids[1], "provider": "test", "model": "test-model", "dims": 4, "embedding": [0.0, 1.0, 0.0, 0.0]},
        ]
        duckdb_provider.insert_embeddings_batch(embeddings)

        # Search with worktree_ids=None (should return all files)
        query_embedding = [1.0, 0.0, 0.0, 0.0]
        results, meta = duckdb_provider.search_semantic(
            query_embedding, provider="test", model="test-model", page_size=10, worktree_ids=None
        )

        # Should find chunks from both worktrees
        assert len(results) == 2
        found_symbols = {r["symbol"] for r in results}
        assert found_symbols == {"main", "feature"}

    def test_search_semantic_all_explicit_searches_all_files_duckdb(self, duckdb_provider, tmp_path):
        """Test worktree_ids=["all"] explicitly searches all files (DuckDB)."""
        # Setup: Create worktrees
        duckdb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        duckdb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create files in different worktrees
        file1 = File(path="main_file.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file2 = File(path="feature_file.py", mtime=456.0, language=Language.PYTHON, size_bytes=200)

        file1_id = duckdb_provider.insert_file(file1)
        file2_id = duckdb_provider.insert_file(file2)

        # Set worktree associations
        _set_file_worktree(duckdb_provider, file1_id, "wt_main")
        _set_file_worktree(duckdb_provider, file2_id, "wt_feature")

        # Create chunks
        chunks = [
            Chunk(file_id=file1_id, code="def main(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="main"),
            Chunk(file_id=file2_id, code="def feature(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="feature"),
        ]
        chunk_ids = duckdb_provider.insert_chunks_batch(chunks)

        # Create embeddings
        embeddings = [
            {"chunk_id": chunk_ids[0], "provider": "test", "model": "test-model", "dims": 4, "embedding": [1.0, 0.0, 0.0, 0.0]},
            {"chunk_id": chunk_ids[1], "provider": "test", "model": "test-model", "dims": 4, "embedding": [0.0, 1.0, 0.0, 0.0]},
        ]
        duckdb_provider.insert_embeddings_batch(embeddings)

        # Search with worktree_ids=["all"] (should return all files)
        query_embedding = [1.0, 0.0, 0.0, 0.0]
        results, meta = duckdb_provider.search_semantic(
            query_embedding, provider="test", model="test-model", page_size=10, worktree_ids=["all"]
        )

        # Should find chunks from both worktrees
        assert len(results) == 2
        found_symbols = {r["symbol"] for r in results}
        assert found_symbols == {"main", "feature"}

    def test_search_semantic_specific_worktree_filters_duckdb(self, duckdb_provider, tmp_path):
        """Test worktree_ids=["specific_id"] filters to that worktree (DuckDB)."""
        # Setup: Create worktrees
        duckdb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        duckdb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create files in different worktrees
        file1 = File(path="main_file.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file2 = File(path="feature_file.py", mtime=456.0, language=Language.PYTHON, size_bytes=200)

        file1_id = duckdb_provider.insert_file(file1)
        file2_id = duckdb_provider.insert_file(file2)

        # Set worktree associations
        _set_file_worktree(duckdb_provider, file1_id, "wt_main")
        _set_file_worktree(duckdb_provider, file2_id, "wt_feature")

        # Create chunks
        chunks = [
            Chunk(file_id=file1_id, code="def main(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="main"),
            Chunk(file_id=file2_id, code="def feature(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="feature"),
        ]
        chunk_ids = duckdb_provider.insert_chunks_batch(chunks)

        # Create embeddings
        embeddings = [
            {"chunk_id": chunk_ids[0], "provider": "test", "model": "test-model", "dims": 4, "embedding": [1.0, 0.0, 0.0, 0.0]},
            {"chunk_id": chunk_ids[1], "provider": "test", "model": "test-model", "dims": 4, "embedding": [0.0, 1.0, 0.0, 0.0]},
        ]
        duckdb_provider.insert_embeddings_batch(embeddings)

        # Search with worktree_ids=["wt_main"] (should only return main worktree files)
        query_embedding = [1.0, 0.0, 0.0, 0.0]
        results, meta = duckdb_provider.search_semantic(
            query_embedding, provider="test", model="test-model", page_size=10, worktree_ids=["wt_main"]
        )

        # Should only find chunks from wt_main
        assert len(results) == 1
        assert results[0]["symbol"] == "main"

    def test_search_regex_none_searches_all_files_duckdb(self, duckdb_provider, tmp_path):
        """Test worktree_ids=None searches all files for regex (DuckDB)."""
        # Setup: Create worktrees
        duckdb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        duckdb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create files in different worktrees
        file1 = File(path="main_file.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file2 = File(path="feature_file.py", mtime=456.0, language=Language.PYTHON, size_bytes=200)

        file1_id = duckdb_provider.insert_file(file1)
        file2_id = duckdb_provider.insert_file(file2)

        # Set worktree associations
        _set_file_worktree(duckdb_provider, file1_id, "wt_main")
        _set_file_worktree(duckdb_provider, file2_id, "wt_feature")

        # Create chunks
        chunks = [
            Chunk(file_id=file1_id, code="def main(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="main"),
            Chunk(file_id=file2_id, code="def feature(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="feature"),
        ]
        duckdb_provider.insert_chunks_batch(chunks)

        # Search with worktree_ids=None (should return all files)
        results, meta = duckdb_provider.search_regex(pattern="def", page_size=10, worktree_ids=None)

        # Should find chunks from both worktrees
        assert len(results) == 2
        found_symbols = {r["name"] for r in results}
        assert found_symbols == {"main", "feature"}

    def test_search_regex_specific_worktree_filters_duckdb(self, duckdb_provider, tmp_path):
        """Test worktree_ids=["specific_id"] filters regex search (DuckDB)."""
        # Setup: Create worktrees
        duckdb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        duckdb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create files in different worktrees
        file1 = File(path="main_file.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file2 = File(path="feature_file.py", mtime=456.0, language=Language.PYTHON, size_bytes=200)

        file1_id = duckdb_provider.insert_file(file1)
        file2_id = duckdb_provider.insert_file(file2)

        # Set worktree associations
        _set_file_worktree(duckdb_provider, file1_id, "wt_main")
        _set_file_worktree(duckdb_provider, file2_id, "wt_feature")

        # Create chunks
        chunks = [
            Chunk(file_id=file1_id, code="def main(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="main"),
            Chunk(file_id=file2_id, code="def feature(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="feature"),
        ]
        duckdb_provider.insert_chunks_batch(chunks)

        # Search with worktree_ids=["wt_feature"] (should only return feature worktree files)
        results, meta = duckdb_provider.search_regex(pattern="def", page_size=10, worktree_ids=["wt_feature"])

        # Should only find chunks from wt_feature
        assert len(results) == 1
        assert results[0]["name"] == "feature"


class TestWorktreeOwnedFileFiltering:
    """Test that search returns files where worktree_id matches."""

    def test_search_semantic_returns_owned_files_duckdb(self, duckdb_provider, tmp_path):
        """Test semantic search returns files owned by specified worktree (DuckDB)."""
        # Setup: Create worktrees
        duckdb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        duckdb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create file owned by wt_feature
        file1 = File(path="feature_owned.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file1_id = duckdb_provider.insert_file(file1)
        _set_file_worktree(duckdb_provider, file1_id, "wt_feature")

        # Create chunk
        chunks = [
            Chunk(file_id=file1_id, code="def feature_owned(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="feature_owned"),
        ]
        chunk_ids = duckdb_provider.insert_chunks_batch(chunks)

        # Create embedding
        embeddings = [
            {"chunk_id": chunk_ids[0], "provider": "test", "model": "test-model", "dims": 4, "embedding": [1.0, 0.0, 0.0, 0.0]},
        ]
        duckdb_provider.insert_embeddings_batch(embeddings)

        # Search with worktree_ids=["wt_feature"]
        query_embedding = [1.0, 0.0, 0.0, 0.0]
        results, meta = duckdb_provider.search_semantic(
            query_embedding, provider="test", model="test-model", page_size=10, worktree_ids=["wt_feature"]
        )

        # Should find the owned file
        assert len(results) == 1
        assert results[0]["symbol"] == "feature_owned"
        assert results[0]["file_path"] == "feature_owned.py"

    def test_search_regex_returns_owned_files_duckdb(self, duckdb_provider, tmp_path):
        """Test regex search returns files owned by specified worktree (DuckDB)."""
        # Setup: Create worktrees
        duckdb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        duckdb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create file owned by wt_main
        file1 = File(path="main_owned.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file1_id = duckdb_provider.insert_file(file1)
        _set_file_worktree(duckdb_provider, file1_id, "wt_main")

        # Create chunk
        chunks = [
            Chunk(file_id=file1_id, code="def main_owned(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="main_owned"),
        ]
        duckdb_provider.insert_chunks_batch(chunks)

        # Search with worktree_ids=["wt_main"]
        results, meta = duckdb_provider.search_regex(pattern="main_owned", page_size=10, worktree_ids=["wt_main"])

        # Should find the owned file
        assert len(results) == 1
        assert results[0]["name"] == "main_owned"


class TestWorktreeInheritedFileFiltering:
    """Test that search returns files inherited by specified worktree."""

    def test_search_semantic_returns_inherited_files_duckdb(self, duckdb_provider, tmp_path):
        """Test semantic search returns inherited files via file_worktree_inheritance (DuckDB)."""
        # Setup: Create worktrees
        duckdb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        duckdb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create file owned by main
        file1 = File(path="shared.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file1_id = duckdb_provider.insert_file(file1)
        _set_file_worktree(duckdb_provider, file1_id, "wt_main")

        # Create inheritance: feature inherits shared.py from main
        duckdb_provider.create_file_inheritances_batch([
            {"worktree_id": "wt_feature", "file_id": file1_id, "source_worktree_id": "wt_main"}
        ])

        # Create chunk
        chunks = [
            Chunk(file_id=file1_id, code="def shared(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="shared"),
        ]
        chunk_ids = duckdb_provider.insert_chunks_batch(chunks)

        # Create embedding
        embeddings = [
            {"chunk_id": chunk_ids[0], "provider": "test", "model": "test-model", "dims": 4, "embedding": [1.0, 0.0, 0.0, 0.0]},
        ]
        duckdb_provider.insert_embeddings_batch(embeddings)

        # Search with worktree_ids=["wt_feature"]
        query_embedding = [1.0, 0.0, 0.0, 0.0]
        results, meta = duckdb_provider.search_semantic(
            query_embedding, provider="test", model="test-model", page_size=10, worktree_ids=["wt_feature"]
        )

        # Should find the inherited file
        assert len(results) == 1
        assert results[0]["symbol"] == "shared"
        assert results[0]["file_path"] == "shared.py"

    def test_search_regex_returns_inherited_files_duckdb(self, duckdb_provider, tmp_path):
        """Test regex search returns inherited files via file_worktree_inheritance (DuckDB)."""
        # Setup: Create worktrees
        duckdb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        duckdb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create file owned by main
        file1 = File(path="utils.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file1_id = duckdb_provider.insert_file(file1)
        _set_file_worktree(duckdb_provider, file1_id, "wt_main")

        # Create inheritance: feature inherits utils.py from main
        duckdb_provider.create_file_inheritances_batch([
            {"worktree_id": "wt_feature", "file_id": file1_id, "source_worktree_id": "wt_main"}
        ])

        # Create chunk
        chunks = [
            Chunk(file_id=file1_id, code="def utility(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="utility"),
        ]
        duckdb_provider.insert_chunks_batch(chunks)

        # Search with worktree_ids=["wt_feature"]
        results, meta = duckdb_provider.search_regex(pattern="utility", page_size=10, worktree_ids=["wt_feature"])

        # Should find the inherited file
        assert len(results) == 1
        assert results[0]["name"] == "utility"


class TestWorktreeCombinedFiltering:
    """Test that search returns BOTH owned AND inherited files."""

    def test_search_semantic_returns_owned_and_inherited_duckdb(self, duckdb_provider, tmp_path):
        """Test semantic search returns both owned and inherited files (DuckDB)."""
        # Setup: Create worktrees
        duckdb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        duckdb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create owned file in feature
        file1 = File(path="owned.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file1_id = duckdb_provider.insert_file(file1)

        # Create inherited file from main
        file2 = File(path="inherited.py", mtime=456.0, language=Language.PYTHON, size_bytes=200)
        file2_id = duckdb_provider.insert_file(file2)

        # Set worktree associations
        _set_file_worktree(duckdb_provider, file1_id, "wt_feature")
        _set_file_worktree(duckdb_provider, file2_id, "wt_main")

        # Create inheritance
        duckdb_provider.create_file_inheritances_batch([
            {"worktree_id": "wt_feature", "file_id": file2_id, "source_worktree_id": "wt_main"}
        ])

        # Create chunks
        chunks = [
            Chunk(file_id=file1_id, code="def owned_func(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="owned_func"),
            Chunk(file_id=file2_id, code="def inherited_func(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="inherited_func"),
        ]
        chunk_ids = duckdb_provider.insert_chunks_batch(chunks)

        # Create embeddings
        embeddings = [
            {"chunk_id": chunk_ids[0], "provider": "test", "model": "test-model", "dims": 4, "embedding": [1.0, 0.0, 0.0, 0.0]},
            {"chunk_id": chunk_ids[1], "provider": "test", "model": "test-model", "dims": 4, "embedding": [0.9, 0.1, 0.0, 0.0]},
        ]
        duckdb_provider.insert_embeddings_batch(embeddings)

        # Search with worktree_ids=["wt_feature"]
        query_embedding = [1.0, 0.0, 0.0, 0.0]
        results, meta = duckdb_provider.search_semantic(
            query_embedding, provider="test", model="test-model", page_size=10, worktree_ids=["wt_feature"]
        )

        # Should find both owned and inherited files
        assert len(results) == 2
        found_symbols = {r["symbol"] for r in results}
        assert found_symbols == {"owned_func", "inherited_func"}

    def test_search_regex_returns_owned_and_inherited_duckdb(self, duckdb_provider, tmp_path):
        """Test regex search returns both owned and inherited files (DuckDB)."""
        # Setup: Create worktrees
        duckdb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        duckdb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create owned file in feature
        file1 = File(path="owned.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file1_id = duckdb_provider.insert_file(file1)

        # Create inherited file from main
        file2 = File(path="inherited.py", mtime=456.0, language=Language.PYTHON, size_bytes=200)
        file2_id = duckdb_provider.insert_file(file2)

        # Set worktree associations
        _set_file_worktree(duckdb_provider, file1_id, "wt_feature")
        _set_file_worktree(duckdb_provider, file2_id, "wt_main")

        # Create inheritance
        duckdb_provider.create_file_inheritances_batch([
            {"worktree_id": "wt_feature", "file_id": file2_id, "source_worktree_id": "wt_main"}
        ])

        # Create chunks with matching pattern
        chunks = [
            Chunk(file_id=file1_id, code="class OwnedClass: pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.CLASS, language=Language.PYTHON, symbol="OwnedClass"),
            Chunk(file_id=file2_id, code="class InheritedClass: pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.CLASS, language=Language.PYTHON, symbol="InheritedClass"),
        ]
        duckdb_provider.insert_chunks_batch(chunks)

        # Search with worktree_ids=["wt_feature"]
        results, meta = duckdb_provider.search_regex(pattern="class.*Class", page_size=10, worktree_ids=["wt_feature"])

        # Should find both owned and inherited files
        assert len(results) == 2
        found_symbols = {r["name"] for r in results}
        assert found_symbols == {"OwnedClass", "InheritedClass"}


class TestWorktreeEmptyResults:
    """Test that search returns empty when worktree has no files."""

    def test_search_semantic_empty_when_no_files_duckdb(self, duckdb_provider, tmp_path):
        """Test semantic search returns empty for worktree with no files (DuckDB)."""
        # Setup: Create worktrees
        duckdb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        duckdb_provider.upsert_worktree("wt_empty", str(tmp_path / "empty"), is_main=False, main_worktree_id="wt_main")

        # Create file in main only
        file1 = File(path="main_only.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file1_id = duckdb_provider.insert_file(file1)
        _set_file_worktree(duckdb_provider, file1_id, "wt_main")

        # Create chunk
        chunks = [
            Chunk(file_id=file1_id, code="def main_only(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="main_only"),
        ]
        chunk_ids = duckdb_provider.insert_chunks_batch(chunks)

        # Create embedding
        embeddings = [
            {"chunk_id": chunk_ids[0], "provider": "test", "model": "test-model", "dims": 4, "embedding": [1.0, 0.0, 0.0, 0.0]},
        ]
        duckdb_provider.insert_embeddings_batch(embeddings)

        # Search with worktree_ids=["wt_empty"]
        query_embedding = [1.0, 0.0, 0.0, 0.0]
        results, meta = duckdb_provider.search_semantic(
            query_embedding, provider="test", model="test-model", page_size=10, worktree_ids=["wt_empty"]
        )

        # Should return empty
        assert len(results) == 0
        assert meta["total"] == 0

    def test_search_regex_empty_when_no_files_duckdb(self, duckdb_provider, tmp_path):
        """Test regex search returns empty for worktree with no files (DuckDB)."""
        # Setup: Create worktrees
        duckdb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        duckdb_provider.upsert_worktree("wt_empty", str(tmp_path / "empty"), is_main=False, main_worktree_id="wt_main")

        # Create file in main only
        file1 = File(path="main_only.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file1_id = duckdb_provider.insert_file(file1)
        _set_file_worktree(duckdb_provider, file1_id, "wt_main")

        # Create chunk
        chunks = [
            Chunk(file_id=file1_id, code="def main_only(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="main_only"),
        ]
        duckdb_provider.insert_chunks_batch(chunks)

        # Search with worktree_ids=["wt_empty"]
        results, meta = duckdb_provider.search_regex(pattern="main_only", page_size=10, worktree_ids=["wt_empty"])

        # Should return empty
        assert len(results) == 0
        assert meta["total"] == 0


# LanceDB-specific tests (same structure, different provider)

class TestWorktreeSearchLanceDB:
    """Test worktree search filtering with LanceDB provider."""

    def test_search_semantic_specific_worktree_lancedb(self, lancedb_provider, tmp_path):
        """Test worktree_ids filtering for semantic search (LanceDB)."""
        # Setup: Create worktrees
        lancedb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        lancedb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create files
        file1 = File(path="main_file.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file2 = File(path="feature_file.py", mtime=456.0, language=Language.PYTHON, size_bytes=200)

        file1_id = lancedb_provider.insert_file(file1)
        file2_id = lancedb_provider.insert_file(file2)

        # Set worktree associations
        _set_file_worktree(lancedb_provider, file1_id, "wt_main")
        _set_file_worktree(lancedb_provider, file2_id, "wt_feature")

        # Create chunks
        chunks = [
            Chunk(file_id=file1_id, code="def main(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="main"),
            Chunk(file_id=file2_id, code="def feature(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="feature"),
        ]
        chunk_ids = lancedb_provider.insert_chunks_batch(chunks)

        # Create embeddings
        embeddings = [
            {"chunk_id": chunk_ids[0], "provider": "test", "model": "test-model", "dims": 4, "embedding": [1.0, 0.0, 0.0, 0.0]},
            {"chunk_id": chunk_ids[1], "provider": "test", "model": "test-model", "dims": 4, "embedding": [0.0, 1.0, 0.0, 0.0]},
        ]
        lancedb_provider.insert_embeddings_batch(embeddings)

        # Search with worktree_ids=["wt_main"]
        query_embedding = [1.0, 0.0, 0.0, 0.0]
        results, meta = lancedb_provider.search_semantic(
            query_embedding, provider="test", model="test-model", page_size=10, worktree_ids=["wt_main"]
        )

        # Should only find main worktree file
        assert len(results) == 1
        assert results[0]["symbol"] == "main"

    def test_search_semantic_inherited_files_lancedb(self, lancedb_provider, tmp_path):
        """Test semantic search returns inherited files (LanceDB)."""
        # Setup: Create worktrees
        lancedb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        lancedb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create file in main
        file1 = File(path="shared.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file1_id = lancedb_provider.insert_file(file1)
        _set_file_worktree(lancedb_provider, file1_id, "wt_main")

        # Create inheritance
        lancedb_provider.create_file_inheritances_batch([
            {"worktree_id": "wt_feature", "file_id": file1_id, "source_worktree_id": "wt_main"}
        ])

        # Create chunk
        chunks = [
            Chunk(file_id=file1_id, code="def shared(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="shared"),
        ]
        chunk_ids = lancedb_provider.insert_chunks_batch(chunks)

        # Create embedding
        embeddings = [
            {"chunk_id": chunk_ids[0], "provider": "test", "model": "test-model", "dims": 4, "embedding": [1.0, 0.0, 0.0, 0.0]},
        ]
        lancedb_provider.insert_embeddings_batch(embeddings)

        # Search with worktree_ids=["wt_feature"]
        query_embedding = [1.0, 0.0, 0.0, 0.0]
        results, meta = lancedb_provider.search_semantic(
            query_embedding, provider="test", model="test-model", page_size=10, worktree_ids=["wt_feature"]
        )

        # Should find inherited file
        assert len(results) == 1
        assert results[0]["symbol"] == "shared"

    def test_search_regex_specific_worktree_lancedb(self, lancedb_provider, tmp_path):
        """Test worktree_ids filtering for regex search (LanceDB)."""
        # Setup: Create worktrees
        lancedb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        lancedb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create files
        file1 = File(path="main_file.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file2 = File(path="feature_file.py", mtime=456.0, language=Language.PYTHON, size_bytes=200)

        file1_id = lancedb_provider.insert_file(file1)
        file2_id = lancedb_provider.insert_file(file2)

        # Set worktree associations
        _set_file_worktree(lancedb_provider, file1_id, "wt_main")
        _set_file_worktree(lancedb_provider, file2_id, "wt_feature")

        # Create chunks
        chunks = [
            Chunk(file_id=file1_id, code="def main(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="main"),
            Chunk(file_id=file2_id, code="def feature(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="feature"),
        ]
        lancedb_provider.insert_chunks_batch(chunks)

        # Search with worktree_ids=["wt_feature"]
        results, meta = lancedb_provider.search_regex(pattern="def", page_size=10, worktree_ids=["wt_feature"])

        # Should only find feature worktree file
        assert len(results) == 1
        assert results[0]["symbol"] == "feature"

    def test_search_regex_inherited_files_lancedb(self, lancedb_provider, tmp_path):
        """Test regex search returns inherited files (LanceDB)."""
        # Setup: Create worktrees
        lancedb_provider.upsert_worktree("wt_main", str(tmp_path / "main"), is_main=True)
        lancedb_provider.upsert_worktree("wt_feature", str(tmp_path / "feature"), is_main=False, main_worktree_id="wt_main")

        # Create file in main
        file1 = File(path="utils.py", mtime=123.0, language=Language.PYTHON, size_bytes=100)
        file1_id = lancedb_provider.insert_file(file1)
        _set_file_worktree(lancedb_provider, file1_id, "wt_main")

        # Create inheritance
        lancedb_provider.create_file_inheritances_batch([
            {"worktree_id": "wt_feature", "file_id": file1_id, "source_worktree_id": "wt_main"}
        ])

        # Create chunk
        chunks = [
            Chunk(file_id=file1_id, code="def utility(): pass", start_line=1, end_line=1,
                  chunk_type=ChunkType.FUNCTION, language=Language.PYTHON, symbol="utility"),
        ]
        lancedb_provider.insert_chunks_batch(chunks)

        # Search with worktree_ids=["wt_feature"]
        results, meta = lancedb_provider.search_regex(pattern="utility", page_size=10, worktree_ids=["wt_feature"])

        # Should find inherited file
        assert len(results) == 1
        assert results[0]["symbol"] == "utility"


# Fixtures

@pytest.fixture
def duckdb_provider(tmp_path):
    """Create DuckDB provider for testing."""
    pytest.importorskip("duckdb")

    from chunkhound.core.config.database_config import DatabaseConfig
    from chunkhound.providers.database.duckdb_provider import DuckDBProvider

    config = DatabaseConfig(path=tmp_path, provider="duckdb")
    db_path = config.get_db_path()

    provider = DuckDBProvider(str(db_path), base_directory=tmp_path)
    provider.connect()

    yield provider

    provider.disconnect()


@pytest.fixture
def lancedb_provider(tmp_path):
    """Create LanceDB provider for testing."""
    pytest.importorskip("lancedb")

    from chunkhound.core.config.database_config import DatabaseConfig
    from chunkhound.providers.database.lancedb_provider import LanceDBProvider

    config = DatabaseConfig(path=tmp_path, provider="lancedb")
    db_path = config.get_db_path()

    provider = LanceDBProvider(str(db_path), base_directory=tmp_path)
    provider.connect()

    yield provider

    provider.disconnect()
