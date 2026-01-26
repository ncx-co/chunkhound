"""PostgreSQL chunk repository implementation - handles chunk CRUD operations."""

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.models import Chunk
from chunkhound.core.types.common import ChunkType, Language

if TYPE_CHECKING:
    from chunkhound.providers.database.postgresql.connection_manager import (
        PostgreSQLConnectionManager,
    )


class PostgreSQLChunkRepository:
    """Repository for chunk CRUD operations in PostgreSQL."""

    def __init__(
        self, connection_manager: "PostgreSQLConnectionManager", provider=None
    ):
        """Initialize chunk repository.

        Args:
            connection_manager: PostgreSQL connection manager instance
            provider: Optional provider instance for transaction-aware connections
        """
        self._connection_manager = connection_manager
        self._provider = provider

    def insert_chunk(self, chunk: Chunk) -> int:
        """Insert chunk record and return chunk ID.

        Args:
            chunk: Chunk model to insert

        Returns:
            Chunk ID
        """
        try:
            # Delegate to provider's executor for thread safety
            if self._provider:
                return self._provider._execute_in_db_thread_sync(
                    "insert_chunk_single", chunk
                )
            else:
                # Fallback for tests
                result = self._connection_manager.fetchval_sync(
                    """
                    INSERT INTO chunks (file_id, chunk_type, symbol, code, start_line, end_line,
                                      start_byte, end_byte, language, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                    """,
                    chunk.file_id,
                    chunk.chunk_type.value if chunk.chunk_type else None,
                    chunk.symbol,
                    chunk.code,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.start_byte,
                    chunk.end_byte,
                    chunk.language.value if chunk.language else None,
                    json.dumps(chunk.metadata) if chunk.metadata else None,
                )

                return result

        except Exception as e:
            logger.error(f"Failed to insert chunk: {e}")
            raise

    def insert_chunks_batch(self, chunks: list[Chunk]) -> list[int]:
        """Insert multiple chunks in batch using optimized PostgreSQL bulk loading.

        Args:
            chunks: List of Chunk models to insert

        Returns:
            List of chunk IDs
        """
        if not chunks:
            return []

        try:
            # Prepare values for bulk INSERT statement
            values_list = []
            for chunk in chunks:
                values_list.append(
                    (
                        chunk.file_id,
                        chunk.chunk_type.value if chunk.chunk_type else None,
                        chunk.symbol,
                        chunk.code,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.start_byte,
                        chunk.end_byte,
                        chunk.language.value if chunk.language else None,
                        json.dumps(chunk.metadata) if chunk.metadata else None,
                    )
                )

            # Use PostgreSQL UNNEST for bulk insert
            placeholders = []
            params = []
            for i, values in enumerate(values_list):
                base = i * 10
                placeholders.append(
                    f"(${base+1}, ${base+2}, ${base+3}, ${base+4}, ${base+5}, "
                    f"${base+6}, ${base+7}, ${base+8}, ${base+9}, ${base+10})"
                )
                params.extend(values)

            query = f"""
                INSERT INTO chunks (file_id, chunk_type, symbol, code, start_line, end_line,
                                  start_byte, end_byte, language, metadata)
                VALUES {', '.join(placeholders)}
                RETURNING id
            """

            # Execute bulk insert and get all IDs
            if self._provider:
                # Provider handles executor thread
                results = self._connection_manager.fetch_sync(query, *params)
            else:
                results = self._connection_manager.fetch_sync(query, *params)

            chunk_ids = [result["id"] for result in results]
            return chunk_ids

        except Exception as e:
            logger.error(f"Failed to insert chunks batch: {e}")
            raise

    def get_chunk_by_id(
        self, chunk_id: int, as_model: bool = False
    ) -> dict[str, Any] | Chunk | None:
        """Get chunk record by ID.

        Args:
            chunk_id: Chunk ID to search for
            as_model: Return Chunk model if True, dict otherwise

        Returns:
            Chunk record or None if not found
        """
        try:
            if self._provider:
                result = self._provider._execute_in_db_thread_sync(
                    "get_chunk_by_id_query", chunk_id
                )
            else:
                result = self._connection_manager.fetch_sync(
                    """
                    SELECT id, file_id, chunk_type, symbol, code, start_line, end_line,
                           start_byte, end_byte, language, created_at, updated_at, metadata
                    FROM chunks
                    WHERE id = $1
                    """,
                    chunk_id,
                )

            if not result or len(result) == 0:
                return None

            row = result[0]
            chunk_dict = {
                "id": row["id"],
                "file_id": row["file_id"],
                "chunk_type": row["chunk_type"],
                "symbol": row["symbol"],
                "code": row["code"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "start_byte": row["start_byte"],
                "end_byte": row["end_byte"],
                "language": row["language"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            }

            if as_model:
                return Chunk(
                    file_id=row["file_id"],
                    chunk_type=ChunkType(row["chunk_type"])
                    if row["chunk_type"]
                    else ChunkType.UNKNOWN,
                    symbol=row["symbol"],
                    code=row["code"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    start_byte=row["start_byte"],
                    end_byte=row["end_byte"],
                    language=Language(row["language"])
                    if row["language"]
                    else Language.UNKNOWN,
                    metadata=chunk_dict["metadata"],
                )

            return chunk_dict

        except Exception as e:
            logger.error(f"Failed to get chunk by ID {chunk_id}: {e}")
            return None

    def get_chunks_by_file_id(
        self, file_id: int, as_model: bool = False
    ) -> list[dict[str, Any] | Chunk]:
        """Get all chunks for a specific file.

        Args:
            file_id: File ID to get chunks for
            as_model: Return Chunk models if True, dicts otherwise

        Returns:
            List of chunk records
        """
        try:
            if self._provider:
                results = self._provider._execute_in_db_thread_sync(
                    "get_chunks_by_file_id_query", file_id
                )
            else:
                results = self._connection_manager.fetch_sync(
                    """
                    SELECT id, file_id, chunk_type, symbol, code, start_line, end_line,
                           start_byte, end_byte, language, created_at, updated_at, metadata
                    FROM chunks
                    WHERE file_id = $1
                    ORDER BY start_line
                    """,
                    file_id,
                )

            chunks = []
            for row in results:
                chunk_dict = {
                    "id": row["id"],
                    "file_id": row["file_id"],
                    "chunk_type": row["chunk_type"],
                    "symbol": row["symbol"],
                    "code": row["code"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "start_byte": row["start_byte"],
                    "end_byte": row["end_byte"],
                    "language": row["language"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }

                if as_model:
                    chunks.append(
                        Chunk(
                            file_id=row["file_id"],
                            chunk_type=ChunkType(row["chunk_type"])
                            if row["chunk_type"]
                            else ChunkType.UNKNOWN,
                            symbol=row["symbol"],
                            code=row["code"],
                            start_line=row["start_line"],
                            end_line=row["end_line"],
                            start_byte=row["start_byte"],
                            end_byte=row["end_byte"],
                            language=Language(row["language"])
                            if row["language"]
                            else Language.UNKNOWN,
                            metadata=chunk_dict["metadata"],
                        )
                    )
                else:
                    chunks.append(chunk_dict)

            return chunks

        except Exception as e:
            logger.error(f"Failed to get chunks for file {file_id}: {e}")
            return []

    def delete_file_chunks(self, file_id: int) -> None:
        """Delete all chunks for a file.

        Args:
            file_id: File ID to delete chunks for
        """
        try:
            # Delegate to provider if available for proper executor handling
            if self._provider:
                self._provider._execute_in_db_thread_sync("delete_file_chunks", file_id)
            else:
                # Fallback for tests - simplified version without embedding cleanup
                self._connection_manager.execute_sync(
                    "DELETE FROM chunks WHERE file_id = $1", file_id
                )

        except Exception as e:
            logger.error(f"Failed to delete chunks for file {file_id}: {e}")
            raise

    def delete_chunk(self, chunk_id: int) -> None:
        """Delete a single chunk by ID.

        Args:
            chunk_id: Chunk ID to delete
        """
        try:
            # Delegate to provider if available for proper executor handling
            if self._provider:
                self._provider._execute_in_db_thread_sync("delete_chunk", chunk_id)
            else:
                # Fallback for tests - simplified version without embedding cleanup
                self._connection_manager.execute_sync(
                    "DELETE FROM chunks WHERE id = $1", chunk_id
                )

        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            raise

    def update_chunk(self, chunk_id: int, **kwargs) -> None:
        """Update chunk record with new values.

        Args:
            chunk_id: Chunk ID to update
            **kwargs: Fields to update
        """
        if not kwargs:
            return

        try:
            # Build dynamic update query
            set_clauses = []
            params = []
            param_num = 1

            valid_fields = [
                "chunk_type",
                "symbol",
                "code",
                "start_line",
                "end_line",
                "start_byte",
                "end_byte",
                "language",
            ]

            for key, value in kwargs.items():
                if key in valid_fields:
                    set_clauses.append(f"{key} = ${param_num}")
                    params.append(value)
                    param_num += 1

            if set_clauses:
                set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                params.append(chunk_id)

                query = (
                    f"UPDATE chunks SET {', '.join(set_clauses)} WHERE id = ${param_num}"
                )
                if self._provider:
                    self._provider._execute_in_db_thread_sync(
                        "update_chunk_query", chunk_id, query, params
                    )
                else:
                    self._connection_manager.execute_sync(query, *params)

        except Exception as e:
            logger.error(f"Failed to update chunk {chunk_id}: {e}")
            raise

    def get_chunks_in_range(
        self, file_id: int, start_line: int, end_line: int
    ) -> list[dict]:
        """Get all chunks overlapping a line range.

        Args:
            file_id: ID of the file to search within
            start_line: Start line of the range
            end_line: End line of the range

        Returns:
            List of chunk dictionaries overlapping the range, ordered by start_line
        """
        try:
            # Overlap condition: chunk overlaps if any of:
            # - chunk start_line is within range
            # - chunk end_line is within range
            # - chunk spans the entire range
            query = """
                SELECT id, file_id, chunk_type, symbol, code, start_line, end_line,
                       start_byte, end_byte, language, created_at, updated_at, metadata
                FROM chunks
                WHERE file_id = $1
                AND (
                    (start_line BETWEEN $2 AND $3) OR
                    (end_line BETWEEN $4 AND $5) OR
                    (start_line <= $6 AND end_line >= $7)
                )
                ORDER BY start_line
            """

            if self._provider:
                results = self._provider._execute_in_db_thread_sync(
                    "get_chunks_in_range_query",
                    file_id,
                    start_line,
                    end_line,
                    query,
                )
            else:
                results = self._connection_manager.fetch_sync(
                    query,
                    file_id,
                    start_line,
                    end_line,
                    start_line,
                    end_line,
                    start_line,
                    end_line,
                )

            chunks = []
            for row in results:
                metadata_json = row["metadata"]
                chunk_dict = {
                    "id": row["id"],
                    "file_id": row["file_id"],
                    "chunk_type": row["chunk_type"],
                    "symbol": row["symbol"],
                    "code": row["code"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "start_byte": row["start_byte"],
                    "end_byte": row["end_byte"],
                    "language": row["language"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "metadata": json.loads(metadata_json) if metadata_json else {},
                }
                chunks.append(chunk_dict)

            return chunks

        except Exception as e:
            logger.error(f"Failed to get chunks in range for file {file_id}: {e}")
            return []

    def get_all_chunks_with_metadata(self) -> list[dict[str, Any]]:
        """Get all chunks with their metadata including file paths.

        Returns:
            List of chunk dictionaries with file paths
        """
        try:
            # Use SQL to get chunks with file paths
            query = """
                SELECT c.id, c.file_id, f.path as file_path, c.code,
                       c.start_line, c.end_line, c.chunk_type, c.language, c.symbol,
                       c.metadata
                FROM chunks c
                JOIN files f ON c.file_id = f.id
                ORDER BY c.id
            """

            if self._provider:
                results = self._provider._execute_in_db_thread_sync(
                    "get_all_chunks_with_metadata_query", query
                )
            else:
                results = self._connection_manager.fetch_sync(query)

            # Convert to list of dictionaries
            result = []
            for row in results:
                result.append(
                    {
                        "id": row["id"],
                        "file_id": row["file_id"],
                        "file_path": row["file_path"],
                        "content": row["code"],
                        "start_line": row["start_line"],
                        "end_line": row["end_line"],
                        "chunk_type": row["chunk_type"],
                        "language": row["language"],
                        "name": row["symbol"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Failed to get all chunks with metadata: {e}")
            return []
