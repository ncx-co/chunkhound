"""PostgreSQL file repository implementation for ChunkHound - handles file CRUD operations."""

from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.models import File
from chunkhound.core.types.common import Language

if TYPE_CHECKING:
    from chunkhound.providers.database.postgresql.connection_manager import (
        PostgreSQLConnectionManager,
    )


class PostgreSQLFileRepository:
    """Repository for file CRUD operations using PostgreSQL."""

    def __init__(
        self, connection_manager: "PostgreSQLConnectionManager", provider=None
    ):
        """Initialize file repository with connection manager.

        Args:
            connection_manager: PostgreSQL connection manager instance
            provider: Optional provider instance for transaction-aware connections
        """
        self.connection_manager = connection_manager
        self._provider = provider

    def _extract_file_id(self, file_record: dict[str, Any] | File) -> int | None:
        """Safely extract file ID from either dict or File model."""
        if isinstance(file_record, File):
            return file_record.id
        elif isinstance(file_record, dict) and "id" in file_record:
            return file_record["id"]
        else:
            return None

    def insert_file(self, file: File, worktree_id: str | None = None) -> int:
        """Insert file record and return file ID.

        If file with same path exists, updates metadata.

        Args:
            file: File model to insert
            worktree_id: Optional worktree identifier for isolation

        Returns:
            File ID
        """
        try:
            # First try to find existing file by path
            existing = self.get_file_by_path(str(file.path), worktree_id)
            if existing:
                # File exists, update it
                file_id = self._extract_file_id(existing)
                if file_id is not None:
                    self.update_file(
                        file_id,
                        size_bytes=file.size_bytes,
                        mtime=file.mtime,
                        content_hash=file.content_hash,
                    )
                    return file_id

            # No existing file, insert new one
            if self._provider:
                # Delegate to provider for proper executor handling
                return self._provider._execute_in_db_thread_sync(
                    "insert_file", file, worktree_id
                )
            else:
                # Fallback for tests
                result = self.connection_manager.fetchval_sync(
                    """
                    INSERT INTO files (path, name, extension, size, modified_time, language, worktree_id)
                    VALUES ($1, $2, $3, $4, TO_TIMESTAMP($5), $6, $7)
                    RETURNING id
                    """,
                    file.path,
                    file.name,
                    file.extension,
                    file.size_bytes,
                    file.mtime,
                    file.language.value if file.language else None,
                    worktree_id,
                )
                return result

        except Exception as e:
            logger.error(f"Failed to insert file {file.path}: {e}")
            # Return existing file ID if constraint error (duplicate)
            if "duplicate key" in str(e).lower():
                existing = self.get_file_by_path(str(file.path), worktree_id)
                if existing and isinstance(existing, dict) and "id" in existing:
                    logger.info(f"Returning existing file ID for {file.path}")
                    return existing["id"]
            raise

    def get_file_by_path(
        self,
        path: str,
        worktree_id: str | None = None,
        as_model: bool = False,
    ) -> dict[str, Any] | File | None:
        """Get file record by path.

        Args:
            path: File path to search for
            worktree_id: Optional worktree identifier for isolation
            as_model: Return File model if True, dict otherwise

        Returns:
            File record or None if not found
        """
        try:
            if self._provider:
                return self._provider._execute_in_db_thread_sync(
                    "get_file_by_path", path, worktree_id, as_model
                )
            else:
                # Fallback for tests
                from chunkhound.core.utils import normalize_path_for_lookup

                base_dir = (
                    self._provider.get_base_directory() if self._provider else None
                )
                lookup_path = normalize_path_for_lookup(path, base_dir)

                if worktree_id:
                    result = self.connection_manager.fetch_sync(
                        """
                        SELECT id, path, name, extension, size, modified_time, language,
                               created_at, updated_at, worktree_id
                        FROM files
                        WHERE path = $1 AND worktree_id = $2
                        """,
                        lookup_path,
                        worktree_id,
                    )
                else:
                    result = self.connection_manager.fetch_sync(
                        """
                        SELECT id, path, name, extension, size, modified_time, language,
                               created_at, updated_at, worktree_id
                        FROM files
                        WHERE path = $1 AND worktree_id IS NULL
                        """,
                        lookup_path,
                    )

            if not result or len(result) == 0:
                return None

            row = result[0]
            file_dict = {
                "id": row["id"],
                "path": row["path"],
                "name": row["name"],
                "extension": row["extension"],
                "size": row["size"],
                "modified_time": row["modified_time"],
                "language": row["language"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

            if as_model:
                return File(
                    path=row["path"],
                    mtime=row["modified_time"].timestamp()
                    if row["modified_time"]
                    else 0,
                    size_bytes=row["size"],
                    language=Language(row["language"])
                    if row["language"]
                    else Language.UNKNOWN,
                )

            return file_dict

        except Exception as e:
            logger.error(f"Failed to get file by path {path}: {e}")
            return None

    def get_file_by_id(
        self, file_id: int, as_model: bool = False
    ) -> dict[str, Any] | File | None:
        """Get file record by ID.

        Args:
            file_id: File ID to search for
            as_model: Return File model if True, dict otherwise

        Returns:
            File record or None if not found
        """
        try:
            if self._provider:
                return self._provider._execute_in_db_thread_sync(
                    "get_file_by_id_query", file_id, as_model
                )
            else:
                # Fallback for tests
                result = self.connection_manager.fetch_sync(
                    """
                    SELECT id, path, name, extension, size, modified_time, language,
                           created_at, updated_at, worktree_id
                    FROM files
                    WHERE id = $1
                    """,
                    file_id,
                )

            if not result or len(result) == 0:
                return None

            row = result[0]
            file_dict = {
                "id": row["id"],
                "path": row["path"],
                "name": row["name"],
                "extension": row["extension"],
                "size": row["size"],
                "modified_time": row["modified_time"],
                "language": row["language"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

            if as_model:
                return File(
                    path=row["path"],
                    mtime=row["modified_time"].timestamp()
                    if row["modified_time"]
                    else 0,
                    size_bytes=row["size"],
                    language=Language(row["language"])
                    if row["language"]
                    else Language.UNKNOWN,
                )

            return file_dict

        except Exception as e:
            logger.error(f"Failed to get file by ID {file_id}: {e}")
            return None

    def update_file(
        self,
        file_id: int,
        size_bytes: int | None = None,
        mtime: float | None = None,
        content_hash: str | None = None,
    ) -> None:
        """Update file record with new values.

        Args:
            file_id: ID of the file to update
            size_bytes: New file size in bytes
            mtime: New modification timestamp
            content_hash: Content hash for change detection
        """
        # Skip if no updates provided
        if size_bytes is None and mtime is None and content_hash is None:
            return

        try:
            # Build dynamic update query
            set_clauses = []
            params = []
            param_num = 1

            # Add size update if provided
            if size_bytes is not None:
                set_clauses.append(f"size = ${param_num}")
                params.append(size_bytes)
                param_num += 1

            # Add timestamp update if provided
            if mtime is not None:
                set_clauses.append(f"modified_time = TO_TIMESTAMP(${param_num})")
                params.append(mtime)
                param_num += 1

            # Add content hash update if provided
            if content_hash is not None:
                set_clauses.append(f"content_hash = ${param_num}")
                params.append(content_hash)
                param_num += 1

            if set_clauses:
                set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                params.append(file_id)

                query = (
                    f"UPDATE files SET {', '.join(set_clauses)} WHERE id = ${param_num}"
                )
                if self._provider:
                    self._provider._execute_in_db_thread_sync(
                        "update_file", file_id, size_bytes, mtime, content_hash
                    )
                else:
                    # Fallback for tests
                    self.connection_manager.execute_sync(query, *params)

        except Exception as e:
            logger.error(f"Failed to update file {file_id}: {e}")
            raise

    def delete_file_completely(self, file_path: str) -> bool:
        """Delete a file and all its chunks/embeddings completely.

        Args:
            file_path: Path of file to delete

        Returns:
            True if file was deleted, False otherwise
        """
        try:
            # Get file ID first
            file_record = self.get_file_by_path(file_path)
            if not file_record:
                return False

            file_id = (
                file_record["id"] if isinstance(file_record, dict) else file_record.id
            )

            # Delete in correct order due to foreign key constraints
            # PostgreSQL cascades should handle embeddings and chunks
            # 1. Delete embeddings first (from all embedding tables)
            if self._provider:
                embedding_tables = self._provider._get_all_embedding_tables()
            else:
                # Fallback for tests
                tables = self.connection_manager.fetch_sync(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_name LIKE 'embeddings_%'
                    AND table_schema = 'public'
                    """
                )
                embedding_tables = [table["table_name"] for table in tables]

            for table_name in embedding_tables:
                self.connection_manager.execute_sync(
                    f"""
                    DELETE FROM {table_name}
                    WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = $1)
                    """,
                    file_id,
                )

            # 2. Delete chunks
            self.connection_manager.execute_sync(
                "DELETE FROM chunks WHERE file_id = $1", file_id
            )

            # 3. Delete file
            self.connection_manager.execute_sync("DELETE FROM files WHERE id = $1", file_id)

            logger.debug(f"File {file_path} and all associated data deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    def get_file_stats(self, file_id: int) -> dict[str, Any]:
        """Get statistics for a specific file.

        Args:
            file_id: File ID to get stats for

        Returns:
            Dictionary with file statistics
        """
        try:
            # Get file info
            file_result = self.connection_manager.fetch_sync(
                """
                SELECT path, name, extension, size, language
                FROM files
                WHERE id = $1
                """,
                file_id,
            )

            if not file_result or len(file_result) == 0:
                return {}

            file_row = file_result[0]

            # Get chunk count and types
            chunk_results = self.connection_manager.fetch_sync(
                """
                SELECT chunk_type, COUNT(*) as count
                FROM chunks
                WHERE file_id = $1
                GROUP BY chunk_type
                """,
                file_id,
            )

            chunk_types = {result["chunk_type"]: result["count"] for result in chunk_results}
            total_chunks = sum(chunk_types.values())

            # Get embedding count across all embedding tables
            embedding_count = 0
            if self._provider:
                embedding_tables = self._provider._get_all_embedding_tables()
            else:
                # Fallback for tests
                tables = self.connection_manager.fetch_sync(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_name LIKE 'embeddings_%'
                    AND table_schema = 'public'
                    """
                )
                embedding_tables = [table["table_name"] for table in tables]

            for table_name in embedding_tables:
                count_result = self.connection_manager.fetchval_sync(
                    f"""
                    SELECT COUNT(*)
                    FROM {table_name} e
                    JOIN chunks c ON e.chunk_id = c.id
                    WHERE c.file_id = $1
                    """,
                    file_id,
                )
                embedding_count += count_result or 0

            return {
                "file_id": file_id,
                "path": file_row["path"],
                "name": file_row["name"],
                "extension": file_row["extension"],
                "size": file_row["size"],
                "language": file_row["language"],
                "chunks": total_chunks,
                "chunk_types": chunk_types,
                "embeddings": embedding_count,
            }

        except Exception as e:
            logger.error(f"Failed to get file stats for {file_id}: {e}")
            return {}

    def _maybe_checkpoint(self, force: bool = False) -> None:
        """Perform checkpoint if needed - no-op for PostgreSQL.

        PostgreSQL handles checkpointing automatically.
        """
        pass
