"""PostgreSQL provider implementation for ChunkHound - concrete database provider using PostgreSQL with pgvector.

# FILE_CONTEXT: Relational database provider with vector search capabilities
# CRITICAL: Single-threaded access enforced by SerialDatabaseProvider
# PERFORMANCE: ivfflat indexes for vector search, bulk operations optimized
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.models import Chunk, Embedding, File
from chunkhound.embeddings import EmbeddingManager
from chunkhound.providers.database.postgresql.chunk_repository import (
    PostgreSQLChunkRepository,
)
from chunkhound.providers.database.postgresql.connection_manager import (
    PostgreSQLConnectionManager,
)
from chunkhound.providers.database.postgresql.embedding_repository import (
    PostgreSQLEmbeddingRepository,
)
from chunkhound.providers.database.postgresql.file_repository import (
    PostgreSQLFileRepository,
)
from chunkhound.providers.database.serial_database_provider import (
    SerialDatabaseProvider,
)

# Type hinting only
if TYPE_CHECKING:
    from chunkhound.core.config.database_config import DatabaseConfig


class PostgreSQLProvider(SerialDatabaseProvider):
    """PostgreSQL implementation of DatabaseProvider protocol.

    # CLASS_CONTEXT: Relational database with pgvector for embeddings
    # CONSTRAINT: Inherits from SerialDatabaseProvider for thread safety
    # PERFORMANCE: Uses asyncpg with sync wrappers, connection pooling
    """

    def __init__(
        self,
        connection_string: str,
        base_directory: Path,
        embedding_manager: "EmbeddingManager | None" = None,
        config: "DatabaseConfig | None" = None,
    ):
        """Initialize PostgreSQL provider.

        Args:
            connection_string: PostgreSQL connection string
            base_directory: Base directory for path normalization
            embedding_manager: Optional embedding manager for vector generation
            config: Database configuration for provider-specific settings
        """
        # Initialize base class with connection string as db_path
        super().__init__(connection_string, base_directory, embedding_manager, config)

        self.provider_type = "postgresql"  # Identify this as PostgreSQL provider
        self._connection_string = connection_string

        # Initialize connection manager
        self._connection_manager = PostgreSQLConnectionManager(
            connection_string, config
        )

        # Initialize repositories with provider reference
        self._file_repository = PostgreSQLFileRepository(
            self._connection_manager, self
        )
        self._chunk_repository = PostgreSQLChunkRepository(
            self._connection_manager, self
        )
        self._embedding_repository = PostgreSQLEmbeddingRepository(
            self._connection_manager, self
        )
        self._embedding_repository.set_provider_instance(self)

    def _create_connection(self) -> Any:
        """Create and return a PostgreSQL connection.

        For PostgreSQL, we use the connection pool from the connection manager.
        This method returns a placeholder since actual connections are managed by asyncpg pool.

        Returns:
            Connection manager instance (connection pool wrapper)
        """
        # PostgreSQL uses connection pooling, so we return the manager itself
        # which will acquire connections from the pool as needed
        return self._connection_manager

    def _get_schema_sql(self) -> list[str]:
        """Get SQL statements for creating the PostgreSQL schema.

        Returns:
            List of SQL statements
        """
        return [
            # Schema version table
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
            """,
            # Files table
            """
            CREATE TABLE IF NOT EXISTS files (
                id SERIAL PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                extension TEXT,
                size INTEGER,
                modified_time TIMESTAMP,
                content_hash TEXT,
                language TEXT,
                worktree_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Chunks table
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
                chunk_type TEXT NOT NULL,
                symbol TEXT,
                code TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                start_byte INTEGER,
                end_byte INTEGER,
                language TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Default embeddings table (1536 dimensions for OpenAI)
            """
            CREATE TABLE IF NOT EXISTS embeddings_1536 (
                id SERIAL PRIMARY KEY,
                chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding vector(1536),
                dims INTEGER NOT NULL DEFAULT 1536,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(chunk_id, provider, model)
            )
            """,
        ]

    @property
    def db_path(self) -> Path | str:
        """Database connection identifier - returns connection string."""
        return self._connection_string

    @property
    def is_connected(self) -> bool:
        """Check if database connection is active."""
        return self._connection_manager.is_connected

    def connect(self) -> None:
        """Establish database connection and initialize schema."""
        try:
            # Initialize connection manager
            self._connection_manager.connect()

            # Call parent connect which handles executor initialization
            super().connect()

        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise

    def _executor_connect(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for connect - runs in DB thread.

        Args:
            conn: Connection manager instance
            state: Executor state dictionary
        """
        try:
            # Create schema
            self._executor_create_schema(conn, state)

            # Create indexes
            self._executor_create_indexes(conn, state)

            logger.info("PostgreSQL database initialization complete in executor thread")

        except Exception as e:
            logger.error(f"PostgreSQL database initialization failed: {e}")
            raise

    def disconnect(self, skip_checkpoint: bool = False) -> None:
        """Close database connection.

        Args:
            skip_checkpoint: Ignored for PostgreSQL (no checkpointing needed)
        """
        try:
            # Call parent disconnect
            super().disconnect(skip_checkpoint)
        finally:
            # Disconnect connection manager
            self._connection_manager.disconnect()

    def _executor_disconnect(
        self, conn: Any, state: dict[str, Any], skip_checkpoint: bool
    ) -> None:
        """Executor method for disconnect - runs in DB thread.

        Args:
            conn: Connection manager instance
            state: Executor state dictionary
            skip_checkpoint: Ignored for PostgreSQL
        """
        # PostgreSQL handles persistence automatically, no checkpoint needed
        logger.info("PostgreSQL connection closing in executor thread")

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information."""
        return self._connection_manager.health_check()

    def get_connection_info(self) -> dict[str, Any]:
        """Get information about the database connection."""
        return self._connection_manager.get_connection_info()

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        return self._execute_in_db_thread_sync("table_exists", table_name)

    def _executor_table_exists(
        self, conn: Any, state: dict[str, Any], table_name: str
    ) -> bool:
        """Executor method for table_exists - runs in DB thread."""
        result = conn.fetchval_sync(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = $1 AND table_schema = 'public'
            """,
            table_name,
        )
        return result > 0

    def _ensure_embedding_table_exists(self, dims: int) -> str:
        """Ensure embedding table for given dimensions exists."""
        return self._execute_in_db_thread_sync("ensure_embedding_table_exists", dims)

    def _executor_ensure_embedding_table_exists(
        self, conn: Any, state: dict[str, Any], dims: int
    ) -> str:
        """Executor method for ensure_embedding_table_exists - runs in DB thread."""
        table_name = f"embeddings_{dims}"

        # Check if table exists
        if not self._executor_table_exists(conn, state, table_name):
            logger.info(f"Creating embedding table {table_name} for {dims} dimensions")

            # Create table with pgvector column
            conn.execute_sync(
                f"""
                CREATE TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    embedding vector({dims}),
                    dims INTEGER NOT NULL DEFAULT {dims},
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(chunk_id, provider, model)
                )
                """
            )

            # Create index on chunk_id for efficient deletions
            conn.execute_sync(
                f"CREATE INDEX idx_{table_name}_chunk_id ON {table_name}(chunk_id)"
            )

            # Create ivfflat index for vector search
            # Note: ivfflat requires sufficient data for training
            # We'll create it later when we have enough embeddings
            logger.info(f"Embedding table {table_name} created successfully")

        return table_name

    def _get_all_embedding_tables(self) -> list[str]:
        """Get list of all embedding tables."""
        return self._execute_in_db_thread_sync("get_all_embedding_tables")

    def _executor_get_all_embedding_tables(
        self, conn: Any, state: dict[str, Any]
    ) -> list[str]:
        """Executor method for _get_all_embedding_tables - runs in DB thread."""
        tables = conn.fetch_sync(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'embeddings_%'
            AND table_schema = 'public'
            """
        )
        return [table["table_name"] for table in tables]

    def create_indexes(self) -> None:
        """Create database indexes for performance optimization."""
        self._execute_in_db_thread_sync("create_indexes")

    def _executor_create_indexes(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for create_indexes - runs in DB thread."""
        logger.info("Creating PostgreSQL indexes")

        try:
            # File indexes
            conn.execute_sync(
                "CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)"
            )
            conn.execute_sync(
                "CREATE INDEX IF NOT EXISTS idx_files_language ON files(language)"
            )
            conn.execute_sync(
                "CREATE INDEX IF NOT EXISTS idx_files_worktree_id ON files(worktree_id)"
            )

            # Chunk indexes
            conn.execute_sync(
                "CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)"
            )
            conn.execute_sync(
                "CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type)"
            )
            conn.execute_sync(
                "CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol)"
            )

            logger.info("PostgreSQL indexes created successfully")

        except Exception as e:
            logger.error(f"Failed to create PostgreSQL indexes: {e}")
            raise

    def create_schema(self) -> None:
        """Create database schema for files, chunks, and embeddings."""
        self._execute_in_db_thread_sync("create_schema")

    def _executor_create_schema(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for create_schema - runs in DB thread."""
        logger.info("Creating PostgreSQL schema")

        try:
            schema_statements = self._get_schema_sql()
            for statement in schema_statements:
                conn.execute_sync(statement)

            # Track schema version
            result = conn.fetchval_sync(
                "SELECT MAX(version) FROM schema_version WHERE version IS NOT NULL"
            )
            current_version = result if result is not None else 0

            if current_version == 0:
                conn.execute_sync(
                    """
                    INSERT INTO schema_version (version, description)
                    VALUES (1, 'Initial schema')
                    """
                )
                logger.info("Schema version initialized to 1")

            logger.info("PostgreSQL schema created successfully")

        except Exception as e:
            logger.error(f"Failed to create PostgreSQL schema: {e}")
            raise

    def create_vector_index(
        self, provider: str, model: str, dims: int, metric: str = "cosine"
    ) -> None:
        """Create ivfflat vector index for specific provider/model/dims combination.

        Args:
            provider: Embedding provider name
            model: Embedding model name
            dims: Vector dimensions
            metric: Distance metric (cosine, l2, ip)
        """
        logger.info(
            f"Creating ivfflat index for {provider}/{model} ({dims}D, {metric})"
        )
        self._execute_in_db_thread_sync(
            "create_vector_index", provider, model, dims, metric
        )

    def _executor_create_vector_index(
        self,
        conn: Any,
        state: dict[str, Any],
        provider: str,
        model: str,
        dims: int,
        metric: str,
    ) -> None:
        """Executor method for create_vector_index - runs in DB thread."""
        try:
            table_name = f"embeddings_{dims}"

            # Ensure the table exists
            self._executor_ensure_embedding_table_exists(conn, state, dims)

            # Convert metric to pgvector operator
            ops_map = {
                "cosine": "vector_cosine_ops",
                "l2": "vector_l2_ops",
                "ip": "vector_ip_ops",
            }
            ops = ops_map.get(metric, "vector_cosine_ops")

            index_name = (
                f"ivfflat_{provider}_{model}_{dims}_{metric}"
                .replace("-", "_")
                .replace(".", "_")
            )

            # Create ivfflat index
            # Note: This requires sufficient data (typically 1000+ vectors)
            conn.execute_sync(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}
                USING ivfflat (embedding {ops})
                WITH (lists = 100)
                """
            )

            logger.info(
                f"ivfflat index {index_name} created successfully on {table_name}"
            )

        except Exception as e:
            logger.error(f"Failed to create ivfflat index: {e}")
            raise

    def drop_vector_index(
        self, provider: str, model: str, dims: int, metric: str = "cosine"
    ) -> str:
        """Drop ivfflat vector index for specific provider/model/dims combination."""
        return self._execute_in_db_thread_sync(
            "drop_vector_index", provider, model, dims, metric
        )

    def _executor_drop_vector_index(
        self,
        conn: Any,
        state: dict[str, Any],
        provider: str,
        model: str,
        dims: int,
        metric: str,
    ) -> str:
        """Executor method for drop_vector_index - runs in DB thread."""
        index_name = (
            f"ivfflat_{provider}_{model}_{dims}_{metric}"
            .replace("-", "_")
            .replace(".", "_")
        )

        try:
            conn.execute_sync(f"DROP INDEX IF EXISTS {index_name}")
            logger.info(f"ivfflat index {index_name} dropped")
            return index_name

        except Exception as e:
            logger.error(f"Failed to drop ivfflat index: {e}")
            raise

    def get_existing_vector_indexes(self) -> list[dict[str, Any]]:
        """Get list of existing ivfflat vector indexes on all embedding tables."""
        return self._execute_in_db_thread_sync("get_existing_vector_indexes")

    def _executor_get_existing_vector_indexes(
        self, conn: Any, state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Executor method for get_existing_vector_indexes - runs in DB thread."""
        try:
            # Query PostgreSQL system tables for ivfflat indexes
            results = conn.fetch_sync(
                """
                SELECT
                    i.indexname as index_name,
                    i.tablename as table_name
                FROM pg_indexes i
                WHERE i.tablename LIKE 'embeddings_%'
                AND i.indexname LIKE 'ivfflat_%'
                AND i.schemaname = 'public'
                """
            )

            indexes = []
            for row in results:
                index_name = row["index_name"]
                table_name = row["table_name"]

                # Parse index name: ivfflat_{provider}_{model}_{dims}_{metric}
                if index_name.startswith("ivfflat_"):
                    parts = index_name[8:].split("_")  # Remove 'ivfflat_' prefix
                    if len(parts) >= 4:
                        metric = parts[-1]
                        dims_str = parts[-2]
                        try:
                            dims = int(dims_str)
                            # Join remaining parts as provider_model
                            provider_model = "_".join(parts[:-2])
                            # Find last underscore to separate provider and model
                            last_underscore = provider_model.rfind("_")
                            if last_underscore > 0:
                                provider = provider_model[:last_underscore]
                                model = provider_model[last_underscore + 1 :]
                            else:
                                provider = provider_model
                                model = ""

                            indexes.append(
                                {
                                    "index_name": index_name,
                                    "table_name": table_name,
                                    "provider": provider,
                                    "model": model,
                                    "dims": dims,
                                    "metric": metric,
                                }
                            )
                        except (ValueError, IndexError):
                            logger.warning(f"Could not parse index name: {index_name}")

            return indexes

        except Exception as e:
            logger.error(f"Failed to get existing vector indexes: {e}")
            return []

    # File operations delegate to file_repository
    def insert_file(self, file: File, worktree_id: str | None = None) -> int:
        """Insert file record."""
        return self._file_repository.insert_file(file, worktree_id)

    def get_file_by_path(
        self, path: str, worktree_id: str | None = None, as_model: bool = False
    ) -> dict[str, Any] | File | None:
        """Get file record by path."""
        return self._file_repository.get_file_by_path(path, worktree_id, as_model)

    def get_file_by_id(
        self, file_id: int, as_model: bool = False
    ) -> dict[str, Any] | File | None:
        """Get file record by ID."""
        return self._file_repository.get_file_by_id(file_id, as_model)

    # Chunk operations delegate to chunk_repository
    def insert_chunk(self, chunk: Chunk) -> int:
        """Insert chunk record."""
        return self._chunk_repository.insert_chunk(chunk)

    def insert_chunks_batch(self, chunks: list[Chunk]) -> list[int]:
        """Insert multiple chunks in batch."""
        return self._chunk_repository.insert_chunks_batch(chunks)

    def get_chunk_by_id(
        self, chunk_id: int, as_model: bool = False
    ) -> dict[str, Any] | Chunk | None:
        """Get chunk record by ID."""
        return self._chunk_repository.get_chunk_by_id(chunk_id, as_model)

    # Embedding operations delegate to embedding_repository
    def insert_embedding(
        self,
        chunk_id_or_embedding: int | Embedding,
        vector: list[float] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> int:
        """Insert embedding record.

        Supports two calling conventions:
        1. New style: insert_embedding(Embedding(...))
        2. Legacy style: insert_embedding(chunk_id, vector, provider, model)

        Args:
            chunk_id_or_embedding: Either an Embedding object or chunk_id (int)
            vector: Embedding vector (required if first arg is int)
            provider: Provider name (required if first arg is int)
            model: Model name (required if first arg is int)

        Returns:
            Embedding ID
        """
        if isinstance(chunk_id_or_embedding, Embedding):
            # New style: Embedding object
            return self._embedding_repository.insert_embedding(chunk_id_or_embedding)
        else:
            # Legacy style: separate parameters
            if vector is None or provider is None or model is None:
                raise ValueError(
                    "When calling with chunk_id, vector, provider, and model are required"
                )
            embedding = Embedding(
                chunk_id=chunk_id_or_embedding,
                vector=vector,
                provider=provider,
                model=model,
                dims=len(vector),
            )
            return self._embedding_repository.insert_embedding(embedding)

    def insert_embeddings_batch(
        self, embeddings_data: list[dict], batch_size: int | None = None
    ) -> int:
        """Insert multiple embeddings in batch."""
        return self._embedding_repository.insert_embeddings_batch(
            embeddings_data, batch_size
        )

    def get_embedding_by_chunk_id(
        self, chunk_id: int, provider: str, model: str
    ) -> Embedding | None:
        """Get embedding for specific chunk."""
        return self._embedding_repository.get_embedding_by_chunk_id(
            chunk_id, provider, model
        )

    # Compatibility methods for tests
    def get_embedding(
        self, chunk_id: int, provider: str, model: str
    ) -> dict[str, Any] | None:
        """Get embedding for specific chunk (test compatibility method).

        Args:
            chunk_id: Chunk ID to get embedding for
            provider: Embedding provider name
            model: Embedding model name

        Returns:
            Dictionary with embedding data or None
        """
        embedding = self.get_embedding_by_chunk_id(chunk_id, provider, model)
        if embedding:
            return {
                "chunk_id": embedding.chunk_id,
                "provider": embedding.provider,
                "model": embedding.model,
                "embedding": embedding.vector,
                "dims": embedding.dims,
            }
        return None

    # Executor methods for database thread operations
    # These are called by SerialDatabaseExecutor via _execute_in_db_thread_sync

    def _executor_insert_file(
        self, conn: Any, state: dict[str, Any], file: File, worktree_id: str | None = None
    ) -> int:
        """Executor method for insert_file - runs in DB thread."""
        from chunkhound.core.utils import normalize_path_for_lookup

        # Normalize path for storage
        base_dir = state.get("base_directory")
        normalized_path = normalize_path_for_lookup(str(file.path), base_dir)

        # Insert new file record
        result = conn.fetchval_sync(
            """
            INSERT INTO files (path, name, extension, size, modified_time, language, worktree_id)
            VALUES ($1, $2, $3, $4, TO_TIMESTAMP($5), $6, $7)
            RETURNING id
            """,
            normalized_path,
            file.name,
            file.extension,
            file.size_bytes,
            file.mtime,
            file.language.value if file.language else None,
            worktree_id,
        )
        return result

    def _executor_get_file_by_path(
        self, conn: Any, state: dict[str, Any], path: str, worktree_id: str | None, as_model: bool
    ) -> dict[str, Any] | File | None:
        """Executor method for get_file_by_path - runs in DB thread."""
        from chunkhound.core.utils import normalize_path_for_lookup
        from chunkhound.core.types.common import Language

        base_dir = state.get("base_directory")
        lookup_path = normalize_path_for_lookup(path, base_dir)

        if worktree_id:
            result = conn.fetch_sync(
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
            result = conn.fetch_sync(
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
                mtime=row["modified_time"].timestamp() if row["modified_time"] else 0,
                size_bytes=row["size"],
                language=Language(row["language"]) if row["language"] else Language.UNKNOWN,
            )

        return file_dict

    def _executor_get_file_by_id_query(
        self, conn: Any, state: dict[str, Any], file_id: int, as_model: bool
    ) -> dict[str, Any] | File | None:
        """Executor method for get_file_by_id - runs in DB thread."""
        from chunkhound.core.types.common import Language

        result = conn.fetch_sync(
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
                mtime=row["modified_time"].timestamp() if row["modified_time"] else 0,
                size_bytes=row["size"],
                language=Language(row["language"]) if row["language"] else Language.UNKNOWN,
            )

        return file_dict

    def _executor_update_file(
        self,
        conn: Any,
        state: dict[str, Any],
        file_id: int,
        size_bytes: int | None,
        mtime: float | None,
        content_hash: str | None,
    ) -> None:
        """Executor method for update_file - runs in DB thread."""
        # Build dynamic update query
        set_clauses = []
        params = []
        param_num = 1

        if size_bytes is not None:
            set_clauses.append(f"size = ${param_num}")
            params.append(size_bytes)
            param_num += 1

        if mtime is not None:
            set_clauses.append(f"modified_time = TO_TIMESTAMP(${param_num})")
            params.append(mtime)
            param_num += 1

        if content_hash is not None:
            set_clauses.append(f"content_hash = ${param_num}")
            params.append(content_hash)
            param_num += 1

        if set_clauses:
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            params.append(file_id)

            query = f"UPDATE files SET {', '.join(set_clauses)} WHERE id = ${param_num}"
            conn.execute_sync(query, *params)

    def _executor_insert_chunk_single(
        self, conn: Any, state: dict[str, Any], chunk: Chunk
    ) -> int:
        """Executor method for insert_chunk - runs in DB thread."""
        import json

        metadata_json = json.dumps(chunk.metadata) if chunk.metadata else None

        result = conn.fetchval_sync(
            """
            INSERT INTO chunks (file_id, chunk_type, symbol, code, start_line, end_line,
                              start_byte, end_byte, language, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
            """,
            chunk.file_id,
            chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else chunk.chunk_type,
            chunk.symbol,
            chunk.code,
            chunk.start_line,
            chunk.end_line,
            chunk.start_byte,
            chunk.end_byte,
            chunk.language.value if hasattr(chunk.language, 'value') else chunk.language,
            metadata_json,
        )
        return result

    def _executor_insert_chunks_batch(
        self, conn: Any, state: dict[str, Any], chunks: list[Chunk]
    ) -> list[int]:
        """Executor method for insert_chunks_batch - runs in DB thread."""
        if not chunks:
            return []

        import json

        # Prepare data for bulk insert
        placeholders = []
        params = []
        for i, chunk in enumerate(chunks):
            base = i * 10
            placeholders.append(
                f"(${base+1}, ${base+2}, ${base+3}, ${base+4}, ${base+5}, ${base+6}, "
                f"${base+7}, ${base+8}, ${base+9}, ${base+10})"
            )
            metadata_json = json.dumps(chunk.metadata) if chunk.metadata else None
            params.extend([
                chunk.file_id,
                chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else chunk.chunk_type,
                chunk.symbol,
                chunk.code,
                chunk.start_line,
                chunk.end_line,
                chunk.start_byte,
                chunk.end_byte,
                chunk.language.value if hasattr(chunk.language, 'value') else chunk.language,
                metadata_json,
            ])

        # Single INSERT statement for all chunks
        query = f"""
            INSERT INTO chunks (file_id, chunk_type, symbol, code, start_line, end_line,
                              start_byte, end_byte, language, metadata)
            VALUES {', '.join(placeholders)}
            RETURNING id
        """

        results = conn.fetch_sync(query, *params)
        return [row["id"] for row in results]

    def _executor_get_chunk_by_id_query(
        self, conn: Any, state: dict[str, Any], chunk_id: int
    ) -> Any:
        """Executor method for get_chunk_by_id query - runs in DB thread."""
        return conn.fetch_sync(
            """
            SELECT id, file_id, chunk_type, symbol, code, start_line, end_line,
                   start_byte, end_byte, language, created_at, updated_at, metadata
            FROM chunks WHERE id = $1
            """,
            chunk_id,
        )

    def _executor_get_chunks_by_file_id_query(
        self, conn: Any, state: dict[str, Any], file_id: int
    ) -> Any:
        """Executor method for get_chunks_by_file_id query - runs in DB thread."""
        return conn.fetch_sync(
            """
            SELECT id, file_id, chunk_type, symbol, code, start_line, end_line,
                   start_byte, end_byte, language, created_at, updated_at, metadata
            FROM chunks WHERE file_id = $1
            """,
            file_id,
        )

    def _executor_delete_file_chunks(
        self, conn: Any, state: dict[str, Any], file_id: int
    ) -> None:
        """Executor method for delete_file_chunks - runs in DB thread."""
        conn.execute_sync("DELETE FROM chunks WHERE file_id = $1", file_id)

    def _executor_delete_chunk(
        self, conn: Any, state: dict[str, Any], chunk_id: int
    ) -> None:
        """Executor method for delete_chunk - runs in DB thread."""
        conn.execute_sync("DELETE FROM chunks WHERE id = $1", chunk_id)

    def _executor_update_chunk_query(
        self, conn: Any, state: dict[str, Any], chunk_id: int, query: str, params: list
    ) -> None:
        """Executor method for update_chunk query - runs in DB thread."""
        conn.execute_sync(query, *params)

    def _executor_get_chunks_in_range_query(
        self, conn: Any, state: dict[str, Any], file_id: int, start_line: int, end_line: int
    ) -> Any:
        """Executor method for get_chunks_in_range query - runs in DB thread."""
        return conn.fetch_sync(
            """
            SELECT id, file_id, chunk_type, symbol, code, start_line, end_line,
                   start_byte, end_byte, language, created_at, updated_at, metadata
            FROM chunks
            WHERE file_id = $1
              AND ((start_line >= $2 AND start_line <= $3)
                   OR (end_line >= $2 AND end_line <= $3)
                   OR (start_line <= $2 AND end_line >= $3))
            ORDER BY start_line
            """,
            file_id,
            start_line,
            end_line,
        )

    def _executor_get_all_chunks_with_metadata_query(
        self, conn: Any, state: dict[str, Any], query: str
    ) -> Any:
        """Executor method for get_all_chunks_with_metadata query - runs in DB thread."""
        return conn.fetch_sync(query)
