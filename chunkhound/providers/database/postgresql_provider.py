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
    def insert_embedding(self, embedding: Embedding) -> int:
        """Insert embedding record."""
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
