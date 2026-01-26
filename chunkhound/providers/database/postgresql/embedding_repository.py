"""PostgreSQL Embedding Repository - handles embedding CRUD operations with pgvector."""

import time
from typing import Any

from loguru import logger

from chunkhound.core.models import Embedding
from chunkhound.providers.database.postgresql.connection_manager import (
    PostgreSQLConnectionManager,
)


class PostgreSQLEmbeddingRepository:
    """Repository for embedding operations in PostgreSQL with pgvector."""

    def __init__(
        self, connection_manager: PostgreSQLConnectionManager, provider=None
    ):
        """Initialize the embedding repository.

        Args:
            connection_manager: PostgreSQL connection manager for database access
            provider: Optional provider instance for transaction-aware connections
        """
        self.connection_manager = connection_manager
        self._provider = provider
        self._provider_instance = None  # Will be set by provider

    def set_provider_instance(self, provider_instance):
        """Set the provider instance for index management operations.

        Args:
            provider_instance: Provider instance for index management
        """
        self._provider_instance = provider_instance

    def insert_embedding(self, embedding: Embedding) -> int:
        """Insert embedding record and return embedding ID.

        Args:
            embedding: Embedding model to insert

        Returns:
            Embedding ID
        """
        try:
            # Ensure appropriate table exists for these dimensions
            if self._provider:
                table_name = self._provider._ensure_embedding_table_exists(
                    embedding.dims
                )
            else:
                # Fallback for tests - create table if needed
                table_name = f"embeddings_{embedding.dims}"
                # Check if table exists
                result = self.connection_manager.fetchval_sync(
                    """
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_name = $1 AND table_schema = 'public'
                    """,
                    table_name,
                )
                if result == 0:
                    self.connection_manager.execute_sync(
                        f"""
                        CREATE TABLE {table_name} (
                            id SERIAL PRIMARY KEY,
                            chunk_id INTEGER REFERENCES chunks(id),
                            provider TEXT NOT NULL,
                            model TEXT NOT NULL,
                            embedding vector({embedding.dims}),
                            dims INTEGER NOT NULL DEFAULT {embedding.dims},
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )

            result = self.connection_manager.fetchval_sync(
                f"""
                INSERT INTO {table_name} (chunk_id, provider, model, embedding, dims)
                VALUES ($1, $2, $3, $4::vector, $5)
                RETURNING id
                """,
                embedding.chunk_id,
                embedding.provider,
                embedding.model,
                str(embedding.vector),
                embedding.dims,
            )

            embedding_id = result
            logger.debug(
                f"Inserted embedding {embedding_id} for chunk {embedding.chunk_id}"
            )
            return embedding_id

        except Exception as e:
            logger.error(f"Failed to insert embedding: {e}")
            raise

    def insert_embeddings_batch(
        self,
        embeddings_data: list[dict],
        batch_size: int | None = None,
        connection=None,
    ) -> int:
        """Insert multiple embedding vectors with ivfflat index optimization.

        For large batches (>= batch_size threshold), uses optimization:
        1. Drop ivfflat indexes to avoid insert slowdown
        2. Use fast INSERT for new embeddings, ON CONFLICT for updates
        3. Recreate ivfflat indexes after bulk operations

        Args:
            embeddings_data: List of dicts with keys: chunk_id, provider, model, embedding, dims
            batch_size: Threshold for index optimization (default: 50)
            connection: Optional database connection (not used for PostgreSQL - uses pool)

        Returns:
            Number of successfully inserted embeddings
        """
        if not embeddings_data:
            return 0

        # Use provided batch_size threshold or default to 50
        index_threshold = batch_size if batch_size is not None else 50
        actual_batch_size = len(embeddings_data)
        logger.debug(
            f"🔄 Starting optimized batch insert of {actual_batch_size} embeddings "
            f"(index threshold: {index_threshold})"
        )

        # Auto-detect embedding dimensions from first embedding
        first_vector = embeddings_data[0]["embedding"]
        detected_dims = len(first_vector)

        # Validate all embeddings have the same dimensions
        for i, embedding_data in enumerate(embeddings_data):
            vector = embedding_data["embedding"]
            if len(vector) != detected_dims:
                raise ValueError(
                    f"Embedding vector {i} has {len(vector)} dimensions, "
                    f"expected {detected_dims} (detected from first embedding)"
                )

        # Ensure appropriate table exists for these dimensions
        if self._provider:
            table_name = self._provider._ensure_embedding_table_exists(detected_dims)
        else:
            # Fallback for tests
            table_name = f"embeddings_{detected_dims}"
            result = self.connection_manager.fetchval_sync(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = $1 AND table_schema = 'public'
                """,
                table_name,
            )
            if result == 0:
                self.connection_manager.execute_sync(
                    f"""
                    CREATE TABLE {table_name} (
                        id SERIAL PRIMARY KEY,
                        chunk_id INTEGER REFERENCES chunks(id),
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        embedding vector({detected_dims}),
                        dims INTEGER NOT NULL DEFAULT {detected_dims},
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

        logger.debug(
            f"Using table {table_name} for {detected_dims}-dimensional embeddings"
        )

        # Extract provider/model for conflict checking
        first_embedding = embeddings_data[0]
        provider = first_embedding["provider"]
        model = first_embedding["model"]

        # Use index optimization only for larger batches
        use_index_optimization = actual_batch_size >= index_threshold

        # Log the optimization decision for debugging
        if use_index_optimization:
            logger.debug(
                f"🚀 Large batch: using index optimization "
                f"({actual_batch_size} >= {index_threshold})"
            )
        else:
            logger.debug(
                f"🔍 Small batch: preserving indexes for semantic search "
                f"({actual_batch_size} < {index_threshold})"
            )

        try:
            total_inserted = 0
            start_time = time.time()

            if use_index_optimization:
                # Drop indexes for bulk operations
                logger.debug(
                    f"🔧 Large batch detected ({actual_batch_size} embeddings >= "
                    f"{index_threshold}), applying index optimization"
                )

                dims = first_embedding["dims"]

                # Step 1: Drop ivfflat index
                if self._provider_instance and hasattr(
                    self._provider_instance, "get_existing_vector_indexes"
                ):
                    existing_indexes = (
                        self._provider_instance.get_existing_vector_indexes()
                    )
                    dropped_indexes = []

                    for index_info in existing_indexes:
                        try:
                            self._provider_instance.drop_vector_index(
                                index_info["provider"],
                                index_info["model"],
                                index_info["dims"],
                                index_info["metric"],
                            )
                            dropped_indexes.append(index_info)
                            logger.debug(f"Dropped index: {index_info['index_name']}")
                        except Exception as e:
                            logger.warning(
                                f"Could not drop index {index_info['index_name']}: {e}"
                            )

                    # Step 2: Prepare batch insert with ON CONFLICT
                    insert_start = time.time()

                    try:
                        # Build VALUES clause for bulk insert
                        placeholders = []
                        params = []
                        for i, embedding_data in enumerate(embeddings_data):
                            base = i * 5
                            placeholders.append(
                                f"(${base+1}, ${base+2}, ${base+3}, ${base+4}::vector, ${base+5})"
                            )
                            params.extend(
                                [
                                    embedding_data["chunk_id"],
                                    embedding_data["provider"],
                                    embedding_data["model"],
                                    str(embedding_data["embedding"]),
                                    embedding_data["dims"],
                                ]
                            )

                        # Single INSERT with ON CONFLICT for upsert semantics
                        query = f"""
                            INSERT INTO {table_name} (chunk_id, provider, model, embedding, dims)
                            VALUES {', '.join(placeholders)}
                            ON CONFLICT (chunk_id, provider, model)
                            DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                dims = EXCLUDED.dims,
                                created_at = CURRENT_TIMESTAMP
                        """

                        self.connection_manager.execute_sync(query, *params)

                        insert_time = time.time() - insert_start
                        logger.debug(
                            f"✅ Bulk INSERT completed in {insert_time:.3f}s "
                            f"({len(embeddings_data) / insert_time:.1f} emb/s)"
                        )
                        total_inserted = len(embeddings_data)

                    except Exception as e:
                        logger.error(f"Bulk INSERT failed: {e}")
                        raise

                    # Step 3: Recreate ivfflat index
                    if dropped_indexes and self._provider_instance:
                        logger.debug("📈 Recreating ivfflat index for fast similarity search")
                        index_start = time.time()
                        for index_info in dropped_indexes:
                            try:
                                self._provider_instance.create_vector_index(
                                    index_info["provider"],
                                    index_info["model"],
                                    index_info["dims"],
                                    index_info["metric"],
                                )
                                logger.debug(
                                    f"Recreated ivfflat index: {index_info['index_name']}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Failed to recreate index {index_info['index_name']}: {e}"
                                )
                        index_time = time.time() - index_start
                        logger.debug(f"✅ ivfflat index recreated in {index_time:.3f}s")

                    logger.debug(f"✅ Stored {actual_batch_size} embeddings successfully")
                else:
                    # Fallback to simple batch insert without optimization
                    logger.warning(
                        "Provider instance not available for index management, "
                        "using simple batch insert"
                    )
                    placeholders = []
                    params = []
                    for i, embedding_data in enumerate(embeddings_data):
                        base = i * 5
                        placeholders.append(
                            f"(${base+1}, ${base+2}, ${base+3}, ${base+4}::vector, ${base+5})"
                        )
                        params.extend(
                            [
                                embedding_data["chunk_id"],
                                embedding_data["provider"],
                                embedding_data["model"],
                                str(embedding_data["embedding"]),
                                embedding_data["dims"],
                            ]
                        )

                    query = f"""
                        INSERT INTO {table_name} (chunk_id, provider, model, embedding, dims)
                        VALUES {', '.join(placeholders)}
                        ON CONFLICT (chunk_id, provider, model)
                        DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            dims = EXCLUDED.dims,
                            created_at = CURRENT_TIMESTAMP
                    """
                    self.connection_manager.execute_sync(query, *params)
                    total_inserted = len(embeddings_data)

            else:
                # Small batch: use simple INSERT with ON CONFLICT
                small_start = time.time()

                try:
                    # Build VALUES clause for small batch
                    placeholders = []
                    params = []
                    for i, embedding_data in enumerate(embeddings_data):
                        base = i * 5
                        placeholders.append(
                            f"(${base+1}, ${base+2}, ${base+3}, ${base+4}::vector, ${base+5})"
                        )
                        params.extend(
                            [
                                embedding_data["chunk_id"],
                                embedding_data["provider"],
                                embedding_data["model"],
                                str(embedding_data["embedding"]),
                                embedding_data["dims"],
                            ]
                        )

                    # Single INSERT with ON CONFLICT
                    query = f"""
                        INSERT INTO {table_name} (chunk_id, provider, model, embedding, dims)
                        VALUES {', '.join(placeholders)}
                        ON CONFLICT (chunk_id, provider, model)
                        DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            dims = EXCLUDED.dims,
                            created_at = CURRENT_TIMESTAMP
                    """

                    self.connection_manager.execute_sync(query, *params)

                    small_time = time.time() - small_start
                    logger.debug(
                        f"✅ Small batch completed in {small_time:.3f}s "
                        f"({len(embeddings_data) / small_time:.1f} emb/s)"
                    )
                    total_inserted = len(embeddings_data)

                except Exception as e:
                    logger.error(f"Small batch failed: {e}")
                    raise

                # Ensure ivfflat indexes exist for semantic search
                if self._provider_instance and hasattr(
                    self._provider_instance, "get_existing_vector_indexes"
                ):
                    existing_indexes = (
                        self._provider_instance.get_existing_vector_indexes()
                    )
                    dims = first_embedding["dims"]

                    # Check if any index exists for this dimension
                    index_exists = any(idx["dims"] == dims for idx in existing_indexes)

                    if not index_exists:
                        logger.warning(
                            f"🔍 No ivfflat index found for {dims}D embeddings, "
                            f"creating one now"
                        )
                        try:
                            self._provider_instance.create_vector_index(
                                provider, model, dims, "cosine"
                            )
                            logger.info(
                                f"✅ Created missing ivfflat index for {provider}/{model} "
                                f"({dims}D)"
                            )
                        except Exception as e:
                            logger.error(
                                f"❌ Failed to create ivfflat index for {provider}/{model} "
                                f"({dims}D): {e}"
                            )

                logger.debug(f"✅ Stored {actual_batch_size} embeddings successfully")

            insert_time = time.time() - start_time
            logger.debug(f"⚡ Batch INSERT completed in {insert_time:.3f}s")

            if use_index_optimization:
                logger.debug(
                    f"🏆 Index-optimized batch insert: {total_inserted} embeddings in "
                    f"{insert_time:.3f}s ({total_inserted / insert_time:.1f} embeddings/sec)"
                )
            else:
                logger.debug(
                    f"🎯 Standard batch insert: {total_inserted} embeddings in "
                    f"{insert_time:.3f}s ({total_inserted / insert_time:.1f} embeddings/sec)"
                )

            return total_inserted

        except Exception as e:
            logger.error(f"💥 CRITICAL: Optimized batch insert failed: {e}")
            raise

    def get_embedding_by_chunk_id(
        self, chunk_id: int, provider: str, model: str
    ) -> Embedding | None:
        """Get embedding for specific chunk, provider, and model.

        Args:
            chunk_id: Chunk ID to get embedding for
            provider: Embedding provider name
            model: Embedding model name

        Returns:
            Embedding or None if not found
        """
        try:
            # Search across all embedding tables
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
                result = self.connection_manager.fetch_sync(
                    f"""
                    SELECT id, chunk_id, provider, model, embedding, dims, created_at
                    FROM {table_name}
                    WHERE chunk_id = $1 AND provider = $2 AND model = $3
                    """,
                    chunk_id,
                    provider,
                    model,
                )

                if result and len(result) > 0:
                    row = result[0]
                    return Embedding(
                        chunk_id=row["chunk_id"],
                        provider=row["provider"],
                        model=row["model"],
                        vector=row["embedding"],
                        dims=row["dims"],
                    )

            return None

        except Exception as e:
            logger.error(f"Failed to get embedding for chunk {chunk_id}: {e}")
            return None

    def get_existing_embeddings(
        self, chunk_ids: list[int], provider: str, model: str
    ) -> set[int]:
        """Get set of chunk IDs that already have embeddings for given provider/model.

        Args:
            chunk_ids: List of chunk IDs to check
            provider: Embedding provider name
            model: Embedding model name

        Returns:
            Set of chunk IDs that have embeddings
        """
        if not chunk_ids:
            return set()

        try:
            # Check all embedding tables since dimensions vary by model
            all_chunk_ids = set()

            # Get all embedding tables
            tables = self.connection_manager.fetch_sync(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name LIKE 'embeddings_%'
                AND table_schema = 'public'
                """
            )

            for row in tables:
                table_name = row["table_name"]
                # Create placeholders for IN clause
                placeholders = ", ".join(f"${i+1}" for i in range(len(chunk_ids)))
                params = list(chunk_ids) + [provider, model]

                results = self.connection_manager.fetch_sync(
                    f"""
                    SELECT DISTINCT chunk_id
                    FROM {table_name}
                    WHERE chunk_id IN ({placeholders})
                    AND provider = ${len(chunk_ids) + 1}
                    AND model = ${len(chunk_ids) + 2}
                    """,
                    *params,
                )

                all_chunk_ids.update(result["chunk_id"] for result in results)

            return all_chunk_ids

        except Exception as e:
            logger.error(f"Failed to get existing embeddings: {e}")
            return set()

    def delete_embeddings_by_chunk_id(self, chunk_id: int) -> None:
        """Delete all embeddings for a specific chunk.

        Args:
            chunk_id: Chunk ID to delete embeddings for
        """
        try:
            # Delete from all embedding tables
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
                    f"DELETE FROM {table_name} WHERE chunk_id = $1", chunk_id
                )

        except Exception as e:
            logger.error(f"Failed to delete embeddings for chunk {chunk_id}: {e}")
            raise
