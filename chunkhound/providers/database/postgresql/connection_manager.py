"""PostgreSQL connection and schema management for ChunkHound.

This module provides connection pooling via asyncpg with a synchronous wrapper
to maintain compatibility with the SerialDatabaseProvider pattern.
"""

import asyncio
from typing import Any
from pathlib import Path

import asyncpg
from loguru import logger


class PostgreSQLConnectionManager:
    """Manages PostgreSQL connections, schema creation, and database operations.

    Uses asyncpg for async connection pooling, wrapped in a sync interface
    to work with ChunkHound's single-threaded SerialDatabaseProvider pattern.
    """

    def __init__(self, connection_string: str, config: Any | None = None):
        """Initialize PostgreSQL connection manager.

        Args:
            connection_string: PostgreSQL connection string
                (e.g., "postgresql://user:pass@localhost:5432/chunkhound")
            config: Database configuration for provider-specific settings
        """
        self._connection_string = connection_string
        self.pool: asyncpg.Pool | None = None
        self.config = config
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    @property
    def db_path(self) -> str:
        """Database connection identifier (connection string)."""
        # Mask password in connection string for logging
        try:
            parts = self._connection_string.split("@")
            if len(parts) > 1:
                # Extract user/pass part and replace password with asterisks
                auth_part = parts[0].split("//")[1]
                if ":" in auth_part:
                    user = auth_part.split(":")[0]
                    return f"postgresql://{user}:***@{parts[1]}"
            return self._connection_string
        except Exception:
            return "postgresql://***"

    @property
    def is_connected(self) -> bool:
        """Check if database connection pool is active."""
        return self.pool is not None

    def connect(self) -> None:
        """Establish database connection pool and initialize schema."""
        logger.info("Connecting to PostgreSQL database")

        try:
            # Create connection pool
            pool_size = 5  # Default pool size
            if self.config and hasattr(self.config, "postgresql_pool_size"):
                pool_size = self.config.postgresql_pool_size

            self.pool = self._loop.run_until_complete(
                asyncpg.create_pool(
                    self._connection_string,
                    min_size=1,
                    max_size=pool_size,
                    command_timeout=60,
                )
            )

            logger.info("PostgreSQL connection pool established")

            # Load pgvector extension
            self._load_extensions()

            logger.info("PostgreSQL connection manager initialization complete")

        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise

    def disconnect(self) -> None:
        """Close database connection pool."""
        if self.pool is not None:
            try:
                self._loop.run_until_complete(self.pool.close())
                self.pool = None
                logger.info("PostgreSQL connection pool closed")
            except Exception as e:
                logger.error(f"Error closing PostgreSQL connection pool: {e}")
                raise

    def _load_extensions(self) -> None:
        """Load required PostgreSQL extensions (pgvector)."""
        logger.info("Loading PostgreSQL extensions")

        if self.pool is None:
            raise RuntimeError("No database connection pool")

        async def load_ext() -> None:
            async with self.pool.acquire() as conn:  # type: ignore
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                logger.info("pgvector extension loaded successfully")

        try:
            self._loop.run_until_complete(load_ext())
        except Exception as e:
            logger.error(f"Failed to load PostgreSQL extensions: {e}")
            raise

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information."""
        status = {
            "provider": "postgresql",
            "connected": self.is_connected,
            "db_path": self.db_path,
            "version": None,
            "extensions": [],
            "tables": [],
            "errors": [],
        }

        if not self.is_connected:
            status["errors"].append("Not connected to database")
            return status

        async def check_health() -> None:
            if self.pool is None:
                status["errors"].append("Database pool is None")
                return

            try:
                async with self.pool.acquire() as conn:
                    # Get PostgreSQL version
                    version_result = await conn.fetchval("SELECT version()")
                    status["version"] = version_result

                    # Check if pgvector extension is loaded
                    extensions_result = await conn.fetch("""
                        SELECT extname, extversion
                        FROM pg_extension
                        WHERE extname = 'vector'
                    """)

                    for row in extensions_result:
                        status["extensions"].append({
                            "name": row["extname"],
                            "version": row["extversion"],
                        })

                    # Check if tables exist
                    tables_result = await conn.fetch("""
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_type = 'BASE TABLE'
                    """)

                    status["tables"] = [row["table_name"] for row in tables_result]

                    # Basic functionality test
                    test_result = await conn.fetchval("SELECT 1")
                    if test_result != 1:
                        status["errors"].append("Basic query test failed")

            except Exception as e:
                status["errors"].append(f"Health check error: {str(e)}")

        try:
            self._loop.run_until_complete(check_health())
        except Exception as e:
            status["errors"].append(f"Health check execution error: {str(e)}")

        return status

    def get_connection_info(self) -> dict[str, Any]:
        """Get information about the database connection."""
        return {
            "provider": "postgresql",
            "db_path": self.db_path,
            "connected": self.is_connected,
            "pool_size": self.config.postgresql_pool_size if self.config and hasattr(self.config, "postgresql_pool_size") else 5,
        }

    def execute_sync(self, query: str, *args: Any) -> Any:
        """Execute a query synchronously.

        This is a helper method for simple queries that don't need
        to return results. For complex operations, use async methods.

        Args:
            query: SQL query to execute
            *args: Query parameters

        Returns:
            Query result
        """
        if self.pool is None:
            raise RuntimeError("No database connection pool")

        async def _execute() -> Any:
            async with self.pool.acquire() as conn:  # type: ignore
                return await conn.execute(query, *args)

        return self._loop.run_until_complete(_execute())

    def fetch_sync(self, query: str, *args: Any) -> list[Any]:
        """Fetch multiple rows synchronously.

        Args:
            query: SQL query to execute
            *args: Query parameters

        Returns:
            List of records
        """
        if self.pool is None:
            raise RuntimeError("No database connection pool")

        async def _fetch() -> list[Any]:
            async with self.pool.acquire() as conn:  # type: ignore
                return await conn.fetch(query, *args)

        return self._loop.run_until_complete(_fetch())

    def fetchval_sync(self, query: str, *args: Any) -> Any:
        """Fetch a single value synchronously.

        Args:
            query: SQL query to execute
            *args: Query parameters

        Returns:
            Single value
        """
        if self.pool is None:
            raise RuntimeError("No database connection pool")

        async def _fetchval() -> Any:
            async with self.pool.acquire() as conn:  # type: ignore
                return await conn.fetchval(query, *args)

        return self._loop.run_until_complete(_fetchval())
