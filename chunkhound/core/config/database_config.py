"""Database configuration for ChunkHound.

This module provides database-specific configuration with support for
multiple database providers and storage backends.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DatabaseConfig(BaseModel):
    """Database configuration with support for multiple providers.

    Configuration can be provided via:
    - Environment variables (CHUNKHOUND_DATABASE_*)
    - Configuration files
    - CLI arguments
    - Default values
    """

    # Database location
    path: Path | None = Field(default=None, description="Path to database directory")

    # Provider selection
    provider: Literal["duckdb", "lancedb", "postgresql"] = Field(
        default="duckdb", description="Database provider to use"
    )

    # LanceDB-specific settings
    lancedb_index_type: Literal["auto", "ivf_hnsw_sq", "ivf_rq"] | None = Field(
        default=None,
        description="LanceDB vector index type: auto (default), ivf_hnsw_sq, or ivf_rq (requires 0.25.3+)",
    )

    lancedb_optimize_fragment_threshold: int = Field(
        default=100,
        ge=0,
        description="Minimum fragment count to trigger optimization (0 = always optimize, 50 = aggressive, 100 = balanced, 500 = conservative)",
    )

    # PostgreSQL-specific settings
    postgresql_host: str | None = Field(
        default=None,
        description="PostgreSQL host (default: localhost)",
    )
    postgresql_port: int | None = Field(
        default=None,
        description="PostgreSQL port (default: 5432)",
    )
    postgresql_database: str | None = Field(
        default=None,
        description="PostgreSQL database name",
    )
    postgresql_user: str | None = Field(
        default=None,
        description="PostgreSQL username",
    )
    postgresql_password: str | None = Field(
        default=None,
        description="PostgreSQL password",
    )
    postgresql_connection_string: str | None = Field(
        default=None,
        description="PostgreSQL connection string (overrides individual fields if provided)",
    )
    postgresql_pool_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description="PostgreSQL connection pool size",
    )

    # Disk usage limits
    max_disk_usage_mb: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum database size in MB before indexing is stopped (None = no limit)",
    )

    @field_validator("path")
    def validate_path(cls, v: Path | None) -> Path | None:
        """Convert string paths to Path objects."""
        if v is not None and not isinstance(v, Path):
            return Path(v)
        return v

    @field_validator("provider")
    def validate_provider(cls, v: str) -> str:
        """Validate database provider selection."""
        valid_providers = ["duckdb", "lancedb", "postgresql"]
        if v not in valid_providers:
            raise ValueError(f"Invalid provider: {v}. Must be one of {valid_providers}")
        return v

    def get_db_path(self) -> Path:
        """Get the actual database location for the configured provider.

        Returns the final path used by the provider, including all
        provider-specific transformations:
        - DuckDB: path/chunks.db (file) or :memory: for in-memory
        - LanceDB: path/lancedb.lancedb/ (directory with .lancedb suffix)

        This is the authoritative source for database location checks.
        """
        if self.path is None:
            raise ValueError("Database path not configured")

        # Skip directory creation for in-memory databases (":memory:" is invalid on Windows)
        is_memory = str(self.path) == ":memory:"

        # Backwards-compatible handling:
        # - Older ChunkHound versions used `database.path` as the direct DuckDB
        #   file location (for example, `.chunkhound/db` as a file).
        # - Newer versions treat `database.path` as a directory and store the
        #   DuckDB file as `path/chunks.db`.
        #
        # When the configured path already exists as a file, we treat it as a
        # legacy DuckDB database file and return it directly instead of trying
        # to create a directory at that location.
        if self.provider == "duckdb" and not is_memory:
            if self.path.exists() and self.path.is_file():
                return self.path

        if not is_memory:
            # For directory-style layouts, ensure the base path exists.
            self.path.mkdir(parents=True, exist_ok=True)

        if self.provider == "duckdb":
            return self.path if is_memory else self.path / "chunks.db"
        elif self.provider == "lancedb":
            # LanceDB adds .lancedb suffix to prevent naming collisions
            # and clarify storage structure (see lancedb_provider.py:111-113)
            lancedb_base = self.path / "lancedb"
            return lancedb_base.parent / f"{lancedb_base.stem}.lancedb"
        else:
            raise ValueError(f"Unknown database provider: {self.provider}")

    def get_postgresql_connection_string(self) -> str:
        """Build PostgreSQL connection string from configuration.

        Returns:
            Connection string for PostgreSQL

        Raises:
            ValueError: If PostgreSQL configuration is incomplete
        """
        # If explicit connection string provided, use it
        if self.postgresql_connection_string:
            return self.postgresql_connection_string

        # Build from individual fields
        host = self.postgresql_host or "localhost"
        port = self.postgresql_port or 5432
        database = self.postgresql_database
        user = self.postgresql_user
        password = self.postgresql_password

        if not database:
            raise ValueError("PostgreSQL database name is required")
        if not user:
            raise ValueError("PostgreSQL username is required")

        # Build connection string
        conn_str = f"postgresql://{user}"
        if password:
            conn_str += f":{password}"
        conn_str += f"@{host}:{port}/{database}"

        return conn_str

    def is_configured(self) -> bool:
        """Check if database is properly configured."""
        if self.provider == "postgresql":
            # For PostgreSQL, check if connection string or required fields are set
            return bool(
                self.postgresql_connection_string
                or (self.postgresql_database and self.postgresql_user)
            )
        return self.path is not None

    @classmethod
    def add_cli_arguments(
        cls, parser: argparse.ArgumentParser, required_path: bool = False
    ) -> None:
        """Add database-related CLI arguments."""
        parser.add_argument(
            "--db",
            "--database-path",
            type=Path,
            help="Database file path (default: from config file or .chunkhound.db)",
            required=required_path,
        )

        parser.add_argument(
            "--database-provider",
            choices=["duckdb", "lancedb", "postgresql"],
            help="Database provider to use",
        )

        parser.add_argument(
            "--postgresql-connection-string",
            type=str,
            help="PostgreSQL connection string (overrides individual PostgreSQL settings)",
        )

        parser.add_argument(
            "--max-disk-usage-gb",
            type=float,
            help="Maximum database size in GB before indexing is stopped",
        )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load database config from environment variables."""
        config = {}
        # Support both new and legacy env var names
        if db_path := (
            os.getenv("CHUNKHOUND_DATABASE__PATH") or os.getenv("CHUNKHOUND_DB_PATH")
        ):
            config["path"] = Path(db_path)
        if provider := os.getenv("CHUNKHOUND_DATABASE__PROVIDER"):
            config["provider"] = provider
        if index_type := os.getenv("CHUNKHOUND_DATABASE__LANCEDB_INDEX_TYPE"):
            config["lancedb_index_type"] = index_type
        if threshold := os.getenv("CHUNKHOUND_DATABASE__LANCEDB_OPTIMIZE_FRAGMENT_THRESHOLD"):
            config["lancedb_optimize_fragment_threshold"] = int(threshold)
        # Disk usage limit from environment
        if max_disk_gb := os.getenv("CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB"):
            try:
                config["max_disk_usage_mb"] = float(max_disk_gb) * 1024.0
            except ValueError:
                # Invalid value - silently ignore
                pass
        # PostgreSQL configuration from environment
        if pg_conn_str := os.getenv("CHUNKHOUND_DATABASE__POSTGRESQL_CONNECTION_STRING"):
            config["postgresql_connection_string"] = pg_conn_str
        if pg_host := os.getenv("CHUNKHOUND_DATABASE__POSTGRESQL_HOST"):
            config["postgresql_host"] = pg_host
        if pg_port := os.getenv("CHUNKHOUND_DATABASE__POSTGRESQL_PORT"):
            config["postgresql_port"] = int(pg_port)
        if pg_db := os.getenv("CHUNKHOUND_DATABASE__POSTGRESQL_DATABASE"):
            config["postgresql_database"] = pg_db
        if pg_user := os.getenv("CHUNKHOUND_DATABASE__POSTGRESQL_USER"):
            config["postgresql_user"] = pg_user
        if pg_pass := os.getenv("CHUNKHOUND_DATABASE__POSTGRESQL_PASSWORD"):
            config["postgresql_password"] = pg_pass
        if pg_pool := os.getenv("CHUNKHOUND_DATABASE__POSTGRESQL_POOL_SIZE"):
            config["postgresql_pool_size"] = int(pg_pool)
        return config

    @classmethod
    def extract_cli_overrides(cls, args: Any) -> dict[str, Any]:
        """Extract database config from CLI arguments."""
        overrides = {}
        if hasattr(args, "db") and args.db:
            overrides["path"] = args.db
        if hasattr(args, "database_path") and args.database_path:
            overrides["path"] = args.database_path
        if hasattr(args, "database_provider") and args.database_provider:
            overrides["provider"] = args.database_provider
        if hasattr(args, "max_disk_usage_gb") and args.max_disk_usage_gb is not None:
            overrides["max_disk_usage_mb"] = args.max_disk_usage_gb * 1024.0
        return overrides

    def __repr__(self) -> str:
        """String representation of database configuration."""
        parts = [f"provider={self.provider}", f"path={self.path}"]
        if self.max_disk_usage_mb is not None:
            parts.append(f"max_disk_usage_mb={self.max_disk_usage_mb}")
        return f"DatabaseConfig({', '.join(parts)})"
