# ChunkHound LLM Context

## PROJECT_IDENTITY
ChunkHound: Semantic and regex search tool for codebases with MCP integration
Built: 100% by AI agents - NO human-written code
Purpose: Transform codebases into searchable knowledge bases for AI assistants

## MODIFICATION_RULES
**NEVER:**
- NEVER Use print() in MCP server (stdio.py, http_server.py, tools.py)
- NEVER Make single-row DB inserts in loops
- NEVER Use forward references (quotes) in type annotations unless needed

**ALWAYS:**
- ALWAYS Run smoke tests before committing: `uv run pytest tests/test_smoke.py`
- ALWAYS Batch embeddings (min: 100, max: provider_limit)
- ALWAYS Drop HNSW indexes for bulk inserts > 50 rows
- ALWAYS Use uv for all Python operations
- ALWAYS Update version via: `uv run scripts/update_version.py`

## KEY_COMMANDS
```bash
# Development
lint:      uv run ruff check chunkhound
typecheck: uv run mypy chunkhound
test:      uv run pytest
smoke:     uv run pytest tests/test_smoke.py -v -n auto  # MANDATORY before commits
format:    uv run ruff format chunkhound

# Running
index:     uv run chunkhound index [directory]
mcp_stdio: uv run chunkhound mcp
mcp_http:  uv run chunkhound mcp http --port 5173
```

## VERSION_MANAGEMENT
Dynamic versioning via hatch-vcs - version derived from git tags.

```bash
# Create release
uv run scripts/update_version.py 4.1.0

# Create pre-release
uv run scripts/update_version.py 4.1.0b1
uv run scripts/update_version.py 4.1.0rc1

# Bump version
uv run scripts/update_version.py --bump minor      # v4.0.1 → v4.1.0
uv run scripts/update_version.py --bump minor b1   # v4.0.1 → v4.1.0b1
```

NEVER manually edit version strings - ALWAYS create git tags instead.

## PUBLISHING_PROCESS
```bash
# 1. Create version tag
uv run scripts/update_version.py X.Y.Z

# 2. Run smoke tests (MANDATORY)
uv run pytest tests/test_smoke.py -v

# 3. Prepare release
./scripts/prepare_release.sh

# 4. Test local install
pip install dist/chunkhound-X.Y.Z-py3-none-any.whl

# 5. Push tag
git push origin vX.Y.Z

# 6. Publish
uv publish
```

## KNOWN_DEPRECATION_WARNINGS
**HDBSCAN + scikit-learn**: `force_all_finite` parameter warning
- Non-breaking, safe to ignore
- Waiting for upstream HDBSCAN fix
- Will break in sklearn 1.8 if not fixed upstream

## DATABASE_PROVIDERS
ChunkHound supports multiple database backends for flexibility and deployment scenarios.

### Supported Providers
- **DuckDB** (default): Single-file analytical database, optimal for local development
- **LanceDB**: Serverless vector database with native MVCC support
- **PostgreSQL**: Production-ready RDBMS with pgvector for vector similarity

### PostgreSQL Provider
**Status**: Available (added in v4.1.0)
**Dependencies**: `asyncpg>=0.29.0`, `pgvector>=0.3.0`

**Configuration**:
```bash
# Using connection string (recommended)
export CHUNKHOUND_DATABASE__PROVIDER=postgresql
export CHUNKHOUND_DATABASE__POSTGRESQL_CONNECTION_STRING="postgresql://user:pass@localhost/chunkhound"

# Using individual fields
export CHUNKHOUND_DATABASE__POSTGRESQL_HOST=localhost
export CHUNKHOUND_DATABASE__POSTGRESQL_PORT=5432
export CHUNKHOUND_DATABASE__POSTGRESQL_DATABASE=chunkhound
export CHUNKHOUND_DATABASE__POSTGRESQL_USER=postgres
export CHUNKHOUND_DATABASE__POSTGRESQL_PASSWORD=postgres
export CHUNKHOUND_DATABASE__POSTGRESQL_POOL_SIZE=5
```

**Setup**:
```bash
# Install PostgreSQL with pgvector
brew install postgresql pgvector

# Create database
createdb chunkhound
psql chunkhound -c "CREATE EXTENSION vector;"

# Run ChunkHound
uv run chunkhound index /path/to/code
```

**Architecture**:
- Extends `SerialDatabaseProvider` for single-threaded database access
- Uses asyncpg connection pooling with sync wrappers
- pgvector extension for vector similarity search (ivfflat indexes)
- Batch optimizations: drop/recreate indexes for bulk inserts (50+ embeddings)
- Worktree isolation via `worktree_id` field

**Performance**:
- Connection pooling reduces overhead for concurrent operations
- ivfflat indexes provide efficient approximate nearest neighbor search
- Batch operations optimized for PostgreSQL's COPY protocol

### Provider Constraints
**CRITICAL**: All providers inherit from `SerialDatabaseProvider`
- Single-threaded database access (enforced via SerialDatabaseExecutor)
- No concurrent database operations
- File parsing is parallelized, storage remains single-threaded
- Prevents corruption in databases requiring exclusive access (DuckDB)
- PostgreSQL handles multi-process safety via MVCC (ChunkHound still serializes per-process)

## PROJECT_MAINTENANCE
- Smoke tests are mandatory guardrails
- Run `uv run mypy chunkhound` during reviews to catch Optional/type boundary issues
- All code patterns should be self-documenting
