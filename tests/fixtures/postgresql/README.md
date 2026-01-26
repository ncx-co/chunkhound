# PostgreSQL Test Fixtures

This directory contains fixtures for testing ChunkHound's PostgreSQL provider.

## Files

- `init.sql` - Initialization script that runs when the PostgreSQL container starts
  - Enables the pgvector extension
  - Grants necessary permissions

## Usage

The easiest way to run PostgreSQL tests is using the provided test runner:

```bash
./scripts/test-postgresql.sh
```

This script will:
1. Start PostgreSQL with pgvector using Docker Compose
2. Wait for the database to be ready
3. Run the PostgreSQL provider tests
4. Clean up containers and volumes

## Manual Setup

If you prefer to manage the PostgreSQL container manually:

```bash
# Start PostgreSQL
docker-compose -f docker-compose.test.yml up -d

# Wait for it to be ready
docker-compose -f docker-compose.test.yml exec postgres pg_isready -U chunkhound -d chunkhound_test

# Set environment variables
export CHUNKHOUND_TEST_PG_HOST=localhost
export CHUNKHOUND_TEST_PG_PORT=5433
export CHUNKHOUND_TEST_PG_DATABASE=chunkhound_test
export CHUNKHOUND_TEST_PG_USER=chunkhound
export CHUNKHOUND_TEST_PG_PASSWORD=chunkhound_test_pass

# Run tests
uv run pytest tests/providers/database/test_postgresql_provider.py -v

# Stop and clean up
docker-compose -f docker-compose.test.yml down -v
```

## Connection Details

When running via Docker Compose:
- **Host**: localhost
- **Port**: 5433 (mapped from container's 5432)
- **Database**: chunkhound_test
- **User**: chunkhound
- **Password**: chunkhound_test_pass

## Customization

You can customize the PostgreSQL configuration by editing `docker-compose.test.yml`.

For persistent data across test runs, comment out the volume cleanup in the test script:
```bash
# Change this line in test-postgresql.sh:
$DOCKER_COMPOSE -f docker-compose.test.yml down -v
# To:
$DOCKER_COMPOSE -f docker-compose.test.yml down
```
