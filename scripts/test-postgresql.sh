#!/usr/bin/env bash
# Test PostgreSQL provider with Docker Compose
# This script starts PostgreSQL with pgvector, runs tests, and cleans up

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}ChunkHound PostgreSQL Provider Test Runner${NC}"
echo "============================================"
echo ""

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: docker-compose or docker is not installed${NC}"
    exit 1
fi

# Determine docker compose command
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Function to cleanup
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    $DOCKER_COMPOSE -f docker-compose.test.yml down -v
}

# Trap EXIT to ensure cleanup runs
trap cleanup EXIT

# Start PostgreSQL
echo -e "${YELLOW}Starting PostgreSQL with pgvector...${NC}"
$DOCKER_COMPOSE -f docker-compose.test.yml up -d

# Wait for PostgreSQL to be healthy
echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
timeout=60
elapsed=0
while ! $DOCKER_COMPOSE -f docker-compose.test.yml exec -T postgres pg_isready -U chunkhound -d chunkhound_test &> /dev/null; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [ $elapsed -ge $timeout ]; then
        echo -e "${RED}Error: PostgreSQL did not start within ${timeout} seconds${NC}"
        exit 1
    fi
done

echo -e "${GREEN}PostgreSQL is ready!${NC}"
echo ""

# Set environment variables for tests
export CHUNKHOUND_TEST_PG_HOST=localhost
export CHUNKHOUND_TEST_PG_PORT=5433
export CHUNKHOUND_TEST_PG_DATABASE=chunkhound_test
export CHUNKHOUND_TEST_PG_USER=chunkhound
export CHUNKHOUND_TEST_PG_PASSWORD=chunkhound_test_pass

# Display connection info
echo -e "${YELLOW}PostgreSQL Connection Info:${NC}"
echo "  Host: $CHUNKHOUND_TEST_PG_HOST"
echo "  Port: $CHUNKHOUND_TEST_PG_PORT"
echo "  Database: $CHUNKHOUND_TEST_PG_DATABASE"
echo "  User: $CHUNKHOUND_TEST_PG_USER"
echo ""

# Run tests
echo -e "${YELLOW}Running PostgreSQL provider tests...${NC}"
echo ""

if uv run pytest tests/providers/database/test_postgresql_provider.py -v "$@"; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
