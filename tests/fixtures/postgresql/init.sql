-- Initialize pgvector extension for ChunkHound testing
-- This script runs automatically when the PostgreSQL container starts

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extension is loaded
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Grant necessary permissions to chunkhound user
GRANT ALL PRIVILEGES ON DATABASE chunkhound_test TO chunkhound;
GRANT ALL PRIVILEGES ON SCHEMA public TO chunkhound;
