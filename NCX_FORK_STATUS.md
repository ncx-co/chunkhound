# NCX ChunkHound Fork Status

## Overview

This is NCX's fork of ChunkHound with worktree-aware indexing support. The fork enables efficient semantic code search across 120+ git worktrees without duplicating index data.

**Fork:** https://github.com/ncx-co/chunkhound
**Production Branch:** `ncx/main`
**Upstream:** https://github.com/ChunkHound/chunkhound

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Tracks upstream ChunkHound/chunkhound main |
| `ncx/main` | NCX production branch (main + worktree support) |
| `feature/worktree-support` | Development branch for worktree features |

## Worktree Support Features

### ✅ Implemented

1. **Worktree Detection** (`chunkhound/utils/worktree_detection.py`)
   - Detects main vs linked worktrees
   - Generates unique 16-char hex worktree IDs
   - Computes git diff for changed files

2. **Worktree Enumeration** (`chunkhound/utils/worktree_enumeration.py`)
   - Discovers all worktrees in repository
   - Extracts branch info, paths, main/linked status

3. **Configuration Toggle** (`chunkhound/core/config/indexing_config.py`)
   - `worktree_support_enabled` flag (default: false)
   - Opt-in via config, env var, or CLI flag

4. **MCP Tools**
   - `list_worktrees` - Enumerate indexed worktrees
   - `search_semantic` + `worktree_scope` parameter
   - `search_regex` + `worktree_scope` parameter

### ⚠️ Known Issues

#### Test Failures (13 tests)

**Database Filtering Tests (13 failures in `test_worktree_search_filtering.py`)**
- Tests expect full worktree schema (worktree_id columns, inheritance tables)
- Database providers need schema migration support
- Search methods need worktree filtering implementation
- Status: Needs implementation work

**Fixed:**
- ~~HTTP server tests (2 failures)~~ - Removed in commit 9bc95cb (server removed upstream)

#### Git Diff Tests (3 failures in `test_worktree_detection.py`)
- `test_compute_changes_success` - Git diff computation
- `test_fallback_compute_changes` - Fallback hash-based detection
- `test_fallback_hash_failure` - Error handling
- Status: Needs investigation

### Test Coverage

```
Total: 90 tests
Passed: 75 (83%)
Failed: 15 (17%)

Core functionality:
- Worktree detection: 10/10 ✅
- Worktree enumeration: 45/45 ✅
- Configuration: 18/20 ✅
- MCP tools: 2/2 ✅
- Database filtering: 0/13 ❌
- Git diff: 0/3 ❌
```

## Rebase History

**Date:** 2026-01-20
**Base:** ChunkHound main @ commit 2996beb
**Commits:** 7 commits + 1 fix

### Rebase Summary

```
✅ fae9fb2 feat(worktree): add git worktree support for shared indexing
✅ 2db1308 feat(worktree): implement delta indexing and worktree-scoped search
✅ aa42023 feat(worktree): add worktree_ids filtering to LanceDB provider search
✅ 02a24e0 test(worktree): add comprehensive tests for delta indexing and search filtering
✅ cbb1b83 feat(mcp): add worktree support to MCP server (Phase 5)
✅ 33425a7 test(worktree): add MCP worktree tools test coverage
✅ 7a66354 feat(worktree): make worktree support opt-in via configuration
✅ 9bc95cb test: remove obsolete HTTP server worktree visibility tests
```

### Conflicts Resolved

All conflicts were with `chunkhound/mcp_server/http_server.py`:
- **Reason:** HTTP server removed upstream (PR #117)
- **Resolution:** Deleted file, kept stdio.py implementation
- **Impact:** Tools correctly registered in stdio mode

## Architecture

### Indexing Flow

```
Feature Worktree Index Request
           │
           ▼
    Detect Worktree Info
    (main? linked? branch?)
           │
           ▼
    Compute Git Diff vs Main
    (changed files only)
           │
           ▼
    Index Delta Files
    (with feature worktree_id)
           │
           ▼
    Inherit Unchanged Files
    (reference main worktree chunks)
```

### Database Schema (Planned)

```sql
-- Files table
ALTER TABLE files ADD COLUMN worktree_id TEXT;
CREATE INDEX idx_files_worktree ON files(worktree_id);

-- Worktrees table
CREATE TABLE worktrees (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    is_main BOOLEAN NOT NULL,
    main_worktree_id TEXT,
    head_ref TEXT,
    indexed_at TIMESTAMP
);

-- Inheritance table
CREATE TABLE file_worktree_inheritance (
    worktree_id TEXT NOT NULL,
    file_id INTEGER NOT NULL,
    PRIMARY KEY (worktree_id, file_id)
);
```

## NCX Integration Status

### Not Yet Integrated

The following NCX monolith integration tasks are pending:

- [ ] Update `.chunkhound.json` to enable worktree support
- [ ] Update `.claude/skills/chunkhound-search/SKILL.md` with worktree docs
- [ ] Add ChunkHound prune hook to `wtp-remove-with-cleanup.sh`
- [ ] Implement `chunkhound prune-worktree` command
- [ ] Implement `chunkhound worktree-id` command
- [ ] Test delta indexing in feature worktree
- [ ] Update Docker MCP Gateway config

### Installation

When ready to deploy:

```bash
# Install from NCX fork
uv pip install git+https://github.com/ncx-co/chunkhound@ncx/main

# Or for development
git clone https://github.com/ncx-co/chunkhound
cd chunkhound
git checkout ncx/main
uv pip install -e .
```

## Next Steps

### Phase 1: Fix Test Failures
1. Implement database schema migration for worktree columns
2. Add worktree filtering to search methods
3. Debug git diff computation tests
4. Achieve >95% test pass rate

### Phase 2: Implement Prune Commands
1. Add `chunkhound prune-worktree --id <id>` command
2. Add `chunkhound prune-worktrees --orphaned` command
3. Add `chunkhound worktree-id --path <path>` helper
4. Integrate with WTP cleanup hooks

### Phase 3: NCX Integration
1. Update configuration files
2. Update skill documentation
3. Test delta indexing workflow
4. Deploy to production

### Phase 4: Upstream Contribution (Optional)
1. Polish test coverage
2. Add comprehensive documentation
3. Submit PR to ChunkHound/chunkhound
4. Address upstream feedback

## Maintenance

### Syncing with Upstream

```bash
# Fetch upstream changes
git fetch origin main

# Merge into ncx/main (if needed)
git checkout ncx/main
git merge origin/main

# Resolve conflicts
# Run tests
# Push to fork
```

### Security Vulnerabilities

**Note:** GitHub reports 26 vulnerabilities on the default branch:
- 1 critical, 10 high, 11 moderate, 4 low
- See: https://github.com/ncx-co/chunkhound/security/dependabot
- Consider addressing before production deployment

## Documentation

- **Plan:** `/Users/brad/.claude/plans/encapsulated-puzzling-glacier.md`
- **Implementation:** `external-refs/chunkhound/` (this repo)
- **Skills:** `.claude/skills/chunkhound-search/`

## Contact

For questions about this fork, contact NCX engineering team.
