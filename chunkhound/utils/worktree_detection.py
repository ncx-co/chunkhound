"""Git worktree detection and change computation utilities.

This module provides utilities for detecting Git worktrees and computing
file changes between worktrees to enable efficient delta-based indexing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import hashlib
import subprocess
from loguru import logger


@dataclass
class WorktreeInfo:
    """Information about a Git worktree.

    Attributes:
        is_main: True if this is the main worktree (has .git directory)
        is_linked: True if this is a linked worktree (has .git file)
        main_worktree_path: Path to main worktree (None if is_main)
        git_dir: Path to Git directory or worktree metadata
        worktree_id: Unique identifier for this worktree
        git_repo_path: Path to main .git directory
    """
    is_main: bool
    is_linked: bool
    main_worktree_path: Optional[Path]
    git_dir: Optional[Path]
    worktree_id: str
    git_repo_path: Optional[Path]


def detect_worktree_info(directory: Path) -> WorktreeInfo:
    """Detect if directory is a Git worktree and identify relationships.

    Args:
        directory: Path to directory to analyze

    Returns:
        WorktreeInfo with detected information

    Examples:
        >>> info = detect_worktree_info(Path("/repo/main"))
        >>> info.is_main
        True

        >>> info = detect_worktree_info(Path("/repo-feature"))
        >>> info.is_linked
        True
        >>> info.main_worktree_path
        Path("/repo/main")
    """
    git_path = directory / ".git"

    # Main worktree: .git is a directory
    if git_path.is_dir():
        worktree_id = compute_worktree_id(directory)
        return WorktreeInfo(
            is_main=True,
            is_linked=False,
            main_worktree_path=None,
            git_dir=git_path,
            worktree_id=worktree_id,
            git_repo_path=git_path
        )

    # Linked worktree: .git is a file
    if git_path.is_file():
        try:
            gitdir_line = git_path.read_text().strip()
            # Parse: "gitdir: /path/to/main/.git/worktrees/name"
            if gitdir_line.startswith("gitdir:"):
                linked_git_dir = Path(gitdir_line.split(":", 1)[1].strip())

                # Navigate: .git/worktrees/name -> .git -> main_worktree/
                if linked_git_dir.parent.parent.name == ".git":
                    main_git_dir = linked_git_dir.parent.parent
                    main_worktree = main_git_dir.parent
                    worktree_id = compute_worktree_id(directory)

                    return WorktreeInfo(
                        is_main=False,
                        is_linked=True,
                        main_worktree_path=main_worktree,
                        git_dir=linked_git_dir,
                        worktree_id=worktree_id,
                        git_repo_path=main_git_dir
                    )
        except Exception as e:
            logger.debug(f"Failed to parse linked worktree .git file: {e}")

    # Not a Git worktree
    worktree_id = compute_worktree_id(directory)
    return WorktreeInfo(
        is_main=False,
        is_linked=False,
        main_worktree_path=None,
        git_dir=None,
        worktree_id=worktree_id,
        git_repo_path=None
    )


def compute_worktree_id(directory: Path) -> str:
    """Generate unique worktree ID from directory path.

    Args:
        directory: Path to worktree directory

    Returns:
        16-character hex string identifier

    Examples:
        >>> compute_worktree_id(Path("/repo/main"))
        'a1b2c3d4e5f6g7h8'
    """
    # Use absolute path hash for consistency
    abs_path = str(directory.resolve())
    return hashlib.sha256(abs_path.encode()).hexdigest()[:16]


def get_git_head_ref(worktree_path: Path) -> Optional[str]:
    """Get current HEAD commit SHA for a worktree.

    Args:
        worktree_path: Path to worktree directory

    Returns:
        40-character commit SHA or None if not a Git repo

    Examples:
        >>> get_git_head_ref(Path("/repo/main"))
        'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0'
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.debug(f"Failed to get Git HEAD ref for {worktree_path}: {e}")
        return None


def compute_changed_files_via_git(
    main_worktree: Path,
    feature_worktree: Path,
    patterns: list[str],
    exclude_patterns: list[str]
) -> dict[str, list[Path]]:
    """Compute which files differ between worktrees using Git.

    Uses `git diff-tree` to efficiently detect changes between the HEAD
    commits of two worktrees.

    Args:
        main_worktree: Path to main (base) worktree
        feature_worktree: Path to feature (delta) worktree
        patterns: File patterns to include (e.g., ["*.py", "*.ts"])
        exclude_patterns: File patterns to exclude (e.g., ["*_test.py"])

    Returns:
        Dictionary with keys:
        - 'added': Files only in feature worktree
        - 'modified': Files changed from main worktree
        - 'deleted': Files removed from main worktree
        - 'unchanged': Files identical to main worktree

    Raises:
        subprocess.CalledProcessError: If git command fails

    Examples:
        >>> changes = compute_changed_files_via_git(
        ...     Path("/repo/main"),
        ...     Path("/repo-feature"),
        ...     ["*.py"],
        ...     []
        ... )
        >>> len(changes['modified'])
        42
    """
    # Get current HEAD for each worktree
    main_ref = get_git_head_ref(main_worktree)
    feature_ref = get_git_head_ref(feature_worktree)

    if not main_ref or not feature_ref:
        raise ValueError("Could not determine HEAD refs for worktrees")

    logger.info(
        f"Computing Git diff: {main_ref[:8]}..{feature_ref[:8]} "
        f"({main_worktree.name} -> {feature_worktree.name})"
    )

    # Run: git diff-tree -r --name-status main_ref feature_ref
    result = subprocess.run(
        ['git', 'diff-tree', '-r', '--name-status', main_ref, feature_ref],
        cwd=main_worktree,
        capture_output=True,
        text=True,
        check=True,
        timeout=30
    )

    added, modified, deleted = [], [], []

    for line in result.stdout.strip().splitlines():
        if not line:
            continue

        parts = line.split('\t', 1)
        if len(parts) != 2:
            continue

        status, path = parts

        # Apply pattern filtering
        if not _matches_patterns(path, patterns, exclude_patterns):
            continue

        if status == 'A':
            added.append(feature_worktree / path)
        elif status == 'M':
            modified.append(feature_worktree / path)
        elif status == 'D':
            # Deleted files referenced from main worktree
            deleted.append(main_worktree / path)

    # Compute unchanged files
    # All files in feature worktree that weren't marked as added/modified
    from chunkhound.utils.file_patterns import discover_files

    all_feature_files = discover_files(
        feature_worktree,
        patterns=patterns,
        exclude_patterns=exclude_patterns
    )

    changed_files = set(added + modified)
    unchanged = [f for f in all_feature_files if f not in changed_files]

    logger.info(
        f"Git diff complete: "
        f"+{len(added)} ~{len(modified)} -{len(deleted)} ={len(unchanged)}"
    )

    return {
        'added': added,
        'modified': modified,
        'deleted': deleted,
        'unchanged': unchanged
    }


def fallback_compute_changed_files(
    main_worktree: Path,
    feature_worktree: Path,
    main_index_files: list[dict],
    patterns: list[str],
    exclude_patterns: list[str]
) -> dict[str, list[Path]]:
    """Fallback: compute changes via content hash comparison.

    This method is used when Git-based detection is unavailable or fails.
    It compares content hashes to determine which files changed.

    Args:
        main_worktree: Path to main (base) worktree
        feature_worktree: Path to feature (delta) worktree
        main_index_files: List of file dicts from main index (with content_hash)
        patterns: File patterns to include
        exclude_patterns: File patterns to exclude

    Returns:
        Dictionary with same structure as compute_changed_files_via_git

    Examples:
        >>> main_files = [
        ...     {'path': 'foo.py', 'content_hash': 'abc123'},
        ...     {'path': 'bar.py', 'content_hash': 'def456'}
        ... ]
        >>> changes = fallback_compute_changed_files(
        ...     Path("/repo/main"),
        ...     Path("/repo-feature"),
        ...     main_files,
        ...     ["*.py"],
        ...     []
        ... )
    """
    from chunkhound.utils.hashing import compute_file_hash
    from chunkhound.utils.file_patterns import discover_files

    logger.info(
        f"Using hash-based change detection (Git fallback) for {feature_worktree.name}"
    )

    # Build hash map from main index
    main_hash_map = {
        f['path']: f['content_hash']
        for f in main_index_files
        if f.get('content_hash')
    }

    # Discover files in feature worktree
    feature_files = discover_files(
        feature_worktree,
        patterns=patterns,
        exclude_patterns=exclude_patterns
    )

    added, modified, unchanged = [], [], []

    for feature_file in feature_files:
        rel_path = str(feature_file.relative_to(feature_worktree))

        if rel_path not in main_hash_map:
            # New file (not in main)
            added.append(feature_file)
        else:
            # Compare content hash
            try:
                feature_hash = compute_file_hash(feature_file)
                if feature_hash != main_hash_map[rel_path]:
                    modified.append(feature_file)
                else:
                    unchanged.append(feature_file)
            except Exception as e:
                logger.warning(f"Failed to hash {feature_file}: {e}, treating as modified")
                modified.append(feature_file)

    # Detect deletions (files in main but not in feature)
    feature_paths = {str(f.relative_to(feature_worktree)) for f in feature_files}
    deleted = [
        feature_worktree / path
        for path in main_hash_map.keys()
        if path not in feature_paths
    ]

    logger.info(
        f"Hash-based diff complete: "
        f"+{len(added)} ~{len(modified)} -{len(deleted)} ={len(unchanged)}"
    )

    return {
        'added': added,
        'modified': modified,
        'deleted': deleted,
        'unchanged': unchanged
    }


def _matches_patterns(
    path: str,
    patterns: list[str],
    exclude_patterns: list[str]
) -> bool:
    """Check if path matches inclusion patterns and doesn't match exclusions.

    Args:
        path: Relative file path to check
        patterns: Inclusion patterns (empty = include all)
        exclude_patterns: Exclusion patterns

    Returns:
        True if path should be included
    """
    from fnmatch import fnmatch

    # Check exclusions first
    for exclude_pattern in exclude_patterns:
        if fnmatch(path, exclude_pattern):
            return False

    # If no inclusion patterns, include everything (not excluded)
    if not patterns:
        return True

    # Check if matches any inclusion pattern
    for pattern in patterns:
        if fnmatch(path, pattern):
            return True

    return False
