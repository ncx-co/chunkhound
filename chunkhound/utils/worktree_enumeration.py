"""Git worktree enumeration utilities.

This module provides utilities for discovering all worktrees associated with
a Git repository, enabling multi-worktree indexing and search operations.
"""

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Optional

from loguru import logger

from chunkhound.utils.worktree_detection import (
    WorktreeInfo,
    detect_worktree_info,
    compute_worktree_id,
    get_git_head_ref,
)


@dataclass
class EnumeratedWorktree:
    """Extended worktree information for enumeration results.

    Attributes:
        info: Base worktree detection info
        path: Absolute path to the worktree
        branch: Current branch name (e.g., "main", "feature/foo")
        head_ref: Current HEAD commit SHA
        is_bare: True if this is a bare worktree entry
        is_detached: True if HEAD is detached (not on a branch)
    """
    info: WorktreeInfo
    path: Path
    branch: Optional[str]
    head_ref: Optional[str]
    is_bare: bool
    is_detached: bool

    @property
    def worktree_id(self) -> str:
        """Get the unique worktree identifier."""
        return self.info.worktree_id

    @property
    def is_main(self) -> bool:
        """Check if this is the main worktree."""
        return self.info.is_main

    @property
    def display_name(self) -> str:
        """Get human-readable name for this worktree."""
        if self.branch:
            return self.branch
        if self.is_detached and self.head_ref:
            return f"detached@{self.head_ref[:8]}"
        return self.path.name


def enumerate_worktrees(directory: Path) -> list[EnumeratedWorktree]:
    """Discover all worktrees for the repository containing directory.

    This function finds all worktrees associated with the Git repository
    that contains the given directory. It works whether called from the
    main worktree or any linked worktree.

    Args:
        directory: Path to any directory within the repository

    Returns:
        List of EnumeratedWorktree objects, sorted with main worktree first,
        then linked worktrees sorted by branch name.

    Examples:
        >>> worktrees = enumerate_worktrees(Path("/repo/main"))
        >>> len(worktrees)
        3
        >>> worktrees[0].is_main
        True
        >>> worktrees[1].branch
        'feature/auth'
    """
    # First, detect what kind of worktree we're in
    current_info = detect_worktree_info(directory.resolve())

    if not current_info.git_repo_path:
        logger.debug(f"Directory {directory} is not in a Git repository")
        return []

    # Determine the main worktree path for running git commands
    if current_info.is_main:
        main_path = directory.resolve()
    elif current_info.main_worktree_path:
        main_path = current_info.main_worktree_path
    else:
        logger.warning(f"Could not determine main worktree for {directory}")
        return []

    # Try git worktree list first (most reliable)
    worktrees = _enumerate_via_git_command(main_path)

    if not worktrees:
        # Fallback: scan .git/worktrees directory
        worktrees = _enumerate_via_filesystem(main_path, current_info.git_repo_path)

    # Sort: main worktree first, then by branch name
    worktrees.sort(key=lambda w: (not w.is_main, w.branch or w.path.name))

    return worktrees


def _enumerate_via_git_command(main_path: Path) -> list[EnumeratedWorktree]:
    """Enumerate worktrees using git worktree list --porcelain.

    This is the preferred method as it provides complete information
    directly from Git.

    Args:
        main_path: Path to main worktree (for running git command)

    Returns:
        List of EnumeratedWorktree objects
    """
    try:
        result = subprocess.run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=main_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug(f"git worktree list failed: {e}")
        return []

    worktrees = []
    current_entry: dict = {}

    for line in result.stdout.splitlines():
        line = line.strip()

        if not line:
            # Empty line marks end of entry
            if current_entry.get('worktree'):
                wt = _parse_worktree_entry(current_entry)
                if wt:
                    worktrees.append(wt)
            current_entry = {}
            continue

        if line.startswith('worktree '):
            current_entry['worktree'] = line[9:]  # Path after "worktree "
        elif line.startswith('HEAD '):
            current_entry['head'] = line[5:]  # SHA after "HEAD "
        elif line.startswith('branch '):
            current_entry['branch'] = line[7:]  # Ref after "branch "
        elif line == 'bare':
            current_entry['bare'] = True
        elif line == 'detached':
            current_entry['detached'] = True

    # Don't forget last entry if file doesn't end with newline
    if current_entry.get('worktree'):
        wt = _parse_worktree_entry(current_entry)
        if wt:
            worktrees.append(wt)

    return worktrees


def _parse_worktree_entry(entry: dict) -> Optional[EnumeratedWorktree]:
    """Parse a single worktree entry from git worktree list output.

    Args:
        entry: Dictionary with keys: worktree, head, branch, bare, detached

    Returns:
        EnumeratedWorktree or None if entry is invalid
    """
    worktree_path = Path(entry.get('worktree', ''))
    if not worktree_path.exists():
        logger.debug(f"Worktree path does not exist: {worktree_path}")
        return None

    # Detect worktree info
    info = detect_worktree_info(worktree_path)

    # Parse branch name (refs/heads/main -> main)
    branch_ref = entry.get('branch', '')
    branch = None
    if branch_ref.startswith('refs/heads/'):
        branch = branch_ref[11:]
    elif branch_ref:
        branch = branch_ref

    return EnumeratedWorktree(
        info=info,
        path=worktree_path.resolve(),
        branch=branch,
        head_ref=entry.get('head'),
        is_bare=entry.get('bare', False),
        is_detached=entry.get('detached', False),
    )


def _enumerate_via_filesystem(
    main_path: Path,
    git_repo_path: Path
) -> list[EnumeratedWorktree]:
    """Fallback: enumerate worktrees by scanning .git/worktrees directory.

    This method is used when git worktree list is unavailable.

    Args:
        main_path: Path to main worktree
        git_repo_path: Path to .git directory

    Returns:
        List of EnumeratedWorktree objects
    """
    worktrees = []

    # Add main worktree
    main_info = detect_worktree_info(main_path)
    main_head = get_git_head_ref(main_path)
    main_branch = _get_current_branch(main_path)

    worktrees.append(EnumeratedWorktree(
        info=main_info,
        path=main_path.resolve(),
        branch=main_branch,
        head_ref=main_head,
        is_bare=False,
        is_detached=main_branch is None and main_head is not None,
    ))

    # Scan .git/worktrees for linked worktrees
    worktrees_dir = git_repo_path / "worktrees"
    if worktrees_dir.is_dir():
        for wt_entry in worktrees_dir.iterdir():
            if not wt_entry.is_dir():
                continue

            # Read the gitdir file to find worktree path
            gitdir_file = wt_entry / "gitdir"
            if not gitdir_file.is_file():
                continue

            try:
                wt_path_str = gitdir_file.read_text().strip()
                # gitdir contains path to worktree's .git file
                # We need the parent directory (the actual worktree)
                wt_git_path = Path(wt_path_str)
                if wt_git_path.name == ".git":
                    wt_path = wt_git_path.parent
                else:
                    wt_path = wt_git_path.parent

                if not wt_path.exists():
                    logger.debug(f"Linked worktree path does not exist: {wt_path}")
                    continue

                wt_info = detect_worktree_info(wt_path)
                wt_head = get_git_head_ref(wt_path)
                wt_branch = _get_current_branch(wt_path)

                worktrees.append(EnumeratedWorktree(
                    info=wt_info,
                    path=wt_path.resolve(),
                    branch=wt_branch,
                    head_ref=wt_head,
                    is_bare=False,
                    is_detached=wt_branch is None and wt_head is not None,
                ))
            except Exception as e:
                logger.debug(f"Failed to parse worktree entry {wt_entry}: {e}")

    return worktrees


def _get_current_branch(worktree_path: Path) -> Optional[str]:
    """Get the current branch name for a worktree.

    Args:
        worktree_path: Path to worktree

    Returns:
        Branch name or None if detached/error
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        branch = result.stdout.strip()
        # "HEAD" means detached
        return None if branch == "HEAD" else branch
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.debug(f"Failed to get branch for {worktree_path}: {e}")
        return None


def get_worktree_by_id(
    directory: Path,
    worktree_id: str
) -> Optional[EnumeratedWorktree]:
    """Find a specific worktree by its ID.

    Args:
        directory: Path to any directory within the repository
        worktree_id: The 16-character worktree identifier

    Returns:
        EnumeratedWorktree if found, None otherwise
    """
    for wt in enumerate_worktrees(directory):
        if wt.worktree_id == worktree_id:
            return wt
    return None


def get_main_worktree(directory: Path) -> Optional[EnumeratedWorktree]:
    """Find the main worktree for the repository.

    Args:
        directory: Path to any directory within the repository

    Returns:
        EnumeratedWorktree for main worktree, or None if not found
    """
    for wt in enumerate_worktrees(directory):
        if wt.is_main:
            return wt
    return None


__all__ = [
    'EnumeratedWorktree',
    'enumerate_worktrees',
    'get_worktree_by_id',
    'get_main_worktree',
]
