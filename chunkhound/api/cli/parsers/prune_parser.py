"""Prune command parser - defines CLI interface for worktree pruning."""

import argparse


def add_prune_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add prune command subparser with all its subcommands.

    Args:
        subparsers: Subparsers action from main parser
    """
    prune_parser = subparsers.add_parser(
        "prune",
        help="Manage worktree index cleanup",
        description="Remove indexed data for deleted worktrees",
    )

    # Create subcommands for prune
    prune_subparsers = prune_parser.add_subparsers(
        dest="prune_command",
        help="Prune subcommands",
    )

    # prune worktree --id <worktree_id>
    worktree_parser = prune_subparsers.add_parser(
        "worktree",
        help="Remove specific worktree from index",
        description="Delete all indexed data for a specific worktree by ID",
    )
    worktree_parser.add_argument(
        "--id",
        dest="worktree_id",
        required=True,
        help="Worktree ID to prune (16-char hex string)",
    )
    worktree_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # prune orphaned
    orphaned_parser = prune_subparsers.add_parser(
        "orphaned",
        help="Remove worktrees that no longer exist on disk",
        description="Find and delete indexed worktrees whose paths no longer exist",
    )
    orphaned_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # worktree-id --path <path>
    id_parser = prune_subparsers.add_parser(
        "worktree-id",
        help="Get worktree ID for a path",
        description="Compute the worktree ID for a given directory path",
    )
    id_parser.add_argument(
        "--path",
        required=True,
        help="Path to compute worktree ID for",
    )
    id_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output the ID (for scripting)",
    )
