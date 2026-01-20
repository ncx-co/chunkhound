"""Prune command module - handles worktree index cleanup operations."""

import argparse
import sys
from pathlib import Path

from loguru import logger

from chunkhound.api.cli.utils import verify_database_exists
from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.registry import configure_registry
from chunkhound.utils.worktree_detection import compute_worktree_id

from ..utils.rich_output import RichOutputFormatter


async def prune_worktree_command(args: argparse.Namespace, config: Config) -> None:
    """Remove all indexed data for a specific worktree.

    Args:
        args: Parsed command-line arguments with worktree_id
        config: Pre-validated configuration instance
    """
    formatter = RichOutputFormatter(verbose=getattr(args, "verbose", False))

    # Verify database exists
    try:
        verify_database_exists(config)
    except (ValueError, FileNotFoundError) as e:
        formatter.error(str(e))
        sys.exit(1)

    # Configure registry
    try:
        configure_registry(config)
    except Exception as e:
        formatter.error(f"Failed to configure registry: {e}")
        sys.exit(1)

    # Create database services
    try:
        services = create_services(db_path=config.database.path, config=config)
        provider = services.provider
    except Exception as e:
        formatter.error(f"Failed to create database services: {e}")
        sys.exit(1)

    worktree_id = args.worktree_id

    # Check if worktree exists
    worktree = provider.get_worktree(worktree_id)
    if not worktree:
        formatter.error(f"Worktree not found: {worktree_id}")
        sys.exit(1)

    # Get file counts before deletion
    try:
        counts = provider.get_worktree_file_count(worktree_id)
        native_count = counts.get("native_files", 0)
        inherited_count = counts.get("inherited_files", 0)
        total_count = native_count + inherited_count
    except Exception:
        total_count = 0
        native_count = 0
        inherited_count = 0

    # Display worktree info
    formatter.section_header(f"Pruning Worktree: {worktree_id}")
    formatter.info(f"Path: {worktree['path']}")
    formatter.info(f"Native files: {native_count}")
    formatter.info(f"Inherited files: {inherited_count}")
    formatter.info(f"Total indexed files: {total_count}")

    # Confirm deletion if not forced
    if not args.force:
        try:
            import questionary

            confirm = questionary.confirm(
                "Are you sure you want to delete this worktree index?",
                default=False,
            ).ask()
            if not confirm:
                formatter.info("Prune cancelled")
                sys.exit(0)
        except Exception:
            # Fallback to simple input if questionary not available
            response = input("Are you sure you want to delete this worktree index? (y/N): ")
            if response.lower() != "y":
                formatter.info("Prune cancelled")
                sys.exit(0)

    # Delete worktree
    try:
        success = provider.delete_worktree(worktree_id)
        if success:
            formatter.success(f"Successfully pruned worktree {worktree_id}")
            formatter.info(f"Removed {total_count} files from index")
        else:
            formatter.error(f"Failed to prune worktree {worktree_id}")
            sys.exit(1)
    except Exception as e:
        formatter.error(f"Error pruning worktree: {e}")
        logger.exception("Prune worktree failed")
        sys.exit(1)


async def prune_orphaned_command(args: argparse.Namespace, config: Config) -> None:
    """Remove indexed data for worktrees that no longer exist on disk.

    Args:
        args: Parsed command-line arguments
        config: Pre-validated configuration instance
    """
    formatter = RichOutputFormatter(verbose=getattr(args, "verbose", False))

    # Verify database exists
    try:
        verify_database_exists(config)
    except (ValueError, FileNotFoundError) as e:
        formatter.error(str(e))
        sys.exit(1)

    # Configure registry
    try:
        configure_registry(config)
    except Exception as e:
        formatter.error(f"Failed to configure registry: {e}")
        sys.exit(1)

    # Create database services
    try:
        services = create_services(db_path=config.database.path, config=config)
        provider = services.provider
    except Exception as e:
        formatter.error(f"Failed to create database services: {e}")
        sys.exit(1)

    # List all worktrees
    try:
        worktrees = provider.list_worktrees()
    except Exception as e:
        formatter.error(f"Failed to list worktrees: {e}")
        sys.exit(1)

    if not worktrees:
        formatter.info("No worktrees found in database")
        sys.exit(0)

    # Find orphaned worktrees (path no longer exists)
    orphaned = []
    for wt in worktrees:
        path = Path(wt["path"])
        if not path.exists():
            orphaned.append(wt)

    if not orphaned:
        formatter.success("No orphaned worktrees found")
        sys.exit(0)

    # Display orphaned worktrees
    formatter.section_header(f"Found {len(orphaned)} Orphaned Worktrees")
    for wt in orphaned:
        formatter.info(f"- {wt['id']}: {wt['path']}")

    # Confirm deletion if not forced
    if not args.force:
        try:
            import questionary

            confirm = questionary.confirm(
                f"Delete {len(orphaned)} orphaned worktree(s)?",
                default=False,
            ).ask()
            if not confirm:
                formatter.info("Prune cancelled")
                sys.exit(0)
        except Exception:
            # Fallback to simple input if questionary not available
            response = input(
                f"Delete {len(orphaned)} orphaned worktree(s)? (y/N): "
            )
            if response.lower() != "y":
                formatter.info("Prune cancelled")
                sys.exit(0)

    # Delete orphaned worktrees
    deleted_count = 0
    total_files_removed = 0
    for wt in orphaned:
        try:
            # Get file count before deletion
            counts = provider.get_worktree_file_count(wt["id"])
            file_count = counts.get("native_files", 0) + counts.get(
                "inherited_files", 0
            )

            success = provider.delete_worktree(wt["id"])
            if success:
                deleted_count += 1
                total_files_removed += file_count
                formatter.info(f"✓ Deleted {wt['id']} ({file_count} files)")
            else:
                formatter.warning(f"✗ Failed to delete {wt['id']}")
        except Exception as e:
            formatter.error(f"Error deleting {wt['id']}: {e}")
            logger.exception(f"Failed to delete worktree {wt['id']}")

    # Summary
    formatter.section_header("Prune Summary")
    formatter.success(f"Deleted {deleted_count}/{len(orphaned)} worktrees")
    formatter.info(f"Removed {total_files_removed} files from index")


async def worktree_id_command(args: argparse.Namespace, config: Config) -> None:
    """Get the worktree ID for a given path.

    Args:
        args: Parsed command-line arguments with path
        config: Pre-validated configuration instance
    """
    formatter = RichOutputFormatter(verbose=getattr(args, "verbose", False))

    path = Path(args.path).resolve()

    if not path.exists():
        formatter.error(f"Path does not exist: {path}")
        sys.exit(1)

    worktree_id = compute_worktree_id(path)

    if args.quiet:
        # Just print the ID for scripting
        print(worktree_id)
    else:
        formatter.info(f"Path: {path}")
        formatter.info(f"Worktree ID: {worktree_id}")
