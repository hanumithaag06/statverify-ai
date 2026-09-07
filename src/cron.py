"""
Scheduled Cron Maintenance Module for StatVerify AI.

This module provides automated cleanup of temporary exports, logs, and cache
files, intended to be executed on a schedule (e.g. via Render Cron Jobs).

Usage:
    python -m src.cron
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from src.config import settings
from src.utils import current_timestamp, logger


def cleanup_directory(directory: Path, max_age_hours: int = 24) -> tuple[int, int]:
    """
    Remove files in `directory` older than `max_age_hours`.

    Parameters
    ----------
    directory : Path
        Directory to scan and clean.
    max_age_hours : int, optional
        Files older than this duration will be unlinked, by default 24.

    Returns
    -------
    tuple[int, int]
        (number_of_files_deleted, total_bytes_freed)
    """
    if not directory.exists() or not directory.is_dir():
        return 0, 0

    now = time.time()
    cutoff = now - (max_age_hours * 3600)
    deleted_count = 0
    bytes_freed = 0

    for file_path in directory.glob("*"):
        if file_path.is_file():
            try:
                stat = file_path.stat()
                if stat.st_mtime < cutoff:
                    file_size = stat.st_size
                    file_path.unlink()
                    deleted_count += 1
                    bytes_freed += file_size
            except Exception as ex:
                logger.warning(f"Could not clean file {file_path}: {ex}")

    return deleted_count, bytes_freed


def run_maintenance(max_age_hours: int = 24) -> dict[str, int]:
    """
    Run complete maintenance pipeline.

    Returns
    -------
    dict[str, int]
        Summary of cleanup actions.
    """
    timestamp = current_timestamp()
    logger.info(f"=== Starting Cron Maintenance Task [{timestamp}] ===")

    # 1. Clean exports directory
    export_deleted, export_bytes = cleanup_directory(settings.export_dir, max_age_hours)
    logger.info(f"Cleaned exports directory: {export_deleted} files removed ({export_bytes / 1024:.2f} KB freed).")

    # 2. Clean logs directory (keep logs from last 7 days)
    log_deleted, log_bytes = cleanup_directory(settings.log_dir, max_age_hours=168)
    logger.info(f"Cleaned logs directory: {log_deleted} stale log files removed.")

    summary = {
        "export_files_removed": export_deleted,
        "export_bytes_freed": export_bytes,
        "log_files_removed": log_deleted,
        "log_bytes_freed": log_bytes,
    }

    logger.info(f"=== Cron Maintenance Task Complete [{current_timestamp()}] ===")
    return summary


if __name__ == "__main__":
    run_maintenance()
