"""
StatVerify AI — Scheduled Cron Maintenance Task

Executed by Render Cron Job (or local cron/task scheduler) for periodic system maintenance:
- Cleaning up old PDF report exports and temporary log files.
- System health checks and telemetry logging.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.config import EXPORT_DIR, LOG_DIR, settings
from src.utils import logger


def cleanup_directory(directory: Path, days_old: int = 7) -> int:
    """
    Remove files in directory older than specified number of days.
    Returns count of removed files.
    """
    if not directory.exists():
        return 0

    cutoff_time = datetime.now() - timedelta(days=days_old)
    removed_count = 0

    for file_path in directory.glob("*"):
        if file_path.is_file():
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    removed_count += 1
                    logger.info(f"Cleaned up old maintenance file: {file_path.name}")
                except Exception as ex:
                    logger.warning(f"Failed to delete {file_path.name}: {ex}")

    return removed_count


def run_maintenance_task() -> bool:
    """
    Main entrypoint for scheduled cron execution.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"Starting StatVerify AI Cron Maintenance [{datetime.now().isoformat()}]")
    logger.info(f"Environment: {settings.app_env} | Version: {settings.app_version}")
    logger.info("=" * 60)

    # 1. Clean up old export PDFs and logs
    logger.info("1/2: Running directory cleanup...")
    exports_removed = cleanup_directory(EXPORT_DIR, days_old=7)
    logs_removed = cleanup_directory(LOG_DIR, days_old=14)
    logger.info(f"Cleanup finished. Removed {exports_removed} exports and {logs_removed} old log files.")

    # 2. System Health Check
    logger.info("2/2: Performing system health check...")
    api_key_configured = bool(settings.gemini_api_key and "your_" not in settings.gemini_api_key)
    logger.info(f"AI Model Configured ({settings.model_name}): {api_key_configured}")

    elapsed = time.time() - start_time
    logger.info(f"Cron Maintenance Completed Successfully in {elapsed:.2f}s")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = run_maintenance_task()
        sys.exit(0 if success else 1)
    except Exception as err:
        logger.error(f"Cron execution failed with unhandled exception: {err}")
        sys.exit(1)
