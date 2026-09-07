"""
Tests for src/cron.py maintenance module.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.cron import cleanup_directory, run_maintenance


def test_cleanup_directory(tmp_path: Path):
    """Verify that cleanup_directory unlinks files older than max_age_hours."""
    old_file = tmp_path / "old_export.pdf"
    old_file.write_text("dummy pdf content")

    new_file = tmp_path / "new_export.pdf"
    new_file.write_text("fresh content")

    # Set mtime of old_file to 48 hours ago
    past_time = old_file.stat().st_mtime - (48 * 3600)
    import os
    os.utime(old_file, (past_time, past_time))

    deleted_count, bytes_freed = cleanup_directory(tmp_path, max_age_hours=24)

    assert deleted_count == 1
    assert bytes_freed > 0
    assert not old_file.exists()
    assert new_file.exists()


def test_run_maintenance():
    """Verify run_maintenance executes without exceptions."""
    result = run_maintenance(max_age_hours=24)
    assert isinstance(result, dict)
    assert "export_files_removed" in result
    assert "log_files_removed" in result
