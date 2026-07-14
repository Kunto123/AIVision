"""
VisionInspect - Retention Manager
Kebijakan retensi data: auto-purge gambar dan history entries.
"""

import time
from pathlib import Path
from typing import Optional

from visioninspect.storage.db import Database
from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")


class RetentionManager:
    """
    Mengelola retensi data:
    - Hapus history entries yang lebih lama dari X hari
    - Hapus gambar yang lebih lama dari X hari (kecuali NG jika save_all_ng=True)
    - Sampling OK images (simpan N% saja)
    """

    def __init__(self, db: Database):
        self._db = db

    def purge_old_data(
        self,
        retention_days: int = 30,
        save_ok_sample_percent: int = 10,
        programs_dir: Optional[Path] = None,
    ) -> dict:
        """
        Purge data yang sudah melebihi batas retensi.
        Returns dict with stats.
        """
        stats = {"deleted_history": 0, "deleted_images": 0, "purged_programs": []}

        if retention_days <= 0:
            logger.info("Retention disabled (days=%d), skipping purge", retention_days)
            return stats

        cutoff = time.time() - (retention_days * 86400)
        cutoff_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cutoff))

        # Delete old history entries
        stats["deleted_history"] = self._db.delete_old_entries(cutoff_str)

        # Delete old images from filesystem
        if programs_dir and programs_dir.exists():
            for prog_dir in programs_dir.iterdir():
                if not prog_dir.is_dir():
                    continue
                prog_stats = self._purge_program_images(prog_dir, cutoff, save_ok_sample_percent)
                if prog_stats["deleted"] > 0:
                    stats["deleted_images"] += prog_stats["deleted"]
                    stats["purged_programs"].append({
                        "program": prog_dir.name,
                        "deleted": prog_stats["deleted"],
                    })

        logger.info(
            "Retention purge: %d history entries, %d images",
            stats["deleted_history"], stats["deleted_images"]
        )
        return stats

    def _purge_program_images(
        self, prog_dir: Path, cutoff: float, save_ok_pct: int
    ) -> dict:
        """Purge images for a single program."""
        deleted = 0

        # Purge old OK images (apply sampling)
        ok_dir = prog_dir / "images" / "ok"
        if ok_dir.exists():
            ok_files = sorted(ok_dir.iterdir(), key=lambda p: p.stat().st_mtime)
            for i, f in enumerate(ok_files):
                if f.stat().st_mtime < cutoff:
                    # Apply sampling: keep some percentage
                    if (i % max(1, 100 // max(1, save_ok_pct))) != 0:
                        f.unlink(missing_ok=True)
                        deleted += 1

        # Purge old NG images
        ng_dir = prog_dir / "images" / "ng"
        if ng_dir.exists():
            for f in ng_dir.iterdir():
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
                    deleted += 1

        return {"deleted": deleted}
