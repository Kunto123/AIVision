"""
VisionInspect - Database (SQLite)
Menyimpan riwayat inspeksi, konfigurasi, dan metadata.
Menggunakan WAL mode untuk performa.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")


class Database:
    """
    SQLite database untuk riwayat inspeksi.
    Thread-safe dengan single connection dan lock.
    Menggunakan WAL mode untuk konkurensi read/write.
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._local = threading.local()
        self._connect()
        self._create_tables()
        logger.info("Database initialized: %s", db_path)

    def _connect(self) -> None:
        """Get or create thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._local.conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,  # We handle threading manually
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn.row_factory = sqlite3.Row

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._connect()
        return self._local.conn

    def _create_tables(self) -> None:
        cursor = self.conn.cursor()

        # Inspection history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inspection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                program TEXT NOT NULL,
                score REAL NOT NULL,
                judgement TEXT NOT NULL CHECK(judgement IN ('OK', 'NG')),
                threshold REAL NOT NULL,
                latency_ms REAL,
                image_path TEXT,
                thumbnail_path TEXT,
                roi_region TEXT,
                corrected INTEGER DEFAULT 0,
                correct_judgement TEXT,
                corrected_at TEXT,
                metadata TEXT
            )
        """)

        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_timestamp
            ON inspection_history(timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_program
            ON inspection_history(program)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_judgement
            ON inspection_history(judgement)
        """)

        # Program counters (cached for fast access)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS program_counters (
                program TEXT PRIMARY KEY,
                total INTEGER DEFAULT 0,
                ok_count INTEGER DEFAULT 0,
                ng_count INTEGER DEFAULT 0,
                last_updated TEXT
            )
        """)

        # Audit log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                program TEXT,
                action TEXT NOT NULL,
                details TEXT
            )
        """)

        self.conn.commit()

    # ---- Inspection History ----

    def add_inspection(self, entry: dict) -> int:
        """Add an inspection result. Returns entry ID."""
        self.conn.execute("""
            INSERT INTO inspection_history
                (timestamp, program, score, judgement, threshold,
                 latency_ms, image_path, thumbnail_path, roi_region, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            entry.get("program", ""),
            entry.get("score", 0.0),
            entry.get("judgement", "OK"),
            entry.get("threshold", 0.5),
            entry.get("latency_ms"),
            entry.get("image_path"),
            entry.get("thumbnail_path"),
            json.dumps(entry.get("roi_region")) if entry.get("roi_region") else None,
            json.dumps(entry.get("metadata")) if entry.get("metadata") else None,
        ))
        self.conn.commit()

        # Update counters
        row_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self._update_counters(
            entry.get("program", ""),
            entry.get("judgement", "OK"),
        )
        return row_id

    def get_history(
        self,
        program: Optional[str] = None,
        judgement: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get inspection history with optional filters."""
        query = "SELECT * FROM inspection_history WHERE 1=1"
        params = []

        if program:
            query += " AND program = ?"
            params.append(program)
        if judgement:
            query += " AND judgement = ?"
            params.append(judgement)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_history_count(
        self,
        program: Optional[str] = None,
        judgement: Optional[str] = None,
    ) -> int:
        """Get count of history entries."""
        query = "SELECT COUNT(*) FROM inspection_history WHERE 1=1"
        params = []
        if program:
            query += " AND program = ?"
            params.append(program)
        if judgement:
            query += " AND judgement = ?"
            params.append(judgement)
        return self.conn.execute(query, params).fetchone()[0]

    def mark_correction(self, entry_id: int, correct_judgement: str) -> None:
        """Mark a history entry as corrected."""
        self.conn.execute("""
            UPDATE inspection_history
            SET corrected = 1,
                correct_judgement = ?,
                corrected_at = ?
            WHERE id = ?
        """, (correct_judgement, time.strftime("%Y-%m-%d %H:%M:%S"), entry_id))
        self.conn.commit()

    def delete_old_entries(self, before_timestamp: str) -> int:
        """Delete entries older than given timestamp. Returns count deleted."""
        cursor = self.conn.execute(
            "DELETE FROM inspection_history WHERE timestamp < ?",
            (before_timestamp,),
        )
        deleted = cursor.rowcount
        self.conn.commit()
        logger.info("Deleted %d old history entries (before %s)", deleted, before_timestamp)
        return deleted

    # ---- Counters ----

    def _update_counters(self, program: str, judgement: str) -> None:
        """Update in-memory and DB counters."""
        self.conn.execute("""
            INSERT INTO program_counters (program, total, ok_count, ng_count, last_updated)
            VALUES (?, 1, ?, ?, ?)
            ON CONFLICT(program) DO UPDATE SET
                total = total + 1,
                ok_count = CASE WHEN ? = 'OK' THEN ok_count + 1 ELSE ok_count END,
                ng_count = CASE WHEN ? = 'NG' THEN ng_count + 1 ELSE ng_count END,
                last_updated = ?
        """, (
            program,
            1 if judgement == "OK" else 0,
            1 if judgement == "NG" else 0,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            judgement, judgement,
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ))
        self.conn.commit()

    def get_counters(self, program: str) -> Dict[str, int]:
        """Get counters for a program."""
        cursor = self.conn.execute(
            "SELECT total, ok_count, ng_count FROM program_counters WHERE program = ?",
            (program,),
        )
        row = cursor.fetchone()
        if row:
            return {"total": row["total"], "ok": row["ok_count"], "ng": row["ng_count"]}
        return {"total": 0, "ok": 0, "ng": 0}

    def reset_counters(self, program: str) -> None:
        """Reset counters for a program."""
        self.conn.execute(
            "DELETE FROM program_counters WHERE program = ?",
            (program,),
        )
        self.conn.commit()

    # ---- Audit ----

    def add_audit(self, program: str, action: str, details: dict = None) -> int:
        """Add audit log entry."""
        cursor = self.conn.execute(
            "INSERT INTO audit_log (timestamp, program, action, details) VALUES (?, ?, ?, ?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), program, action,
             json.dumps(details) if details else None),
        )
        self.conn.commit()
        return cursor.lastrowid

    # ---- Maintenance ----

    def vacuum(self) -> None:
        """VACUUM the database to reclaim space."""
        self.conn.execute("VACUUM")
        logger.info("Database vacuumed")

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
