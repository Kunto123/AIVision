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

        # Push outbox (C3 — antrian tahan-restart untuk sink PostgreSQL)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS push_outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0
            )
        """)

        # Users (authentication, roles, RFID)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT 'operator'
                    CHECK(role IN ('admin', 'operator')),
                rfid_uid TEXT UNIQUE,
                rfid_bound_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        self.conn.commit()

        # C4: migrasi kolom must_change_password (SQLite ADD COLUMN sekali)
        try:
            self.conn.execute(
                "ALTER TABLE users ADD COLUMN must_change_password INTEGER DEFAULT 0")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # kolom sudah ada

        # Seed default admin if no users exist
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            self._seed_default_admin()

    # ---- Inspection History ----

    @staticmethod
    def _json_or_str(value):
        """Normalisasi kolom JSON: str diteruskan apa adanya (hindari double-
        encode), dict/list di-dump, None/empty → None."""
        if value is None:
            return None
        if isinstance(value, str):
            return value if value.strip() else None
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return None

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
            self._json_or_str(entry.get("roi_region")),
            self._json_or_str(entry.get("metadata")),
        ))
        self.conn.commit()

        # Update counters
        row_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self._update_counters(
            entry.get("program", ""),
            entry.get("judgement", "OK"),
        )
        return row_id

    def get_history_entry(self, entry_id: int) -> Optional[Dict[str, Any]]:
        """Get a single inspection history entry by id."""
        cursor = self.conn.execute(
            "SELECT * FROM inspection_history WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

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
            # Filter mengikuti hasil TERKOREKSI (kolom tampilan)
            query += " AND COALESCE(correct_judgement, judgement) = ?"
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
        """Mark a history entry as corrected.

        Catatan: kolom `judgement` (hasil asli inspeksi) TIDAK ditimpa —
        nilai koreksi disimpan di `correct_judgement`. Tampilan tabel
        memakai COALESCE(correct_judgement, judgement).
        """
        self.conn.execute("""
            UPDATE inspection_history
            SET corrected = 1,
                correct_judgement = ?,
                corrected_at = ?
            WHERE id = ?
        """, (correct_judgement, time.strftime("%Y-%m-%d %H:%M:%S"), entry_id))
        self.conn.commit()

    def update_roi_region(self, entry_id: int, roi_region) -> None:
        """Update per-ROI breakdown (kolom roi_region) — dipakai setelah
        koreksi tuning agar kolom Per-ROI menampilkan hasil terkoreksi."""
        self.conn.execute(
            "UPDATE inspection_history SET roi_region = ? WHERE id = ?",
            (self._json_or_str(roi_region), entry_id))
        self.conn.commit()

    def rollback_correction(self, entry_id: int) -> None:
        """Revert a previously corrected entry back to original state."""
        self.conn.execute("""
            UPDATE inspection_history
            SET corrected = 0,
                correct_judgement = NULL,
                corrected_at = NULL
            WHERE id = ?
        """, (entry_id,))
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

    # ---- Users / Authentication ----

    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash password with SHA-256 + pepper."""
        import hashlib
        return hashlib.sha256(f"visioninspect_2024_{password}".encode()).hexdigest()

    def _seed_default_admin(self):
        """Create default admin account on first run (C4: wajib ganti password)."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.conn.execute("""
            INSERT INTO users (username, password_hash, display_name, role,
                               must_change_password, created_at, updated_at)
            VALUES (?, ?, ?, ?, 1, ?, ?)
        """, ("admin", self._hash_password("admin"), "Administrator", "admin", now, now))
        self.conn.commit()
        logger.info("Default admin account created (admin/admin — WAJIB ganti password saat login pertama)")

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """Verify credentials. Returns user dict or None."""
        cursor = self.conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row and dict(row)["password_hash"] == self._hash_password(password):
            return dict(row)
        return None

    def get_user_by_rfid(self, rfid_uid: str) -> Optional[dict]:
        """Look up user by RFID UID."""
        cursor = self.conn.execute(
            "SELECT * FROM users WHERE rfid_uid = ?", (rfid_uid,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_users(self) -> List[Dict[str, Any]]:
        """List all users (without password hashes)."""
        cursor = self.conn.execute(
            "SELECT id, username, display_name, role, rfid_uid, rfid_bound_at, "
            "created_at, updated_at FROM users ORDER BY id")
        return [dict(row) for row in cursor.fetchall()]

    def add_user(self, username: str, password: str, display_name: str = "",
                 role: str = "operator") -> int:
        """Add a new user. Returns user ID."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.conn.execute("""
            INSERT INTO users (username, password_hash, display_name, role, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username, self._hash_password(password), display_name, role, now, now))
        self.conn.commit()
        logger.info("User added: %s (role=%s)", username, role)
        return cursor.lastrowid

    def update_user(self, user_id: int, display_name: str = None,
                    password: str = None, role: str = None,
                    must_change_password: bool = None) -> bool:
        """Update user fields. Returns True if changed."""
        fields = []
        values = []
        if display_name is not None:
            fields.append("display_name = ?")
            values.append(display_name)
        if password is not None:
            fields.append("password_hash = ?")
            values.append(self._hash_password(password))
            # Password baru di-set → flag paksa-ganti dimatikan (C4)
            fields.append("must_change_password = 0")
        if role is not None:
            fields.append("role = ?")
            values.append(role)
        if must_change_password is not None:
            fields.append("must_change_password = ?")
            values.append(1 if must_change_password else 0)
        if not fields:
            return False
        fields.append("updated_at = ?")
        values.append(time.strftime("%Y-%m-%d %H:%M:%S"))
        values.append(user_id)
        self.conn.execute(
            f"UPDATE users SET {', '.join(fields)} WHERE id = ?", values)
        self.conn.commit()
        return True

    def bind_rfid(self, user_id: int, rfid_uid: str) -> bool:
        """Bind RFID UID to a user. Returns False if UID already used."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.conn.execute("""
                UPDATE users SET rfid_uid = ?, rfid_bound_at = ?, updated_at = ?
                WHERE id = ?
            """, (rfid_uid, now, now, user_id))
            self.conn.commit()
            logger.info("RFID %s bound to user id=%d", rfid_uid, user_id)
            return True
        except Exception:
            return False

    def unbind_rfid(self, user_id: int) -> bool:
        """Remove RFID binding from a user."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.conn.execute("""
            UPDATE users SET rfid_uid = NULL, rfid_bound_at = NULL, updated_at = ? WHERE id = ?
        """, (now, user_id))
        self.conn.commit()
        return True

    def delete_user(self, user_id: int) -> bool:
        """Delete a user. Returns False if last admin."""
        # Check if this is the last admin
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM users WHERE role = 'admin'")
        admin_count = cursor.fetchone()[0]
        cursor = self.conn.execute("SELECT role FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row and row["role"] == "admin" and admin_count <= 1:
            logger.warning("Cannot delete last admin user")
            return False
        self.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.conn.commit()
        logger.info("User deleted: id=%d", user_id)
        return True

    # ---- Push Outbox (C3 — antrian tahan-restart) ----

    def add_outbox(self, entry: dict) -> int:
        """Simpan satu entry push ke antrian outbox SQLite.

        Entry adalah dict kwargs untuk ``PostgresDB.push_inspection``.
        Tahan-restart: bila aplikasi mati sebelum sink selesai, entry tetap
        ada dan di-flush saat startup berikutnya.
        """
        self.conn.execute("""
            INSERT INTO push_outbox (entry_json, created_at)
            VALUES (?, ?)
        """, (json.dumps(entry), time.strftime("%Y-%m-%d %H:%M:%S")))
        self.conn.commit()
        return self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_outbox(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Ambil batch outbox (tertua dulu). Returns [{id, entry, attempts}, ...]."""
        rows = self.conn.execute("""
            SELECT id, entry_json, attempts FROM push_outbox
            ORDER BY id LIMIT ?
        """, (limit,)).fetchall()
        result = []
        for r in rows:
            try:
                entry = json.loads(r["entry_json"])
            except (json.JSONDecodeError, TypeError, ValueError):
                entry = {}
            result.append({"id": r["id"], "entry": entry, "attempts": r["attempts"]})
        return result

    def delete_outbox(self, ids: List[int]) -> None:
        """Hapus entry outbox yang sudah berhasil di-push (C3: nol duplikat)."""
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self.conn.execute(
            f"DELETE FROM push_outbox WHERE id IN ({placeholders})", ids)
        self.conn.commit()

    def bump_outbox_attempts(self, ids: List[int]) -> None:
        """Naikkan hitungan percobaan entry yang gagal (retry tick berikutnya)."""
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self.conn.execute(
            f"UPDATE push_outbox SET attempts = attempts + 1 "
            f"WHERE id IN ({placeholders})", ids)
        self.conn.commit()

    def count_outbox(self) -> int:
        """Jumlah entry outbox yang belum ter-sink."""
        return self.conn.execute(
            "SELECT COUNT(*) AS c FROM push_outbox").fetchone()["c"]

    def drop_oldest_outbox(self, n: int) -> None:
        """Buang n entry tertua dari outbox (antrian bounded, C3).

        Dilakukan hanya bila antrian membengkak (mis. PG mati berhari-hari);
        entry terbaru dipertahankan. Baris yang dibuang dicatat ke audit log.
        """
        if n <= 0:
            return
        rows = self.conn.execute(
            "SELECT id FROM push_outbox ORDER BY id LIMIT ?", (n,)).fetchall()
        ids = [r["id"] for r in rows]
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self.conn.execute(
            f"DELETE FROM push_outbox WHERE id IN ({placeholders})", ids)
        self.conn.commit()
        logger.warning("Outbox bounded: %d entry tertua dibuang (antrian penuh)", n)

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
