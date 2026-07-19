"""
VisionInspect - PostgreSQL Database Layer
Koneksi ke PostgreSQL 18 untuk autentikasi (qc_user_accounts)
dan push hasil inspeksi (qc_inspection_push).

Tabel sudah ada di database — class ini hanya query, tidak create.

Tabel skema:
  qc_user_accounts:
    id              BIGINT PK
    username        TEXT NOT NULL UNIQUE
    password_hash   TEXT NOT NULL
    role            TEXT NOT NULL (admin/operator)
    is_active       BOOLEAN DEFAULT TRUE
    created_at      TIMESTAMPTZ
    updated_at      TIMESTAMPTZ
    last_login_at   TIMESTAMPTZ
    rfid_uid_hash   TEXT (hashed RFID)

  qc_inspection_push:
    id              BIGINT PK
    partname        TEXT
    datecheckmc     TIMESTAMPTZ
    mpcheck         TEXT (OK/NG)
    data1           DOUBLE PRECISION (part ready confidence)
    data2           DOUBLE PRECISION (anomaly score)
"""

import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    logger.warning("psycopg2 not installed — PostgreSQL tidak tersedia")


# ── Constants ──────────────────────────────────────────────────────────

PASSWORD_PEPPER = "visioninspect_2024_"


# ── Helpers ────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    """Hash password dengan SHA-256 + pepper (sama dengan SQLite Database)."""
    return hashlib.sha256(f"{PASSWORD_PEPPER}{password}".encode()).hexdigest()


def _hash_rfid(rfid_uid: str) -> str:
    """Hash RFID UID untuk disimpan di qc_user_accounts.rfid_uid_hash."""
    return hashlib.sha256(f"{PASSWORD_PEPPER}{rfid_uid}".encode()).hexdigest()


def _now() -> str:
    """Return ISO 8601 timestamp string with timezone."""
    return datetime.now(timezone.utc).isoformat()


# ── PostgresDB Class ───────────────────────────────────────────────────


class PostgresError(Exception):
    """Base exception for PostgreSQL errors."""
    pass


class PostgresConnectionError(PostgresError):
    """Koneksi gagal."""
    pass


class PostgresDB:
    """
    Koneksi ke PostgreSQL untuk autentikasi dan push inspeksi.

    Menggunakan connection-per-call (tanpa pooling) karena eksekusi
    dari Qt event loop — reconnect otomatis tiap query.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Dict dengan key:
                enabled      - bool
                host         - str
                port         - int
                dbname       - str
                user         - str
                password     - str
                sslmode      - str
                connect_timeout - int
        """
        self._cfg = config
        self._enabled = config.get("enabled", False) and HAS_PSYCOPG2

        if self._enabled:
            logger.info(
                "PostgreSQL configured: %s@%s:%d/%s (enabled=%s)",
                config.get("user"), config.get("host"),
                config.get("port"), config.get("dbname"),
                self._enabled,
            )
        else:
            reason = "psycopg2 not installed" if not HAS_PSYCOPG2 else "disabled in config"
            logger.info("PostgreSQL %s — auth & inspection push will use SQLite fallback", reason)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ── Connection ───────────────────────────────────────────────────

    def _connect(self):
        """Create a new connection. Raises PostgresConnectionError on failure."""
        if not self._enabled:
            raise PostgresConnectionError("PostgreSQL not enabled")
        try:
            conn = psycopg2.connect(
                host=self._cfg.get("host", "localhost"),
                port=self._cfg.get("port", 5432),
                dbname=self._cfg.get("dbname", "visioninspect"),
                user=self._cfg.get("user", "postgres"),
                password=self._cfg.get("password", ""),
                sslmode=self._cfg.get("sslmode", "prefer"),
                connect_timeout=self._cfg.get("connect_timeout", 10),
            )
            conn.autocommit = False
            return conn
        except Exception as e:
            raise PostgresConnectionError(f"Koneksi PostgreSQL gagal: {e}")

    def _execute(self, query: str, params: tuple = None,
                 fetch: bool = False, fetch_one: bool = False,
                 returning: bool = False) -> Any:
        """
        Execute query with auto-connect + retry.

        Args:
            query: SQL query string
            params: Query parameters
            fetch: Return all rows as list[dict]
            fetch_one: Return single row as dict or None
            returning: Commit and return cursor.rowcount

        Returns:
            List[dict], dict, int, or None
        """
        if not self._enabled:
            return [] if fetch else None

        conn = None
        try:
            conn = self._connect()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params or ())

                if returning:
                    conn.commit()
                    return cur.rowcount

                if fetch:
                    rows = cur.fetchall()
                    conn.commit()
                    return [dict(r) for r in rows]

                if fetch_one:
                    row = cur.fetchone()
                    # WAJIB commit: INSERT ... RETURNING id memakai fetch_one
                    # (push_inspection, add_user). Tanpa ini baris di-rollback
                    # saat koneksi ditutup (autocommit=False) → DB tetap kosong.
                    conn.commit()
                    return dict(row) if row else None

                conn.commit()
                return cur.rowcount

        except PostgresConnectionError:
            raise
        except Exception as e:
            logger.warning("PostgreSQL query error: %s", e)
            raise PostgresError(f"Query gagal: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    # ── Readiness ────────────────────────────────────────────────────

    def ensure_ready(self) -> bool:
        """Pastikan DB siap pakai setelah terhubung.

        Verifikasi (dan buat bila belum ada) tabel yang dibutuhkan, lalu seed
        admin default bila tabel user kosong. Dipanggil setelah koneksi
        berhasil (startup & simpan settings) agar kegagalan push/login tidak
        terjadi diam-diam. Returns True bila DB siap.
        """
        if not self._enabled:
            return False
        try:
            # 1) Buat tabel bila belum ada (IF NOT EXISTS = no-op bila sudah ada,
            #    jadi skema produksi yang sudah ada tidak diubah).
            self._execute("""
                CREATE TABLE IF NOT EXISTS qc_user_accounts (
                    id BIGSERIAL PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'operator',
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ,
                    last_login_at TIMESTAMPTZ,
                    rfid_uid_hash TEXT,
                    rfid_uid_last4 TEXT,
                    rfid_bound_at TIMESTAMPTZ
                )""")
            self._execute("""
                CREATE TABLE IF NOT EXISTS qc_inspection_push (
                    id BIGSERIAL PRIMARY KEY,
                    partname TEXT,
                    datecheckmc TIMESTAMPTZ,
                    mpcheck TEXT,
                    data1 DOUBLE PRECISION,
                    data2 DOUBLE PRECISION,
                    line TEXT
                )""")

            # 2) Verifikasi tabel benar-benar ada
            missing = []
            for t in ("qc_user_accounts", "qc_inspection_push"):
                row = self._execute("SELECT to_regclass(%s) AS reg", (t,),
                                    fetch_one=True)
                if not row or not row.get("reg"):
                    missing.append(t)
            if missing:
                logger.error("PostgreSQL BELUM siap — tabel hilang: %s", missing)
                return False

            # 3) Seed admin default bila belum ada user (agar selalu bisa login)
            cnt = self._execute("SELECT COUNT(*) AS c FROM qc_user_accounts",
                                fetch_one=True)
            if cnt and int(cnt.get("c", 0)) == 0:
                now = _now()
                self._execute(
                    """INSERT INTO qc_user_accounts
                       (username, password_hash, role, is_active, created_at, updated_at)
                       VALUES (%s, %s, 'admin', TRUE, %s, %s)""",
                    ("admin", _hash_password("admin"), now, now))
                logger.info("Seed admin default (admin/admin) ke qc_user_accounts")

            logger.info("PostgreSQL SIAP: tabel qc_user_accounts & "
                        "qc_inspection_push OK")
            return True
        except Exception as e:
            logger.error("PostgreSQL ensure_ready gagal: %s", e)
            return False

    # ── Authentication ──────────────────────────────────────────────

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """
        Authenticate user against qc_user_accounts.

        Returns user dict (with keys: id, username, role, display_name=username)
        or None if credentials invalid / user not active.

        On success, updates last_login_at.
        """
        if not self._enabled:
            return None

        pw_hash = _hash_password(password)
        try:
            user = self._execute(
                """SELECT id, username, role, is_active, created_at
                   FROM qc_user_accounts
                   WHERE username = %s AND password_hash = %s""",
                (username, pw_hash),
                fetch_one=True,
            )
        except PostgresError as e:
            logger.error("Auth query error: %s", e)
            return None

        if not user:
            return None

        if not user.get("is_active"):
            logger.warning("Login ditolak: user '%s' tidak aktif", username)
            return None

        # Update last_login_at
        try:
            self._execute(
                "UPDATE qc_user_accounts SET last_login_at = %s, updated_at = %s WHERE id = %s",
                (_now(), _now(), user["id"]),
            )
        except PostgresError as e:
            logger.warning("Gagal update last_login_at: %s", e)

        return {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["username"],  # qc_user_accounts tidak punya display_name
            "role": user["role"],
        }

    def get_user_by_rfid(self, rfid_uid: str) -> Optional[dict]:
        """
        Look up user by RFID UID hash.

        Returns user dict or None.
        """
        if not self._enabled:
            return None

        rfid_hash = _hash_rfid(rfid_uid)
        try:
            user = self._execute(
                """SELECT id, username, role, is_active, created_at
                   FROM qc_user_accounts
                   WHERE rfid_uid_hash = %s""",
                (rfid_hash,),
                fetch_one=True,
            )
        except PostgresError as e:
            logger.error("RFID query error: %s", e)
            return None

        if not user:
            return None

        if not user.get("is_active"):
            logger.warning("RFID login ditolak: user '%s' tidak aktif", user["username"])
            return None

        # Update last_login_at
        try:
            self._execute(
                "UPDATE qc_user_accounts SET last_login_at = %s, updated_at = %s WHERE id = %s",
                (_now(), _now(), user["id"]),
            )
        except PostgresError as e:
            logger.warning("Gagal update last_login_at via RFID: %s", e)

        return {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["username"],
            "role": user["role"],
        }

    # ── User Management ─────────────────────────────────────────────

    def list_users(self) -> List[Dict[str, Any]]:
        """List all active users from qc_user_accounts."""
        if not self._enabled:
            return []

        try:
            rows = self._execute(
                """SELECT id, username, role, is_active,
                          created_at, updated_at, last_login_at,
                          rfid_uid_hash
                   FROM qc_user_accounts
                   ORDER BY id""",
                fetch=True,
            )
            # Hide rfid_uid_hash from UI, expose as boolean "has_rfid"
            result = []
            for r in rows:
                result.append({
                    "id": r["id"],
                    "username": r["username"],
                    "display_name": r["username"],
                    "role": r["role"],
                    "is_active": r.get("is_active", True),
                    "has_rfid": bool(r.get("rfid_uid_hash")),
                    'rfid_uid': 'Bound' if r.get('rfid_uid_hash') else '',
                    "created_at": str(r.get("created_at", "")),
                    "updated_at": str(r.get("updated_at", "")),
                    "last_login_at": str(r.get("last_login_at", "")),
                })
            return result
        except PostgresError as e:
            logger.error("List users error: %s", e)
            return []

    def add_user(self, username: str, password: str,
                 display_name: str = "", role: str = "operator") -> int:
        """Add a new user to qc_user_accounts. Returns user ID."""
        if not self._enabled:
            raise PostgresError("PostgreSQL not enabled")

        pw_hash = _hash_password(password)
        now = _now()
        try:
            row = self._execute(
                """INSERT INTO qc_user_accounts
                   (username, password_hash, role, is_active, created_at, updated_at)
                   VALUES (%s, %s, %s, TRUE, %s, %s)
                   RETURNING id""",
                (username, pw_hash, role.lower(), now, now),
                fetch_one=True,
            )
            if row:
                uid = row.get("id") or row.get("id", 0)
                logger.info("User added to PostgreSQL: %s (role=%s)", username, role)
                return int(uid)
            return 0
        except PostgresError as e:
            logger.error("Add user error: %s", e)
            raise PostgresError(f"Gagal menambah user: {e}")

    def update_user(self, user_id: int, display_name: str = None,
                    password: str = None, role: str = None,
                    is_active: bool = None) -> bool:
        """Update user fields. display_name is accepted but not stored (no column)."""
        if not self._enabled:
            return False

        fields = []
        values = []

        if password is not None:
            fields.append("password_hash = %s")
            values.append(_hash_password(password))
        if role is not None:
            fields.append("role = %s")
            values.append(role.lower())
        if is_active is not None:
            fields.append("is_active = %s")
            values.append(is_active)
        if not fields:
            return False

        fields.append("updated_at = %s")
        values.append(_now())
        values.append(user_id)

        try:
            self._execute(
                f"UPDATE qc_user_accounts SET {', '.join(fields)} WHERE id = %s",
                tuple(values),
            )
            return True
        except PostgresError as e:
            logger.error("Update user error: %s", e)
            return False

    def delete_user(self, user_id: int) -> bool:
        """
        Delete a user from qc_user_accounts.
        Prevents deleting the last admin.
        """
        if not self._enabled:
            return False

        try:
            # Check if this is the last admin
            row = self._execute(
                "SELECT role FROM qc_user_accounts WHERE id = %s",
                (user_id,),
                fetch_one=True,
            )
            if not row:
                return False

            if row["role"] == "admin":
                admin_count = self._execute(
                    "SELECT COUNT(*) AS cnt FROM qc_user_accounts WHERE role = 'admin'",
                    fetch_one=True,
                )
                if admin_count and admin_count.get("cnt", 0) <= 1:
                    logger.warning("Tidak bisa menghapus admin terakhir")
                    return False

            self._execute("DELETE FROM qc_user_accounts WHERE id = %s", (user_id,))
            logger.info("User deleted from PostgreSQL: id=%d", user_id)
            return True
        except PostgresError as e:
            logger.error("Delete user error: %s", e)
            return False

    def bind_rfid(self, user_id: int, rfid_uid: str) -> bool:
        """Bind RFID UID hash to a user."""
        if not self._enabled:
            return False

        rfid_hash = _hash_rfid(rfid_uid)
        now = _now()
        try:
            # Check if hash already used
            existing = self._execute(
                "SELECT id FROM qc_user_accounts WHERE rfid_uid_hash = %s",
                (rfid_hash,),
                fetch_one=True,
            )
            if existing:
                logger.warning("RFID hash already bound to user id=%d", existing["id"])
                return False

            self._execute(
                "UPDATE qc_user_accounts SET rfid_uid_hash = %s, updated_at = %s WHERE id = %s",
                (rfid_hash, now, user_id),
            )
            logger.info("RFID bound to user id=%d (hash=%s...)", user_id, rfid_hash[:12])
            return True
        except PostgresError as e:
            logger.error("Bind RFID error: %s", e)
            return False

    def unbind_rfid(self, user_id: int) -> bool:
        """Remove RFID binding from a user."""
        if not self._enabled:
            return False

        now = _now()
        try:
            self._execute(
                "UPDATE qc_user_accounts SET rfid_uid_hash = NULL, updated_at = %s WHERE id = %s",
                (now, user_id),
            )
            return True
        except PostgresError as e:
            logger.error("Unbind RFID error: %s", e)
            return False

    # ── Inspection Push ─────────────────────────────────────────────

    def push_inspection(self, partname: str, mpcheck: str,
                        data1: float = 0.0, data2: float = 0.0) -> Optional[int]:
        """
        Push inspection result to qc_inspection_push.

        Args:
            partname: Nama part/program/template
            mpcheck: "OK" or "NG"
            data1: Part ready confidence (0.0 = not ready, 1.0 = ready)
                   — dari conf part ready / part check
            data2: Anomaly score — dari conf OK/NG inference

        Returns:
            Inserted row ID, or None on failure.
        """
        if not self._enabled:
            return None

        now = _now()
        try:
            row = self._execute(
                """INSERT INTO qc_inspection_push
                   (partname, datecheckmc, mpcheck, data1, data2)
                   VALUES (%s, %s, %s, %s, %s)
                   RETURNING id""",
                (partname, now, mpcheck, float(data1), float(data2)),
                fetch_one=True,
            )
            if row:
                rid = row.get("id") or row.get("id", 0)
                logger.debug("Inspection pushed: id=%s part=%s mpcheck=%s data1=%.3f data2=%.3f",
                             rid, partname, mpcheck, data1, data2)
                return int(rid)
            return None
        except PostgresError as e:
            logger.warning("Push inspection error: %s", e)
            return None

    def get_history(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get inspection history from qc_inspection_push.
        Returns list of dicts (for HistoryPage display).
        """
        if not self._enabled:
            return []

        try:
            rows = self._execute(
                """SELECT id, partname, datecheckmc, mpcheck, data1, data2
                   FROM qc_inspection_push
                   ORDER BY datecheckmc DESC
                   LIMIT %s OFFSET %s""",
                (limit, offset),
                fetch=True,
            )
            result = []
            for r in rows:
                result.append({
                    "id": r["id"],
                    "timestamp": str(r.get("datecheckmc", "")),
                    "program": r.get("partname", ""),
                    "score": r.get("data2", 0.0),      # data2 = anomaly score
                    "judgement": r.get("mpcheck", ""),
                    "data1": r.get("data1", 0.0),       # part ready confidence
                    "image_path": "",
                    "corrected": False,
                })
            return result
        except PostgresError as e:
            logger.warning("Get history error: %s", e)
            return []

    def get_history_count(self) -> int:
        """Get total count of inspection records."""
        if not self._enabled:
            return 0

        try:
            row = self._execute(
                "SELECT COUNT(*) AS cnt FROM qc_inspection_push",
                fetch_one=True,
            )
            return row.get("cnt", 0) if row else 0
        except PostgresError as e:
            logger.warning("Get history count error: %s", e)
            return 0
