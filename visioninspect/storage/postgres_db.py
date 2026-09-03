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

  qc_inspection_push  (SKEMA ASLI — lima kolom, jangan ditambah):
    id              BIGINT PK
    partname        TEXT (nama part = nama template aktif)
    datecheckmc     TIMESTAMPTZ (waktu INSPEKSI, bukan waktu insert)
    mpcheck         TEXT (MP/ManPower = nama akun operator yang login)
    data1           DOUBLE PRECISION (skor part-check; 0 bila tidak aktif)
    data2           DOUBLE PRECISION (skor ROI penentu)

  Isi tabel ini HANYA hasil inspeksi OK dari operator view. NG tidak pernah
  dikirim — bukti cacat, gambar, threshold, latensi, dan koreksi operator
  semuanya tersimpan di SQLite lokal.
"""

import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from visioninspect.storage import secret_store
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
        # C4: password non-plaintext — decrypt token "enc:v1:" (DPAPI/Fernet).
        # Token plaintext lama di-pass-through (migrasi saat save settings).
        self._password = secret_store.decrypt(config.get("password", ""))

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

    def _connect(self, timeout: Optional[float] = None):
        """Create a new connection. Raises PostgresConnectionError on failure.

        Args:
            timeout: connect_timeout override (detik). None = pakai config.
        """
        if not self._enabled:
            raise PostgresConnectionError("PostgreSQL not enabled")
        try:
            conn = psycopg2.connect(
                host=self._cfg.get("host", "localhost"),
                port=self._cfg.get("port", 5432),
                dbname=self._cfg.get("dbname", "visioninspect"),
                user=self._cfg.get("user", "postgres"),
                password=self._password,
                sslmode=self._cfg.get("sslmode", "prefer"),
                connect_timeout=(timeout if timeout is not None
                                 else self._cfg.get("connect_timeout", 10)),
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
            # Skema ASLI tabel produksi — lima kolom, tidak lebih. Aplikasi
            # ini pernah menambahkan 9 kolom lain (line/operator/image_path/
            # threshold/latency_ms/local_id/corrected*); semuanya dicabut
            # kembali, lihat _drop_legacy_push_columns().
            self._execute("""
                CREATE TABLE IF NOT EXISTS qc_inspection_push (
                    id BIGSERIAL PRIMARY KEY,
                    partname TEXT,
                    datecheckmc TIMESTAMPTZ,
                    mpcheck TEXT,
                    data1 DOUBLE PRECISION,
                    data2 DOUBLE PRECISION
                )""")

            # C4: kolom must_change_password untuk akun seed (paksa ganti).
            # Dijalankan SEBELUM pembersihan kolom lama: tanpa kolom ini login
            # gagal total, jadi ia tidak boleh bergantung pada langkah lain
            # yang sifatnya hanya kebersihan skema.
            try:
                self._execute(
                    "ALTER TABLE qc_user_accounts ADD COLUMN IF NOT EXISTS "
                    "must_change_password BOOLEAN NOT NULL DEFAULT FALSE")
            except PostgresError as e:
                logger.warning("Migrasi kolom must_change_password gagal: %s", e)

            self._drop_legacy_push_columns()

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
                       (username, password_hash, role, is_active,
                        must_change_password, created_at, updated_at)
                       VALUES (%s, %s, 'admin', TRUE, TRUE, %s, %s)""",
                    ("admin", _hash_password("admin"), now, now))
                logger.info("Seed admin default ke qc_user_accounts "
                            "(admin/admin — WAJIB ganti password saat login pertama)")

            logger.info("PostgreSQL SIAP: tabel qc_user_accounts & "
                        "qc_inspection_push OK")
            return True
        except Exception as e:
            logger.error("PostgreSQL ensure_ready gagal: %s", e)
            return False

    # ── Authentication ──────────────────────────────────────────────

    def sync_users_from_sqlite(self, sqlite_db) -> int:
        """[DEPRECATED — 2026-08-07] Sinkronkan user SQLite → PG (upsert one-way).

        TIDAK DIPANGGIL LAGI sejak PG dijadikan satu-satunya sumber akun
        (main_window.py). Method ini menimpa role/password qc_user_accounts
        dengan isi SQLite tiap startup, sehingga akun yang dibuat/diedit di
        pgAdmin4 selalu dikembalikan ke state SQLite. Dipertahankan hanya
        sebagai utilitas migrasi manual bila suatu saat diperlukan.

        C2: SQLite & PG adalah dua auth source terpisah. User (dengan
        password custom) hidup di SQLite; PG hanya punya seed admin/admin.
        Begitu PG "hidup" (lihat fix is_alive 2026-08-07), login beralih ke
        PG dan user SQLite tak ada di sana → login gagal. Pepper hash sama
        (``visioninspect_2024_``) sehingga password_hash bisa disalin
        langsung. Dipanggil sekali saat startup setelah ``ensure_ready``.

        Returns jumlah user yang di-upsert (0 bila disabled/gagal).
        """
        if not self._enabled:
            return 0
        try:
            users = sqlite_db.list_users_full()
            if not users:
                return 0
            now = _now()
            n = 0
            for u in users:
                # ON CONFLICT (username) butuh UNIQUE constraint — tabel lama
                # hasil CREATE IF NOT EXISTS bisa tidak punya → pakai
                # SELECT→INSERT/UPDATE manual (robust terhadap skema apa pun).
                exists = self._execute(
                    "SELECT id FROM qc_user_accounts WHERE username = %s",
                    (u["username"],), fetch_one=True)
                if exists:
                    self._execute(
                        """UPDATE qc_user_accounts
                           SET password_hash = %s, role = %s,
                               must_change_password = %s, updated_at = %s
                           WHERE username = %s""",
                        (u["password_hash"], u["role"],
                         bool(u.get("must_change_password", False)),
                         now, u["username"]))
                else:
                    self._execute(
                        """INSERT INTO qc_user_accounts
                           (username, password_hash, role, is_active,
                            must_change_password, created_at, updated_at)
                           VALUES (%s, %s, %s, TRUE, %s, %s, %s)""",
                        (u["username"], u["password_hash"], u["role"],
                         bool(u.get("must_change_password", False)),
                         now, now))
                n += 1
            logger.info("Sinkronisasi user SQLite → PG: %d user", n)
            return n
        except Exception as e:
            logger.warning("Sinkronisasi user ke PG gagal: %s", e)
            return 0

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
                """SELECT id, username, role, is_active, must_change_password, created_at
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
            # C4: password baru di-set → flag paksa-ganti dimatikan
            fields.append("must_change_password = FALSE")
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

    #: Kolom yang DULU ditambahkan aplikasi ini ke tabel produksi milik
    #: perusahaan. Dicabut agar tabel kembali ke bentuk aslinya.
    _LEGACY_PUSH_COLUMNS = (
        "line", "operator", "image_path", "threshold", "latency_ms",
        "local_id", "corrected", "correct_judgement", "corrected_at",
    )

    def _drop_legacy_push_columns(self) -> None:
        """Hapus permanen kolom yang dulu ditambahkan aplikasi ini.

        PERMANEN: data di kolom-kolom itu ikut hilang dan tidak bisa
        dikembalikan tanpa backup. Dijalankan sekali — `DROP COLUMN IF EXISTS`
        bersifat idempoten, jadi startup berikutnya tidak melakukan apa-apa.

        Semua informasi itu TETAP ADA di SQLite lokal (image_path, threshold,
        latency_ms, verdict, koreksi, skor per ROI), jadi yang hilang hanya
        salinannya di PostgreSQL.
        """
        # Except LEBAR (bukan hanya PostgresError): langkah ini semata-mata
        # kebersihan skema. Kalau ia gagal karena sebab apa pun, sisa
        # ensure_ready() — termasuk migrasi kolom yang dibutuhkan LOGIN —
        # tetap harus jalan. (Pernah terjadi: TypeError di sini membuat
        # `must_change_password` tidak pernah dibuat dan login mati total.)
        try:
            existing = self._execute(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_name = 'qc_inspection_push'""",
                fetch=True) or []
            have = {r.get("column_name") for r in existing}

            to_drop = [c for c in self._LEGACY_PUSH_COLUMNS if c in have]
            if not to_drop:
                return
            logger.warning(
                "Menghapus PERMANEN %d kolom tambahan dari qc_inspection_push: "
                "%s — data di kolom ini hilang. Tabel kembali ke skema asli "
                "(partname, datecheckmc, mpcheck, data1, data2).",
                len(to_drop), ", ".join(to_drop))
            for col in to_drop:
                try:
                    self._execute(
                        "ALTER TABLE qc_inspection_push "
                        f"DROP COLUMN IF EXISTS {col}")
                except Exception as e:
                    logger.error("Gagal menghapus kolom %s: %s", col, e)
        except Exception as e:
            logger.warning(
                "Pembersihan kolom lama qc_inspection_push dilewati: %s", e)

    # ── Inspection Push ─────────────────────────────────────────────

    def push_inspection(self, partname: str, mpcheck: str,
                        data1: float = 0.0, data2: float = 0.0,
                        datecheckmc: Optional[str] = None) -> Optional[int]:
        """Push hasil inspeksi OK ke qc_inspection_push (skema asli, 5 kolom).

        Args:
            partname: Nama part (dari nama template aktif)
            mpcheck: MP (ManPower) yang memeriksa — NAMA AKUN OPERATOR yang
                login di operator view. Bukan verdict OK/NG: tabel ini hanya
                menerima hasil OK, jadi verdict-nya tersirat.
            data1: Skor part-check (0 bila part-check tidak aktif)
            data2: Skor inspeksi ROI penentu
            datecheckmc: Waktu INSPEKSI (bukan waktu insert). Wajib diisi
                pemanggil — kalau outbox tertahan karena PG mati, memakai
                jam insert akan menggeser seluruh baris tertunda ke waktu
                koneksi pulih. None hanya sebagai jaring pengaman.

        Returns:
            Inserted row ID, atau None on failure.
        """
        if not self._enabled:
            return None

        when = datecheckmc or _now()
        try:
            row = self._execute(
                """INSERT INTO qc_inspection_push
                   (partname, datecheckmc, mpcheck, data1, data2)
                   VALUES (%s, %s, %s, %s, %s)
                   RETURNING id""",
                (partname, when, mpcheck, float(data1), float(data2)),
                fetch_one=True,
            )
            if row:
                rid = row.get("id") or row.get("id", 0)
                logger.debug("Inspection pushed: id=%s part=%s mp=%s "
                             "data1=%.3f data2=%.3f",
                             rid, partname, mpcheck, data1, data2)
                return int(rid)
            return None
        except PostgresError as e:
            logger.warning("Push inspection error: %s", e)
            return None

    # CATATAN: propagasi koreksi ke PostgreSQL (mark_correction_pg /
    # rollback_correction_pg) DIHAPUS. Kolom penopangnya (local_id, corrected,
    # correct_judgement, corrected_at) sudah tidak ada, dan tabel ini hanya
    # menerima hasil OK — sementara koreksi hampir selalu menyangkut NG.
    # Koreksi operator tetap tercatat lengkap di SQLite lokal.

    # ── Liveness (C2) ──────────────────────────────────────────────

    def is_alive(self, timeout: Optional[float] = None) -> bool:
        """Cek apakah PostgreSQL benar-benar terjangkau (C2).

        ``is_enabled`` hanya membaca flag config — server bisa saja mati.
        Query ringan SELECT 1 dengan connect_timeout singkat; dipakai untuk
        login fallback ke SQLite lokal dan indikator sink di DIAGNOSTICS.

        Catatan (fix 2026-08-07): JANGAN panggil dengan timeout kecil
        (mis. 2.0) di host ``localhost``/Windows — resolve IPv6 ``::1``
        dulu bisa makan budget, lalu libpq fallback ke 127.0.0.1; kalau
        timeout habis, is_alive false-negative padahal PG hidup (inisialisasi
        yang pakai connect_timeout config = 10s tetap sukses). None = pakai
        config ``connect_timeout`` supaya konsisten dengan jalur init.
        """
        if not self._enabled:
            return False
        try:
            conn = self._connect(timeout=timeout)
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                return True
            finally:
                conn.close()
        except Exception:
            return False

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
