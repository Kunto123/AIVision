"""
ExternalDB - fasad DB eksternal (dikonfigurasi db.txt).

Menyediakan surface yang sama dengan store user lain: authenticate,
get_user_by_rfid, list_users, add_user, update_user, delete_user, bind/unbind.
Plus push_row() untuk hasil inspeksi dan check() untuk validasi startup.
"""

from typing import List, Optional

from visioninspect.storage import credentials, db_txt
from visioninspect.storage.db_adapters import get_adapter
from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")


class CheckReport:
    def __init__(self):
        self.lines: List[str] = []
        self.ok = True

    def info(self, msg):
        self.lines.append(f"[ ok ] {msg}")

    def warn(self, msg):
        self.lines.append(f"[warn] {msg}")

    def error(self, msg):
        self.lines.append(f"[ERR ] {msg}")
        self.ok = False


class ExternalDB:
    """DB eksternal siap pakai. `is_enabled` selalu True bila objek ini dibuat."""

    def __init__(self, settings: db_txt.DbSettings):
        self.s = settings
        self._adapter = get_adapter(settings)

    @property
    def is_enabled(self) -> bool:
        return True

    def test_connection(self):
        return self._adapter.test_connection()

    def is_alive(self, timeout=None) -> bool:
        ok, _ = self._adapter.test_connection()
        return ok

    # ── Auth ──────────────────────────────────────────────────────────

    def _connect(self):
        """Dipakai LoginDialog untuk membedakan 'DB mati' vs 'password salah'."""
        ok, msg = self._adapter.test_connection()
        if not ok:
            raise RuntimeError(msg)

        class _Dummy:
            def close(self):
                pass

        return _Dummy()

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        if not self.s.user_table:
            return None
        try:
            user = self._adapter.authenticate(
                self.s.user_table, username, credentials.hash_password(password))
        except Exception as e:
            logger.warning("Auth DB eksternal gagal: %s", e)
            return None
        if user:
            self._adapter.touch_last_login(self.s.user_table, user["id"])
        return user

    def get_user_by_rfid(self, rfid_uid: str) -> Optional[dict]:
        if not self.s.user_table:
            return None
        try:
            user = self._adapter.get_user_by_rfid(
                self.s.user_table, credentials.hash_rfid(rfid_uid))
        except Exception as e:
            logger.warning("RFID DB eksternal gagal: %s", e)
            return None
        if user:
            self._adapter.touch_last_login(self.s.user_table, user["id"])
        return user

    # ── Manajemen user (dipakai CLI; account page pakai store lokal) ──

    def list_users(self) -> List[dict]:
        if not self.s.user_table:
            return []
        try:
            return self._adapter.list_users(self.s.user_table)
        except Exception as e:
            logger.warning("list_users DB eksternal gagal: %s", e)
            return []

    def add_user(self, username: str, password: str, display_name: str = "",
                 role: str = "operator") -> None:
        self._adapter.add_user(self.s.user_table, username,
                               credentials.hash_password(password), role)

    def set_password(self, username: str, password: str) -> int:
        return self._adapter.set_password(
            self.s.user_table, username, credentials.hash_password(password))

    def delete_user(self, username: str) -> int:
        return self._adapter.delete_user(self.s.user_table, username)

    def ensure_user_table(self):
        """Buat tabel user bila belum ada. (dibuat_atau_ada, ddl)."""
        if not self.s.user_table:
            return False, ""
        return self._adapter.ensure_user_table(self.s.user_table)

    # ── Push inspeksi ────────────────────────────────────────────────

    def push_row(self, snapshot: dict) -> bool:
        """Resolve mapping db.txt -> INSERT dinamis. True bila 1 baris masuk."""
        if not (self.s.inspection_table and self.s.mapping):
            return False
        cols, vals, server_time = db_txt.resolve_row(self.s.mapping, snapshot)
        res = self._adapter.insert(self.s.inspection_table, cols, vals, server_time)
        return bool(res)

    # ── Validasi ────────────────────────────────────────────────────

    def check(self) -> CheckReport:
        r = CheckReport()
        s = self.s
        r.lines.append(
            f"engine={s.engine} host={s.host}:{s.port} db={s.name or '(file)'}")
        ok, msg = self._adapter.test_connection()
        if not ok:
            r.error(f"koneksi gagal: {msg}")
            return r
        r.info("koneksi OK")

        # Tabel inspeksi + mapping
        if s.mapping:
            try:
                cols = {c.name: c for c in
                        self._adapter.describe_table(s.inspection_table)}
            except Exception as e:
                r.error(f"describe_table('{s.inspection_table}') gagal: {e}")
                cols = {}
            if not cols:
                r.error(f"tabel inspeksi '{s.inspection_table}' tidak ditemukan / kosong")
            else:
                r.info(f"tabel '{s.inspection_table}' ada ({len(cols)} kolom)")
                mapped = set(s.mapping.keys())
                for col in s.mapping:
                    if col not in cols:
                        r.error(f"MAP kolom '{col}' tidak ada di tabel")
                    else:
                        src = s.mapping[col]
                        r.info(f"  {col} <- {src}")
                for name, c in cols.items():
                    if (not c.nullable and not c.has_default
                            and not c.is_pk and name not in mapped):
                        r.error(f"kolom '{name}' NOT NULL tanpa default & tidak di-MAP")
        else:
            r.warn("tidak ada MAP_* — push inspeksi TIDAK aktif (tak ada kolom dipetakan)")

        # Tabel user
        if s.user_table:
            created, ddl = self._adapter.ensure_user_table(s.user_table)
            if created:
                r.info(f"tabel user '{s.user_table}' siap")
            else:
                r.warn(f"tabel user '{s.user_table}' harus dibuat manual:\n{ddl}")
        else:
            r.warn("DB_USER_TABLE kosong — login hanya pakai akun lokal")
        return r
