"""
Kontrak adapter DB. Connection-per-call (tanpa pooling), sama gaya postgres_db.py.
Subclass override: _connect, ph, q(), server_now(), list_tables(), describe_table(),
user_table_ddl().
"""

from typing import Dict, List, Optional

from visioninspect.storage.db_txt import DbSettings, USER_TABLE_COLUMNS
from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")


class Column:
    """Satu kolom hasil describe_table."""

    __slots__ = ("name", "type", "nullable", "has_default", "is_pk")

    def __init__(self, name, type="", nullable=True, has_default=False, is_pk=False):
        self.name = name
        self.type = type
        self.nullable = nullable
        self.has_default = has_default
        self.is_pk = is_pk

    def __repr__(self):
        return f"Column({self.name} null={self.nullable} default={self.has_default} pk={self.is_pk})"


class BaseAdapter:
    engine = ""
    ph = "%s"                     # placeholder parameter
    driver_hint = ""              # nama paket pip untuk pesan error

    def __init__(self, settings: DbSettings):
        self.s = settings

    # ── di-override per engine ──────────────────────────────────────────

    def _connect(self):
        """Buka koneksi DBAPI baru (raise bila gagal)."""
        raise NotImplementedError

    def q(self, ident: str) -> str:
        """Quote identifier (nama tabel/kolom)."""
        return '"' + ident.replace('"', '""') + '"'

    def server_now(self) -> str:
        """Ekspresi SQL waktu server untuk kolom @server_now."""
        return "CURRENT_TIMESTAMP"

    def list_tables(self) -> List[str]:
        raise NotImplementedError

    def describe_table(self, name: str) -> List[Column]:
        raise NotImplementedError

    def user_table_ddl(self, name: str) -> str:
        raise NotImplementedError

    # ── util eksekusi ──────────────────────────────────────────────────

    def _run(self, sql: str, params=(), fetch: Optional[str] = None):
        """fetch: None | 'one' | 'all' | 'affected' (jumlah baris; 1 bila driver
        tak melaporkan). JANGAN pakai cursor.lastrowid — psycopg2 mengembalikan
        0 (OID) untuk tabel biasa, itu bikin INSERT sukses dianggap gagal."""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            if fetch == "one":
                row = cur.fetchone()
                conn.commit()
                return row
            if fetch == "all":
                rows = cur.fetchall()
                conn.commit()
                return rows
            conn.commit()
            rc = cur.rowcount
            if fetch == "affected":
                # -1/None = driver tak melapor (mis. pyodbc NOCOUNT) → anggap 1
                return 1 if rc in (-1, None) else rc
            return rc
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ── kontrak publik ────────────────────────────────────────────────

    def test_connection(self):
        """(ok: bool, pesan: str)."""
        try:
            self._run("SELECT 1", fetch="one")
            return True, "OK"
        except Exception as e:
            return False, str(e).strip()

    def insert(self, table: str, cols: List[str], vals: List[object],
               server_time_cols: List[str]) -> int:
        """INSERT dinamis; kolom di-quote per engine, nilai parameterized.
        Return jumlah baris masuk (1 = sukses)."""
        all_cols = list(cols) + list(server_time_cols)
        if not all_cols:
            return 0
        placeholders = ([self.ph] * len(cols)
                        + [self.server_now()] * len(server_time_cols))
        sql = (f"INSERT INTO {self.q(table)} "
               f"({', '.join(self.q(c) for c in all_cols)}) "
               f"VALUES ({', '.join(placeholders)})")
        return self._run(sql, tuple(vals), fetch="affected")

    def _select_user(self, table: str, where_col: str, value) -> Optional[dict]:
        sql = (f"SELECT {', '.join(self.q(c) for c in USER_TABLE_COLUMNS)} "
               f"FROM {self.q(table)} WHERE {self.q(where_col)} = {self.ph}")
        row = self._run(sql, (value,), fetch="one")
        if not row:
            return None
        return {
            "id": row[0],
            "username": row[1],
            "password_hash": row[2],
            "role": (row[3] or "operator"),
            "rfid": row[4],
            "last_login": row[5],
            "display_name": row[1],
        }

    def authenticate(self, table: str, username: str, pw_hash: str) -> Optional[dict]:
        sql = (f"SELECT {', '.join(self.q(c) for c in USER_TABLE_COLUMNS)} "
               f"FROM {self.q(table)} WHERE {self.q('username')} = {self.ph} "
               f"AND {self.q('password_hash')} = {self.ph}")
        row = self._run(sql, (username, pw_hash), fetch="one")
        if not row:
            return None
        return {"id": row[0], "username": row[1], "role": (row[3] or "operator"),
                "display_name": row[1], "last_login": row[5]}

    def get_user_by_rfid(self, table: str, rfid_hash: str) -> Optional[dict]:
        u = self._select_user(table, "rfid", rfid_hash)
        if u:
            u.pop("password_hash", None)
        return u

    def touch_last_login(self, table: str, user_id) -> None:
        sql = (f"UPDATE {self.q(table)} SET {self.q('last_login')} = {self.server_now()} "
               f"WHERE {self.q('id')} = {self.ph}")
        try:
            self._run(sql, (user_id,))
        except Exception as e:
            logger.warning("touch_last_login gagal: %s", e)

    def ensure_user_table(self, name: str):
        """(dibuat_atau_ada: bool, ddl: str). False = buat manual pakai ddl."""
        ddl = self.user_table_ddl(name)
        try:
            self._run(ddl)
            return True, ddl
        except Exception as e:
            logger.warning("ensure_user_table('%s') gagal (%s) — buat manual", name, e)
            return False, ddl

    def list_users(self, table: str) -> List[dict]:
        sql = (f"SELECT {', '.join(self.q(c) for c in USER_TABLE_COLUMNS)} "
               f"FROM {self.q(table)} ORDER BY {self.q('id')}")
        rows = self._run(sql, fetch="all") or []
        return [{"id": r[0], "username": r[1], "display_name": r[1],
                 "role": (r[3] or "operator"),
                 "rfid_uid": "Bound" if r[4] else "",
                 "rfid_bound_at": "", "created_at": "",
                 "last_login": r[5]} for r in rows]

    def add_user(self, table: str, username: str, pw_hash: str,
                 role: str = "operator", rfid_hash: Optional[str] = None) -> None:
        cols = ["username", "password_hash", "role"]
        vals = [username, pw_hash, role]
        if rfid_hash:
            cols.append("rfid")
            vals.append(rfid_hash)
        sql = (f"INSERT INTO {self.q(table)} "
               f"({', '.join(self.q(c) for c in cols)}) "
               f"VALUES ({', '.join([self.ph] * len(cols))})")
        self._run(sql, tuple(vals))

    def set_password(self, table: str, username: str, pw_hash: str) -> int:
        sql = (f"UPDATE {self.q(table)} SET {self.q('password_hash')} = {self.ph} "
               f"WHERE {self.q('username')} = {self.ph}")
        return self._run(sql, (pw_hash, username))

    def delete_user(self, table: str, username: str) -> int:
        sql = f"DELETE FROM {self.q(table)} WHERE {self.q('username')} = {self.ph}"
        return self._run(sql, (username,))
