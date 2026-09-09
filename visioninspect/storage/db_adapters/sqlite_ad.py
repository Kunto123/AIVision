"""Adapter SQLite (stdlib sqlite3) — DB_NAME = path file .db."""

from pathlib import Path
from typing import List

from visioninspect.storage.db_adapters.base import BaseAdapter, Column


class SQLiteAdapter(BaseAdapter):
    """Adapter SQLite (stdlib sqlite3) — DB_NAME = path file .db."""

    engine = "sqlite"
    ph = "?"
    driver_hint = "sqlite3 (stdlib)"

    def _connect(self):
        import sqlite3
        p = Path(self.s.name).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), timeout=float(self.s.connect_timeout or 10))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def q(self, ident: str) -> str:
        """Quote identifier gaya SQLite (double-quote)."""
        return '"' + ident.replace('"', '""') + '"'

    def server_now(self) -> str:
        """Ekspresi waktu server SQLite (UTC, presisi detik)."""
        return "CURRENT_TIMESTAMP"

    def list_tables(self) -> List[str]:
        """Nama tabel dari sqlite_master."""
        rows = self._run(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name",
            fetch="all") or []
        return [r[0] for r in rows]

    def describe_table(self, name: str) -> List[Column]:
        """Kolom tabel via PRAGMA table_info."""
        rows = self._run(f"PRAGMA table_info({self.q(name)})", fetch="all") or []
        # (cid, name, type, notnull, dflt_value, pk)
        return [Column(r[1], r[2], not bool(r[3]), r[4] is not None or bool(r[5]),
                       is_pk=bool(r[5])) for r in rows]

    def user_table_ddl(self, name: str) -> str:
        """DDL tabel user standar (INTEGER AUTOINCREMENT PK)."""
        return (
            f"CREATE TABLE IF NOT EXISTS {self.q(name)} (\n"
            "    id            INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "    username      TEXT NOT NULL UNIQUE,\n"
            "    password_hash TEXT NOT NULL,\n"
            "    role          TEXT NOT NULL DEFAULT 'operator',\n"
            "    rfid          TEXT UNIQUE,\n"
            "    last_login    TEXT\n"
            ")")
