"""Adapter PostgreSQL (psycopg2)."""

from typing import List

from visioninspect.storage.db_adapters.base import BaseAdapter, Column


class PostgresAdapter(BaseAdapter):
    engine = "postgresql"
    ph = "%s"
    driver_hint = "psycopg2-binary"

    def _connect(self):
        import psycopg2
        kw = dict(host=self.s.host, port=self.s.port, dbname=self.s.name,
                  user=self.s.user, password=self.s.password,
                  connect_timeout=self.s.connect_timeout)
        # DB_EXTRA diteruskan apa adanya (sslmode, options, dst.)
        for k, v in self.s.extra.items():
            kw[k] = v
        conn = psycopg2.connect(**kw)
        conn.autocommit = False
        return conn

    def q(self, ident: str) -> str:
        return '"' + ident.replace('"', '""') + '"'

    def server_now(self) -> str:
        return "CURRENT_TIMESTAMP"

    def list_tables(self) -> List[str]:
        rows = self._run(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema NOT IN ('pg_catalog','information_schema') "
            "ORDER BY table_name", fetch="all") or []
        return [r[0] for r in rows]

    def describe_table(self, name: str) -> List[Column]:
        rows = self._run(
            "SELECT column_name, data_type, is_nullable, column_default, is_identity "
            "FROM information_schema.columns WHERE table_name = %s "
            "ORDER BY ordinal_position", (name,), fetch="all") or []
        out = []
        for cn, dt, nn, dflt, ident in rows:
            has_def = dflt is not None or str(ident).upper() == "YES"
            out.append(Column(cn, dt, str(nn).upper() == "YES", has_def,
                              is_pk=False))
        return out

    def user_table_ddl(self, name: str) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self.q(name)} (\n"
            "    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,\n"
            "    username      TEXT NOT NULL UNIQUE,\n"
            "    password_hash TEXT NOT NULL,\n"
            "    role          TEXT NOT NULL DEFAULT 'operator',\n"
            "    rfid          TEXT UNIQUE,\n"
            "    last_login    TIMESTAMPTZ\n"
            ")")
