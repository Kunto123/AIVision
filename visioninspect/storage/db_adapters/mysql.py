"""Adapter MySQL / MariaDB (pymysql)."""

from typing import List

from visioninspect.storage.db_adapters.base import BaseAdapter, Column


class MySQLAdapter(BaseAdapter):
    engine = "mysql"
    ph = "%s"
    driver_hint = "pymysql"

    def _connect(self):
        import pymysql
        kw = dict(host=self.s.host, port=self.s.port, database=self.s.name,
                  user=self.s.user, password=self.s.password,
                  connect_timeout=self.s.connect_timeout, autocommit=False)
        for k, v in self.s.extra.items():
            kw[k] = v
        return pymysql.connect(**kw)

    def q(self, ident: str) -> str:
        return "`" + ident.replace("`", "``") + "`"

    def server_now(self) -> str:
        return "NOW(6)"

    def list_tables(self) -> List[str]:
        rows = self._run(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s ORDER BY table_name",
            (self.s.name,), fetch="all") or []
        return [r[0] for r in rows]

    def describe_table(self, name: str) -> List[Column]:
        rows = self._run(
            "SELECT column_name, data_type, is_nullable, column_default, "
            "extra, column_key FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (self.s.name, name), fetch="all") or []
        out = []
        for cn, dt, nn, dflt, extra, ckey in rows:
            has_def = dflt is not None or "auto_increment" in (extra or "").lower()
            out.append(Column(cn, dt, str(nn).upper() == "YES", has_def,
                              is_pk=(ckey == "PRI")))
        return out

    def user_table_ddl(self, name: str) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self.q(name)} (\n"
            "    id            BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,\n"
            "    username      VARCHAR(64) NOT NULL UNIQUE,\n"
            "    password_hash VARCHAR(64) NOT NULL,\n"
            "    role          VARCHAR(16) NOT NULL DEFAULT 'operator',\n"
            "    rfid          VARCHAR(64) NULL UNIQUE,\n"
            "    last_login    DATETIME(6) NULL\n"
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4")
