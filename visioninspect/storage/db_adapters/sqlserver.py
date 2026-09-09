"""
Adapter Microsoft SQL Server (pyodbc + ODBC Driver 18).
Butuh driver ODBC di-install manual di PC edge (di luar pip).
DB_EXTRA=TrustServerCertificate=yes bila server tanpa sertifikat TLS tepercaya.
"""

from typing import List

from visioninspect.storage.db_adapters.base import BaseAdapter, Column


class SQLServerAdapter(BaseAdapter):
    """Adapter Microsoft SQL Server (pyodbc + ODBC Driver 18)."""

    engine = "sqlserver"
    ph = "?"
    driver_hint = "pyodbc (+ Microsoft ODBC Driver 18 for SQL Server)"

    def _connect(self):
        import pyodbc
        driver = self.s.extra.get("Driver", "ODBC Driver 18 for SQL Server")
        parts = [
            f"DRIVER={{{driver}}}",
            f"SERVER={self.s.host},{self.s.port}",
            f"DATABASE={self.s.name}",
            f"UID={self.s.user}",
            f"PWD={self.s.password}",
        ]
        for k, v in self.s.extra.items():
            if k.lower() == "driver":
                continue
            parts.append(f"{k}={v}")
        return pyodbc.connect(";".join(parts), timeout=self.s.connect_timeout)

    def q(self, ident: str) -> str:
        """Quote identifier gaya T-SQL (bracket)."""
        return "[" + ident.replace("]", "]]") + "]"

    def server_now(self) -> str:
        """Ekspresi waktu server UTC (DATETIME2)."""
        return "SYSUTCDATETIME()"

    def list_tables(self) -> List[str]:
        """Base table dari information_schema."""
        rows = self._run(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_type = 'BASE TABLE' ORDER BY table_name", fetch="all") or []
        return [r[0] for r in rows]

    def describe_table(self, name: str) -> List[Column]:
        """Kolom tabel dari information_schema + sys.identity_columns untuk PK."""
        rows = self._run(
            "SELECT column_name, data_type, is_nullable, column_default "
            "FROM information_schema.columns WHERE table_name = ? "
            "ORDER BY ordinal_position", (name,), fetch="all") or []
        idcols = set()
        try:
            irows = self._run(
                "SELECT c.name FROM sys.identity_columns c "
                "WHERE c.object_id = OBJECT_ID(?)", (name,), fetch="all") or []
            idcols = {r[0] for r in irows}
        except Exception:
            pass
        out = []
        for cn, dt, nn, dflt in rows:
            has_def = dflt is not None or cn in idcols
            out.append(Column(cn, dt, str(nn).upper() == "YES", has_def,
                              is_pk=(cn in idcols)))
        return out

    def ensure_user_table(self, name: str):
        """Buat tabel user bila belum ada (dibungkus IF OBJECT_ID karena T-SQL
        tak punya CREATE TABLE IF NOT EXISTS) → (dibuat_atau_ada, ddl)."""
        ddl = self.user_table_ddl(name)
        guarded = (
            f"IF OBJECT_ID(N'{name}', N'U') IS NULL\nBEGIN\n{ddl}\nEND")
        try:
            self._run(guarded)
            return True, ddl
        except Exception as e:
            from visioninspect.utils.logging_setup import get_logger
            get_logger("app").warning(
                "ensure_user_table('%s') gagal (%s) — buat manual", name, e)
            return False, ddl

    def user_table_ddl(self, name: str) -> str:
        """DDL tabel user standar (IDENTITY PK, NVARCHAR, username & rfid UNIQUE)."""
        return (
            f"CREATE TABLE {self.q(name)} (\n"
            "    id            BIGINT IDENTITY(1,1) PRIMARY KEY,\n"
            "    username      NVARCHAR(64) NOT NULL UNIQUE,\n"
            "    password_hash NVARCHAR(64) NOT NULL,\n"
            "    role          NVARCHAR(16) NOT NULL DEFAULT 'operator',\n"
            "    rfid          NVARCHAR(64) NULL UNIQUE,\n"
            "    last_login    DATETIME2 NULL\n"
            ")")
