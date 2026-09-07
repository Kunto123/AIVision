"""Adapter DB per engine — kontrak tipis (bukan SQLAlchemy)."""

from visioninspect.storage.db_txt import DbSettings


def get_adapter(settings: DbSettings):
    """Buat adapter sesuai settings.engine."""
    e = settings.engine
    if e == "postgresql":
        from visioninspect.storage.db_adapters.pg import PostgresAdapter
        return PostgresAdapter(settings)
    if e == "mysql":
        from visioninspect.storage.db_adapters.mysql import MySQLAdapter
        return MySQLAdapter(settings)
    if e == "sqlserver":
        from visioninspect.storage.db_adapters.sqlserver import SQLServerAdapter
        return SQLServerAdapter(settings)
    if e == "sqlite":
        from visioninspect.storage.db_adapters.sqlite_ad import SQLiteAdapter
        return SQLiteAdapter(settings)
    raise ValueError(f"DB_ENGINE tidak dikenal: {e}")
