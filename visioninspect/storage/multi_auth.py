"""
MultiAuth - login dua sumber.

Cek akun LOKAL dulu (users.json — offline-safe, tanpa timeout), lalu DB
eksternal. Ketemu di salah satu = login sukses. TIDAK ada sinkronisasi
antar-store. Manajemen akun (add/edit/hapus/RFID) diarahkan ke store lokal.
"""

from typing import Optional

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")


class MultiAuth:
    def __init__(self, local, external=None):
        self._local = local           # UserFileStore
        self._external = external      # ExternalDB | None

    # ── Login ────────────────────────────────────────────────────────

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        user = self._local.authenticate(username, password)
        if user:
            user["_source"] = "local"
            return user
        if self._external is not None:
            user = self._external.authenticate(username, password)
            if user:
                user["_source"] = "db"
                logger.info("Login via DB eksternal: %s", username)
                return user
        return None

    def get_user_by_rfid(self, rfid_uid: str) -> Optional[dict]:
        user = self._local.get_user_by_rfid(rfid_uid)
        if user:
            user["_source"] = "local"
            return user
        if self._external is not None:
            user = self._external.get_user_by_rfid(rfid_uid)
            if user:
                user["_source"] = "db"
                return user
        return None

    def _connect(self):
        """LoginDialog memakai ini untuk pesan 'koneksi DB gagal' yang tepat."""
        if self._external is not None:
            return self._external._connect()

        class _Dummy:
            def close(self):
                pass

        return _Dummy()

    # ── Manajemen (store lokal) ─────────────────────────────────────

    def list_users(self):
        return self._local.list_users()

    def add_user(self, *a, **kw):
        return self._local.add_user(*a, **kw)

    def update_user(self, *a, **kw):
        return self._local.update_user(*a, **kw)

    def delete_user(self, *a, **kw):
        return self._local.delete_user(*a, **kw)

    def bind_rfid(self, *a, **kw):
        return self._local.bind_rfid(*a, **kw)

    def unbind_rfid(self, *a, **kw):
        return self._local.unbind_rfid(*a, **kw)
