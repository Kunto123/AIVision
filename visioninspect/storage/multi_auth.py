"""
MultiAuth - login dua sumber.

Cek akun LOKAL dulu (users.json — offline-safe, tanpa timeout), lalu DB
eksternal. Ketemu di salah satu = login sukses. TIDAK ada sinkronisasi
antar-store. Manajemen akun (add/edit/hapus/RFID) diarahkan per `source`
("lokal" | "db") — tab Akun me-list gabungan keduanya + kolom Lokasi.
"""

from typing import Optional

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")


class MultiAuth:
    """Login dua sumber: cek UserFileStore lokal dulu, lalu ExternalDB (bila ada)."""

    def __init__(self, local, external=None):
        """local = UserFileStore; external = ExternalDB atau None."""
        self._local = local           # UserFileStore
        self._external = external      # ExternalDB | None

    # ── Login ────────────────────────────────────────────────────────

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """Login: coba lokal lalu DB; dict user + `_source`, atau None."""
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
        """Cari user via RFID: coba lokal lalu DB; dict + `_source`, atau None."""
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

    @property
    def has_external(self) -> bool:
        """True bila DB eksternal terpasang sebagai sumber login kedua."""
        return self._external is not None

    # ── Manajemen — routing per `source` ("lokal" | "db") ──────────

    def list_users(self):
        """Gabungan lokal + DB, tiap baris ditandai `source`."""
        out = [dict(u, source="lokal") for u in self._local.list_users()]
        if self._external is not None:
            out += [dict(u, source="db") for u in self._external.list_users()]
        return out

    def add_user(self, username, password, display_name="", role="operator",
                 source="lokal"):
        """Tambah user ke store sesuai `source` ("lokal" | "db")."""
        if source == "db" and self._external is not None:
            self._external.add_user(username, password, display_name, role)
            return None
        return self._local.add_user(username, password, display_name, role)

    def update_user(self, user_id, display_name=None, password=None, role=None,
                    source="lokal", username=None):
        """Ubah user di store sesuai `source` (DB hanya password & role)."""
        if source == "db" and self._external is not None:
            if password is not None:
                self._external.set_password(username, password)
            if role is not None:
                self._external.set_role(username, role)
            return True
        return self._local.update_user(user_id, display_name=display_name,
                                       password=password, role=role)

    def delete_user(self, user_id, source="lokal", username=None):
        """Hapus user dari store sesuai `source` ("lokal" | "db")."""
        if source == "db" and self._external is not None:
            return self._external.delete_user(username)
        return self._local.delete_user(user_id)

    def bind_rfid(self, user_id, rfid_uid, source="lokal", username=None):
        """Ikat kartu RFID ke user di store sesuai `source`."""
        if source == "db" and self._external is not None:
            return self._external.bind_rfid(username, rfid_uid)
        return self._local.bind_rfid(user_id, rfid_uid)

    def unbind_rfid(self, user_id, source="lokal", username=None):
        """Lepas kartu RFID dari user di store sesuai `source`."""
        if source == "db" and self._external is not None:
            return bool(self._external.unbind_rfid(username))
        return self._local.unbind_rfid(user_id)
