"""
UserFileStore - store akun lokal berbasis file JSON (users.json).

Salah satu dari dua sumber login (yang lain: tabel DB via ExternalDB).
Surface method sama dengan Database (authenticate, list_users, add_user, ...).
Migrasi otomatis dari tabel SQLite `users` saat file belum ada.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from visioninspect.storage import credentials
from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class UserFileStore:
    """users.json: {"version": 1, "users": [ {...}, ... ]}."""

    def __init__(self, path: Path, sqlite_db=None):
        """Muat users.json; bila kosong, migrasi dari SQLite lalu seed admin/admin."""
        self._path = Path(path)
        self._data: Dict = {"version": 1, "users": []}
        existed = self._path.exists()
        self._load()
        if not self._data["users"]:
            # Baru pertama kali: coba migrasi akun dari tabel SQLite `users`
            # (jangan kehilangan akun operator + binding RFID yang sudah dibuat).
            if sqlite_db is not None and not existed:
                self._migrate_from_sqlite(sqlite_db)
            if not self._data["users"]:
                self._seed_default_admin()

    # ── IO ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists() and self._path.stat().st_size > 0:
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
                self._data.setdefault("version", 1)
                self._data.setdefault("users", [])
                return
            except (json.JSONDecodeError, OSError) as e:
                logger.error("users.json rusak (%s) — mulai kosong", e)
        self._data = {"version": 1, "users": []}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self._path.parent,
            suffix=".tmp", delete=False)
        try:
            json.dump(self._data, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
            try:
                os.fsync(tmp.fileno())
            except (OSError, AttributeError):
                pass
            tmp.close()
            os.replace(tmp.name, str(self._path))
        except Exception:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            raise

    # ── Seed & migrasi ───────────────────────────────────────────────

    def _next_id(self) -> int:
        return (max((u.get("id", 0) for u in self._data["users"]), default=0) + 1)

    def _seed_default_admin(self) -> None:
        self._data["users"].append({
            "id": 1, "username": "admin",
            "password_hash": credentials.hash_password("admin"),
            "display_name": "Administrator", "role": "admin",
            "rfid_uid": None, "rfid_bound_at": None,
            "created_at": _now(), "updated_at": _now(),
        })
        self._save()
        logger.info("users.json: seed admin/admin (ganti password lewat tab Akun)")

    def _migrate_from_sqlite(self, sqlite_db) -> int:
        """Salin tabel SQLite `users` -> users.json (dipanggil dari __init__)."""
        try:
            full = {u["id"]: u for u in sqlite_db.list_users_full()}
            meta = {u["id"]: u for u in sqlite_db.list_users()}
        except Exception as e:
            logger.warning("Migrasi users.json dari SQLite gagal: %s", e)
            return 0
        if not full:
            return 0
        users = []
        for uid, u in full.items():
            m = meta.get(uid, {})
            users.append({
                "id": uid, "username": u["username"],
                "password_hash": u["password_hash"],
                "display_name": m.get("display_name", ""),
                "role": u.get("role", "operator"),
                "rfid_uid": m.get("rfid_uid") or None,
                "rfid_bound_at": m.get("rfid_bound_at") or None,
                "created_at": m.get("created_at", _now()),
                "updated_at": _now(),
            })
        self._data = {"version": 1, "users": users}
        self._save()
        logger.info("users.json: migrasi %d akun dari SQLite", len(users))
        return len(users)

    # ── Query ────────────────────────────────────────────────────────

    def _find(self, **kw) -> Optional[dict]:
        for u in self._data["users"]:
            if all(u.get(k) == v for k, v in kw.items()):
                return u
        return None

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """Cek username + password → dict user, atau None bila salah."""
        u = self._find(username=username)
        if u and u["password_hash"] == credentials.hash_password(password):
            return dict(u)
        return None

    def get_user_by_rfid(self, rfid_uid: str) -> Optional[dict]:
        """Cari user berdasarkan UID kartu RFID → dict, atau None."""
        u = self._find(rfid_uid=rfid_uid)
        return dict(u) if u else None

    def list_users(self) -> List[dict]:
        """Daftar user dengan field aman untuk UI (tanpa password_hash)."""
        return [{
            "id": u["id"], "username": u["username"],
            "display_name": u.get("display_name", ""),
            "role": u.get("role", "operator"),
            "rfid_uid": u.get("rfid_uid") or "",
            "rfid_bound_at": u.get("rfid_bound_at") or "",
            "created_at": u.get("created_at", ""),
            "updated_at": u.get("updated_at", ""),
        } for u in self._data["users"]]

    def list_users_full(self) -> List[dict]:
        """Daftar user lengkap termasuk password_hash (untuk migrasi/sync)."""
        return [dict(u) for u in self._data["users"]]

    # ── Mutasi ───────────────────────────────────────────────────────

    def add_user(self, username: str, password: str, display_name: str = "",
                 role: str = "operator") -> int:
        """Tambah user baru → id; ValueError bila username sudah ada."""
        if self._find(username=username):
            raise ValueError(f"Username '{username}' sudah ada")
        uid = self._next_id()
        self._data["users"].append({
            "id": uid, "username": username,
            "password_hash": credentials.hash_password(password),
            "display_name": display_name, "role": role,
            "rfid_uid": None, "rfid_bound_at": None,
            "created_at": _now(), "updated_at": _now(),
        })
        self._save()
        logger.info("users.json: tambah user %s (role=%s)", username, role)
        return uid

    def update_user(self, user_id: int, display_name: str = None,
                    password: str = None, role: str = None) -> bool:
        """Ubah display_name/password/role yang non-None; False bila user tak ada."""
        u = self._find(id=user_id)
        if not u:
            return False
        if display_name is not None:
            u["display_name"] = display_name
        if password is not None:
            u["password_hash"] = credentials.hash_password(password)
        if role is not None:
            u["role"] = role
        u["updated_at"] = _now()
        self._save()
        return True

    def delete_user(self, user_id: int) -> bool:
        """Hapus user; tolak (False) bila itu admin terakhir atau user tak ada."""
        u = self._find(id=user_id)
        if not u:
            return False
        if u.get("role") == "admin":
            admins = [x for x in self._data["users"] if x.get("role") == "admin"]
            if len(admins) <= 1:
                logger.warning("users.json: tidak bisa hapus admin terakhir")
                return False
        self._data["users"] = [x for x in self._data["users"] if x["id"] != user_id]
        self._save()
        logger.info("users.json: hapus user id=%d", user_id)
        return True

    def bind_rfid(self, user_id: int, rfid_uid: str) -> bool:
        """Ikat UID kartu RFID ke user; False bila UID sudah dipakai / user tak ada."""
        if self._find(rfid_uid=rfid_uid):
            return False
        u = self._find(id=user_id)
        if not u:
            return False
        u["rfid_uid"] = rfid_uid
        u["rfid_bound_at"] = _now()
        u["updated_at"] = _now()
        self._save()
        return True

    def unbind_rfid(self, user_id: int) -> bool:
        """Lepas ikatan kartu RFID dari user."""
        u = self._find(id=user_id)
        if not u:
            return False
        u["rfid_uid"] = None
        u["rfid_bound_at"] = None
        u["updated_at"] = _now()
        self._save()
        return True
