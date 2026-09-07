"""
VisionInspect - Hash kredensial (dipakai bersama semua store user).
SHA-256 + pepper. Sama persis dengan Database._hash_password lama.
"""

import hashlib

# Pepper aplikasi — HARUS sama dengan storage/db.py & storage/postgres_db.py.
PEPPER = "visioninspect_2024_"


def hash_password(password: str) -> str:
    """Hash password: SHA-256(pepper + password) hex."""
    return hashlib.sha256(f"{PEPPER}{password}".encode()).hexdigest()


def hash_rfid(rfid_uid: str) -> str:
    """Hash RFID UID untuk kolom `rfid` di tabel user DB."""
    return hashlib.sha256(f"{PEPPER}{rfid_uid}".encode()).hexdigest()
