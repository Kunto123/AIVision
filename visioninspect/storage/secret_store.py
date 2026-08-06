"""
VisionInspect - Secret Store (C4: kredensial non-plaintext)

Menyimpan kredensial (password PostgreSQL, dst.) TIDAK sebagai plaintext
di disk. Format token: ``enc:v1:<base64>``.

Strategi per platform (machine-bound key, bukan "base64 palsu"):
  * Windows  -> DPAPI (CryptProtectData / CryptUnprotectData, scope user)
                via ctypes — terikat user + mesin, tanpa key di disk.
  * Linux/WSL-> Kunci acak 32-byte di ``~/.visioninspect/secret.key``
                (mode 0600) + enkripsi Fernet (cryptography). Bila
                cryptography tidak tersedia, fallback XOR stream
                SHA-256 (stdlib-only) — tetap bukan plaintext.

Catatan migrasi: token lama yang berupa plaintext di-pass-through oleh
``decrypt()``; migrasi otomatis ke ``enc:v1:`` terjadi saat settings
disimpan (main_window._on_settings_save) atau saat startup.
"""

import base64
import ctypes
import hashlib
import os
import sys
from ctypes import wintypes
from pathlib import Path

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")

PREFIX = "enc:v1:"
_KEY_DIR = Path.home() / ".visioninspect"
_KEY_FILE = _KEY_DIR / "secret.key"

try:
    from cryptography.fernet import Fernet
    HAS_FERNET = True
except ImportError:  # pragma: no cover - fallback stdlib
    HAS_FERNET = False


def _is_windows() -> bool:
    return sys.platform == "win32"


# ── Windows DPAPI ────────────────────────────────────────────────────

class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _dpapi_protect(data: bytes) -> bytes:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    blob_in = _DATA_BLOB(len(data), ctypes.cast(
        ctypes.create_string_buffer(data), ctypes.POINTER(ctypes.c_char)))
    blob_out = _DATA_BLOB()
    # CRYPTPROTECT_UI_FORBIDDEN = 0x1
    if not crypt32.CryptProtectData(
            ctypes.byref(blob_in), None, None, None, None, 0x1,
            ctypes.byref(blob_out)):
        raise RuntimeError("CryptProtectData gagal")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def _dpapi_unprotect(data: bytes) -> bytes:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    blob_in = _DATA_BLOB(len(data), ctypes.cast(
        ctypes.create_string_buffer(data), ctypes.POINTER(ctypes.c_char)))
    blob_out = _DATA_BLOB()
    if not crypt32.CryptUnprotectData(
            ctypes.byref(blob_in), None, None, None, None, 0x1,
            ctypes.byref(blob_out)):
        raise RuntimeError("CryptUnprotectData gagal")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


# ── Linux/WSL key ────────────────────────────────────────────────────

def _get_machine_key() -> bytes:
    """Kunci acak per-user/mesin (mode 0600). Dibuat sekali, dipakai terus."""
    _KEY_DIR.mkdir(parents=True, exist_ok=True)
    if _KEY_FILE.exists():
        return _KEY_FILE.read_bytes()
    key = os.urandom(32)
    _KEY_FILE.write_bytes(key)
    try:
        os.chmod(_KEY_FILE, 0o600)
    except OSError:
        pass
    logger.info("Secret key dibuat: %s", _KEY_FILE)
    return key


def _xor_stream(data: bytes, key: bytes) -> bytes:
    """XOR stream dengan key SHA-256 — fallback stdlib (bukan plaintext)."""
    k = hashlib.sha256(key).digest()
    return bytes(b ^ k[i % len(k)] for i, b in enumerate(data))


def _encrypt_bytes(data: bytes) -> bytes:
    if _is_windows():
        return _dpapi_protect(data)
    key = _get_machine_key()
    if HAS_FERNET:
        return Fernet(base64.urlsafe_b64encode(key)).encrypt(data)
    return _xor_stream(data, key)


def _decrypt_bytes(raw: bytes) -> bytes:
    if _is_windows():
        return _dpapi_unprotect(raw)
    key = _get_machine_key()
    if HAS_FERNET:
        return Fernet(base64.urlsafe_b64encode(key)).decrypt(raw)
    return _xor_stream(raw, key)


# ── Public API ───────────────────────────────────────────────────────

def encrypt(plaintext: str) -> str:
    """Enkripsi string -> token ``enc:v1:<base64>``. String kosong -> kosong."""
    if not plaintext:
        return ""
    return PREFIX + base64.b64encode(_encrypt_bytes(plaintext.encode("utf-8"))).decode("ascii")


def decrypt(token: str) -> str:
    """Dekripsi token ``enc:v1:`` -> plaintext. Token plaintext lama
    (tanpa prefix) di-pass-through untuk migrasi."""
    if not token:
        return ""
    if not token.startswith(PREFIX):
        return token
    raw = base64.b64decode(token[len(PREFIX):])
    return _decrypt_bytes(raw).decode("utf-8")


def is_encrypted(token: str) -> bool:
    return bool(token) and token.startswith(PREFIX)
