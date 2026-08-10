#!/usr/bin/env python3
"""
VisionInspect — Tambah/Update User PostgreSQL (qc_user_accounts)

Sejak 2026-08-07 PostgreSQL adalah SATU-SATUNYA sumber akun aplikasi
(sync SQLite → PG dihapus). Script ini membuat/mengubah akun langsung di
tabel qc_user_accounts, dengan format hash yang sama persis dengan aplikasi
(SHA-256 + pepper `visioninspect_2024_`), sehingga akun langsung bisa
dipakai login.

Koneksi PG diambil dari data/config.json (postgresql.*). Password koneksi
otomatis di-decrypt via secret_store (DPAPI Windows / Fernet WSL). Bila
decrypt gagal (mis. token DPAPI Windows dibaca dari WSL), berikan override:

  python tools/pg_add_user.py --username admin --password rahasia123 --role admin \
      --pg-password "password-koneksi-postgres"

Usage:
  # Tambah user baru (password ditanya interaktif, tidak tampil di layar)
  python tools/pg_add_user.py --username operator1 --role operator

  # Tambah admin (password via argumen — hati-hati terlihat di history shell)
  python tools/pg_add_user.py --username admin --password rahasia123 --role admin

  # Ubah user yang sudah ada (role/password/display-name)
  python tools/pg_add_user.py --username test --role admin --update

  # Lihat daftar user yang ada
  python tools/pg_add_user.py --list

Catatan: jalankan dengan Python yang punya psycopg2 (venv `.vision` di
Windows, atau `.venv` WSL bila jaringan ke PG host memungkinkan).
"""

import argparse
import getpass
import hashlib
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_DATA_DIR = _PROJECT_ROOT / "data"

PASSWORD_PEPPER = "visioninspect_2024_"
VALID_ROLES = ("admin", "operator")


def _hash_password(password: str) -> str:
    """Hash password — sama persis dengan Database & PostgresDB."""
    return hashlib.sha256(f"{PASSWORD_PEPPER}{password}".encode()).hexdigest()


def _load_pg_config(data_dir: Path) -> dict:
    """Baca config postgresql dari data/config.json (password tetap terenkripsi)."""
    cfg_path = data_dir / "config.json"
    if not cfg_path.exists():
        sys.exit(f"❌ {cfg_path} tidak ditemukan. Jalankan dari project root atau "
                 "set env VISIONINSPECT_DATA.")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"❌ config.json rusak: {e}")
    pg = cfg.get("postgresql", {})
    if not pg.get("enabled"):
        sys.exit("❌ PostgreSQL disabled di config.json (postgresql.enabled=false). "
                 "Aktifkan dulu di Settings aplikasi, atau edit config.json.")
    return pg


def _decrypt_password(token: str) -> str:
    """Decrypt token enc:v1: via secret_store. Plaintext lama di-pass-through."""
    if not token:
        return ""
    try:
        from visioninspect.storage import secret_store
        return secret_store.decrypt(token)
    except Exception as e:
        print(f"⚠️  Gagal decrypt password koneksi via secret_store ({e}).")
        print("   Pakai --pg-password (atau env VISIONINSPECT_PG_PASSWORD) sebagai override.")
        return ""


def _connect(pg_cfg: dict, pg_pw_override: str = ""):
    """Konek ke PostgreSQL. Returns psycopg2 connection."""
    try:
        import psycopg2
    except ImportError:
        sys.exit("❌ psycopg2 tidak terinstall. Install dulu: pip install psycopg2-binary")

    password = pg_pw_override or _decrypt_password(pg_cfg.get("password", ""))
    if not password:
        print("ℹ️  Password koneksi PostgreSQL tidak tersedia (gagal decrypt / tidak di-set).")
        print("   Di Windows jalankan dengan venv .vision agar DPAPI bisa decrypt otomatis;")
        print("   di WSL berikan --pg-password (atau env VISIONINSPECT_PG_PASSWORD).")
        try:
            password = getpass.getpass("Password koneksi PostgreSQL: ")
        except (EOFError, KeyboardInterrupt):
            sys.exit("❌ Tidak ada password koneksi. Ulangi dengan --pg-password.")

    try:
        conn = psycopg2.connect(
            host=pg_cfg.get("host", "localhost"),
            port=pg_cfg.get("port", 5432),
            dbname=pg_cfg.get("dbname", "visioninspect"),
            user=pg_cfg.get("user", "postgres"),
            password=password,
            sslmode=pg_cfg.get("sslmode", "prefer"),
            connect_timeout=pg_cfg.get("connect_timeout", 10),
        )
        conn.autocommit = False
        return conn
    except Exception as e:
        sys.exit(f"❌ Koneksi PostgreSQL gagal: {e}")


def _ensure_table(conn) -> None:
    """Pastikan tabel qc_user_accounts ada + kolom must_change_password (idempotent)."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS qc_user_accounts (
                id BIGSERIAL PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'operator',
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ,
                last_login_at TIMESTAMPTZ,
                rfid_uid_hash TEXT,
                rfid_uid_last4 TEXT,
                rfid_bound_at TIMESTAMPTZ
            )""")
        cur.execute("""
            ALTER TABLE qc_user_accounts ADD COLUMN IF NOT EXISTS
                must_change_password BOOLEAN NOT NULL DEFAULT FALSE""")
    conn.commit()


def add_user(conn, username: str, password: str, role: str,
             is_active: bool = True,
             must_change_password: bool = False, update: bool = False) -> None:
    """Insert atau update (--update) satu user. Exit non-zero bila sudah ada.

    Catatan: qc_user_accounts TIDAK punya kolom display_name (PostgresDB
    selalu menampilkan display_name=username), jadi tidak di-set di sini.
    """
    now_sql = "now()"
    pw_hash = _hash_password(password)

    with conn.cursor() as cur:
        cur.execute("SELECT id, role FROM qc_user_accounts WHERE username = %s",
                    (username,))
        existing = cur.fetchone()

        if existing and not update:
            sys.exit(f"❌ User '{username}' SUDAH ada (id={existing[0]}, role={existing[1]}).\n"
                     "   Gunakan --update untuk mengubah role/password-nya.")

        if existing:
            cur.execute("""
                UPDATE qc_user_accounts
                   SET password_hash = %s, role = %s, is_active = %s,
                       must_change_password = %s, updated_at = %s
                 WHERE username = %s""",
                        (pw_hash, role, is_active, must_change_password, now_sql, username))
            print(f"✏️  User '{username}' diupdate (role={role}, is_active={is_active}).")
        else:
            cur.execute("""
                INSERT INTO qc_user_accounts
                    (username, password_hash, role, is_active,
                     must_change_password, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        (username, pw_hash, role, is_active,
                         must_change_password, now_sql, now_sql))
            print(f"✅ User '{username}' dibuat (role={role}, is_active={is_active}).")
    conn.commit()


def list_users(conn) -> None:
    """Tampilkan semua user."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, username, role, is_active, must_change_password,
                   created_at::text, last_login_at::text
              FROM qc_user_accounts ORDER BY id""")
        rows = cur.fetchall()
    if not rows:
        print("ℹ️  Belum ada user di qc_user_accounts.")
        return
    print(f"{'ID':>3}  {'USERNAME':<16} {'ROLE':<9} {'AKTIF':<6} {'GANTI PW':<8} CREATED")
    print("-" * 72)
    for r in rows:
        uid, uname, role, active, must, created, last = r
        print(f"{uid:>3}  {uname:<16} {role:<9} "
              f"{'ya' if active else 'TIDAK':<6} {'ya' if must else '':<8} {created}")
    print(f"\n{len(rows)} user(s).")


def main():
    parser = argparse.ArgumentParser(
        description="Tambah/ubah user VisionInspect di PostgreSQL (qc_user_accounts).")
    parser.add_argument("--username", help="Username akun (wajib untuk add/update)")
    parser.add_argument("--password", help="Password akun. Bila kosong → ditanya interaktif")
    parser.add_argument("--role", choices=VALID_ROLES, default="operator",
                        help="Role akun (default: operator)")
    parser.add_argument("--inactive", action="store_true",
                        help="Set is_active=FALSE (akun nonaktif, tidak bisa login)")
    parser.add_argument("--must-change-password", action="store_true",
                        help="Paksa ganti password saat login pertama")
    parser.add_argument("--update", action="store_true",
                        help="Izinkan mengubah user yang sudah ada")
    parser.add_argument("--list", action="store_true",
                        help="Tampilkan daftar user lalu keluar")
    parser.add_argument("--data-dir", default=str(_DATA_DIR),
                        help="Direktori data (tempat config.json). Default: <root>/data")
    # Override koneksi PG
    parser.add_argument("--pg-host", default=None)
    parser.add_argument("--pg-port", type=int, default=None)
    parser.add_argument("--pg-dbname", default=None)
    parser.add_argument("--pg-user", default=None)
    parser.add_argument("--pg-password", default=None,
                        help="Password koneksi PG (override decrypt config). "
                             "Alternatif: env VISIONINSPECT_PG_PASSWORD")
    args = parser.parse_args()

    pg_cfg = _load_pg_config(Path(args.data_dir))
    for key, attr in (("host", "pg_host"), ("port", "pg_port"),
                      ("dbname", "pg_dbname"), ("user", "pg_user")):
        val = getattr(args, attr)
        if val is not None:
            pg_cfg[key] = val

    pg_pw = args.pg_password or os.environ.get("VISIONINSPECT_PG_PASSWORD", "")

    # Fail fast: validasi argumen sebelum menyentuh jaringan/DB
    if not args.list and not args.username:
        parser.error("--username wajib diisi (atau pakai --list)")

    conn = _connect(pg_cfg, pg_pw)
    try:
        _ensure_table(conn)

        if args.list:
            list_users(conn)
            return

        # Password akun: argumen > interaktif (2x konfirmasi)
        password = args.password
        if password is None:
            try:
                password = getpass.getpass(f"Password untuk '{args.username}': ")
                confirm = getpass.getpass("Ulangi password: ")
            except (EOFError, KeyboardInterrupt):
                sys.exit("❌ Dibatalkan. Berikan --password untuk mode non-interaktif.")
            if password != confirm:
                sys.exit("❌ Password tidak cocok.")
        if not password:
            sys.exit("❌ Password tidak boleh kosong.")

        add_user(conn, args.username, password, args.role,
                 is_active=not args.inactive,
                 must_change_password=args.must_change_password,
                 update=args.update)
        print("ℹ️  Verifikasi:")
        list_users(conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
