"""
CLI ringan untuk db.txt — dijalankan tanpa Qt / torch.
  --check-db              : validasi db.txt lalu keluar
  --encrypt-secret TEKS   : cetak token enc:v2: untuk DB_PASSWORD
  --db-user-add U P [ROLE]: tambah akun ke tabel user DB
"""

import sys


def _encrypt(text: str) -> int:
    from visioninspect.storage import secret_store
    print(secret_store.encrypt_portable(text))
    return 0


def check_db() -> int:
    from visioninspect.storage import db_txt
    s = db_txt.load()
    if not s.engine and not s.errors:
        print("db.txt tidak ada / DB_ENGINE kosong — DB eksternal mati "
              "(auth SQLite lokal saja, tidak ada push inspeksi).")
        return 0
    for e in s.errors:
        print(f"[ERR ] {e}")
    if s.errors:
        return 1
    from visioninspect.storage.external_db import ExternalDB
    try:
        rep = ExternalDB(s).check()
    except Exception as e:  # driver hilang, dll.
        print(f"[ERR ] {e}")
        return 1
    for line in rep.lines:
        print(line)
    print("HASIL:", "OK" if rep.ok else "ADA ERROR")
    return 0 if rep.ok else 1


def db_user_add(argv) -> int:
    if len(argv) < 2:
        print("pakai: --db-user-add <username> <password> [role]")
        return 2
    username, password = argv[0], argv[1]
    role = argv[2] if len(argv) > 2 else "operator"
    from visioninspect.storage import db_txt
    from visioninspect.storage.external_db import ExternalDB
    s = db_txt.load()
    if not s.is_configured or not s.user_table:
        print("db.txt belum dikonfigurasi / DB_USER_TABLE kosong")
        return 1
    ext = ExternalDB(s)
    ext.ensure_user_table()
    try:
        ext.add_user(username, password, role=role)
    except Exception as e:
        print(f"gagal: {e}")
        return 1
    print(f"user '{username}' (role={role}) ditambahkan ke {s.user_table}")
    return 0


def main(ns=None) -> int:
    """ns = argparse.Namespace dari run.py; atau parse sys.argv sendiri."""
    args = sys.argv[1:]
    if ns is not None and getattr(ns, "encrypt_secret", ""):
        return _encrypt(ns.encrypt_secret)
    if "--encrypt-secret" in args:
        i = args.index("--encrypt-secret")
        return _encrypt(args[i + 1] if i + 1 < len(args) else "")
    if "--db-user-add" in args:
        i = args.index("--db-user-add")
        return db_user_add(args[i + 1:])
    if (ns is not None and getattr(ns, "check_db", False)) or "--check-db" in args:
        return check_db()
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
