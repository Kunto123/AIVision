"""
VisionInspect - Konfigurasi DB eksternal via file db.txt.

File teks KEY=VALUE di sebelah executable (root proyek saat dev). Tanpa GUI.
Env var `VI_<KEY>` meng-override tiap baris. Lihat db.txt.example.

Tanpa db.txt / DB_ENGINE kosong -> fitur DB eksternal mati:
auth hanya tabel SQLite `users` lokal, tidak ada push inspeksi.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")

FILENAME = "db.txt"
VALID_ENGINES = ("postgresql", "mysql", "sqlserver", "sqlite")
DEFAULT_PORT = {"postgresql": 5432, "mysql": 3306, "sqlserver": 1433, "sqlite": 0}

# 6 kolom tetap tabel user DB — skema milik VisionInspect, bukan sistem customer.
USER_TABLE_COLUMNS = ("id", "username", "password_hash", "role", "rfid", "last_login")

# Field sumber yang bisa dipetakan di sisi kanan MAP_<kolom>= (lihat design doc).
SOURCE_FIELDS = (
    "partname", "program", "template_id", "operator", "judgement",
    "score_worst", "score_avg", "threshold", "part_check_score",
    "latency_ms", "num_rois", "station_id", "timestamp_edge",
)

_SIMPLE_KEYS = (
    "DB_ENGINE", "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD",
    "DB_EXTRA", "DB_CONNECT_TIMEOUT", "DB_INSPECTION_TABLE", "DB_USER_TABLE",
    "STATION_ID",
)


def base_dir() -> Path:
    """Folder tempat db.txt dicari: sebelah .exe (frozen) atau root proyek (dev)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def config_path() -> Path:
    return base_dir() / FILENAME


@dataclass
class DbSettings:
    engine: str = ""
    host: str = ""
    port: int = 0
    name: str = ""          # sqlite: path file .db
    user: str = ""
    password: str = ""      # sudah di-decrypt
    extra: Dict[str, str] = field(default_factory=dict)
    connect_timeout: int = 10
    inspection_table: str = ""
    user_table: str = ""
    station_id: str = ""
    mapping: Dict[str, str] = field(default_factory=dict)   # kolom_db -> source
    source_path: str = ""
    errors: List[str] = field(default_factory=list)

    @property
    def is_configured(self) -> bool:
        """True bila db.txt valid & DB_ENGINE diisi (fitur DB eksternal aktif)."""
        return bool(self.engine) and not self.errors

    @property
    def has_mapping(self) -> bool:
        return bool(self.mapping)

    @property
    def has_user_table(self) -> bool:
        return bool(self.user_table)


def _strip_inline_comment(key: str, value: str) -> str:
    """Buang komentar inline ' # ...' — KECUALI untuk DB_PASSWORD (bisa berisi #)."""
    if key == "DB_PASSWORD":
        return value
    idx = value.find(" #")
    return value[:idx].strip() if idx >= 0 else value


def _read_raw(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        # MAP_<kolom> — case kolom DB dipertahankan; key lain di-uppercase.
        k = ("MAP_" + k[4:]) if k[:4].upper() == "MAP_" else k.upper()
        out[k] = _strip_inline_comment(k, v.strip())
    return out


def _apply_env(raw: Dict[str, str]) -> None:
    """Override tiap key dari env `VI_<KEY>` (dev / docker)."""
    keys = set(raw.keys()) | set(_SIMPLE_KEYS)
    for k in list(keys):
        env = os.environ.get("VI_" + k)
        if env is not None:
            raw[k] = env


def resolve_row(mapping: Dict[str, str], snapshot: dict
                ) -> Tuple[List[str], List[object], List[str]]:
    """Petakan snapshot -> (kolom, nilai, kolom_waktu_server) untuk INSERT dinamis."""
    cols: List[str] = []
    vals: List[object] = []
    server_time: List[str] = []
    for col, src in mapping.items():
        if src == "@server_now":
            server_time.append(col)
        elif src.startswith("@const:"):
            cols.append(col)
            vals.append(src[len("@const:"):])
        else:
            cols.append(col)
            vals.append(snapshot.get(src))
    return cols, vals, server_time


def load() -> DbSettings:
    """Baca db.txt + env. Selalu sukses; masalah dikumpulkan di `.errors`."""
    s = DbSettings()
    path = config_path()
    raw: Dict[str, str] = {}
    if path.exists():
        try:
            raw = _read_raw(path)
            s.source_path = str(path)
        except Exception as e:
            s.errors.append(f"Gagal baca {path}: {e}")
            return s
    _apply_env(raw)

    engine = raw.get("DB_ENGINE", "").strip().lower()
    if not engine:
        return s  # tidak dikonfigurasi -> perilaku lama
    if engine not in VALID_ENGINES:
        s.errors.append(
            f"DB_ENGINE tidak dikenal: '{engine}' (pilih: {', '.join(VALID_ENGINES)})")
        return s
    s.engine = engine

    s.host = raw.get("DB_HOST", "").strip()
    s.name = raw.get("DB_NAME", "").strip()
    s.user = raw.get("DB_USER", "").strip()
    s.station_id = raw.get("STATION_ID", "").strip()
    s.inspection_table = raw.get("DB_INSPECTION_TABLE", "").strip()
    s.user_table = raw.get("DB_USER_TABLE", "").strip()

    try:
        s.port = int(raw.get("DB_PORT", "").strip() or 0)
    except ValueError:
        s.errors.append(f"DB_PORT bukan angka: '{raw.get('DB_PORT')}'")
    if not s.port:
        s.port = DEFAULT_PORT[engine]
    try:
        s.connect_timeout = int(raw.get("DB_CONNECT_TIMEOUT", "").strip() or 10)
    except ValueError:
        s.connect_timeout = 10

    # DB_PASSWORD: plain / enc:v1: / enc:v2: -> selalu jadi plaintext di memori
    pw_raw = raw.get("DB_PASSWORD", "")
    try:
        from visioninspect.storage import secret_store
        s.password = secret_store.decrypt(pw_raw)
    except Exception as e:
        s.errors.append(f"Dekripsi DB_PASSWORD gagal: {e}")

    # DB_EXTRA: "k=v;k2=v2" (param koneksi tambahan; SQL Server: TrustServerCertificate=yes)
    for part in raw.get("DB_EXTRA", "").split(";"):
        part = part.strip()
        if part and "=" in part:
            ek, _, ev = part.partition("=")
            s.extra[ek.strip()] = ev.strip()

    # MAP_<kolom_db>=<source>
    for k, v in raw.items():
        if k.startswith("MAP_"):
            col = k[4:].strip()
            if col:
                s.mapping[col] = v.strip()

    _validate(s)
    return s


def _validate(s: DbSettings) -> None:
    if s.engine == "sqlite":
        if not s.name:
            s.errors.append("DB_NAME wajib untuk sqlite (path file .db)")
    else:
        if not s.host:
            s.errors.append("DB_HOST wajib")
        if not s.name:
            s.errors.append("DB_NAME wajib")
        if not s.user:
            s.errors.append("DB_USER wajib")
    if s.mapping and not s.inspection_table:
        s.errors.append("MAP_* diisi tapi DB_INSPECTION_TABLE kosong")
    # Peringatan source tak dikenal (bukan error — @const/@server_now bebas)
    for col, src in s.mapping.items():
        if src.startswith("@"):
            if src != "@server_now" and not src.startswith("@const:"):
                s.errors.append(f"MAP_{col}: token '{src}' tidak dikenal")
        elif src not in SOURCE_FIELDS:
            s.errors.append(
                f"MAP_{col}: source '{src}' bukan field yang dikenal "
                f"({', '.join(SOURCE_FIELDS)})")


TEMPLATE = """\
# ============================================================================
#  db.txt - konfigurasi database eksternal VisionInspect
#  Letakkan file ini di folder yang sama dengan VisionInspect.exe.
#  Hapus / kosongkan DB_ENGINE untuk mematikan fitur (perilaku lama dipakai).
#  Cek konfigurasi tanpa buka aplikasi:  VisionInspect.exe --check-db
# ============================================================================

# ---- Koneksi ----
DB_ENGINE=postgresql            # postgresql | mysql | sqlserver | sqlite
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=qc_central             # sqlite: isi path file .db
DB_USER=visioninspect
DB_PASSWORD=                   # plain, atau token enc:v2: dari --encrypt-secret
DB_EXTRA=                      # k=v;k=v  (SQL Server: TrustServerCertificate=yes)
DB_CONNECT_TIMEOUT=10

# ---- Tabel ----
DB_INSPECTION_TABLE=production_qc_log
DB_USER_TABLE=vi_user_accounts

# ---- Identitas stasiun ----
STATION_ID=STN-01

# ---- Pemetaan kolom: MAP_<kolom_db>=<source> ----
# source = nama field di bawah, atau @server_now, atau @const:<teks>
#   partname program template_id operator judgement
#   score_worst score_avg threshold part_check_score latency_ms num_rois
#   station_id timestamp_edge
# Kolom DB yang tidak di-MAP tidak dikirim (pakai DEFAULT / NULL milik DB).
MAP_part_name=partname
MAP_checked_at=@server_now
MAP_operator=operator
MAP_qc_score=score_worst
MAP_gate_score=part_check_score
MAP_station=station_id
"""


def write_template(path: Path = None) -> Path:
    """Tulis db.txt.example (tidak menimpa db.txt asli)."""
    target = path or (base_dir() / "db.txt.example")
    target.write_text(TEMPLATE, encoding="utf-8")
    return target
