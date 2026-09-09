# VisionInspect

Aplikasi desktop untuk **inspeksi visual industri berbasis AI**, berjalan secara lokal. Model dilatih sendiri dari foto part.

- **Teaching** — latih model dari data foto part OK NG.
- **3 engine inferensi** — YOLO klasifikasi (dianjurkan), PatchCore, atau EfficientAd. Runtime produksi memakai **OpenVINO**.
- **Redefinition loop** — admin koreksi hasil salah → rebuild.
- **Komunikasi PLC** — Mitsubishi FX lewat *Computer Link*.

> Detail teknis: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Panduan operator: [docs/MANUAL_OPERATOR.md](docs/MANUAL_OPERATOR.md) · Wiring & PLC: [docs/MANUAL_TEKNISI.md](docs/MANUAL_TEKNISI.md)

---

## Quick Start

```batch
git clone <repo-url> VisionInspect
cd VisionInspect
run.bat
```

Saat pertama dijalankan, `run.bat` otomatis membuat venv `.vision\`, meng-install `requirements.txt`, dan mengarahkan folder data ke `data\` di dalam proyek — lalu menjalankan aplikasi. Perlu koneksi internet saat pertama (install deps + unduh backbone pretrained saat training pertama).

Setup manual (kalau perlu):

```batch
python -m venv .vision
.vision\Scripts\python.exe -m pip install -r requirements.txt
```

> ⚠️ Jangan pakai launcher `py`. Selalu `.vision\Scripts\python.exe` atau aktifkan venv dulu (`.vision\Scripts\activate`).

Login pertama kali: user `admin`, password `admin` — wajib ganti saat login pertama. Akun disimpan lokal (tabel SQLite `users`, atau `data\users.json` kalau `db.txt` aktif). DB eksternal untuk auth + push diatur lewat `db.txt` (lihat di bawah).

## Persyaratan

| Komponen | Minimum | Rekomendasi |
|----------|---------|-------------|
| OS | Windows 10 | Windows 11 |
| CPU | Intel i5 gen 10, 4 core | Intel i7 gen 12+ |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB kosong | 10 GB SSD |
| Kamera | USB (UVC) | USB 3.0 / GigE |
| Python | 3.10+ | 3.11 |

WSL hanya dipakai untuk training (lihat *Training*). Aplikasi utama berjalan di Windows native.

## Requirements

Isi paket sama semua (runtime OpenVINO/GUI/auth/Flask/`fxplc`/driver DB + training `ultralytics`/`lightning`/`nncf` + test). Hanya wheel `torch`/`torchvision` yang beda — dua file, pilih sesuai mesin:

| File | Untuk | torch |
|------|-------|-------|
| `requirements.txt` | PC edge (Windows), mesin tanpa GPU | `+cpu` |
| `requirements-gpu.txt` | PC dev / WSL dengan GPU NVIDIA | `+cu124` |

```batch
:: Windows — run.bat bikin venv .vision\ + install otomatis saat pertama
run.bat
```
```bash
# WSL — sekali di awal, dari root proyek
python3 -m venv .venv
.venv/bin/pip install -r requirements-gpu.txt
.venv/bin/python run.py
```

Semua versi **di-pin** ke kombinasi yang teruji E2E. Kedua file wajib identik kecuali 3 baris (index + torch + torchvision) — cek: `python tools/check_reqs_sync.py`.

## Menjalankan

```batch
run.bat                                    :: cara normal
.vision\Scripts\python.exe run.py --log-level DEBUG
```

Opsi CLI (`run.py` / `visioninspect/main.py`): `--config <path>`, `--data-dir <path>`, `--log-level DEBUG|INFO|WARNING|ERROR`, `--version`, `--check-db`, `--encrypt-secret <teks>`.

**edge_mode** — set `"edge_mode": true` di `data\config.json` supaya `torch` tidak ikut dimuat saat start. PC edge inference-only wajib pakai ini.

## Database eksternal (db.txt)

Satu-satunya jalur DB eksternal. Kalau file **`db.txt`** ada di root proyek (di samping `run.py`), aplikasi push hasil inspeksi OK ke tabel DB customer dan mengecek login ke tabel user di DB itu. **Tanpa `db.txt`**: auth hanya tabel SQLite `users` lokal, tidak ada push. (Tidak ada lagi setting PostgreSQL di tab Settings — semua di `db.txt`.)

- **Engine**: PostgreSQL, MySQL/MariaDB, SQL Server, atau SQLite. Driver PostgreSQL/MySQL sudah di `requirements.txt`; SQL Server juga butuh *Microsoft ODBC Driver 18* di-install manual di PC edge.
- **Konfigurasi + pemetaan kolom** semua di `db.txt` — salin `db.txt.example`, isi, lalu:
  ```batch
  run.bat --check-db                    :: validasi koneksi + tabel + mapping
  run.bat --encrypt-secret "passwordDB" :: -> token enc:v2: untuk DB_PASSWORD
  run.bat --db-user-add budi rahasia operator
  ```
- **Login dua sumber**: akun lokal (`users.json`, migrasi otomatis dari SQLite saat pertama) **dan** tabel `DB_USER_TABLE`. Cocok di salah satu = masuk. Manajemen akun di tab Akun menulis ke store lokal; akun DB dibuat via `--db-user-add`.
- **`enc:v2:`**: token terikat ke `%USERPROFILE%\.visioninspect\secret.key` di mesin tempat `--encrypt-secret` dijalankan. Jalankan di tiap PC edge, atau copy `secret.key` bareng `db.txt`. Password plain juga boleh.

## Training

Training memerlukan PyTorch. Di Windows PyTorch sering bermasalah (WinError 1114), jadi ada dua jalur:

| Jalur | Kapan | Cara |
|-------|-------|------|
| **Dalam aplikasi** (tab TEACH) | PyTorch Windows jalan, atau pakai engine *Simple* (tanpa torch) | Tombol **TRAIN** di tab TEACH |
| **CLI via WSL** | PyTorch Windows tidak jalan | Di WSL (venv terpisah): `python tools/train_cli.py --program <Program> --template <TemplateID>` — tanpa Qt |

Kalau PyTorch tidak tersedia sama sekali, engine otomatis jatuh ke `SimpleThresholdTrainer` (statistik piksel, akurasi terbatas).

Detail jalur WSL: [docs/MANUAL_TEKNISI.md](docs/MANUAL_TEKNISI.md#training-pytorch).

## Struktur proyek

```
visioninspect/
├── main.py            Entry point (argparse, logging, Qt, MainWindow)
├── core/              Logika inti (kamera, inferensi, training, program, dll)
├── plc/               FX Computer Link + pemetaan I/O
├── api/               Flask REST API (opsional, 127.0.0.1)
├── gui/               PySide6 — main_window, pages/, widgets/, dialogs/, worker thread
├── storage/           SQLite (WAL) + PostgreSQL opsional + retensi
└── utils/             config, i18n, logging

tools/                 train_cli, check_reqs_sync, fx_probe, pg_add_user
tests/                 pytest (core, part_check, soak)
docs/                  ARCHITECTURE + manual operator + manual teknisi
data/                  config.json, logs/, programs/, database (dibuat saat runtime)
```