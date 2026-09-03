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

Saat pertama dijalankan, `run.bat` otomatis membuat venv `.vision\`, meng-install `requirements.txt`, meng-set `HF_HUB_OFFLINE=1`, dan mengarahkan folder data ke `data\` di dalam proyek — lalu menjalankan aplikasi.

Setup manual (kalau perlu):

```batch
python -m venv .vision
.vision\Scripts\python.exe -m pip install -r requirements.txt
```

> ⚠️ Jangan pakai launcher `py`. Selalu `.vision\Scripts\python.exe` atau aktifkan venv dulu (`.vision\Scripts\activate`).

Login pertama kali (tanpa PostgreSQL): user `admin`, password `admin` — wajib ganti saat login pertama. Kalau PostgreSQL aktif, akun diambil dari `qc_user_accounts` (tambah user lewat `tools/pg_add_user.py`).

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

## Struktur requirements

| File | Untuk | Isi |
|------|-------|-----|
| `requirements.txt` | Umum / PC edge | Inferensi OpenVINO + GUI + auth + Flask API + `fxplc`. Sudah termasuk `anomalib` + `torch` CPU (dipakai saat import). |
| `requirements_edge.txt` | PC edge (inference-only) | Praktis sama dengan `requirements.txt`. Training YOLO tetap di PC dev. |
| `requirements_dev.txt` | PC dev / training | Tambahan `ultralytics` (YOLO), `lightning`, `nncf` (INT8) + tooling test. |

## Menjalankan

```batch
run.bat                                    :: cara normal
.vision\Scripts\python.exe run.py --log-level DEBUG
```

Opsi CLI (`run.py` / `visioninspect/main.py`): `--config <path>`, `--data-dir <path>`, `--log-level DEBUG|INFO|WARNING|ERROR`, `--version`.

**edge_mode** — set `"edge_mode": true` di `data\config.json` supaya `torch` tidak ikut dimuat saat start. PC edge inference-only wajib pakai ini.

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

tools/                 train_cli, bundling_weights, fx_probe, pg_add_user, offline bundle
tests/                 pytest (core, part_check, soak)
docs/                  ARCHITECTURE + manual operator + manual teknisi
data/                  config.json, logs/, programs/, database (dibuat saat runtime)
```