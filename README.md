# VisionInspect — Sistem Inspeksi Visual Industri Berbasis AI

**VisionInspect** adalah aplikasi desktop untuk inspeksi visual industri berbasis AI yang berjalan **100% lokal di satu PC tanpa GPU**. Menggunakan **Anomalib** (PatchCore/EfficientAd) sebagai fondasi model dan **OpenVINO** untuk inferensi CPU real-time, dengan GUI PySide6 yang responsif.

## Quick Start (PC Baru — dengan internet)

```batch
:: 1. Clone project
git clone <repo-url> VisionInspect
cd VisionInspect

:: 2. Setup otomatis (buat venv + install semua dependencies)
setup.bat

:: 3. Download pretrained weights untuk backbone model
.vision\Scripts\python.exe tools\bundling_weights.py

:: 4. Jalankan aplikasi
run.bat
```

> ⚠️ **PENTING**: Jangan gunakan `py` launcher! Selalu gunakan `.vision\Scripts\python.exe` atau aktivasi venv dulu:
> ```batch
> .vision\Scripts\activate
> python tools\bundling_weights.py
> ```

### Kalau Tidak Punya Internet (Offline Deployment)

Lihat [Offline Deployment](#offline-deployment) untuk panduan lengkap.

---

## Fitur Utama

- 🔍 **Teaching few-shot**: Cukup 10–30 gambar OK untuk membuat model inspeksi
- ⚡ **Inferensi CPU real-time**: OpenVINO INT8 < 100ms per frame (ROI 256×256)
- 🔄 **Redefinition Loop**: Koreksi hasil salah → rebuild model cepat → hot-swap
- 🏭 **Komunikasi PLC**: RS232/RS485 via Modbus RTU atau ASCII protocol
- 📊 **History & Audit Trail**: SQLite WAL mode, retensi configurable
- 🔌 **Flask API internal** (opsional): REST endpoint di 127.0.0.1
- 🎨 **GUI Dark Navy**: 5 tab (RUN, TEACH, HISTORY, SETTINGS, DIAGNOSTICS)
- 🌐 **Bahasa Indonesia + English**: i18n built-in

## Persyaratan Sistem

| Komponen | Minimum | Rekomendasi |
|----------|---------|-------------|
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| CPU | Intel i5 gen 10, 4 core | Intel i7 gen 12+ |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB free | 10 GB SSD |
| Kamera | USB (UVC) | USB 3.0 atau GigE |
| Python | 3.10+ | 3.11 |

## Instalasi Cepat

Lihat [Quick Start](#quick-start-pc-baru--dengan-internet) di atas untuk langkah cepat.

Atau manual:

```bash
# 1. Clone atau extract project
cd VisionInspect

# 2. Buat virtual environment
python -m venv .vision
# Windows: .vision\Scripts\activate
# Linux/WSL: source .vision/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Download Pretrained Weights (Offline)

Anomalib (via `TimmFeatureExtractor`) membutuhkan backbone pretrained dari HuggingFace Hub.
Untuk offline deployment, download weights di PC development lalu copy ke edge PC.

```bash
# Di PC development (ada internet), jalankan dari root proyek:
source .vision/bin/activate
python tools/bundling_weights.py

# Hasil: cache tersimpan di ~/.cache/huggingface/hub/
# Copy folder ini ke PC target:
#   ~/.cache/huggingface/  →  PC target di path yang sama
```

Atau gunakan script bundle otomatis:
```bash
# Di PC development:
tools/prepare_offline_bundle.bat    # Windows

# Hasil: folder offline_bundle/ — siap di-copy ke USB
```

### Offline Deployment (Edge PC — tanpa internet)

```bash
# Di edge PC, dari folder hasil bundle:
offline_bundle\install.bat

# Atau manual:
python -m venv .vision
.vision\Scripts\python.exe -m pip install --no-index --find-links=offline_bundle\wheels -r offline_bundle\requirements.txt
# Copy folder offline_bundle\hf_cache\* ke %USERPROFILE%\.cache\huggingface\
```

Setelah itu jalankan via `run.bat` (yang sudah otomatis set `HF_HUB_OFFLINE=1`).

## Menjalankan

```bash
source .vision/bin/activate
python run.py
```

Atau dengan opsi:
```bash
python run.py --log-level DEBUG --config /path/to/config.json
```

## Testing

```bash
source .vision/bin/activate
pytest tests/ -v

# Soak test (simulasi 10.000 frame)
python tests/test_soak.py
```

## Struktur Proyek

```
visioninspect/
├── main.py                  # Entry point
├── core/                    # Logika inti
│   ├── camera.py            # Thread kamera
│   ├── inference.py         # OpenVINO engine
│   ├── training.py          # Anomalib pipeline
│   ├── program.py           # Manajemen program
│   ├── redefinition.py      # Koreksi & rebuild
│   └── watchdog.py          # Thread monitor
├── plc/                     # Komunikasi PLC
│   ├── serial_manager.py    # RS232/RS485
│   ├── modbus_rtu.py        # Modbus register map
│   └── ascii_protocol.py    # ASCII STX/ETX
├── api/
│   └── flask_app.py         # REST API (opsional)
├── gui/
│   ├── main_window.py       # Window utama
│   ├── pages/               # 5 halaman tab
│   │   ├── run_page.py
│   │   ├── teach_page.py
│   │   ├── history_page.py
│   │   ├── settings_page.py
│   │   └── diagnostics_page.py
│   ├── widgets/             # Widget reusable
│   └── theme.qss            # Dark navy theme
├── storage/
│   ├── db.py                # SQLite WAL
│   └── retention.py         # Retensi data
└── utils/
    ├── config.py            # Manajemen konfigurasi
    ├── i18n.py              # Internasionalisasi
    └── logging_setup.py     # Logging terstruktur
tools/
├── plc_simulator.py         # Simulator PLC
└── bundling_weights.py      # Download weight offline
docs/
├── MANUAL_OPERATOR.md       # Panduan operator
├── MANUAL_TEKNISI.md        # Wiring & troubleshooting
└── ARCHITECTURE.md          # Dokumentasi teknis
```

## PLC Communication

### Modbus RTU Register Map

| Register | Address | Description |
|----------|---------|-------------|
| Holding 0 | 0x00 | System status (0=idle, 1=running, 2=training, 3=error) |
| Holding 1 | 0x01 | Last result (0=none, 1=OK, 2=NG) |
| Holding 2 | 0x02 | Last score × 100 (integer) |
| Holding 3 | 0x03 | Total counter (16-bit rolling) |
| Holding 4 | 0x04 | NG counter (16-bit rolling) |
| Holding 10 | 0x0A | Active program number |
| Coil 0 | 0x00 | Trigger inspection (set → reset) |
| Coil 1 | 0x01 | Reset counters (set → reset) |

### ASCII Protocol

Format frame: `STX <CMD> [DATA] ETX <CHECKSUM>`

- `TRG` — Trigger inspeksi
- `RES,OK,<score>` — Hasil OK
- `RES,NG,<score>` — Hasil NG
- `PRG,<n>` — Ganti program
- `STA` — Status request

## Testing dengan Simulator PLC

```bash
# 1. Install socat
sudo apt-get install socat   # Linux/WSL

# 2. Buat virtual serial pair
socat -d -d PTY,link=/tmp/ttyV0 PTY,link=/tmp/ttyV1

# 3. Jalankan simulator (terminal 1)
python tools/plc_simulator.py --port /tmp/ttyV0 --protocol ascii

# 4. Jalankan VisionInspect dengan port /tmp/ttyV1 (terminal 2)
```

## Performa Target

| Metrik | Target | Keterangan |
|--------|--------|------------|
| Inferensi (OpenVINO INT8) | < 100 ms | ROI 256×256, PatchCore/resnet18 |
| Inferensi (EfficientAd-S) | < 30 ms | ROI 256×256 |
| Rebuild (50 gambar) | < 2 menit | CPU 4-core |
| RAM idle | < 800 MB | |
| RAM running | < 1.5 GB | |
| Start-to-ready | < 15 detik | |

## Lisensi

Proprietary — Internal use.
