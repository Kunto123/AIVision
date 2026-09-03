# VisionInspect — Arsitektur

Aplikasi **satu proses, multi-thread**. Semua komunikasi antar-thread lewat Qt signal/slot + antrean berukuran terbatas (drop-oldest). GUI tidak pernah memanggil fungsi blocking.

## Thread

| Thread | Isi | Catatan |
|--------|-----|---------|
| Main (GUI) | Event loop PySide6, semua halaman | `gui/main_window.py` |
| Camera | Grab frame OpenCV, hitung FPS | `gui/camera_worker.py` → `core/camera.py` |
| Inference | Jalankan engine per ROI, hasil dikirim via signal | `gui/inference_worker.py` → `core/inference.py` |
| PLC poll | Baca/tulis coil FX, reconnect otomatis | `plc/fx_link.py` |
| Watchdog | Pantau thread; restart kalau hang | `core/watchdog.py` |
| Training | Dibuat saat tombol TRAIN ditekan, dibuang setelah selesai | `gui/training_worker.py` → `core/training.py` |
| Video replay | Opsional — putar ulang rekaman untuk uji | `gui/video_replay_worker.py` |
| Flask API | Opsional, bind 127.0.0.1 saja | `api/flask_app.py` |

## Pipeline inspeksi (mode RUN)

```
Camera frame
  │
  ├─ 1. Part Presence Check  (core/part_check.py) — gate CV klasik sebelum QC
  │      • disabled   → langsung ke QC (kompatibilitas lama)
  │      • incomplete → BLOK QC + stop timer NG  (fail-safe: cegah NG palsu)
  │      • active     → evaluate_part_presence()
  │            ready=False → "⏳ Menunggu Part", skip QC
  │            ready=True  → lanjut
  │
  ├─ 2. (opsional) YOLO class pre-filter  (core/yolo_filter.py)
  │      Kalau expected_classes diset & part kelasnya tidak terdeteksi → NG (class mismatch),
  │      tanpa masuk scoring. Butuh `ultralytics`; kalau tak terpasang, langkah ini dilewati.
  │
  ├─ 3. QC inference per ROI  (core/inference.py)
  │      Engine dipilih otomatis dari file meta di samping model.xml:
  │        • yolo      → klasifikasi OK/NG per crop (probabilitas kelas)   [dianjurkan]
  │        • anomaly   → PatchCore / EfficientAd → skor anomali → similarity + heatmap
  │        • simple    → z-score statistik piksel (fallback tanpa PyTorch)
  │      Output per ROI: score [0..1] (1.0 = mirip OK), judgement OK/NG, heatmap (anomaly saja)
  │
  ├─ 4. Agregasi ROI → OK/NG keseluruhan
  │
  └─ 5. Kirim ke PLC + update counter + simpan history
         • ke PLC: HANYA pulse coil OK. NG diputuskan PLC dari ketiadaan OK.
         • history: semua NG disimpan; OK di-sampling (default 10%)
```

**Fail-safe:** error inferensi apa pun → judgement NG. Part-check incomplete/exception → blok QC. Kamera lepas / model tak termuat → heartbeat ke PLC berhenti → PLC menyalakan lampu fault (lini berhenti memvonis).

## Engine training

| Engine | Butuh | Output | Kapan |
|--------|-------|--------|-------|
| **YOLO** (klasifikasi) | `ultralytics` (PC dev) | `model/openvino/` + `yolo_meta.json` | Default, dianjurkan |
| PatchCore | `anomalib` + torch | OpenVINO IR + `norm.json` (kalibrasi skor) | Few-shot, ada heatmap |
| EfficientAd | `anomalib` + torch | OpenVINO IR + `norm.json` | Lebih cepat dari PatchCore |
| Simple | numpy/opencv saja | `mean.npy` + `std.npy` | Fallback kalau torch mati |

Semua engine di-export ke **OpenVINO** untuk runtime (opsional INT8 PTQ). Model lama dipakai terus sampai model baru siap → **hot-swap atomik** (`InferenceEngine.load_model`).

Training di Windows sering gagal (`WinError 1114` saat torch load). Jalur alternatif: jalankan `tools/train_cli.py` di venv WSL, tanpa Qt (lihat [MANUAL_TEKNISI.md](MANUAL_TEKNISI.md#training-pytorch)).

## Program & Template

```
data/programs/<nama-program>/
├── config.json              program-level (kamera, PLC)
├── metadata.json
└── templates/<template_id>/
    ├── config.json          roi, threshold, algorithm, part-check, yolo_pretrained
    ├── images/{ok,ng}/       foto training
    ├── images/corrections/{ok,ng}/   dari redefinition loop
    ├── model/
    │   ├── openvino/         model.xml + norm.json / yolo_meta.json / model_meta.json
    │   └── openvino_int8/    versi INT8 (opsional)
    └── versions/v1, v2, ...  snapshot untuk rollback
```

- **Program** = satu setup lini. **Template** = satu jenis part dalam program itu.
- Multi-ROI (maks 4), tiap ROI bisa punya threshold sendiri + polygon mask (`core/roi_mask.py`).
- Ganti template lewat GUI atau sinyal PLC (`switch_template`).

## Redefinition loop

```
Pilih baris history salah → tandai koreksi (OK↔NG) → gambar masuk corrections/{ok,ng}
  → Rebuild (dataset gabungan) → versi model baru → hot-swap → audit_log
```

## Penyimpanan

- **SQLite (WAL)** — `storage/db.py`: `inspection_history`, `users`, `counters`, `audit_log`. Zero-config, tahan crash.
- **PostgreSQL** — `storage/postgres_db.py`: opsional, aktif lewat config `postgresql.enabled`. Untuk akun (`qc_user_accounts`) + push hasil terpusat multi-PC.
- **Autentikasi:** kalau PostgreSQL aktif **dan** terjangkau → login lewat PG; kalau tidak → fallback ke tabel `users` SQLite lokal (supaya lini tidak berhenti saat server DB mati).
- **Retensi** — `storage/retention.py`: auto-purge per umur, sampling gambar OK.
- **Config** — JSON di `data/config.json` (atau `~/.visioninspect/config.json`). Atomic write (temp + rename). Default hardcoded di `utils/config.py::Config.DEFAULTS`, di-merge dengan config user saat load.

## PLC — Mitsubishi FX Computer Link

Modbus RTU & ASCII **sudah dihapus**: FX3U tanpa adaptor `-MB` tidak menyediakan MODBUS slave. Transport sekarang memakai protokol port pemrograman (jalur yang sama dengan GX Works2), lewat library `fxplc`.

- Format serial terkunci **7E1**; hanya port + baudrate yang bisa diatur.
- Nomor coil dipetakan langsung ke relay M (coil 1 → M1). Peta di `plc/io_map.py`, bisa di-override lewat tab **I/O Settings**.
- Kontrak dengan ladder: sistem hanya kirim OK + heartbeat; PLC yang pegang timing, urutan, dan vonis NG.

Detail wiring, nomor coil, dan pemulihan lampu fault: [MANUAL_TEKNISI.md](MANUAL_TEKNISI.md).

## REST API (opsional)

Flask di `127.0.0.1:<port>`, auth API key. Aktif lewat config `flask_api.enabled`.

| Endpoint | Method | Fungsi |
|----------|--------|--------|
| `/health` | GET | Ping |
| `/status` | GET | Status sistem |
| `/last_result` | GET | Hasil inspeksi terakhir |
| `/trigger` | POST | Picu satu siklus inspeksi |
| `/history` | GET | Riwayat inspeksi |
| `/program/<name>/activate` | POST | Aktifkan program |

## Stack

Python 3.11 · PySide6 (Qt) · OpenVINO (runtime) · anomalib / ultralytics (training) · OpenCV (kamera) · fxplc (PLC) · SQLite / PostgreSQL · Flask · PyInstaller (packaging one-folder, spec dikelola per-mesin).
