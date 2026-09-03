# VisionInspect — Manual Teknisi

## PLC — Mitsubishi FX Computer Link

VisionInspect berkomunikasi dengan PLC FX lewat **protokol port pemrograman** (Computer Link) — jalur serial yang sama dipakai GX Works2. Library: `fxplc`.

> Modbus RTU & protokol ASCII **sudah tidak dipakai**. FX3U tanpa adaptor `-MB` khusus tidak menyediakan MODBUS slave (`D8400`/`D8401` tetap nol). Computer Link bekerja di port yang sudah terbukti hidup.

### Parameter serial

| Parameter | Nilai | Bisa diatur? |
|-----------|-------|--------------|
| Format | **7E1** (7 data bit, even parity, 1 stop) | Tidak — terkunci protokol |
| Port | `COM1`, `COM3`, … | Ya (SETTINGS → PLC) |
| Baudrate | 9600 (umum) | Ya |

Konverter: USB-to-RS232 atau USB-to-RS485 (sesuai port PLC). Setelah driver terpasang, port muncul sebagai `COMx` di Device Manager → Ports.

### Wiring RS232 (point-to-point)

```
PC (DB9)          PLC (port pemrograman)
  TX (pin 3) ───── RX
  RX (pin 2) ───── TX
  GND (pin 5) ──── GND
```

### Wiring RS485 (half-duplex, kalau PLC pakai port RS485)

```
PC (USB-RS485)     PLC (RS485)
  A / D+ ────────── A / D+
  B / D- ────────── B / D-
  GND ───────────── GND
```

- Pasang resistor terminasi **120 Ω** di kedua ujung bus kalau kabel > 10 m.
- Kabel panjang: pakai shielded twisted pair.
- Kalau data korup: turunkan baudrate, tambah delay TX.

## Pemetaan I/O (coil ↔ relay M)

Nomor coil dipetakan **langsung** ke relay M PLC (coil 1 → M1). Default di `visioninspect/plc/io_map.py`, bisa di-override lewat tab **I/O Settings**.

### Output — sistem tulis, PLC baca

| Coil | Relay | Nama | Arti |
|------|-------|------|------|
| 1 | M1 | `result_ok` | Part OK (pulse). **Tidak ada coil NG** — lihat catatan. |
| 3 | M3 | `part_ready` | Part terdeteksi di gate (pulse saat transisi) |
| 4 | M4 | `busy` | Sistem sedang inspeksi |
| 7 | M7 | `heartbeat` | Di-toggle ±1 Hz selama sistem sehat |
| 9 | M9 | `session_reset` | Pulse saat operator masuk RUN (opt-in) |

### Input — PLC tulis, sistem baca

| Coil | Relay | Nama | Arti |
|------|-------|------|------|
| 0 | M0 | `trigger` | Minta 1 siklus inspeksi |
| 5 | M5 | `reset_result` | Reset counter produksi OK/NG |
| 6 | M6 | `switch_template` | Ganti template aktif (nomor dari D10) |
| 8 | M8 | `ng_from_plc` | PLC memvonis NG → sistem tambah counter NG + bersihkan state siklus |
| D10 | — | `program_register` | Nomor template tujuan (1 = template pertama) |

### Kontrak dengan ladder

1. **Sistem hanya mengirim OK.** NG diputuskan PLC dari **ketiadaan** sinyal OK dalam jendela waktunya sendiri.
2. **Heartbeat** memisahkan "part cacat" dari "sistem rusak". Ladder memantau *perubahan* coil heartbeat; diam > N detik = sistem rusak → nyalakan lampu fault, hentikan vonis.
3. Mode output hasil (I/O Settings): `one_shot` (default, pulse per part) atau `latching` (level sampai hasil berikutnya).
4. Mode `switch_template`: `cycle` (tiap sinyal maju satu template) atau `register` (pindah ke nomor di D10).

## Probe / uji koneksi PLC

```batch
.vision\Scripts\python.exe tools\fx_probe.py --port COM11
.vision\Scripts\python.exe tools\fx_probe.py --port COM11 --baud 9600
```

Membaca blok special relay M8000+ untuk memastikan PLC menjawab. Kalau `fxplc` belum terpasang: `pip install "fxplc @ git+https://github.com/KrystianD/fxplc.git"` (atau dari `vendor/`).

Di tab **DIAGNOSTICS** ada juga tombol tes kirim sinyal ke PLC.

## Training (PyTorch)

Engine YOLO / PatchCore / EfficientAd butuh PyTorch. Di Windows, torch sering gagal load (`WinError 1114` — TLS slot exhaustion). Dua jalur:

| Jalur | Cara |
|-------|------|
| Windows langsung | Tombol TRAIN di TEACH — jalan kalau torch Windows sehat |
| **WSL** (disarankan untuk training) | Jalankan `tools/train_cli.py` di venv WSL, tanpa Qt (lihat langkah di bawah) |
| Tanpa torch | Engine otomatis jatuh ke `SimpleThresholdTrainer` (z-score piksel, akurasi terbatas) |

### Training via WSL

Setup venv WSL sekali (dari root proyek di dalam WSL):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements_dev.txt
```

Tiap kali retrain:

```bash
# 1. Capture OK/NG lewat GUI di Windows, lalu TUTUP aplikasi
# 2. Di WSL, dari root proyek:
source .venv/bin/activate
python tools/train_cli.py --program <Program> --template <TemplateID>
# 3. Buka lagi aplikasi di Windows — model baru otomatis dipakai
```

Nama template ada di `data/programs/<Program>/templates/`.

`edge_mode: true` di `data\config.json` → torch tidak dimuat saat start (PC edge inference-only, hemat RAM + waktu boot).

## Deployment offline

Di PC dev (ada internet):

```batch
.vision\Scripts\python.exe tools\bundling_weights.py    :: unduh backbone pretrained → cache HuggingFace
tools\prepare_offline_bundle.bat                       :: buat offline_bundle\ (wheels + cache HF)
```

Di edge PC (tanpa internet): copy `offline_bundle\`, jalankan `tools\install_offline.bat`, lalu `run.bat` (`run.bat` sudah set `HF_HUB_OFFLINE=1`).

## PostgreSQL (opsional)

Untuk deployment multi-PC: akun aplikasi (`qc_user_accounts`) dan push hasil inspeksi (`qc_inspection_push`) dipusatkan di PostgreSQL. Aktifkan di config `postgresql.enabled` + isi host/db/user/password (password otomatis dienkripsi via DPAPI/Fernet, tidak plaintext di disk).

- Tabel harus **sudah ada** di server — aplikasi hanya query, tidak `CREATE`.
- Tambah/ubah user: `.vision\Scripts\python.exe tools\pg_add_user.py` (hash SHA-256 + pepper, kompatibel dengan login aplikasi).
- Kalau server PG tidak terjangkau saat login, aplikasi fallback ke akun SQLite lokal supaya lini tidak berhenti.

## Troubleshooting

### Serial tidak terdeteksi
- Cek driver USB-Serial (FTDI, CH340, CP210x) di Device Manager → Ports (COM & LPT).

### PLC tidak menjawab
- Cek wiring TX/RX (sering terbalik).
- Pastikan format **7E1** — PLC harus di-set sama (parameter port pemrograman FX default sudah 7E1).
- Uji dengan `tools\fx_probe.py`.
- Pastikan PLC dalam RUN.

### Aplikasi tidak bisa start
- Cek `data\logs\app.log`.
- Pastikan port kamera tidak dipakai aplikasi lain.
- Reset config: hapus `data\config.json` (akan dibuat ulang dari default).

### Training gagal di Windows
- Kemungkinan `WinError 1114`. Pakai jalur WSL (lihat *Training via WSL* di atas).

## Performa — target

| Metrik | Target | Catatan |
|--------|--------|---------|
| Inferensi | < 100 ms | ROI 256×256, PatchCore INT8 |
| Inferensi | < 30 ms | EfficientAd-S / YOLO-cls kecil |
| Rebuild model | < 2 menit | ~50 gambar, CPU 4-core |
| RAM idle | < 800 MB | edge_mode |
| RAM running | < 1.5 GB | |
| Start-to-ready | < 15 detik | |
| Uptime | 24/7 | tanpa restart |

Monitor lewat tab **DIAGNOSTICS**: RAM (cek memory leak), latensi inferensi (avg + P95), FPS kamera, status thread.

## Packaging

Build one-folder pakai PyInstaller. File `.spec` dikelola per-mesin build (tidak ikut repo karena berisi path lokal):

```batch
.vision\Scripts\python.exe -m PyInstaller <VisionInspect.spec>
```

Hasil di `dist\VisionInspect\`.
