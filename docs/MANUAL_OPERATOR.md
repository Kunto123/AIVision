# VisionInspect — Manual Operator

## Pengenalan

VisionInspect adalah sistem inspeksi visual industri berbasis AI. Aplikasi ini membantu operator memeriksa produk secara otomatis menggunakan kamera dan model AI yang telah dilatih dengan contoh produk OK (dan opsional NG).

## Antarmuka

Aplikasi memiliki 5 halaman utama:

### 1. RUN — Mode Inspeksi

**Tampilan utama operator.** Menampilkan:
- **Live View** (besar): Gambar langsung dari kamera
- **Judgement** (raksasa): OK (hijau) atau NG (merah) — terbaca dari jarak 2-3 meter
- **Skor Anomali**: Angka 0.0–1.0 (semakin tinggi = semakin anomali)
- **Counter**: Total inspeksi, jumlah OK, jumlah NG
- **Status PLC**: Terhubung/terputus (hijau/merah)

**Mode Trigger:**
- **Kontinu**: Inspeksi otomatis setiap frame
- **PLC**: Inspeksi dipicu dari sinyal PLC
- **Manual**: Tekan tombol "Trigger Sekarang"

### 2. TEACH — Teaching & Training

**Untuk membuat dan melatih model AI.**

**Langkah-langkah:**
1. **Capture OK**: Ambil gambar produk yang baik (minimal 1, disarankan 10-30)
2. **Capture NG**: (Opsional) Ambil gambar produk cacat
3. **Train**: Tekan tombol TRAIN untuk memulai training
   - Progress bar menunjukkan status
   - Setelah selesai, threshold otomatis dikalibrasi
4. **Slider Threshold**: Geser untuk mengubah sensitivitas deteksi

### 3. HISTORY — Riwayat Inspeksi

**Daftar semua hasil inspeksi.**

- **Filter**: Lihat semua, OK saja, atau NG saja
- **Koreksi**: Pilih hasil yang salah → klik "Tandai OK" atau "Tandai NG"
- **Rebuild Model**: Setelah koreksi, rebuild untuk meningkatkan akurasi
- **Rollback**: Kembali ke versi model sebelumnya

### 4. SETTINGS — Pengaturan

**Konfigurasi sistem:**
- **Kamera**: Device index, resolusi, FPS, exposure
- **ROI**: Posisi dan ukuran region yang diperiksa
- **PLC**: Mode (RS232/RS485), port, baudrate, protokol (Modbus/ASCII)
- **Model AI**: Algoritma (PatchCore/EfficientAd), backbone
- **Riwayat**: Retensi data (hari), sampling OK
- **Flask API**: Aktif/nonaktif, port
- **Bahasa**: Indonesia/English

### 5. DIAGNOSTICS — Diagnostik

**Untuk troubleshooting:**
- **Live Logs**: Output log real-time
- **Performa**: RAM, CPU, FPS kamera, latensi inferensi
- **Thread Status**: Status camera, inference, PLC, training
- **Tes PLC**: Kirim frame uji ke PLC

## Alur Kerja Teaching

```
1. Setup kamera → atur posisi dan fokus
2. Buka tab TEACH
3. Capture 10-30 gambar OK
4. (Opsional) Capture beberapa gambar NG
5. Tekan TRAIN → tunggu selesai
6. Cek threshold di histogram
7. Buka tab RUN untuk verifikasi
```

## Alur Koreksi (Redefinition)

```
1. Buka tab HISTORY
2. Pilih hasil yang salah
3. Klik "Tandai OK" atau "Tandai NG"
4. Klik "Rebuild Model" → model baru dibuat dengan data koreksi
5. Model baru otomatis digunakan (hot-swap)
```

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Kamera tidak terdeteksi | Cek device index di SETTINGS, coba 0, 1, 2 |
| Gambar gelap/terlalu terang | Atur exposure di SETTINGS |
| Training gagal | Pastikan ada minimal 1 gambar OK |
| False positive (OK dinyatakan NG) | Turunkan threshold (geser ke kiri) |
| False negative (NG dinyatakan OK) | Naikkan threshold (geser ke kanan) |
| PLC tidak terhubung | Cek kabel, port COM, baudrate |
| Aplikasi lambat | Turunkan resolusi kamera, gunakan EfficientAd |
