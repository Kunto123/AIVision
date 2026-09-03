# VisionInspect — Manual Operator

Sistem inspeksi visual otomatis. Kamera memotret part, model AI (dilatih dari contoh part OK) memutuskan **OK** atau **NG**, hasil dikirim ke PLC.

## Login

Saat aplikasi dibuka, masuk dengan username/password atau tap kartu RFID.

- **Operator** — hanya bisa membuka tab **RUN**.
- **Admin** — semua tab (TEACH, HISTORY, SETTINGS, DIAGNOSTICS, Akun, I/O Settings).

Login pertama kali (instalasi baru tanpa PostgreSQL): `admin` / `admin`, lalu wajib ganti password. Kalau sistem pakai PostgreSQL terpusat, akun dibuat teknisi/admin di server.

## Halaman

### RUN — inspeksi (operator)

- **Live View** — gambar kamera + kotak ROI (hijau = area QC, biru = gate part-check).
- **Judgement 2 baris** — baris atas = tahap (`MENUNGGU PART` / `JUDGEMENT`), baris bawah = status OK/NG tahap itu (besar, hijau/merah).
- **Skor** — 0.00–1.00 (semakin tinggi = semakin mirip part OK).
- **Counter OK / NG**.
- **Status PLC** — hijau terhubung / merah terputus.
- **Selektor template** — pilih jenis part yang sedang diperiksa (harus sama dengan pilihan di TEACH).
- Tahap `MENUNGGU PART` = gate part-check belum melihat part di posisi. `PART-CHECK BELUM LENGKAP` = konfigurasi gate belum selesai, minta admin melengkapi di TEACH.
- Kalau **Konfirmasi OK** diaktifkan (Settings, N > 1): baris bawah menampilkan progres mis. `OK 2/3` (kuning) sampai N frame OK terkumpul, baru jadi `OK` hijau.
- Tahap `PART TERHALANG` = part sempat tidak terbaca beberapa frame (tangan/bayangan lewat). Sistem menahan verdict — kalau part muncul lagi dalam N frame, pemeriksaan lanjut tanpa mengulang.

Mode trigger (diatur admin di SETTINGS): **Kontinu** (inspeksi tiap frame) atau **PLC** (inspeksi saat ada sinyal trigger).

### TEACH — teaching & training (admin)

1. **Capture / Import OK** — kumpulkan foto part baik (min. beberapa, disarankan 10–30).
2. **Capture / Import NG** — opsional, foto part cacat.
3. Atur **ROI** (area yang diperiksa) dan opsional **Gate ROI** + Part Presence Check.
4. Tekan **TRAIN** — tunggu progress selesai; threshold dikalibrasi otomatis.
5. Cek **histogram** skor OK vs NG; geser **slider threshold** kalau perlu.
6. Buka **RUN** untuk verifikasi.

Engine (YOLO / PatchCore / EfficientAd) dipilih di TEACH. **YOLO dianjurkan** untuk hasil paling stabil.

### HISTORY — riwayat & koreksi (admin)

- Tabel semua hasil, filter OK / NG / semua.
- **Koreksi**: pilih baris yang salah → "Tandai OK" / "Tandai NG".
- **Rebuild Model**: setelah beberapa koreksi, rebuild → model baru otomatis dipakai (hot-swap).
- **Rollback**: kembali ke versi model sebelumnya.

### SETTINGS — konfigurasi (admin)

Kamera (device, resolusi, FPS, exposure) · ROI · PLC (port, baudrate) · model & threshold · retensi history · Flask API · bahasa.

**Penghitungan Part** — *Jarak minimum antar hitungan* (cooldown) dan *Konfirmasi OK — N frame berturut* (gate + judgement baru mengeluarkan OK setelah N hasil infer OK berturut; NG apa pun mereset; 1 = mati; mode Auto Sequence saja).

### DIAGNOSTICS — troubleshooting (admin)

Log live, RAM/CPU, FPS kamera, latensi inferensi, status thread, tes kirim sinyal PLC. Auto-refresh ±2 detik.

### Akun / I/O Settings (admin)

CRUD user + bind RFID · pemetaan coil PLC + mode output hasil (latching / one-shot) + monitor coil live.

## Alur kerja singkat

**Teaching:** setup kamera → tab TEACH → capture 10–30 OK → (opsional NG) → TRAIN → cek histogram → verifikasi di RUN.

**Koreksi:** tab HISTORY → pilih hasil salah → tandai → Rebuild Model → model baru dipakai otomatis.

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Kamera tidak terdeteksi | SETTINGS → coba device index 0, 1, 2 |
| Gambar gelap / terlalu terang | SETTINGS → atur exposure |
| Training gagal | Pastikan ada cukup gambar OK; untuk YOLO/anomali butuh PyTorch (lihat teknisi) |
| OK dinyatakan NG (false NG) | Turunkan threshold (slider ke kiri) atau tambah foto OK lalu retrain |
| NG dinyatakan OK (false OK) | Naikkan threshold (slider ke kanan) atau tambah foto NG lalu retrain |
| Selalu "⏳ Menunggu Part" | Gate ROI / Part Presence Check salah setel — minta admin cek di TEACH |
| PLC tidak terhubung | Cek kabel, port COM, baudrate (lihat teknisi) |
| Aplikasi lambat | Turunkan resolusi kamera; pakai engine YOLO atau EfficientAd |

## Lampu Fault PLC menyala — prosedur pemulihan

Lampu fault di panel menyala = PLC mendeteksi sistem inspeksi tidak sehat (aplikasi mati, kamera lepas, model tak termuat, atau komunikasi putus > ±10 detik). Selama lampu menyala, **lini tidak memvonis apa pun** — ini disengaja supaya part tidak lolos tanpa diperiksa.

**Cara pulihkan:**

1. Pastikan VisionInspect jalan normal lagi (kamera aktif, model termuat, status PLC hijau di layar RUN).
2. Tekan tombol **reset di panel** selama **1–2 detik**, lalu lepas.
3. Lampu fault padam — lini siap menerima part lagi.

**Catatan:**

- Untuk sekalian me-nol-kan counter OK/NG di panel: tahan tombol reset **≥ 3 detik**.
- Lampu fault **tidak padam sendiri** meski aplikasi sudah normal — wajib tekan reset. Ini benar, bukan rusak.
- Saat admin melakukan training, heartbeat berhenti → lampu fault menyala setelah ±10 detik. Normal. Pulihkan dengan langkah di atas setelah training selesai.
- Kalau lampu fault menyala berulang tanpa training: periksa kamera (sering lepas?) dan `data/logs/app.log`, laporkan ke teknisi.
