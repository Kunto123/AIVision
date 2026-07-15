# Progress — VisionInspect

## Status Tahap

| Tahap | Status | Catatan |
|-------|--------|---------|
| Part Presence Check — Edge Ratio Fix | ✅ Selesai 2026-07-16 | Lihat di bawah |

---

## Sesi: 2026-07-16 — Part Presence Check: Edge Ratio + Fail-Safe

### Masalah
Di RUN page, metode "Tepi (Canny)" pada Part Presence Check selalu mengembalikan "part ready" meski kamera dihalangi jari, sehingga pipeline langsung lanjut ke Tahap 2 (QC) dan Tahap 1 seolah di-skip.

### Diagnostik (Tahap 1)
Penyebab terkonfirmasi:
- **Konfig**: `edge_threshold` di template config = **0.8** (bukan 0.08). Spinbox range 0.001–1.0 memungkinkan nilai ini.
- **Akibat**: `edge_score = |live_edge - master_edge|` maksimum skenario cuma ~0.37. Dengan threshold 0.8, semua kondisi jadi "ready" (`0.37 < 0.8`).
- **Master stats real**: `master_edge_density = 0.05765`, sehingga background kosong punya diff = 0.05765 yang < 0.08 (absolute diff), jadi tetap "ready".

### Perbaikan (Tahap 2)

1. **Metrik edge diubah ke rasio relatif** (`part_check.py::evaluate_part_presence`):
   - `edge_score = |live - master| / max(master, MIN_EDGE_FLOOR=0.01, EPSILON=1e-7)`
   - Lebih stabil lintas kondisi pencahayaan dan ukuran ROI.
   - Denominator dicegah blow-up via `MIN_EDGE_FLOOR` (0.01, setara 1% edge density).
   - Default `edge_threshold` diubah jadi **0.5** (50% perubahan relatif).

2. **Conditional computation**:
   - Method `"edge"`: hanya hitung edge score, **tidak akses** `master_mean_bgr`/`std_bgr`.
   - Method `"color"`: hanya hitung color score, **tidak akses** `master_edge_density`.
   - Method `"both"`: hitung keduanya.
   - Master field yang None → return `PartCheckResult(ready=False, reason="no_master")`, bukan exception.

3. **Fail-safe** (`main_window.py`):
   - Sudah benar: `pc_result is None` → `set_waiting_for_part()` + return.
   - Exception `evaluate_part_presence` → block QC (tidak fail-open). ✓
   - Hanya ditambahkan log di exception handler, sudah diverifikasi.

4. **UI** (`teach_page.py`):
   - Spinbox `_pc_edge_th_spin`: range 0.001–10.0, default 0.5, step 0.1.
   - Tooltip Bahasa Indonesia: "Ambang perubahan edge relatif (rasio)..."
   - Setter di `set_part_check_config` pakai default 0.5.

5. **Konfig template** diupdate: `edge_threshold`: 0.5.

6. **Logging DEBUG** dihapus dari semua file.

### File yang Diubah
- `visioninspect/core/part_check.py` — ratio metric, conditional computation, MIN_EDGE_FLOOR
- `visioninspect/gui/main_window.py` — hapus DEBUG logs
- `visioninspect/gui/pages/teach_page.py` — spinbox default, range, tooltip
- `data/programs/Default/templates/template_1/config.json` — edge_threshold 0.5
- `tests/test_part_check.py` — 8 test baru (threshold boundary, None fields)
- `tests/test_core.py` — update assertion 0.08→0.5

### Test Status
- **68/68 PASS** (32 part_check + 36 core + lainnya)
- 8 test baru:
  - `test_edge_ready_near_threshold`
  - `test_edge_missing_edge_density`
  - `test_color_missing_mean_bgr`
  - `test_color_missing_std_bgr`
  - `test_edge_method_ignores_color_fields`
  - `test_color_method_ignores_edge_fields`
  - plus 2 existing extended

### Definition of Done Checklist
- [x] Fungsi berjalan end-to-end ✓
- [x] Error handling + logging di setiap boundary ✓
- [x] Tidak memblokir main thread GUI ✓ (pure numpy/cv2)
- [x] Unit/integration test terkait ditulis dan PASS (68/68) ✓
- [x] Tidak melanggar §3 (satu proses multi-thread, PySide6) dan §4 (scope) ✓
- [x] Parameter dapat dikonfigurasi (spinbox) ✓
- [x] Tidak hardcode; nilai default di DEFAULT_PART_CHECK_CONFIG ✓
- [x] Tidak merusak metode "color" dan "both" ✓

### Langkah Berikutnya
- User perlu restart aplikasi untuk memuat konfigurasi baru (atau switch template sekali untuk trigger `_refresh_part_check_gate_cache`).
- Monitor log untuk edge ratio aktual di kamera nyata.

---

## Sesi: 2026-07-16 (sesi 2) — Diagnostik Akar B + Peringatan Edge Method

### Diagnostik Final (data kamera nyata)

Dari log `PC_EVAL` dengan 3 kondisi gate:

| Kondisi | live_edge | edge_score | edge_ready | Verdict |
|---------|-----------|------------|------------|---------|
| Part terpasang | 0.053–0.061 | 0.006–0.15 | True | ✅ Normal |
| Area kosong | 0.000 | 1.0 | False | ✅ Benar |
| Part bergeser | ~0.058 | ~0.08 | True | ❌ **False positive** |

**Akar B terkonfirmasi**: edge density Canny tidak bisa membedakan part di posisi benar vs bergeser. Part dengan tekstur permukaan homogen menghasilkan edge density sama di mana pun posisinya dalam gate ROI.

### Perbaikan (untuk Akar B)

1. **Peringatan di `_on_capture_master()`**: log warning saat user capture master dengan metode "edge" — menjelaskan keterbatasan deteksi posisi.
2. **Tooltip metode combo** di TEACH: jelaskan perbedaan metode (Warna vs Tepi vs Keduanya).
3. **`docs/KNOWN_ISSUES.md`**: dokumentasi keterbatasan metode tepi + data referensi.

### File yang Diubah
- `visioninspect/gui/main_window.py` — warning saat capture master dg metode edge
- `visioninspect/gui/pages/teach_page.py` — tooltip metode combo
- `docs/KNOWN_ISSUES.md` — dibuat (dokumentasi keterbatasan)
- `docs/PROGRESS.md` — update sesi ini
- `visioninspect/utils/logging_setup.py` — restore ke konfigurasi normal (DEBUG sementara dihapus)

### Test Status
- **68/68 PASS** — semua test lama dan baru tetap lolos
