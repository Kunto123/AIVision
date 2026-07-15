# Known Issues — VisionInspect

## Part Presence Check — Metode "Tepi (Canny)" tidak bisa deteksi pergeseran posisi

**Akar masalah:** `compute_edge_density()` menghitung rasio pixel tepi terhadap total pixel di area gate ROI — hanya JUMLAH, bukan POSISI tepi. Akibatnya:

- Part dengan tekstur permukaan homogen (metal, plastik, PCB) menghasilkan `live_edge` yang hampir sama di mana pun part berada dalam ROI.
- Part bergeser 10–50px dalam ROI → `edge_score` < threshold → tetap dianggap "ready".
- Metode hanya bisa membedakan "ada objek" vs "kosong" (background polos), TIDAK bisa membedakan "posisi benar" vs "bergeser".

**Saran:**
1. Gunakan metode **Warna (mean/std)** bila part punya warna kontras dengan latar.
2. Gunakan metode **Keduanya (AND)** untuk kombinasi warna + tepi.
3. Jika harus pakai metode Tepi, pastikan gate ROI hanya mencakup area dengan fitur tepi yang unik (mis. sudut part, lubang, text/code) — bukan permukaan rata.

**Referensi data (2026-07-16):**
- Master edge density: 0.053
- Part di posisi: live_edge 0.053–0.061 → edge_score 0.006–0.15 → ready ✅
- Part bergeser: live_edge ~0.058 → edge_score ~0.08 → ready ❌
- Area kosong: live_edge 0.0 → edge_score 1.0 → not-ready ✅
