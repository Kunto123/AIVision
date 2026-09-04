"""
VisionInspect — Pemetaan I/O PLC (netral terhadap transport)
============================================================
Penomoran coil dan perilaku output hasil. Dipisahkan dari implementasi
transport supaya nomor yang sama berlaku apa pun jalurnya — pada FX Computer
Link, nomor coil dipakai langsung sebagai nomor relay M (coil 1 → M1).

⚠️ GANTI DI SINI (atau lewat UI I/O Settings) kalau alamat berubah.
"""

from typing import Optional


def build_io_map(plc_config: Optional[dict]) -> dict:
    """IO map default (coil/register), override dari config. Kontrak ladder:
    sistem HANYA kirim OK + heartbeat; NG diputuskan PLC dari ketiadaan OK."""
    default = {
        "outputs": {
            "result_ok": 1,       # M1: ON = part OK (pulse)
            # Coil result_ng DIHAPUS — NG diputuskan PLC dari ketiadaan OK
            # dalam jendela waktunya sendiri.
            "part_ready": 3,      # M3: ON = part terdeteksi (pulse saat transisi)
            "busy": 4,            # M4: ON = sistem sedang inspeksi
            # Heartbeat: di-TOGGLE selama sistem sehat. Ladder pantau PERUBAHANnya
            # — diam > N detik = sistem rusak, BUKAN part cacat.
            "heartbeat": 7,
            # M9: pulse saat operator masuk RUN → ladder bersihkan state basi
            # (dijaga kontak /M100). Opt-in lewat plc.reset_on_run_entry.
            "session_reset": 9,
        },
        "inputs": {
            "trigger": 0,         # M0: PLC minta 1 siklus inspeksi
            "reset_result": 5,    # M5: reset counter produksi OK/NG
            # M6: ganti TEMPLATE aktif (nomor dari program_register).
            # Nama lama "switch_program" masih dikenali untuk config lama.
            "switch_template": 6,
            # M8: PLC memvonis NG → sistem menambah counter NG di layar dan
            # membersihkan state siklus (siap trigger part berikutnya).
            "ng_from_plc": 8,
        },
        "program_register": 10,   # D10: nomor template tujuan
    }
    io = plc_config.get("io_map") if plc_config else None
    # Guard: config `plc` boleh tanpa key `io_map` (config lama/ditulis tangan)
    if not isinstance(io, dict):
        return default
    for section in ("outputs", "inputs"):
        if isinstance(io.get(section), dict):
            default[section].update(io[section])
    if isinstance(io.get("program_register"), int):
        default["program_register"] = io["program_register"]
    return default


# Mode output hasil OK (ala Keyence IV3): latching = LEVEL sampai hasil
# berikutnya/reset PLC; one_shot = pulse singkat.
DEFAULT_IO_MODE: dict = {
    # DEFAULT one_shot: tiap part = SATU kedipan, "diam" berarti gagal.
    # Latching bikin vonis part sebelumnya bisa terbaca sebagai vonis baru.
    "output_mode": "one_shot",         # "latching" | "one_shot"
    "one_shot_on_time_ms": 300,        # durasi coil ON pd one_shot
    "one_shot_delay_ms": 0,            # tunda sebelum coil ON (one_shot)
    "part_ready_output": False,        # nyalakan utk kirim coil part_ready
    "busy_output": False,              # nyalakan utk kirim coil busy
}


def build_io_mode(plc_config: Optional[dict] = None) -> dict:
    """Perilaku I/O default, override dari config `plc.io_mode`.
    Field dijamin lengkap (backward-compat config lama tanpa io_mode)."""
    out = DEFAULT_IO_MODE.copy()
    pl = plc_config or {}
    mode = pl.get("io_mode")
    if isinstance(mode, dict):
        for key, value in mode.items():
            if key in out and value is not None:
                out[key] = value
    if out["output_mode"] not in ("latching", "one_shot"):
        out["output_mode"] = "latching"
    out["one_shot_on_time_ms"] = max(0, int(out["one_shot_on_time_ms"]))
    out["one_shot_delay_ms"] = max(0, int(out["one_shot_delay_ms"]))
    out["part_ready_output"] = bool(out["part_ready_output"])
    out["busy_output"] = bool(out["busy_output"])
    return out
