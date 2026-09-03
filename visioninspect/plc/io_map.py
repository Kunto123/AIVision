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
    """IO map default (coil/register) — override dari config.

    Kontrak dengan ladder:
      - Sistem HANYA mengirim OK. NG diputuskan PLC dari ketiadaan OK dalam
        jendela waktunya sendiri.
      - Heartbeat memisahkan "part cacat" dari "sistem rusak"; tanpa itu
        kedua-duanya terlihat sama (sama-sama tidak ada OK).

    outputs  : coil yang SISTEM tulis → PLC baca
               result_ok / heartbeat / part_ready / busy
    inputs   : coil yang PLC tulis → sistem baca
               trigger / reset_result / switch_template / ng_from_plc
    program_register : data register berisi nomor template tujuan
    """
    default = {
        "outputs": {
            "result_ok": 1,       # M1: ON = part OK (pulse)
            # CATATAN: coil result_ng DIHAPUS. NG sepenuhnya diputuskan PLC
            # dari KETIADAAN sinyal OK dalam jendela waktu miliknya, jadi
            # sistem tidak pernah lagi mengirim NG.
            "part_ready": 3,      # M3: ON = part terdeteksi (pulse saat transisi)
            "busy": 4,            # M4: ON = sistem sedang inspeksi
            # Heartbeat: di-TOGGLE berkala selama sistem sehat (model termuat
            # + kamera jalan). Tanpa ini, kamera lepas / aplikasi mati terlihat
            # sama seperti part cacat di sisi PLC — lini terus membuang
            # produksi bagus tanpa ada yang tahu. Ladder memantau
            # PERUBAHANnya: tidak berubah > N detik = sistem rusak, bukan NG.
            "heartbeat": 7,
            # M9: pulse saat operator masuk RUN — ladder membersihkan
            # M100/M110/M114/Y000/Y001/C0/C1/C2 kalau tidak sedang di
            # tengah siklus (ladder yang menjaga lewat kontak /M100, bukan
            # aplikasi). Opt-in lewat plc.reset_on_run_entry — lihat
            # blueprint .claude/blueprint/kenapa-ok-tidak-sampai-plc.md.
            "session_reset": 9,
        },
        "inputs": {
            "trigger": 0,         # M0: PLC minta 1 siklus inspeksi
            "reset_result": 5,    # M5: reset counter produksi OK/NG
            # M6: ganti TEMPLATE aktif; nomornya dibaca dari program_register.
            # Nama lama "switch_program" masih dikenali (config lama) — lihat
            # _on_plc_poll_tick di main_window.
            "switch_template": 6,
            # M8: PLC memvonis NG (tidak ada OK dalam jendela waktunya).
            # Dipakai sistem untuk (1) menambah counter NG di layar supaya
            # cocok dengan lampu, dan (2) membersihkan state siklus agar siap
            # menerima trigger part berikutnya.
            "ng_from_plc": 8,
        },
        "program_register": 10,   # D10: nomor template tujuan
    }
    io = plc_config.get("io_map") if plc_config else None
    # Guard: config `plc` tanpa key `io_map` bukan hal aneh (config lama /
    # ditulis tangan). Sebelum ini baris program_register di bawah membaca
    # `io` tanpa cek dan meledak dengan AttributeError.
    if not isinstance(io, dict):
        return default
    for section in ("outputs", "inputs"):
        if isinstance(io.get(section), dict):
            default[section].update(io[section])
    if isinstance(io.get("program_register"), int):
        default["program_register"] = io["program_register"]
    return default


# Mode output hasil (mirip "Output Settings" di sensor Keyence IV3).
# Aplikasi sudah jadi "sensor": PLC yg pegang timing/urutan. Hasil OK bisa:
#   - latching: dibiarkan LEVEL sampai hasil berikutnya / PLC reset.
#   - one_shot : pulse singkat (`one_shot_delay` lalu ON selama `one_shot_on_time`).
DEFAULT_IO_MODE: dict = {
    # DEFAULT one_shot (bukan latching) — kontrak yang disepakati dengan
    # ladder: tiap part menghasilkan SATU kedipan tersendiri, dan "diam"
    # berarti gagal. Latching membuat hasil part sebelumnya tetap terbaca
    # selama sistem masih menghitung, sehingga vonis basi bisa dianggap baru.
    "output_mode": "one_shot",         # "latching" | "one_shot"
    "one_shot_on_time_ms": 300,        # durasi coil ON pd one_shot
    "one_shot_delay_ms": 0,            # tunda sebelum coil ON (one_shot)
    "part_ready_output": False,        # nyalakan utk kirim coil part_ready
    "busy_output": False,              # nyalakan utk kirim coil busy
}


def build_io_mode(plc_config: Optional[dict] = None) -> dict:
    """Perilaku I/O (default), override dari config `plc.io_mode`.

    Menjamin field selalu lengkap — backward-compat untuk config lama yang
    belum punya io_mode.
    """
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
