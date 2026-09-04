"""Probe FX Computer Link (protokol port pemrograman) — alternatif MODBUS.

    python tools/fx_probe.py --port COM11
    python tools/fx_probe.py --port COM11 --baud 38400
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

try:
    from fxplc.client.FXPLCClient import FXPLCClient
    from fxplc.transports.TransportSerial import TransportSerial
except ImportError:
    sys.exit(
        "fxplc belum terpasang.\n"
        "  pip install git+https://github.com/KrystianD/fxplc.git"
    )


#: Byte pertama blok special relay M8000+ di peta FX Computer Link.
#: M8000 = byte ini bit 0; M8013 = byte berikutnya (offset 1) bit 5.
SPECIAL_M_BASE = 0x01E0


async def probe(port: str, baud: int, timeout: float) -> int:
    print(f"Membuka {port} @ {baud} …")
    try:
        transport = TransportSerial(port, baudrate=baud, timeout=timeout)
    except Exception as e:
        print(f"  GAGAL membuka port: {e}")
        print("\n  Cek: nomor COM benar? GX Works2 masih memegang port?")
        return 1

    client = FXPLCClient(transport)
    ok_any = False

    # ── 1. Blok special relay M8000+ — baca RAW di alamat aslinya.
    print(f"\n[1] Special relay M8000+ — read_bytes(0x{SPECIAL_M_BASE:04X}, 2)")
    try:
        raw = await client.read_bytes(SPECIAL_M_BASE, 2)
        m8000 = bool(raw[0] & 0x01)
        print(f"    mentah: {raw.hex()}")
        if m8000:
            print("    M8000 ON — PLC menjawab dan sedang RUN.")
            ok_any = True
        else:
            print("    Menjawab, tapi M8000 OFF — PLC kemungkinan STOP.")
            ok_any = True
    except Exception as e:
        print(f"    GAGAL: {e!r}")

    # ── 2. M8013 — clock 1 detik
    print("\n[2] M8013 (clock 1 detik) — diamati 3 detik")
    seen = set()
    try:
        for _ in range(12):
            raw = await client.read_bytes(SPECIAL_M_BASE, 2)
            seen.add(bool(raw[1] & (1 << 5)))
            await asyncio.sleep(0.25)
        print(f"    nilai yang muncul: {sorted(int(v) for v in seen)}")
        if len(seen) > 1:
            print("    BERUBAH — ini bukti kuat komunikasi sungguhan.")
            ok_any = True
        else:
            print("    tidak berubah — bisa jadi hanya nilai default/echo.")
    except Exception as e:
        print(f"    GAGAL: {e!r}")

    # ── 3. Device yang dipakai kontrak.
    print("\n[3] Relay kontrak VisionInspector")
    labels = [("M1", "result_ok"), ("M3", "part_ready"),
              ("M7", "heartbeat"), ("M8", "ng_from_plc")]
    for label, nama in labels:
        try:
            v = await client.read_bit(label)
            print(f"    {label:5s} ({nama:12s}) = {v}")
            ok_any = True
        except Exception as e:
            print(f"    {label:5s} ({nama:12s}) GAGAL: {e!r}")

    # ── 4. Tulis — yang paling menentukan untuk.
    print("\n[4] Uji TULIS ke M1 (coil result_ok)")
    try:
        awal = await client.read_bit("M1")
        await client.write_bit("M1", not awal)
        time.sleep(0.1)
        baru = await client.read_bit("M1")
        await client.write_bit("M1", awal)          # kembalikan
        print(f"    {awal} → tulis {not awal} → terbaca {baru} → dikembalikan")
        if baru != awal:
            print("    TULIS BERHASIL — jalur ini bisa dipakai penuh.")
            ok_any = True
        else:
            print("    Tulis tidak berpengaruh — mungkin hanya baca yang didukung.")
    except Exception as e:
        print(f"    GAGAL: {e!r}")

    try:
        transport.close()
    except Exception:
        pass

    print("\n" + "=" * 60)
    if ok_any:
        print("PLC MENJAWAB lewat protokol FX.")
        print("Jalur ini layak dipakai — MODBUS tidak diperlukan sama sekali.")
    else:
        print("TIDAK ADA jawaban sama sekali.")
        print("Cek: port benar? PLC RUN? GX Works2 masih memegang port?")
    return 0 if ok_any else 2


async def bench(port: str, baud: int, timeout: float) -> int:
    """
    Ukur waktu satu siklus poll
    """
    print(f"Membuka {port} @ {baud} …")
    try:
        transport = TransportSerial(port, baudrate=baud, timeout=timeout)
    except Exception as e:
        print(f"  GAGAL membuka port: {e}")
        return 1
    client = FXPLCClient(transport)

    INPUTS = ["M0", "M5", "M6", "M8"]      # trigger, reset, switch, ng
    N = 20

    print(f"\nMengukur {N} siklus (4 baca + 1 tulis per siklus)…")
    per_read, per_write, per_cycle = [], [], []
    gagal = 0
    try:
        for _ in range(N):
            t_cycle = time.perf_counter()
            for lbl in INPUTS:
                t0 = time.perf_counter()
                try:
                    await client.read_bit(lbl)
                    per_read.append((time.perf_counter() - t0) * 1000)
                except Exception:
                    gagal += 1
            t0 = time.perf_counter()
            try:
                await client.write_bit("M7", False)
                per_write.append((time.perf_counter() - t0) * 1000)
            except Exception:
                gagal += 1
            per_cycle.append((time.perf_counter() - t_cycle) * 1000)
    finally:
        try:
            transport.close()
        except Exception:
            pass

    if not per_cycle:
        print("Tidak ada siklus yang selesai.")
        return 2

    def stat(xs, nama):
        if not xs:
            print(f"  {nama:12s} —")
            return
        xs = sorted(xs)
        p95 = xs[min(len(xs) - 1, int(len(xs) * 0.95))]
        print(f"  {nama:12s} rata-rata {sum(xs)/len(xs):6.1f} ms   "
              f"p95 {p95:6.1f} ms   maks {xs[-1]:6.1f} ms")

    print()
    stat(per_read, "baca 1 bit")
    stat(per_write, "tulis 1 bit")
    stat(per_cycle, "SATU SIKLUS")
    if gagal:
        print(f"  transaksi gagal: {gagal}")

    p95_cycle = sorted(per_cycle)[min(len(per_cycle) - 1,
                                      int(len(per_cycle) * 0.95))]
    print("\n" + "=" * 60)
    if p95_cycle < 100:
        print(f"SANGGUP. Satu siklus p95 {p95_cycle:.0f} ms, jauh di bawah "
              "interval poll 200 ms.")
    elif p95_cycle < 180:
        print(f"CUKUP, tapi ketat. Siklus p95 {p95_cycle:.0f} ms vs poll 200 ms — "
              "margin tipis.")
    else:
        print(f"TIDAK SANGGUP di 200 ms. Siklus p95 {p95_cycle:.0f} ms.")
        print("Pilihan: naikkan baudrate, turunkan laju poll, atau baca "
              "banyak bit sekaligus lewat read_int.")
    return 0


#: Dari source fxplc — registers_map_bit_images["M"] = (0x0100, 8).
#: byte_addr = 0x0100 + nomor//8, bit = nomor%8. Jadi M0-M15 muat di 2 byte.
M_BASE = 0x0100


async def batch(port: str, baud: int, timeout: float) -> int:
    """Buktikan satu read_bytes bisa menggantikan empat read_bit.
    """
    print(f"Membuka {port} @ {baud} …")
    try:
        transport = TransportSerial(port, baudrate=baud, timeout=timeout)
    except Exception as e:
        print(f"  GAGAL membuka port: {e}")
        return 1
    client = FXPLCClient(transport)

    CEK = [0, 1, 3, 5, 6, 7, 8]          # relay yang dipakai kontrak
    rc = 2
    try:
        print(f"\n[1] Borongan: read_bytes(0x{M_BASE:04X}, 2) → M0-M15")
        t0 = time.perf_counter()
        raw = await client.read_bytes(M_BASE, 2)
        t_batch = (time.perf_counter() - t0) * 1000
        borong = {n: bool(raw[n // 8] & (1 << (n % 8))) for n in CEK}
        print(f"    mentah: {raw.hex()}   ({t_batch:.1f} ms)")
        print("    " + "  ".join(f"M{n}={int(v)}" for n, v in borong.items()))

        print("\n[2] Satuan: read_bit per relay")
        t0 = time.perf_counter()
        satuan = {n: await client.read_bit(f"M{n}") for n in CEK}
        t_single = (time.perf_counter() - t0) * 1000
        print("    " + "  ".join(f"M{n}={int(v)}" for n, v in satuan.items())
              + f"   ({t_single:.1f} ms)")

        print("\n[3] Kecocokan")
        beda = [n for n in CEK if borong[n] != satuan[n]]
        if beda:
            print(f"    TIDAK COCOK di: {['M%d' % n for n in beda]}")
            print("    Jangan pakai optimasi borongan — perhitungan bit meleset.")
        else:
            print("    COCOK SEMUA — borongan sah dipakai.")
            rc = 0

        print("\n[4] Verifikasi pola: set M1 dan M3, byte0 harus 0x0A")
        awal = {n: await client.read_bit(f"M{n}") for n in (1, 3)}
        try:
            await client.write_bit("M1", True)
            await client.write_bit("M3", True)
            raw2 = await client.read_bytes(M_BASE, 2)
            b0 = raw2[0]
            print(f"    byte0 = 0x{b0:02X} (biner {b0:08b})")
            if b0 & 0x0A == 0x0A:
                print("    bit 1 dan bit 3 menyala di posisi yang benar.")
                if b0 & ~0x0A:
                    print(f"    catatan: bit lain juga ON (0x{b0 & ~0x0A:02X}) "
                          "— wajar kalau ladder ikut menyalakannya.")
                rc = 0
            else:
                print("    SALAH POSISI — perhitungan bit meleset.")
                print("    JANGAN pakai borongan; pakai read_bit satuan.")
                rc = 3
        finally:
            for n, v in awal.items():
                try:
                    await client.write_bit(f"M{n}", v)
                except Exception:
                    pass
            print(f"    dikembalikan: M1={int(awal[1])} M3={int(awal[3])}")

        print("\n[5] Perkiraan satu siklus poll (borongan + 1 tulis)")
        t0 = time.perf_counter()
        await client.write_bit("M7", False)
        t_write = (time.perf_counter() - t0) * 1000
        print(f"    read_bytes {t_batch:.1f} ms + write_bit {t_write:.1f} ms "
              f"= {t_batch + t_write:.1f} ms")
        print(f"    dibanding cara satuan (4 baca + 1 tulis): "
              f"~{t_single/len(CEK)*4 + t_write:.1f} ms")
    except Exception as e:
        print(f"    GAGAL: {e!r}")
    finally:
        try:
            transport.close()
        except Exception:
            pass
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe FX Computer Link via fxplc.")
    ap.add_argument("--port", required=True, help="mis. COM11")
    ap.add_argument("--baud", type=int, default=9600,
                    help="port pemrograman FX umumnya 9600 (default)")
    ap.add_argument("--timeout", type=float, default=1.0)
    ap.add_argument("--bench", action="store_true",
                    help="ukur waktu satu siklus poll, bukan uji fungsi")
    ap.add_argument("--batch", action="store_true",
                    help="buktikan satu read_bytes menggantikan 4 read_bit")
    args = ap.parse_args()
    fn = batch if args.batch else bench if args.bench else probe
    return asyncio.run(fn(args.port, args.baud, args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
