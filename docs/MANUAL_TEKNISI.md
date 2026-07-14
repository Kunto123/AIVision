# VisionInspect — Manual Teknisi

## Wiring Serial

### RS232 (Point-to-Point)

```
PC (DB9)         PLC (DB9)
  TX (pin 3)  ─── RX (pin 2)
  RX (pin 2)  ─── TX (pin 3)
  GND (pin 5) ─── GND (pin 5)
```

Umumnya menggunakan konverter **USB-to-RS232**. Setelah driver terinstall, port akan muncul sebagai `COM3` (Windows) atau `/dev/ttyUSB0` (Linux).

### RS485 (Half-Duplex)

Kabel **2-wire** (A/B atau D+/D-):

```
PC (USB-RS485)    PLC (RS485)
  A/D+ ─────────── A/D+
  B/D- ─────────── B/D-
  GND  ─────────── GND
```

**Penting untuk RS485:**
1. **Terminasi**: Pasang resistor 120Ω di kedua ujung bus jika kabel > 10m
2. **Auto-direction vs RTS**: Beberapa konverter USB-RS485 mendeteksi arah otomatis. Jika tidak, gunakan mode RTS-controlled di pengaturan
3. **Delay**: Jika data korup, coba tambah delay before/after TX (0.1–10 ms)

## Register Map Modbus RTU

VisionInspect sebagai **Modbus slave** (default ID=1).

| Address | Type | Name | R/W | Description |
|---------|------|------|-----|-------------|
| 0x0000 | Holding | System Status | R | 0=idle, 1=running, 2=training, 3=error |
| 0x0001 | Holding | Last Result | R | 0=none, 1=OK, 2=NG |
| 0x0002 | Holding | Last Score | R | Score × 100 (0–10000) |
| 0x0003 | Holding | Total Counter | R | Rolling 16-bit |
| 0x0004 | Holding | NG Counter | R | Rolling 16-bit |
| 0x000A | Holding | Active Program | R/W | Tulis nomor program untuk switch |
| 0x0000 | Coil | Trigger | R/W | Set 1 → inspeksi trigger → reset 0 |
| 0x0001 | Coil | Reset Counter | R/W | Set 1 → reset semua counter → reset 0 |

### Contoh Pembacaan (Python dengan pymodbus)

```python
from pymodbus.client import ModbusSerialClient

client = ModbusSerialClient(port='COM3', baudrate=9600)
client.connect()

# Baca status
rr = client.read_holding_registers(0, 5, slave=1)
status = rr.registers[0]
result = rr.registers[1]  # 0=none, 1=OK, 2=NG
score = rr.registers[2] / 100.0

# Trigger inspeksi
client.write_coil(0, True, slave=1)
client.write_coil(0, False, slave=1)

client.close()
```

## Protokol ASCII (untuk PLC Lama)

**Format Frame:**
```
STX (0x02) <CMD> [DATA] ETX (0x03) <CHECKSUM>
```

Checksum: XOR seluruh byte dari STX hingga ETX (inklusif).

### Perintah

| Command | Data | Deskripsi | Response |
|---------|------|-----------|----------|
| `TRG` | — | Trigger inspeksi | `ACK` |
| `RES` | `OK,<score>` | Hasil OK dikirim PLC | `ACK` |
| `RES` | `NG,<score>` | Hasil NG dikirim PLC | `ACK` |
| `PRG` | `<n>` | Ganti program ke-n | `ACK` |
| `STA` | — | Request status | `STA,<code>,<text>` |

### Contoh Hex

```
Trigger:      02 54 52 47 03 54
Result OK:    02 52 45 53 2c 4f 4b 2c 30 2e 39 35 30 30 03 B6
Result NG:    02 52 45 53 2c 4e 47 2c 30 2e 38 35 30 30 03 AE
```

## Testing dengan PLC Simulator

### Setup Virtual Serial Pair (Linux/WSL)

```bash
# Install socat
sudo apt-get install socat

# Buat virtual serial pair
socat -d -d PTY,link=/tmp/ttyV0 PTY,link=/tmp/ttyV1

# Terminal 1: Jalankan simulator
python tools/plc_simulator.py --port /tmp/ttyV0 --protocol ascii

# Terminal 2: Konfigurasi VisionInspect ke port /tmp/ttyV1
# Atau kirim test langsung:
python -c "
import serial
ser = serial.Serial('/tmp/ttyV1', 9600, timeout=1)
# Kirim trigger
ser.write(b'\x02TRG\x03\x54')
print('Trigger sent')
# Baca response
resp = ser.read(10)
print(f'Response: {resp.hex()}')
"
```

### Windows (com0com)

1. Install com0com dari [sourceforge](https://sourceforge.net/projects/com0com/)
2. Setup virtual pair: `COM3` ↔ `COM4`
3. Jalankan simulator ke `COM3`, VisionInspect ke `COM4`

## Troubleshooting

### Serial Tidak Terdeteksi
- Cek driver USB-to-Serial (FTDI, CH340, CP210x)
- Di Linux: `ls /dev/ttyU*` atau `dmesg | grep tty`
- Di Windows: Device Manager → Ports (COM & LPT)

### Data Korup / CRC Error
- Turunkan baudrate (9600 atau 19200)
- Untuk RS485: pasang resistor terminasi 120Ω
- Untuk kabel panjang > 10m: gunakan shielded twisted pair
- Cek delay before/after TX (RS485)

### PLC Tidak Merespon
- Cek wiring (TX/RX terbalik?)
- Cek parity dan stop bits (harus match dengan PLC)
- Uji dengan serial terminal (HTerm, RealTerm, screen)
- Gunakan fitur "Kirim Frame Uji" di tab DIAGNOSTICS

### VisionInspect Tidak Bisa Start
- Cek `logs/app.log` untuk error detail
- Pastikan port kamera tidak dipakai aplikasi lain
- Hapus `~/.visioninspect/config.json` untuk reset konfigurasi

## Performa

### Target

| Metrik | Target | Catatan |
|--------|--------|---------|
| Inferensi Frame | < 100 ms | 256×256 ROI, PatchCore, OpenVINO INT8 |
| Inferensi Frame | < 30 ms | 256×256 ROI, EfficientAd-S |
| Rebuild Model | < 2 menit | 50 gambar OK, CPU 4-core |
| RAM (idle) | < 800 MB | |
| RAM (running) | < 1.5 GB | |
| Start-to-ready | < 15 detik | |
| Uptime | 24/7 | Tanpa restart |

### Monitoring

Gunakan tab DIAGNOSTICS untuk memonitor:
- **RAM Usage**: Pastikan tidak ada memory leak
- **Inference Latency**: Rolling average + P95
- **Camera FPS**: Frame rate aktual
- **Thread Status**: Semua thread harus RUNNING

## Packaging (PyInstaller)

```bash
# Build one-folder executable
pip install pyinstaller
pyinstaller --onefile --name VisionInspect run.py
```

Hasil ada di `dist/VisionInspect.exe` (Windows) atau `dist/VisionInspect` (Linux).
