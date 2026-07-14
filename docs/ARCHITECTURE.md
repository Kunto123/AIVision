# VisionInspect — Architecture Document

## Overview

VisionInspect is a single-process, multi-threaded desktop application for industrial visual inspection. It runs **100% locally on CPU** using Anomalib for anomaly detection and OpenVINO for inference.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    VisionInspect (satu proses)            │
│                                                          │
│  ┌────────────┐   frame    ┌─────────────────┐          │
│  │ Camera     │──queue────▶│ Inference Engine │          │
│  │ Thread     │            │ (OpenVINO, CPU)  │          │
│  └────────────┘            └───────┬─────────┘          │
│                                    │ result (score,     │
│                                    │ heatmap, OK/NG)    │
│                     ┌──────────────┼──────────────┐     │
│                     ▼              ▼              ▼     │
│              ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│              │ GUI      │   │ PLC I/O  │   │ Logger & │ │
│              │ (PySide6 │   │ Thread   │   │ History  │ │
│              │ main     │   │ (serial) │   │ (SQLite) │ │
│              │ thread)  │   └──────────┘   └──────────┘ │
│              └──────────┘                               │
│                                                         │
│  ┌──────────────────────────────────────────────┐       │
│  │ Training/Rebuild Worker (QThread terpisah,   │       │
│  │ dipicu manual: teaching & redefinition)      │       │
│  └──────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────┘
```

## Threading Model

| Thread | Purpose | Priority |
|--------|---------|----------|
| **Main (GUI)** | PySide6 event loop | Normal |
| **Camera** | Frame grabbing via OpenCV | High |
| **Inference** | OpenVINO inference | High |
| **PLC I/O** | Serial communication + reconnect | Normal |
| **Training** | Anomalib training (spawned on demand) | Low |
| **Watchdog** | Monitor thread health | Normal |
| **Flask API** | (Optional) REST API | Low |

**Communication:**
- Camera → Inference: `queue.Queue(maxsize=2)` with drop-oldest-frame policy
- Inference → GUI: Qt signals (thread-safe `Qt.QueuedConnection`)
- PLC → GUI: Qt signals + callbacks
- Training → GUI: Progress callbacks + hot-swap model pointer

## Data Flow

### Inspection Flow (RUN mode)
```
Camera → frame → crop ROI → resize → OpenVINO infer → 
  score + heatmap → judge (OK/NG) → update GUI → send to PLC → save history
```

### Training Flow (TEACH mode)
```
User captures OK/NG images → saves to programs/<name>/images/{ok,ng} →
  Anomalib Folder datamodule → PatchCore.fit() → 
  calibrate threshold → export OpenVINO IR → INT8 PTQ → 
  save version → hot-swap model atomically
```

### Redefinition Flow
```
Select history entry → mark correction → image → corrections/{ok,ng} →
  Rebuild model (same as training, combined dataset) →
  new version → hot-swap (old model still serving) → audit trail
```

## Module Map

### `visioninspect/core/` — Business Logic

| File | Responsibility |
|------|---------------|
| `camera.py` | Camera device abstraction, frame grabbing thread, FPS counter |
| `inference.py` | OpenVINO inference engine, model hot-swap, heatmap overlay |
| `training.py` | Anomalib pipeline, threshold calibration, INT8 quantization |
| `program.py` | Program CRUD, versioning, image management, atomic writes |
| `redefinition.py` | Correction logic, rebuild orchestrator, audit trail |
| `watchdog.py` | Thread health monitoring, auto-restart on hang |

### `visioninspect/plc/` — PLC Communication

| File | Responsibility |
|------|---------------|
| `serial_manager.py` | RS232/RS485 serial, auto-reconnect, RTS control, RX/TX logging |
| `modbus_rtu.py` | Modbus register map, coil operations, command polling |
| `ascii_protocol.py` | STX/ETX framing, XOR checksum, command parsing |

### `visioninspect/gui/` — User Interface

| File | Responsibility |
|------|---------------|
| `main_window.py` | Main window with tabs, menu, status bar, theme loading |
| `pages/run_page.py` | Operator screen: live view, big judgement, counters |
| `pages/teach_page.py` | Teaching: capture, gallery, train, threshold, histogram |
| `pages/history_page.py` | History table, filter, correction actions |
| `pages/settings_page.py` | All configuration: camera, ROI, PLC, model, retention |
| `pages/diagnostics_page.py` | Live logs, performance metrics, PLC test |
| `theme.qss` | Dark navy Qt stylesheet |

### `visioninspect/storage/` — Data Persistence

| File | Responsibility |
|------|---------------|
| `db.py` | SQLite with WAL mode, inspection history, counters, audit |
| `retention.py` | Auto-purge old data by age, OK image sampling |

### `visioninspect/api/` — Optional REST API

| File | Responsibility |
|------|---------------|
| `flask_app.py` | Flask on 127.0.0.1, API key auth, status/trigger/history endpoints |

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.11 | Ecosystem, team skill |
| GUI | PySide6 (Qt) | Native feel, threading, GPU rendering |
| AI | Anomalib (PatchCore) | Few-shot anomaly detection |
| Inference | OpenVINO | CPU optimization, INT8 quantization |
| Camera | OpenCV | USB/GigE, wide driver support |
| PLC | pyserial + pymodbus | RS232/RS485, Modbus RTU |
| Database | SQLite (WAL) | Zero-config, embedded |
| API | Flask | Lightweight, localhost only |
| Packaging | PyInstaller | Single-folder deployment |

## Configuration

Configuration is stored as JSON at `~/.visioninspect/config.json` with atomic writes (write-to-temp → rename). Defaults are hardcoded and merged with user config on load.

Key sections:
- `camera`: device_index, resolution, fps, exposure
- `roi`: region of interest coordinates
- `model`: algorithm, backbone, threshold
- `plc`: serial parameters, protocol selection
- `flask_api`: enabled, port, api_key
- `history`: retention policy
- `logging`: level, file rotation

## Data Storage

### Directory Structure
```
~/.visioninspect/
├── config.json
├── logs/
│   ├── app.log
│   ├── plc.log
│   ├── inference.log
│   ├── camera.log
│   └── training.log
├── database.db (SQLite WAL)
└── programs/
    ├── <program-name>/
    │   ├── config.json
    │   ├── metadata.json
    │   ├── model/
    │   │   ├── openvino/
    │   │   └── openvino_int8/
    │   ├── images/
    │   │   ├── ok/
    │   │   ├── ng/
    │   │   └── corrections/
    │   │       ├── ok/
    │   │       └── ng/
    │   ├── versions/
    │   │   ├── v1/
    │   │   ├── v2/
    │   │   └── ...
    │   └── audit/
    └── ...
```

## Safety & Fail-Safe

- **Inference error** → judgement = NG + alarm (fail-safe)
- **Serial error** → buffer results, retry, GUI alarm
- **Camera disconnected** → auto-retry, clear display
- **Training error** → log full trace, restore previous model
- **Thread hang** → watchdog auto-restart after timeout
- **Config write** → atomic (temp file + rename)
- **Database** → WAL mode for crash recovery
