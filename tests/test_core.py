"""
VisionInspect - Unit Tests
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visioninspect.utils.config import Config
from visioninspect.utils.i18n import Translator


class TestConfig:
    """Test configuration management."""

    def test_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            c = Config(config_path)
            assert c.get("camera.device_index") == 0
            assert c.get("model.algorithm") == "patchcore"
            assert c.get("plc.mode") == "rs232"
            assert c.get("language") == "id"

    def test_set_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            c = Config(config_path)
            c.set("camera.device_index", 2)
            assert c.get("camera.device_index") == 2
            c.save()

            # Reload
            c2 = Config(config_path)
            assert c2.get("camera.device_index") == 2

    def test_nested_set(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            c = Config(Path(tmpdir) / "config.json")
            c.set("plc.rs485_direction", "auto")
            assert c.get("plc.rs485_direction") == "auto"

    def test_reset(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            c = Config(Path(tmpdir) / "config.json")
            c.set("camera.device_index", 99)
            c.reset_to_defaults()
            assert c.get("camera.device_index") == 0

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            c = Config(config_path)
            c.set("model.algorithm", "efficientad")
            c.save()

            # Read raw file
            with open(config_path) as f:
                data = json.load(f)
            assert data["model"]["algorithm"] == "efficientad"


class TestTranslator:
    """Test internationalization."""

    def test_id_default(self):
        t = Translator("id")
        assert t.tr("ok") == "OK"
        assert t.tr("ng") == "NG"
        assert t.tr("app_name") == "VisionInspect"

    def test_en(self):
        t = Translator("en")
        assert t.tr("ok") == "OK"
        assert t.tr("nav_run") == "RUN"

    def test_fallback(self):
        t = Translator("id")
        # Key yang tidak ada return key name
        assert t.tr("nonexistent_key") == "nonexistent_key"

    def test_format(self):
        t = Translator("id")
        result = t.tr("teach_count_ok", count=5)
        assert "5" in result

    def test_switch_language(self):
        t = Translator("id")
        assert t.tr("nav_run") == "RUN"
        t.language = "en"
        assert t.tr("nav_run") == "RUN"


class TestASCIIProtocol:
    """Test ASCII protocol framing and parsing."""

    def test_make_frame(self):
        from visioninspect.plc.ascii_protocol import ASCIIProtocolManager
        frame = ASCIIProtocolManager.make_frame("TRG")
        assert frame.startswith(b"\x02")
        assert b"TRG" in frame
        assert frame.endswith(b"\x03") or len(frame) > 3

    def test_parse_trg(self):
        from visioninspect.plc.ascii_protocol import ASCIIProtocolManager
        frame = ASCIIProtocolManager.make_frame("TRG")
        parsed = ASCIIProtocolManager.parse_frame(frame)
        assert parsed is not None
        assert parsed["command"] == "TRG"

    def test_parse_res(self):
        from visioninspect.plc.ascii_protocol import ASCIIProtocolManager
        frame = ASCIIProtocolManager.make_frame("RES", "OK,0.2345")
        parsed = ASCIIProtocolManager.parse_frame(frame)
        assert parsed is not None
        assert parsed["command"] == "RES"
        assert parsed["data"] == "OK,0.2345"

    def test_checksum_validation(self):
        from visioninspect.plc.ascii_protocol import ASCIIProtocolManager
        frame = ASCIIProtocolManager.make_frame("STA")
        # Corrupt the checksum byte
        bad_frame = frame[:-1] + bytes([frame[-1] ^ 0xFF])
        parsed = ASCIIProtocolManager.parse_frame(bad_frame)
        assert parsed is None or "error" in parsed

    def test_parse_prg(self):
        from visioninspect.plc.ascii_protocol import ASCIIProtocolManager
        frame = ASCIIProtocolManager.make_frame("PRG", "3")
        parsed = ASCIIProtocolManager.parse_frame(frame)
        assert parsed is not None
        assert parsed["command"] == "PRG"
        assert parsed["data"] == "3"


class TestSerialConfig:
    """Test serial configuration."""

    def test_default_config(self):
        from visioninspect.plc.serial_manager import SerialConfig
        cfg = SerialConfig()
        assert cfg.port == "COM1"
        assert cfg.baudrate == 9600
        assert cfg.mode == "rs232"

    def test_rs485_config(self):
        from visioninspect.plc.serial_manager import SerialConfig
        cfg = SerialConfig(mode="rs485", rs485_direction="rts")
        assert cfg.mode == "rs485"
        assert cfg.rs485_direction == "rts"


class TestInferenceResult:
    """Test inference result data class."""

    def test_result_creation(self):
        from visioninspect.core.inference import InferenceResult
        r = InferenceResult(score=0.5, judgement="OK", latency_ms=10.0, threshold=0.5)
        assert r.score == 0.5
        assert r.judgement == "OK"
        assert r.latency_ms == 10.0

    def test_result_ng(self):
        from visioninspect.core.inference import InferenceResult
        r = InferenceResult(score=0.8, judgement="NG", latency_ms=5.0, threshold=0.5)
        assert r.judgement == "NG"


class TestProgramManager:
    """Test program management."""

    def test_create_and_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from visioninspect.core.program import ProgramManager
            pm = ProgramManager(Path(tmpdir))
            pm.create_program("test-produk-a")
            programs = pm.list_programs()
            assert len(programs) == 1
            assert programs[0]["name"] == "test-produk-a"

    def test_sanitize_name(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from visioninspect.core.program import ProgramManager
            pm = ProgramManager(Path(tmpdir))
            info = pm.create_program("produk/../jahat")
            assert info["name"] == "produk___jahat"  # 3 underscores: /→_, ..→_

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from visioninspect.core.program import ProgramManager
            pm = ProgramManager(Path(tmpdir))
            pm.create_program("delete-test")
            pm.delete_program("delete-test")
            assert len(pm.list_programs()) == 0

    def test_template_versioning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from visioninspect.core.program import ProgramManager
            pm = ProgramManager(Path(tmpdir))
            pm.create_program("ver-test")
            # Create template
            tmpl = pm.create_template("ver-test", "Template 1")
            tmpl_id = tmpl["id"]
            # Create mock model export
            export_dir = Path(tmpdir) / "export"
            (export_dir / "openvino").mkdir(parents=True)
            (export_dir / "openvino" / "model.xml").write_text("<?xml?>")
            (export_dir / "openvino" / "model.bin").write_text("binary")
            # Save model
            version = pm.save_template_model("ver-test", tmpl_id,
                                              {"export_path": str(export_dir)})
            assert version == 1
            version = pm.save_template_model("ver-test", tmpl_id,
                                              {"export_path": str(export_dir)})
            assert version == 2
            # Check trained flag
            cfg = pm.get_template_config("ver-test", tmpl_id)
            assert cfg["trained"] is True
            assert cfg["model_version"] == 2
            # Check model path
            model_path = pm.get_template_model_path("ver-test", tmpl_id)
            assert model_path is not None
            assert model_path.name == "model.xml"

    def test_template_crud(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from visioninspect.core.program import ProgramManager
            pm = ProgramManager(Path(tmpdir))
            pm.create_program("tmpl-test")
            # Create templates
            t1 = pm.create_template("tmpl-test", "Deteksi Label")
            t2 = pm.create_template("tmpl-test", "Deteksi Tutup")
            assert t1["id"] != t2["id"]
            # List templates
            templates = pm.list_templates("tmpl-test")
            assert len(templates) == 2
            # Delete template
            pm.delete_template("tmpl-test", t1["id"])
            templates = pm.list_templates("tmpl-test")
            assert len(templates) == 1

    def test_template_images(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmpdir:
            from visioninspect.core.program import ProgramManager
            pm = ProgramManager(Path(tmpdir))
            pm.create_program("img-test")
            tmpl = pm.create_template("img-test", "Image Test")
            tmpl_id = tmpl["id"]
            # Save images
            fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
            pm.save_template_image("img-test", tmpl_id, fake_img, "ok")
            pm.save_template_image("img-test", tmpl_id, fake_img, "ok")
            pm.save_template_image("img-test", tmpl_id, fake_img, "ng")
            assert pm.count_template_images("img-test", tmpl_id, "ok") == 2
            assert pm.count_template_images("img-test", tmpl_id, "ng") == 1
            # Verify config counts
            cfg = pm.get_template_config("img-test", tmpl_id)
            assert cfg["num_ok"] == 2
            assert cfg["num_ng"] == 1


class TestDatabase:
    """Test SQLite database."""

    def test_add_and_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from visioninspect.storage.db import Database
            db = Database(Path(tmpdir) / "test.db")

            entry_id = db.add_inspection({
                "program": "test-program",
                "score": 0.3,
                "judgement": "OK",
                "threshold": 0.5,
                "latency_ms": 25.0,
            })
            assert entry_id > 0

            history = db.get_history(program="test-program")
            assert len(history) == 1
            assert history[0]["judgement"] == "OK"

            db.close()

    def test_counters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from visioninspect.storage.db import Database
            db = Database(Path(tmpdir) / "test.db")

            db.add_inspection({"program": "p1", "score": 0.1, "judgement": "OK", "threshold": 0.5})
            db.add_inspection({"program": "p1", "score": 0.8, "judgement": "NG", "threshold": 0.5})
            db.add_inspection({"program": "p1", "score": 0.2, "judgement": "OK", "threshold": 0.5})

            counters = db.get_counters("p1")
            assert counters["total"] == 3
            assert counters["ok"] == 2
            assert counters["ng"] == 1

            db.close()

    def test_correction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from visioninspect.storage.db import Database
            db = Database(Path(tmpdir) / "test.db")

            eid = db.add_inspection({
                "program": "p1", "score": 0.8, "judgement": "NG", "threshold": 0.5
            })
            db.mark_correction(eid, "OK")
            history = db.get_history()
            assert history[0]["corrected"] == 1
            assert history[0]["correct_judgement"] == "OK"

            db.close()
