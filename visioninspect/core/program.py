"""
VisionInspect - Program Manager
Manajemen program inspeksi: create, switch, config, model versioning.
Program = folder terstruktur dengan config, model, images.

Struktur folder per program:
    programs/<name>/
        config.json                    ← program-level config (camera, PLC)
        metadata.json                  ← program metadata
        templates/
            <template_id>/
                config.json            ← template config (roi, threshold, algorithm)
                images/
                    ok/   (*.png, *.jpg)
                    ng/   (*.png, *.jpg)
                model/
                    openvino/model.xml
                    openvino_int8/model.xml (optional)
"""

import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")


class ProgramError(Exception):
    pass


class ProgramManager:
    """
    Manajemen program inspeksi + template di dalamnya.
    """

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # PROGRAM LEVEL
    # =====================================================================

    def list_programs(self) -> List[Dict[str, Any]]:
        """List all programs with metadata."""
        programs = []
        if not self._base_dir.exists():
            return programs

        for folder in sorted(self._base_dir.iterdir()):
            if folder.is_dir() and not folder.name.startswith("."):
                info = self.get_program_info(folder.name)
                if info:
                    programs.append(info)
        return programs

    def get_program_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get program metadata + list templates."""
        config_path = self._get_config_path(name)
        if not config_path.exists():
            return None

        config = self._load_json(config_path, {})
        meta = self._load_json(self._get_meta_path(name), {})

        return {
            "name": name,
            "config": config,
            "metadata": meta,
            "path": str(self._get_program_dir(name)),
            "templates": self.list_templates(name),
        }

    def create_program(self, name: str, config: Optional[dict] = None) -> dict:
        """Create a new inspection program."""
        safe_name = self._sanitize_name(name)
        prog_dir = self._get_program_dir(safe_name)

        if prog_dir.exists():
            raise ProgramError(f"Program '{safe_name}' sudah ada")

        # Create directories
        (prog_dir / "templates").mkdir(parents=True)

        # Default config
        default_config = {
            "name": safe_name,
            "plc": {},
            "camera": {},
        }
        if config:
            default_config.update(config)

        self._atomic_write(self._get_config_path(safe_name), default_config)

        meta = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "active_template": "",
        }
        self._atomic_write(self._get_meta_path(safe_name), meta)

        logger.info("Program created: %s", safe_name)
        return {"name": safe_name, "path": str(prog_dir), "config": default_config}

    def delete_program(self, name: str) -> None:
        """Delete a program and all its data."""
        prog_dir = self._get_program_dir(name)
        if not prog_dir.exists():
            raise ProgramError(f"Program '{name}' tidak ditemukan")
        shutil.rmtree(prog_dir)
        logger.info("Program deleted: %s", name)

    def rename_program(self, old_name: str, new_name: str) -> None:
        """Rename a program."""
        old_dir = self._get_program_dir(old_name)
        new_dir = self._get_program_dir(new_name)
        if not old_dir.exists():
            raise ProgramError(f"Program '{old_name}' tidak ditemukan")
        if new_dir.exists():
            raise ProgramError(f"Program '{new_name}' sudah ada")
        old_dir.rename(new_dir)
        config = self._load_json(self._get_config_path(new_name), {})
        config["name"] = new_name
        self._atomic_write(self._get_config_path(new_name), config)
        logger.info("Program renamed: %s → %s", old_name, new_name)

    # =====================================================================
    # TEMPLATE LEVEL
    # =====================================================================

    def list_templates(self, program: str) -> List[Dict[str, Any]]:
        """List all templates in a program."""
        tmpl_dir = self._get_template_dir(program)
        if not tmpl_dir.exists():
            return []

        templates = []
        for folder in sorted(tmpl_dir.iterdir()):
            if folder.is_dir() and not folder.name.startswith("."):
                cfg = self._load_json(folder / "config.json", {})
                templates.append({
                    "id": folder.name,
                    "name": cfg.get("name", folder.name),
                    "config": cfg,
                    "path": str(folder),
                })
        return templates

    def create_template(self, program: str, template_name: str,
                        config: Optional[dict] = None) -> dict:
        """
        Create a new template in a program.
        Returns template info.
        """
        safe_name = self._sanitize_name(template_name)
        tmpl_id = self._next_template_id(program)
        tmpl_dir = self._get_template_dir(program) / tmpl_id

        # Create folders
        (tmpl_dir / "images" / "ok").mkdir(parents=True)
        (tmpl_dir / "images" / "ng").mkdir(parents=True)
        (tmpl_dir / "model").mkdir(parents=True)

        from visioninspect.core.part_check import DEFAULT_PART_CHECK_CONFIG

        # Default template config
        default_config = {
            "id": tmpl_id,
            "name": safe_name,
            "algorithm": "patchcore",
            "backbone": "resnet18",
            "input_size": 256,
            "threshold": 0.5,
            "threshold_mode": "adaptive",
            "coreset_sampling_ratio": 0.1,
            "roi": {
                "x": 0, "y": 0,
                "width": 256, "height": 256,
            },
            "num_ok": 0,
            "num_ng": 0,
            "trained": False,
            "model_version": 0,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "part_check": DEFAULT_PART_CHECK_CONFIG.copy(),
        }
        if config:
            default_config.update(config)

        self._atomic_write(tmpl_dir / "config.json", default_config)
        logger.info("Template '%s' (%s) created in program '%s'", safe_name, tmpl_id, program)

        return {
            "id": tmpl_id,
            "name": safe_name,
            "config": default_config,
        }

    def delete_template(self, program: str, template_id: str) -> None:
        """Delete a template."""
        tmpl_dir = self._get_template_dir(program) / template_id
        if not tmpl_dir.exists():
            raise ProgramError(f"Template '{template_id}' tidak ditemukan")
        shutil.rmtree(tmpl_dir)
        logger.info("Template '%s' deleted from program '%s'", template_id, program)

    def get_template_config(self, program: str, template_id: str) -> dict:
        """Get template configuration."""
        return self._load_json(
            self._get_template_dir(program) / template_id / "config.json", {}
        )

    def update_template_config(self, program: str, template_id: str,
                                updates: dict) -> None:
        """Update template configuration."""
        cfg_path = self._get_template_dir(program) / template_id / "config.json"
        cfg = self._load_json(cfg_path, {})
        cfg.update(updates)
        self._atomic_write(cfg_path, cfg)

    def set_active_template(self, program: str, template_id: str) -> None:
        """Set active template for a program."""
        meta = self._load_json(self._get_meta_path(program), {})
        meta["active_template"] = template_id
        self._atomic_write(self._get_meta_path(program), meta)

    def get_active_template(self, program: str) -> Optional[str]:
        """Get active template ID for a program."""
        meta = self._load_json(self._get_meta_path(program), {})
        return meta.get("active_template", "")

    # =====================================================================
    # TEMPLATE IMAGE MANAGEMENT
    # =====================================================================

    def save_template_image(self, program: str, template_id: str,
                             image: Any, label: str,
                             update_count: bool = True) -> Path:
        """
        Save an image to the template's image directory.
        label = "ok" or "ng"
        """
        base = (self._get_template_dir(program) / template_id
                / "images" / label)
        base.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{uuid.uuid4().hex[:8]}.png"
        dest = base / filename

        cv2.imwrite(str(dest), image)

        # Update count (skip saat import batch — diakumulasi dan ditulis sekali di akhir)
        if update_count:
            cfg = self.get_template_config(program, template_id)
            if label == "ok":
                cfg["num_ok"] = cfg.get("num_ok", 0) + 1
            else:
                cfg["num_ng"] = cfg.get("num_ng", 0) + 1
            self.update_template_config(program, template_id, cfg)
        else:
            logger.debug("Image saved (batch mode, count deferred): %s", dest)

        logger.debug("Image saved: %s", dest)
        return dest

    def count_template_images(self, program: str, template_id: str,
                               label: str) -> int:
        """Count images in template by label."""
        base = (self._get_template_dir(program) / template_id
                / "images" / label)
        if not base.exists():
            return 0
        return len(list(base.glob("*.png")) + list(base.glob("*.jpg")))

    def list_template_images(self, program: str, template_id: str,
                              label: str) -> List[Path]:
        """List image paths in template by label."""
        base = (self._get_template_dir(program) / template_id
                / "images" / label)
        if not base.exists():
            return []
        files = sorted(base.glob("*.png")) + sorted(base.glob("*.jpg"))
        return files

    # =====================================================================
    # MODEL VERSIONING (per template)
    # =====================================================================

    def save_template_model(self, program: str, template_id: str,
                             model_artifacts: dict) -> int:
        """
        Save trained model artifacts to a template.
        `model_artifacts` harus berisi path ke folder export (openvino/ atau openvino_int8/).
        Returns version number.
        """
        cfg = self.get_template_config(program, template_id)
        version = cfg.get("model_version", 0) + 1

        model_dir = (self._get_template_dir(program) / template_id / "model")

        # Clear old model
        if model_dir.exists():
            for child in model_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

        # Copy specific subfolders only (openvino, openvino_int8)
        copied = False
        for key in ["export_path", "int8_path"]:
            src_str = model_artifacts.get(key, "")
            if src_str:
                src = Path(src_str)
                if src.exists() and src.is_dir():
                    for subdir in ["openvino", "openvino_int8", "simple_model", "torch"]:
                        sub_src = src / subdir
                        if sub_src.exists() and sub_src.is_dir():
                            shutil.copytree(sub_src, model_dir / subdir,
                                            dirs_exist_ok=True)
                            copied = True

        if not copied:
            logger.warning("No model artifacts found to copy for template '%s'", template_id)

        # Update template config
        cfg["model_version"] = version
        cfg["trained"] = True
        cfg["threshold"] = model_artifacts.get("threshold", cfg.get("threshold", 0.5))
        cfg["last_trained"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.update_template_config(program, template_id, cfg)

        logger.info("Model v%d saved to template '%s'", version, template_id)
        return version

    def get_template_model_path(self, program: str, template_id: str) -> Optional[Path]:
        """
        Get path to the trained model file (OpenVINO XML) for a template.
        Returns None if no model.
        """
        model_dir = self._get_template_dir(program) / template_id / "model"

        # Check int8 first, then openvino
        for sub in ["openvino_int8", "openvino"]:
            xml = model_dir / sub / "model.xml"
            if xml.exists():
                return xml
        # Check simple model format (no-PyTorch fallback)
        simple = model_dir / "simple_model" / "mean.npy"
        if simple.exists():
            return simple
        return None

    def is_template_trained(self, program: str, template_id: str) -> bool:
        """Check if a template has a trained model."""
        cfg = self.get_template_config(program, template_id)
        return cfg.get("trained", False)

    # =====================================================================
    # PART CHECK
    # =====================================================================

    def get_part_check_config(self, program: str, template_id: str) -> dict:
        """Get part check config for a template, with safe defaults for old templates."""
        from visioninspect.core.part_check import DEFAULT_PART_CHECK_CONFIG
        cfg = self.get_template_config(program, template_id)
        pc = cfg.get("part_check", {})
        merged = DEFAULT_PART_CHECK_CONFIG.copy()
        merged.update(pc)
        return merged

    def update_part_check_config(self, program: str, template_id: str,
                                  updates: dict) -> dict:
        """Update part check config fields (read-modify-merge-write)."""
        pc = self.get_part_check_config(program, template_id)
        pc.update(updates)
        tmpl_cfg = self.get_template_config(program, template_id)
        tmpl_cfg["part_check"] = pc
        self.update_template_config(program, template_id, tmpl_cfg)
        return pc

    def save_part_check_master(self, program: str, template_id: str,
                                image, gate_roi: dict,
                                canny_low: int = 50,
                                canny_high: int = 150) -> dict:
        """Crop image to gate_roi, compute master stats, save to disk + config."""
        from visioninspect.core.part_check import compute_master_stats, crop_roi
        import time

        cropped = crop_roi(image, gate_roi)
        if cropped is None or cropped.size == 0 or cropped.shape[0] < 2 or cropped.shape[1] < 2:
            raise ProgramError("Gate ROI terlalu kecil untuk foto master")

        stats = compute_master_stats(cropped, canny_low, canny_high)
        tmpl_dir = self._get_template_dir(program) / template_id
        pc_dir = tmpl_dir / "part_check"
        pc_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(pc_dir / "master.png"), cropped)

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        return self.update_part_check_config(program, template_id, {
            "has_master": True,
            "gate_roi": gate_roi,
            "master_mean_bgr": stats["mean_bgr"],
            "master_std_bgr": stats["std_bgr"],
            "master_edge_density": stats["edge_density"],
            "master_captured_at": now,
            "master_image_size": stats["image_size"],
        })

    def get_part_check_master_image_path(self, program: str,
                                          template_id: str) -> Optional[Path]:
        """Get path to saved master image, or None."""
        pc_dir = self._get_template_dir(program) / template_id / "part_check"
        master = pc_dir / "master.png"
        return master if master.exists() else None

    # =====================================================================
    # INTERNAL
    # =====================================================================

    def _get_program_dir(self, name: str) -> Path:
        return self._base_dir / name

    def _get_config_path(self, name: str) -> Path:
        return self._get_program_dir(name) / "config.json"

    def _get_meta_path(self, name: str) -> Path:
        return self._get_program_dir(name) / "metadata.json"

    def _get_template_dir(self, program: str) -> Path:
        return self._get_program_dir(program) / "templates"

    def _next_template_id(self, program: str) -> str:
        """Generate next sequential template ID."""
        tmpl_dir = self._get_template_dir(program)
        tmpl_dir.mkdir(parents=True, exist_ok=True)

        existing = [d.name for d in tmpl_dir.iterdir()
                    if d.is_dir() and d.name.startswith("template_")]
        nums = []
        for name in existing:
            try:
                nums.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                pass

        next_num = max(nums) + 1 if nums else 1
        return f"template_{next_num}"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip())
        safe = safe.replace("..", "_")
        if not safe:
            raise ProgramError("Nama tidak valid (kosong)")
        return safe

    @staticmethod
    def _load_json(path: Path, default: Any = None) -> Any:
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    @staticmethod
    def _atomic_write(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8",
            dir=path.parent, suffix=".tmp", delete=False,
        )
        try:
            json.dump(data, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
            try:
                os.fsync(tmp.fileno())
            except (OSError, AttributeError):
                pass
            tmp.close()
            os.replace(tmp.name, str(path))
        except Exception:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            raise
