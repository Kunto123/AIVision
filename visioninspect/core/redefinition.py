"""
VisionInspect - Redefinition Loop
Logika koreksi hasil inspeksi, rebuild model, versioning & rollback.
"""

import json
import time
from pathlib import Path
from typing import Callable, Optional

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")


class RedefinitionError(Exception):
    pass


class RedefinitionManager:
    """
    Mengelola redefinition loop:
    - Koreksi hasil (operator menandai OK → NG atau NG → OK)
    - Rebuild model dengan data koreksi
    - Versioning dan rollback
    - Audit trail
    """

    def __init__(self, program_manager, training_pipeline, inference_engine):
        self._pm = program_manager
        self._training = training_pipeline
        self._engine = inference_engine
        self._progress_callback: Optional[Callable[[int, str], None]] = None

    def set_progress_callback(self, cb: Optional[Callable[[int, str], None]]) -> None:
        self._progress_callback = cb
        if self._training:
            self._training.set_progress_callback(cb)

    def correct_result(self, program: str, image_path: Path,
                       original_judgement: str, correct_judgement: str) -> dict:
        """
        Koreksi hasil inspeksi yang salah.
        - Pindahkan gambar ke corrections/{ok,ng}
        - Catat audit trail
        """
        if original_judgement == correct_judgement:
            logger.warning("Koreksi sama dengan asli, diabaikan")
            return {"status": "skipped"}

        # Determine correction label
        if correct_judgement == "OK":
            label = "ok"
        elif correct_judgement == "NG":
            label = "ng"
        else:
            raise RedefinitionError(f"Invalid judgement: {correct_judgement}")

        # Copy image to corrections directory
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            raise RedefinitionError(f"Gambar tidak ditemukan: {image_path}")

        dest = self._pm.save_image(program, img, label, correction=True)

        # Audit trail
        audit = self._log_audit(program, {
            "action": "correction",
            "original_judgement": original_judgement,
            "correct_judgement": correct_judgement,
            "image_source": str(image_path),
            "image_dest": str(dest),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        logger.info(
            "Koreksi: %s → %s (program: %s, image: %s)",
            original_judgement, correct_judgement, program, dest
        )

        return {
            "status": "corrected",
            "label": label,
            "dest": str(dest),
            "audit_id": audit.get("id"),
        }

    def rebuild_model(self, program: str, output_dir: Path) -> dict:
        """
        Rebuild model dengan menggabungkan data asli + koreksi.
        Model lama tetap melayani inferensi sampai model baru siap (hot-swap).
        """
        prog_dir = Path(self._pm.get_program_info(program)["path"])
        ok_dirs = [
            prog_dir / "images" / "ok",
            prog_dir / "images" / "corrections" / "ok",
        ]
        ng_dirs = [
            prog_dir / "images" / "ng",
            prog_dir / "images" / "corrections" / "ng",
        ]

        # Collect all OK and NG images
        all_ok = []
        for d in ok_dirs:
            if d.exists():
                all_ok.extend(list(d.glob("*.png")) + list(d.glob("*.jpg")))
        all_ng = []
        for d in ng_dirs:
            if d.exists():
                all_ng.extend(list(d.glob("*.png")) + list(d.glob("*.jpg")))

        if len(all_ok) < 1:
            raise RedefinitionError("Tidak ada gambar OK untuk training")

        logger.info(
            "Rebuild model: program=%s, %d OK, %d NG",
            program, len(all_ok), len(all_ng)
        )

        # Run training pipeline
        result = self._training.train(
            ok_dir=prog_dir / "images" / "ok",
            ng_dir=prog_dir / "images" / "ng" if (prog_dir / "images" / "ng").exists() else None,
            output_dir=output_dir,
        )

        # Save as new version
        version = self._pm.save_model_version(program, result)
        result["version"] = version

        # Audit trail
        self._log_audit(program, {
            "action": "rebuild",
            "version": version,
            "num_ok": len(all_ok),
            "num_ng": len(all_ng),
            "threshold": result.get("threshold"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Hot-swap model in inference engine
        model_path = Path(result.get("int8_path") or result.get("export_path", ""))
        if model_path.exists():
            try:
                self._engine.hot_swap(model_path, threshold=result.get("threshold"))
                logger.info("Hot-swap model berhasil: %s", model_path)
            except Exception as e:
                logger.error("Hot-swap gagal: %s", e)
        else:
            logger.warning("Model path tidak ditemukan untuk hot-swap: %s", model_path)

        return result

    def rollback_model(self, program: str, version: int) -> None:
        """Rollback model ke versi tertentu."""
        self._pm.rollback_to_version(program, version)

        # Reload model di inference engine
        prog_dir = Path(self._pm.get_program_info(program)["path"])
        model_dir = prog_dir / "model"

        # Try INT8 first, then OpenVINO
        int8_path = model_dir / "openvino_int8" / "model.xml"
        ov_path = model_dir / "openvino" / "model.xml"

        if int8_path.exists():
            self._engine.hot_swap(int8_path)
        elif ov_path.exists():
            self._engine.hot_swap(ov_path)

        # Audit trail
        self._log_audit(program, {
            "action": "rollback",
            "version": version,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        logger.info("Rollback program '%s' ke v%d selesai", program, version)

    # ---- Audit Trail ----

    def _log_audit(self, program: str, entry: dict) -> dict:
        """Log audit entry ke file JSON."""
        prog_dir = Path(self._pm.get_program_info(program)["path"])
        audit_dir = prog_dir / "audit"
        audit_dir.mkdir(exist_ok=True)

        entry["id"] = f"{int(time.time())}_{__import__('uuid').uuid4().hex[:8]}"

        audit_path = audit_dir / f"{entry['id']}.json"
        with open(audit_path, "w") as f:
            json.dump(entry, f, indent=2)

        return entry

    def get_audit_trail(self, program: str, limit: int = 100) -> list[dict]:
        """Get audit trail entries."""
        prog_dir = Path(self._pm.get_program_info(program)["path"])
        audit_dir = prog_dir / "audit"
        if not audit_dir.exists():
            return []

        entries = []
        for f in sorted(audit_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                with open(f, "r") as fp:
                    entries.append(json.load(fp))
            except Exception:
                pass
        return entries
