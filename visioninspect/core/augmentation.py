"""
VisionInspect - Data Augmentation
Generate variasi tambahan dari gambar OK/NG training via transformasi
klasik (rotasi, flip, translasi, brightness, contrast) — sengaja TIDAK
menyediakan cutout/random-erasing/distorsi berat/blur berat, karena semua
itu bisa menyerupai defect asli (scratch, kontaminasi) dan justru mengajari
model bahwa tampilan mirip-cacat itu normal.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import numpy.typing as npt

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("training")

# ── Constants ──────────────────────────────────────────────────────────

AUGMENTATION_TYPES = (
    "rotation", "flip_horizontal", "flip_vertical",
    "translation", "brightness", "contrast",
)

# Rentang default dipakai saat parameter max_* dikosongkan user (mode Acak).
DEFAULT_RANGES = {
    "rotation": 15.0,      # ± derajat
    "translation": 10.0,   # ± persen lebar/tinggi gambar
    "brightness": 20.0,    # ± persen
    "contrast": 20.0,      # ± persen
}

DEFAULT_AUGMENTATION_CONFIG: dict = {
    "count_per_type": 5,
    "rotation": {"enabled": False, "max_degrees": None},
    "flip_horizontal": {"enabled": False},
    "flip_vertical": {"enabled": False},
    "translation": {"enabled": False, "max_percent": None},
    "brightness": {"enabled": False, "max_percent": None},
    "contrast": {"enabled": False, "max_percent": None},
    "generated_config_hash": None,
    "generated_at": None,
}


# ── Transform Functions ─────────────────────────────────────────────────

def _rotate(img: npt.NDArray, max_degrees: Optional[float]) -> npt.NDArray:
    limit = max_degrees if max_degrees else DEFAULT_RANGES["rotation"]
    angle = random.uniform(-limit, limit)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _flip_horizontal(img: npt.NDArray, _unused: Optional[float] = None) -> npt.NDArray:
    return cv2.flip(img, 1)


def _flip_vertical(img: npt.NDArray, _unused: Optional[float] = None) -> npt.NDArray:
    return cv2.flip(img, 0)


def _translate(img: npt.NDArray, max_percent: Optional[float]) -> npt.NDArray:
    limit = max_percent if max_percent else DEFAULT_RANGES["translation"]
    h, w = img.shape[:2]
    dx = random.uniform(-limit, limit) / 100.0 * w
    dy = random.uniform(-limit, limit) / 100.0 * h
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _brightness(img: npt.NDArray, max_percent: Optional[float]) -> npt.NDArray:
    limit = max_percent if max_percent else DEFAULT_RANGES["brightness"]
    beta = random.uniform(-limit, limit) / 100.0 * 255.0
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


def _contrast(img: npt.NDArray, max_percent: Optional[float]) -> npt.NDArray:
    limit = max_percent if max_percent else DEFAULT_RANGES["contrast"]
    alpha = 1.0 + random.uniform(-limit, limit) / 100.0
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


# Tiap entri: (fungsi, nama parameter rentang di config atau None kalau
# augmentasi-nya deterministik/tanpa parameter seperti flip).
_TRANSFORMS: dict = {
    "rotation": (_rotate, "max_degrees"),
    "flip_horizontal": (_flip_horizontal, None),
    "flip_vertical": (_flip_vertical, None),
    "translation": (_translate, "max_percent"),
    "brightness": (_brightness, "max_percent"),
    "contrast": (_contrast, "max_percent"),
}


# ── Config Hash (untuk deteksi skip-jika-tidak-berubah) ─────────────────

def compute_config_hash(config: dict, rois: Optional[list] = None) -> str:
    """Hash deterministik dari field augmentasi yang relevan — mengecualikan
    generated_config_hash/generated_at sendiri supaya tidak muter.

    rois (opsional) diikutsertakan sebagai fingerprint posisi/ukuran ROI saat
    ini — augmentasi geometris (rotasi/flip/translasi) dijalankan di atas
    hasil crop ROI, jadi kalau ROI digeser/diubah ukurannya, cache augmentasi
    lama harus otomatis dianggap basi juga, bukan cuma kalau setting
    augmentasi sendiri yang berubah.
    """
    relevant = {k: v for k, v in config.items()
                if k not in ("generated_config_hash", "generated_at")}
    if rois:
        relevant["_rois_fingerprint"] = sorted(
            (r.get("uid"), r.get("x"), r.get("y"), r.get("width"), r.get("height"))
            for r in rois)
    payload = json.dumps(relevant, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


# ── Orchestrator ─────────────────────────────────────────────────────────

def _clear_dir(d: Path) -> None:
    if not d.exists():
        return
    for f in d.iterdir():
        if f.is_file():
            f.unlink()


def _augment_one_dir(src_dir: Path, out_dir: Path, config: dict,
                      progress_cb: Optional[Callable[[int, str], None]],
                      progress_base: int, progress_span: int) -> int:
    """Generate augmented images dari semua gambar di src_dir ke out_dir.
    Returns jumlah gambar yang di-generate."""
    images = (list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg"))
              + list(src_dir.glob("*.jpeg"))) if src_dir and src_dir.exists() else []
    if not images:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    count_per_type = max(1, int(config.get("count_per_type", 5)))
    enabled_types = [t for t in AUGMENTATION_TYPES if config.get(t, {}).get("enabled")]
    if not enabled_types:
        return 0

    # Augmentasi tanpa parameter rentang (flip) itu deterministik — generate
    # lebih dari 1 salinan identik itu sia-sia (buang disk, dan membuat
    # memory bank PatchCore bias ke 1 varian yang sama berkali-kali).
    def _count_for(aug_type: str) -> int:
        return count_per_type if _TRANSFORMS[aug_type][1] else 1

    total_ops = len(images) * sum(_count_for(t) for t in enabled_types)
    done = 0
    generated = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Augmentasi: gagal baca %s, dilewati", img_path)
            continue
        for aug_type in enabled_types:
            fn, param_key = _TRANSFORMS[aug_type]
            param_val = config[aug_type].get(param_key) if param_key else None
            for i in range(_count_for(aug_type)):
                out_img = fn(img, param_val)
                dest = out_dir / f"aug_{aug_type}_{img_path.stem}_{i}.png"
                cv2.imwrite(str(dest), out_img)
                generated += 1
                done += 1
                if progress_cb and total_ops:
                    pct = progress_base + int(progress_span * done / total_ops)
                    progress_cb(pct, f"Augmentasi {aug_type}: {done}/{total_ops}")
    return generated


def generate_augmentations(
    ok_dir: Path, ng_dir: Optional[Path],
    ok_out_dir: Path, ng_out_dir: Path,
    config: dict, rois: Optional[list] = None, force: bool = False,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> dict:
    """Generate augmented images dari ok_dir/ng_dir ke ok_out_dir/ng_out_dir,
    skip kalau config (+ rois, lihat compute_config_hash) tidak berubah sejak
    generate terakhir (kecuali force=True).

    ok_dir/ng_dir HARUS sudah berupa hasil crop ROI (kalau template punya
    ROI) — augmentasi geometris (rotasi/flip/translasi) yang dijalankan di
    full-frame lalu di-crop pakai kotak ROI yang diam di tempat akan salah
    sasaran (part bisa tergeser keluar kotak crop, atau flip malah menangkap
    area yang tidak berhubungan sama sekali). Kalau template tidak punya ROI,
    ok_dir/ng_dir boleh full-frame (tidak ada kotak crop tetap yang bisa
    salah sasaran dalam kasus itu).

    Returns: {"generated": bool, "ok_count": int, "ng_count": int, "config_hash": str}
    """
    current_hash = compute_config_hash(config, rois)
    stored_hash = config.get("generated_config_hash")
    has_existing = ok_out_dir.exists() and any(ok_out_dir.iterdir())

    if not force and has_existing and current_hash == stored_hash:
        logger.info("Augmentasi: skip (config tidak berubah, hash=%s)", current_hash)
        return {"generated": False, "ok_count": 0, "ng_count": 0,
                "config_hash": current_hash}

    logger.info("Augmentasi: generate ulang (force=%s, hash %s -> %s)",
                force, stored_hash, current_hash)
    _clear_dir(ok_out_dir)
    _clear_dir(ng_out_dir)

    ok_count = _augment_one_dir(ok_dir, ok_out_dir, config, progress_cb, 0, 50)
    ng_count = (_augment_one_dir(ng_dir, ng_out_dir, config, progress_cb, 50, 50)
                if ng_dir else 0)

    logger.info("Augmentasi selesai: %d OK, %d NG di-generate", ok_count, ng_count)
    return {"generated": True, "ok_count": ok_count, "ng_count": ng_count,
            "config_hash": current_hash}
