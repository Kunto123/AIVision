"""
VisionInspect — Polygon Mask per ROI
Menolkan piksel di luar polygon (kontur part) di dalam crop ROI, sebelum resize.

Wajib diterapkan IDENTIK di training (`training_worker.py`) dan inference
(`inference.py`) — kalau hanya satu sisi, area yang ter-mask di sisi lain tidak
punya referensi "normal" → false NG, bukan perbaikan.

Koordinat polygon selalu ruang ROI-lokal (0,0 = pojok kiri-atas crop, ukuran
asli sebelum resize), jadi tetap valid walau input_size berubah — hanya perlu
digambar ulang kalau kotak ROI (x/y/width/height) berubah.
"""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def apply_polygon_mask(
    crop_bgr: np.ndarray,
    polygon: Optional[Sequence[Tuple[int, int]]],
    fill_value: int = 0,
) -> np.ndarray:
    """Nolkan piksel di luar `polygon` (koordinat ROI-lokal, sebelum resize).
    `polygon` None/kosong → no-op (ROI tanpa mask berperilaku seperti dulu)."""
    if not polygon or len(polygon) < 3:
        return crop_bgr
    mask = np.zeros(crop_bgr.shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    out = crop_bgr.copy()
    out[mask == 0] = fill_value
    return out


def resolve_polygon_for_image(
    roi: dict,
    overrides: Optional[dict],
    image_key: str,
) -> Optional[List[Tuple[int, int]]]:
    """Polygon untuk satu gambar: override kalau ada, fallback ke default ROI.
    `overrides` = `image_mask_overrides[roi_uid]` ({image_key: polygon})."""
    if overrides:
        override = overrides.get(image_key)
        if override:
            return [tuple(p) for p in override]
    default = roi.get("mask_polygon")
    return [tuple(p) for p in default] if default else None
