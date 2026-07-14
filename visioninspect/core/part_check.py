"""
VisionInspect - Part Presence Check (Step 1)
Classical computer vision gate sebelum QC OK/NG (Step 2).
Mendeteksi apakah part sudah terpasang di area gate ROI
menggunakan mean/std warna dan/atau Canny edge density.

Threshold default perlu di-tuning terhadap kamera & lighting asli.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")

# ── Constants ──────────────────────────────────────────────────────────

PART_CHECK_METHODS = ("color", "edge", "both")
METHOD_LABELS = {
    "color": "Warna (mean/std)",
    "edge": "Tepi (Canny)",
    "both": "Keduanya (AND)",
}

DEFAULT_PART_CHECK_CONFIG: dict = {
    "enabled": False,
    "method": "both",
    "gate_roi": None,  # {x, y, width, height} or None
    "color_threshold": 0.35,
    "edge_threshold": 0.08,
    "canny_low": 50,
    "canny_high": 150,
    "has_master": False,
    "master_mean_bgr": None,  # [b, g, r] float
    "master_std_bgr": None,  # [b, g, r] float (clipped)
    "master_edge_density": None,  # float [0,1]
    "master_captured_at": None,  # ISO timestamp
    "master_image_size": None,  # [height, width]
}

MIN_STD = 5.0  # floor for std_bgr (0-255 scale), prevents div-by-zero


# ── Data Classes ───────────────────────────────────────────────────────

@dataclass
class PartCheckResult:
    """Result of a single part presence evaluation."""
    ready: bool
    method: str
    color_score: Optional[float] = None
    color_ready: Optional[bool] = None
    edge_score: Optional[float] = None
    edge_ready: Optional[bool] = None
    live_edge_density: Optional[float] = None
    reason: str = ""  # "no_master", "invalid_gate_roi", "mismatch"


# ── Core Functions ─────────────────────────────────────────────────────

def crop_roi(frame: npt.NDArray, roi: dict) -> Optional[npt.NDArray]:
    """Crop frame to ROI with bounds checking (min 1px)."""
    if roi is None:
        return None
    h_img, w_img = frame.shape[:2]
    x = max(0, min(int(roi["x"]), w_img - 1))
    y = max(0, min(int(roi["y"]), h_img - 1))
    w = max(1, min(int(roi.get("width", 64)), w_img - x))
    h = max(1, min(int(roi.get("height", 64)), h_img - y))
    return frame[y:y + h, x:x + w]


def compute_edge_density(
    image_bgr: npt.NDArray,
    canny_low: int = 50,
    canny_high: int = 150,
) -> float:
    """Compute edge density using Canny. Gray → Canny → fraction of edge pixels.
    Returns float in [0, 1]."""
    if image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    return float(np.count_nonzero(edges)) / float(edges.size)


def compute_master_stats(
    master_bgr: npt.NDArray,
    canny_low: int = 50,
    canny_high: int = 150,
) -> dict:
    """Compute master statistics from a BGR image (already cropped to gate ROI).
    Returns dict with:
        mean_bgr (list[float] 3), std_bgr (list[float] 3, floored),
        edge_density (float), image_size (list[int] h,w)
    """
    mean_bgr = cv2.mean(master_bgr)[:3]  # (B, G, R) float
    # Manual std per channel (cv2.meanStdDev returns weird format)
    f32 = master_bgr.astype(np.float32)
    std_bgr = [float(np.std(f32[:, :, c])) for c in range(3)]
    std_bgr = [max(s, MIN_STD) for s in std_bgr]
    return {
        "mean_bgr": list(mean_bgr),
        "std_bgr": std_bgr,
        "edge_density": compute_edge_density(master_bgr, canny_low, canny_high),
        "image_size": list(master_bgr.shape[:2]),
    }


def evaluate_part_presence(
    frame_bgr: npt.NDArray,
    gate_roi: dict,
    part_check_cfg: dict,
) -> PartCheckResult:
    """Evaluate whether a part is present at the gate ROI.
    Compares live frame against master statistics.

    Args:
        frame_bgr: Full camera frame (BGR).
        gate_roi: Dict with x, y, width, height.
        part_check_cfg: Part check config dict.

    Returns:
        PartCheckResult with ready flag.
    """
    if not part_check_cfg.get("has_master"):
        return PartCheckResult(
            ready=False, method=part_check_cfg.get("method", "both"),
            reason="no_master",
        )

    # Crop gate ROI
    cropped = crop_roi(frame_bgr, gate_roi)
    if cropped is None or cropped.size == 0 or cropped.shape[0] < 2 or cropped.shape[1] < 2:
        return PartCheckResult(
            ready=False, method=part_check_cfg.get("method", "both"),
            reason="invalid_gate_roi",
        )

    method = part_check_cfg.get("method", "both")
    color_threshold = float(part_check_cfg.get("color_threshold", 0.35))
    edge_threshold = float(part_check_cfg.get("edge_threshold", 0.08))
    canny_low = int(part_check_cfg.get("canny_low", 50))
    canny_high = int(part_check_cfg.get("canny_high", 150))

    master_mean = np.array(part_check_cfg["master_mean_bgr"], dtype=np.float32)
    master_std = np.array(part_check_cfg["master_std_bgr"], dtype=np.float32)

    # ── Color score: normalized z-score of mean BGR ──
    live_mean = cv2.mean(cropped)[:3]
    live_mean_arr = np.array(live_mean, dtype=np.float32)
    z = np.abs(live_mean_arr - master_mean) / master_std
    color_score = float(np.mean(z))
    color_ready = color_score < color_threshold

    # ── Edge score: absolute difference in edge density ──
    live_edge = compute_edge_density(cropped, canny_low, canny_high)
    master_edge = float(part_check_cfg["master_edge_density"])
    edge_score = abs(live_edge - master_edge)
    edge_ready = edge_score < edge_threshold

    # ── Combine ──
    if method == "color":
        ready = color_ready
    elif method == "edge":
        ready = edge_ready
    else:  # "both" (AND)
        ready = color_ready and edge_ready

    reason = "" if ready else "mismatch"

    return PartCheckResult(
        ready=ready,
        method=method,
        color_score=color_score,
        color_ready=color_ready,
        edge_score=edge_score,
        edge_ready=edge_ready,
        live_edge_density=live_edge,
        reason=reason,
    )
