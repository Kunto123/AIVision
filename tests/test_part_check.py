"""
Unit tests for Part Presence Check module (core/part_check.py).
Pure numpy/cv2 — no Qt dependency, runs in any environment.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from visioninspect.core.part_check import (
    crop_roi,
    compute_edge_density,
    compute_master_stats,
    evaluate_part_presence,
    part_check_state,
    PartCheckResult,
    MIN_STD,
)


class TestCropROI:
    """crop_roi() bounds checking and degenerate cases."""

    def test_normal_crop(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8)
        roi = {"x": 50, "y": 30, "width": 200, "height": 150}
        cropped = crop_roi(frame, roi)
        assert cropped is not None
        assert cropped.shape == (150, 200, 3)

    def test_clamp_out_of_bounds(self):
        """ROI partially outside frame — clamp, not crash."""
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        roi = {"x": 80, "y": 80, "width": 100, "height": 100}
        cropped = crop_roi(frame, roi)
        assert cropped is not None
        h, w = cropped.shape[:2]
        assert h > 0 and w > 0
        assert h <= 20 and w <= 20  # clamped to available area

    def test_negative_coordinates(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        roi = {"x": -50, "y": -30, "width": 200, "height": 150}
        cropped = crop_roi(frame, roi)
        assert cropped is not None
        h, w = cropped.shape[:2]
        assert h > 0 and w > 0

    def test_minimum_1px(self):
        """Degenerate ROI (0-dim) clamped to at least 1px."""
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        roi = {"x": 10, "y": 10, "width": 0, "height": 0}
        cropped = crop_roi(frame, roi)
        assert cropped is not None
        assert cropped.shape[0] >= 1 and cropped.shape[1] >= 1

    def test_none_roi(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        assert crop_roi(frame, None) is None


class TestEdgeDensity:
    """compute_edge_density() bounds and behavior."""

    def test_flat_image_near_zero(self):
        img = np.ones((50, 50, 3), dtype=np.uint8) * 200
        d = compute_edge_density(img)
        assert 0.0 <= d <= 0.01

    def test_image_with_edges(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.rectangle(img, (5, 5), (25, 25), (255, 255, 255), -1)
        d = compute_edge_density(img)
        assert d > 0.01

    def test_range_bounds(self):
        img = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        d = compute_edge_density(img)
        assert 0.0 <= d <= 1.0

    def test_canny_params_edge(self):
        """Higher Canny threshold = fewer edges on noisy image."""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        d_low = compute_edge_density(img, canny_low=10, canny_high=30)
        d_high = compute_edge_density(img, canny_low=200, canny_high=255)
        assert d_low >= d_high  # lower threshold = more edges detected

    def test_empty_image(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert compute_edge_density(empty) == 0.0


class TestMasterStats:
    """compute_master_stats() correctness."""

    def test_flat_color(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        stats = compute_master_stats(img)
        assert len(stats["mean_bgr"]) == 3
        assert all(abs(m - 128.0) < 1.0 for m in stats["mean_bgr"])
        assert all(s >= MIN_STD for s in stats["std_bgr"])
        assert stats["edge_density"] < 0.01
        assert stats["image_size"] == [100, 100]

    def test_std_floor(self):
        """Identical pixels produce std < MIN_STD, should be clamped."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 200
        stats = compute_master_stats(img)
        for s in stats["std_bgr"]:
            assert s >= MIN_STD, f"std {s} should be floored to {MIN_STD}"

    def test_checkerboard_edges(self):
        """High-contrast pattern should produce edge density > 0."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[::2, ::2] = [255, 255, 255]  # checkerboard
        stats = compute_master_stats(img)
        assert stats["edge_density"] > 0.01  # definitely some edges


class TestEvaluatePartPresence:
    """evaluate_part_presence() logic."""

    @pytest.fixture
    def master_frame(self):
        return np.ones((480, 640, 3), dtype=np.uint8) * 200

    @pytest.fixture
    def gate_roi(self):
        return {"x": 100, "y": 80, "width": 200, "height": 150}

    @pytest.fixture
    def master_stats(self, master_frame, gate_roi):
        cropped = crop_roi(master_frame, gate_roi)
        return compute_master_stats(cropped)

    def _make_cfg(self, master_stats, method="both", **overrides):
        cfg = {
            "has_master": True,
            "method": method,
            "color_threshold": 0.35,
            "edge_threshold": 0.5,  # ratio-based (default baru)
            "canny_low": 50,
            "canny_high": 150,
            "master_mean_bgr": master_stats["mean_bgr"],
            "master_std_bgr": master_stats["std_bgr"],
            "master_edge_density": master_stats["edge_density"],
        }
        cfg.update(overrides)
        return cfg

    def test_same_image_ready(self, master_frame, gate_roi, master_stats):
        """When live == master, all methods should return ready=True."""
        cfg = self._make_cfg(master_stats)
        result = evaluate_part_presence(master_frame, gate_roi, cfg)
        assert result.ready
        assert result.reason == ""

    def test_color_method(self, master_frame, gate_roi, master_stats):
        """color method only checks color_score, ignores edge."""
        cfg = self._make_cfg(master_stats, method="color",
                             color_threshold=0.35)
        # Same color → ready
        result = evaluate_part_presence(master_frame, gate_roi, cfg)
        assert result.ready
        assert result.color_ready

    def test_edge_method(self, master_frame, gate_roi, master_stats):
        """edge method only checks edge_score (rasio)."""
        cfg = self._make_cfg(master_stats, method="edge",
                             edge_threshold=0.5)
        # Same edges → ratio = 0 → ready
        result = evaluate_part_presence(master_frame, gate_roi, cfg)
        assert result.ready
        assert result.edge_ready

    def test_color_mismatch(self, master_frame, gate_roi, master_stats):
        """Different color → color_ready=False."""
        different = np.ones_like(master_frame) * 50
        cfg = self._make_cfg(master_stats, color_threshold=0.1)
        result = evaluate_part_presence(different, gate_roi, cfg)
        assert not result.color_ready
        assert result.color_score is not None

    def test_both_method_and_condition(self, master_frame, gate_roi, master_stats):
        """both: ready = color_ready AND edge_ready (not OR)."""
        # Master is flat 200-gray — almost no edges (edge_density ≈ 0)
        # Create frame with SAME mean color but noise (adds edges)
        noisy = master_frame.copy()
        y, x = gate_roi["y"], gate_roi["x"]
        h, w = gate_roi["height"], gate_roi["width"]
        # Noise centered at 0, so mean doesn't shift
        noise = np.random.randint(-30, 30, (h, w, 3), dtype=np.int16)
        gate_area = noisy[y:y+h, x:x+w].astype(np.int16)
        noisy[y:y+h, x:x+w] = np.clip(gate_area + noise, 0, 255).astype(np.uint8)

        cfg = self._make_cfg(master_stats, method="both",
                             color_threshold=5.0,  # lenient on color
                             edge_threshold=0.001)  # strict on edge
        result = evaluate_part_presence(noisy, gate_roi, cfg)
        assert result.color_ready   # same mean color
        assert not result.edge_ready  # added noise = more edges
        assert not result.ready     # AND → not ready

    def test_no_master(self, master_frame, gate_roi, master_stats):
        cfg = {"has_master": False, "method": "both"}
        result = evaluate_part_presence(master_frame, gate_roi, cfg)
        assert not result.ready
        assert result.reason == "no_master"

    def test_invalid_gate_roi_degenerate(self, master_frame, master_stats):
        """ROI too small → reason='invalid_gate_roi'."""
        bad_roi = {"x": 0, "y": 0, "width": 1, "height": 1}
        cfg = self._make_cfg(master_stats)
        result = evaluate_part_presence(master_frame, bad_roi, cfg)
        assert not result.ready
        assert result.reason == "invalid_gate_roi"

    def test_edge_ready_different_edge(self, master_frame, gate_roi, master_stats):
        """Edge method should fail when edge ratio exceeds threshold."""
        # Master is flat 200-gray (edge_density ≈ 0)
        # Create frame with noise in gate region (max edges)
        noisy = master_frame.copy()
        y, x = gate_roi["y"], gate_roi["x"]
        h, w = gate_roi["height"], gate_roi["width"]
        noisy[y:y+h, x:x+w] = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        cfg = self._make_cfg(master_stats, method="edge",
                             edge_threshold=0.001)  # very strict
        result = evaluate_part_presence(noisy, gate_roi, cfg)
        assert not result.edge_ready

    def test_clamp_out_of_bounds_gate(self, master_frame, master_stats):
        """Gate ROI outside frame should clamp and not crash."""
        far_roi = {"x": 5000, "y": 5000, "width": 100, "height": 100}
        cfg = self._make_cfg(master_stats)
        # Should not raise
        result = evaluate_part_presence(master_frame, far_roi, cfg)
        # Result may be ready or not, but should not crash
        assert isinstance(result, PartCheckResult)

    # ── New tests: edge ratio threshold boundary ──

    def test_edge_ready_near_threshold(self, master_frame, gate_roi, master_stats):
        """Edge ratio: threshold above vs below actual ratio → ready toggles."""
        # Master is flat 200-gray → master_edge ≈ 0
        # Use white rectangle di gate untuk edge density terukur
        y, x = gate_roi["y"], gate_roi["x"]
        h, w = gate_roi["height"], gate_roi["width"]
        high_contrast = master_frame.copy()
        cv2.rectangle(high_contrast, (x, y), (x + w - 1, y + h - 1),
                      (255, 255, 255), -1)

        cropped = crop_roi(high_contrast, gate_roi)
        actual_live_edge = compute_edge_density(cropped, 50, 150)
        master_edge = float(master_stats["edge_density"])
        denom = max(master_edge, 0.01, 1e-7)
        expected_ratio = abs(actual_live_edge - master_edge) / denom

        # Threshold di bawah ratio → reject
        cfg_below = self._make_cfg(
            master_stats, method="edge",
            edge_threshold=expected_ratio * 0.5)
        r_below = evaluate_part_presence(high_contrast, gate_roi, cfg_below)
        assert not r_below.edge_ready, (
            f"ratio={expected_ratio:.4f} harus > threshold={expected_ratio*0.5:.4f}")

        # Threshold di atas ratio → accept
        cfg_above = self._make_cfg(
            master_stats, method="edge",
            edge_threshold=expected_ratio * 2.0 + 0.1)
        r_above = evaluate_part_presence(high_contrast, gate_roi, cfg_above)
        assert r_above.edge_ready, (
            f"ratio={expected_ratio:.4f} harus < threshold={expected_ratio*2.0+0.1:.4f}")

    # ── New tests: None master fields → ready=False, bukan exception ──

    def test_edge_missing_edge_density(self, master_frame, gate_roi):
        """edge method: master_edge_density=None → ready=False."""
        cfg = {
            "has_master": True, "method": "edge",
            "master_edge_density": None,
            "edge_threshold": 0.5,
            "canny_low": 50, "canny_high": 150,
        }
        result = evaluate_part_presence(master_frame, gate_roi, cfg)
        assert not result.ready
        assert result.reason == "no_master"

    def test_color_missing_mean_bgr(self, master_frame, gate_roi, master_stats):
        """color method: master_mean_bgr=None → ready=False."""
        cfg = {
            "has_master": True, "method": "color",
            "master_mean_bgr": None,
            "master_std_bgr": master_stats["std_bgr"],
            "color_threshold": 0.35,
        }
        result = evaluate_part_presence(master_frame, gate_roi, cfg)
        assert not result.ready
        assert result.reason == "no_master"

    def test_color_missing_std_bgr(self, master_frame, gate_roi, master_stats):
        """color method: master_std_bgr=None → ready=False."""
        cfg = {
            "has_master": True, "method": "color",
            "master_mean_bgr": master_stats["mean_bgr"],
            "master_std_bgr": None,
            "color_threshold": 0.35,
        }
        result = evaluate_part_presence(master_frame, gate_roi, cfg)
        assert not result.ready
        assert result.reason == "no_master"

    def test_edge_method_ignores_color_fields(self, master_frame, gate_roi, master_stats):
        """edge method: master_mean_bgr/std_bgr=None TIDAK menyebabkan error."""
        cfg = {
            "has_master": True, "method": "edge",
            "master_edge_density": master_stats["edge_density"],
            "master_mean_bgr": None,
            "master_std_bgr": None,
            "edge_threshold": 0.5,
            "canny_low": 50, "canny_high": 150,
        }
        result = evaluate_part_presence(master_frame, gate_roi, cfg)
        assert result.ready  # edge-only, tidak sentuh color fields

    def test_color_method_ignores_edge_fields(self, master_frame, gate_roi, master_stats):
        """color method: master_edge_density=None TIDAK menyebabkan error."""
        cfg = {
            "has_master": True, "method": "color",
            "master_mean_bgr": master_stats["mean_bgr"],
            "master_std_bgr": master_stats["std_bgr"],
            "master_edge_density": None,
            "color_threshold": 0.35,
        }
        result = evaluate_part_presence(master_frame, gate_roi, cfg)
        assert result.ready  # color-only, tidak sentuh edge fields


class TestPartCheckState:
    """part_check_state() returns correct state label."""

    def test_disabled_when_not_enabled(self):
        assert part_check_state({"enabled": False}) == "disabled"
        assert part_check_state({}) == "disabled"
        assert part_check_state({"enabled": True, "has_master": False, "gate_roi": None}) == "incomplete"
        assert part_check_state({"enabled": True, "has_master": True, "gate_roi": None}) == "incomplete"
        assert part_check_state({"enabled": True, "has_master": False, "gate_roi": {"x": 0}}) == "incomplete"

    def test_incomplete_missing_master(self):
        cfg = {"enabled": True, "has_master": False, "gate_roi": {"x": 0, "y": 0, "width": 100, "height": 100}}
        assert part_check_state(cfg) == "incomplete"

    def test_incomplete_missing_gate_roi(self):
        cfg = {"enabled": True, "has_master": True, "gate_roi": None}
        assert part_check_state(cfg) == "incomplete"

    def test_active_fully_configured(self):
        cfg = {"enabled": True, "has_master": True, "gate_roi": {"x": 0, "y": 0, "width": 100, "height": 100}}
        assert part_check_state(cfg) == "active"
