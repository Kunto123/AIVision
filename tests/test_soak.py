#!/usr/bin/env python3
"""
VisionInspect — Soak Test
Simulasi ribuan frame untuk menguji stabilitas dan memory leak.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psutil

from visioninspect.core.inference import InferenceEngine
from visioninspect.utils.logging_setup import setup_logging, get_logger


def main():
    setup_logging(Path("visioninspect-soak-logs"))
    logger = get_logger("app")

    N_FRAMES = 10000
    frame_size = (480, 640, 3)

    print(f"=== VisionInspect Soak Test ===")
    print(f"Simulasi {N_FRAMES} frame, ukuran {frame_size}")
    print()

    # Test inference engine (no model loaded — just API test)
    engine = InferenceEngine(input_size=256)

    process = psutil.Process()
    start_mem = process.memory_info().rss / 1024 / 1024
    print(f"Memory awal: {start_mem:.1f} MB")

    latencies = []
    start = time.perf_counter()

    for i in range(N_FRAMES):
        # Generate synthetic frame
        frame = np.random.randint(0, 256, frame_size, dtype=np.uint8)

        # Mock ROI
        roi = {"x": 100, "y": 50, "width": 256, "height": 256}

        # Run inference (no model → returns OK with 0 score)
        result = engine.infer(frame, roi)
        latencies.append(result.latency_ms)

        if (i + 1) % 1000 == 0:
            elapsed = time.perf_counter() - start
            fps = (i + 1) / elapsed
            mem = process.memory_info().rss / 1024 / 1024
            avg_lat = sum(latencies[-1000:]) / len(latencies[-1000:])
            print(f"  Frame {i+1:5d}/{N_FRAMES} | "
                  f"FPS: {fps:.1f} | "
                  f"Lat: {avg_lat:.2f}ms | "
                  f"Mem: {mem:.1f}MB")

    elapsed = time.perf_counter() - start
    avg_lat = sum(latencies) / len(latencies)
    final_mem = process.memory_info().rss / 1024 / 1024
    mem_delta = final_mem - start_mem

    print()
    print(f"=== Hasil ===")
    print(f"Total frame:    {N_FRAMES}")
    print(f"Waktu:          {elapsed:.1f}s")
    print(f"FPS rata-rata:  {N_FRAMES/elapsed:.1f}")
    print(f"Latensi avg:    {avg_lat:.3f}ms")
    print(f"Memory awal:    {start_mem:.1f} MB")
    print(f"Memory akhir:   {final_mem:.1f} MB")
    print(f"Memory delta:   {mem_delta:+.1f} MB")

    if mem_delta > 50:
        print("⚠️  PERINGATAN: Potensi memory leak (delta > 50 MB)!")
        return 1
    else:
        print("✅ Memory stabil (delta < 50 MB)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
