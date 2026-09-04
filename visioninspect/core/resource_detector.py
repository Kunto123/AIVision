"""
VisionInspect - Resource Detector
Mendeteksi kemampuan hardware (GPU, RAM, CPU cores) dan memberikan
rekomendasi parameter training yang optimal untuk hardware tersebut.
"""

import os
import platform
from dataclasses import dataclass, field
from typing import Optional

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("resource")


@dataclass
class ResourceProfile:
    """Profil hardware yang terdeteksi."""

    # Deteksi hardware
    has_cuda: bool = False
    cuda_device_name: str = ""
    cuda_compute_capability: str = ""
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0

    # Rekomendasi
    device: str = "cpu"  # "cuda" | "cpu"
    mode: str = "balanced"  # "full" | "balanced" | "lightweight"
    batch_size: int = 8
    num_workers: int = 0
    precision: str = "32"  # "32" | "16-mixed"
    enable_mixed_precision: bool = False
    max_images: int = 9999  # batas aman jumlah gambar

    # Peringatan
    warnings: list[str] = field(default_factory=list)

    # Memory estimation
    estimated_peak_gb: float = 0.0
    safe_to_train: bool = True


def _get_ram_gb() -> float:
    """Deteksi RAM total dalam GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass

    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 * 1024), 1)
        elif platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            return round(mem.ullTotalPhys / (1024 ** 3), 1)
    except Exception:
        pass
    return 0.0


def _get_available_ram_gb() -> float:
    """Deteksi RAM tersedia dalam GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().available / (1024 ** 3), 1)
    except ImportError:
        return _get_ram_gb() * 0.5  # estimasi kasar


def _get_cpu_cores() -> tuple[int, int]:
    """Deteksi jumlah CPU cores (physical, logical)."""
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        return physical, logical
    except ImportError:
        logical = os.cpu_count() or 1
        return max(1, logical // 2), logical


def _detect_cuda() -> tuple[bool, str, str]:
    """Deteksi CUDA availability via PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
            return True, name, cap
        logger.info("CUDA tidak tersedia (PyTorch terdeteksi tapi cuda.is_available()=False)")
    except ImportError:
        logger.info("CUDA tidak tersedia (PyTorch tidak terinstall)")
    except Exception as e:
        logger.info("CUDA detection error: %s", e)
    return False, "", ""


def estimate_peak_memory(
    num_ok: int,
    num_ng: int,
    algorithm: str,
    aug_factor: int = 1,
    input_size: int = 256,
    coreset_ratio: float = 0.1,
) -> float:
    """Estimasi peak memory training (GB). PatchCore: memory bank O(n×d), ±4 MB
    per gambar (resnet18 256²), puncak SEBELUM coreset. EfficientAd jauh ringan."""
    total = (num_ok + num_ng) * aug_factor

    if algorithm == "efficientad":
        # EfficientAd: lightweight, ~1-2 GB peak
        base_per_100 = 1.5
    elif algorithm == "patchcore":
        if input_size <= 224:
            base_per_100 = 2.0  # 224px → lebih sedikit patch
        else:
            base_per_100 = 3.5  # 256px → ~1.7× lebih banyak patch dari 224
    else:
        base_per_100 = 2.5

    peak = max(0.5, (total / 100.0) * base_per_100)
    return round(peak, 1)


def detect_resource() -> ResourceProfile:
    """Deteksi hardware dan rekomendasikan parameter training optimal."""
    prof = ResourceProfile()

    # === Deteksi hardware ===
    prof.has_cuda, prof.cuda_device_name, prof.cuda_compute_capability = _detect_cuda()
    prof.total_ram_gb = _get_ram_gb()
    prof.available_ram_gb = _get_available_ram_gb()
    prof.cpu_cores_physical, prof.cpu_cores_logical = _get_cpu_cores()

    # === Logika rekomendasi ===

    # -- Device --
    if prof.has_cuda:
        prof.device = "cuda"
        logger.info("GPU terdeteksi: %s (Compute Capability: %s)",
                    prof.cuda_device_name, prof.cuda_compute_capability)

        # Cek kemampuan mixed precision (compute capability >= 7.0)
        try:
            cc_major = int(prof.cuda_compute_capability.split(".")[0])
            if cc_major >= 7:
                prof.enable_mixed_precision = True
                prof.precision = "16-mixed"
        except (ValueError, IndexError):
            pass

    # -- RAM-based mode --
    ram = prof.total_ram_gb
    if ram == 0.0:
        # Fallback jika tidak bisa deteksi RAM
        prof.mode = "balanced"
        prof.warnings.append("Tidak bisa mendeteksi RAM. Mode: balanced.")
    elif ram >= 24 and prof.has_cuda:
        prof.mode = "full"
        prof.batch_size = 16 if prof.has_cuda else 8
        prof.num_workers = min(8, prof.cpu_cores_logical)
        prof.max_images = 9999
    elif ram >= 16:
        prof.mode = "balanced"
        prof.batch_size = 8
        prof.num_workers = min(6, prof.cpu_cores_logical)
        prof.max_images = 500
    elif ram >= 12:
        prof.mode = "balanced"
        prof.batch_size = 8
        prof.num_workers = min(4, prof.cpu_cores_logical)
        prof.max_images = 200
    elif ram >= 8:
        prof.mode = "lightweight"
        prof.batch_size = 4
        prof.num_workers = min(2, prof.cpu_cores_logical)
        prof.max_images = 100
        prof.warnings.append(
            f"RAM {ram}GB terbatas. Training > 100 gambar + augmentasi berisiko crash. "
            "Disarankan train di PC Dev, lalu deploy model ke sini."
        )
    else:
        prof.mode = "lightweight"
        prof.batch_size = 2
        prof.num_workers = 0
        prof.max_images = 30
        prof.warnings.append(
            f"RAM {ram}GB sangat terbatas. Hanya training ringan dengan ≤ 30 gambar. "
            "Sebaiknya train di PC Dev dan deploy model."
        )

    # Pastikan num_workers tidak lebih dari CPU cores fisik (hindari oversubscription)
    prof.num_workers = min(prof.num_workers, max(1, prof.cpu_cores_physical))

    # Untuk PatchCore di RAM rendah, paksa num_workers=0 karena
    # multiprocessing data loading bisa double memory usage
    if not prof.has_cuda and ram < 12:
        prof.num_workers = 0

    return prof


def check_training_safety(
    profile: ResourceProfile,
    num_ok: int,
    algorithm: str,
    aug_count: int,
    aug_factor: int,
) -> ResourceProfile:
    """Periksa apakah training aman dijalankan dengan resource yang ada.
    Mengembalikan profile yang sudah di-update dengan warning dan saran.
    """
    total_images = (num_ok + (0)) * aug_factor
    if aug_count > 0:
        total_images = total_images * aug_count  # per type × count_per_type
    else:
        total_images = num_ok * aug_factor

    # Estimasi peak memory
    peak = estimate_peak_memory(
        num_ok=num_ok,
        num_ng=0,
        algorithm=algorithm,
        aug_factor=aug_factor,
    )
    profile.estimated_peak_gb = peak

    # Cek safety
    available = profile.available_ram_gb if profile.available_ram_gb > 0 else profile.total_ram_gb * 0.5
    safe_threshold = available * 0.7  # 70% dari RAM available

    if peak > safe_threshold:
        profile.safe_to_train = False
        profile.warnings.append(
            f"⚠️ Estimasi peak memory {peak}GB melebihi batas aman {safe_threshold:.1f}GB "
            f"(70% dari RAM available {available}GB). "
            "Kurangi jumlah gambar atau augmentasi."
        )

        # Rekomendasi max gambar yang aman
        safe_ratio = safe_threshold / peak
        safe_images = max(1, int(total_images * safe_ratio))
        profile.warnings.append(
            f"Saran: maksimal ~{safe_images} gambar total (termasuk augmentasi)."
        )
        profile.warnings.append(
            f"Atau: train di PC Dev dengan spesifikasi lebih tinggi, "
            f"lalu deploy model hasil export ke PC ini."
        )

    # Cek apakah batch_size perlu diturunkan untuk RAM kecil
    if profile.total_ram_gb < 12 and profile.batch_size > 8:
        profile.batch_size = 8
    if profile.total_ram_gb < 8 and profile.batch_size > 4:
        profile.batch_size = 4

    # Cek num_workers untuk RAM kecil
    if profile.total_ram_gb < 10 and profile.num_workers > 2:
        profile.num_workers = 2

    return profile
