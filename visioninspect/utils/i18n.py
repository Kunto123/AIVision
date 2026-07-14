"""
VisionInspect - Internationalization (i18n)
Menyediakan terjemahan Bahasa Indonesia dan Inggris.
Struktur sederhana untuk memudahkan penambahan bahasa lain.
"""

from typing import Optional


class Translator:
    """
    Translator sederhana dengan dictionary.
    Default: Bahasa Indonesia.
    """

    _strings: dict[str, dict[str, str]] = {}

    # ==================== ID = Bahasa Indonesia ====================
    _id: dict[str, str] = {
        # General
        "app_name": "VisionInspect",
        "app_title": "VisionInspect — Sistem Inspeksi Visual Industri",
        "ok": "OK",
        "ng": "NG",
        "warning": "Peringatan",
        "error": "Error",
        "save": "Simpan",
        "cancel": "Batal",
        "delete": "Hapus",
        "confirm": "Konfirmasi",
        "loading": "Memuat...",
        "ready": "Siap",
        "running": "Berjalan",
        "stopped": "Berhenti",
        "connected": "Terhubung",
        "disconnected": "Terputus",

        # Navigation
        "nav_run": "RUN",
        "nav_teach": "TEACH",
        "nav_history": "HISTORY",
        "nav_settings": "SETTINGS",
        "nav_diagnostics": "DIAGNOSTIK",

        # Run page
        "run_title": "Mode Inspeksi",
        "run_status": "Status",
        "run_score": "Skor Anomali",
        "run_latency": "Latensi",
        "run_fps": "FPS",
        "run_judgement": "Hasil",
        "run_counter_total": "Total Inspeksi",
        "run_counter_ok": "OK",
        "run_counter_ng": "NG",
        "run_plc_status": "Status PLC",
        "run_trigger_mode": "Mode Trigger",
        "run_trigger_continuous": "Kontinu",
        "run_trigger_plc": "PLC",
        "run_trigger_manual": "Manual",
        "run_trigger_now": "Trigger Sekarang",
        "run_no_camera": "Tidak ada kamera",
        "run_no_model": "Belum ada model terlatih",

        # Teach page
        "teach_title": "Teaching & Training",
        "teach_capture_ok": "Capture OK",
        "teach_capture_ng": "Capture NG",
        "teach_import": "Import File",
        "teach_gallery_ok": "Galeri OK",
        "teach_gallery_ng": "Galeri NG",
        "teach_count_ok": "OK: {count} gambar",
        "teach_count_ng": "NG: {count} gambar",
        "teach_train": "TRAIN",
        "teach_rebuild": "REBUILD MODEL",
        "teach_training_progress": "Training: {percent}%",
        "teach_training_done": "Training selesai!",
        "teach_training_failed": "Training gagal: {error}",
        "teach_threshold": "Threshold",
        "teach_threshold_adaptive": "Adaptif",
        "teach_threshold_manual": "Manual",
        "teach_histogram": "Distribusi Skor",
        "teach_model_version": "Versi Model: v{version}",
        "teach_min_samples": "Minimal dibutuhkan {n} gambar OK. Saat ini: {count}",
        "teach_accuracy_warning": "Akurasi lebih baik dengan 10-30 gambar OK",

        # History page
        "history_title": "Riwayat Inspeksi",
        "history_filter": "Filter",
        "history_all": "Semua",
        "history_filter_ok": "OK Saja",
        "history_filter_ng": "NG Saja",
        "history_date": "Tanggal",
        "history_score": "Skor",
        "history_judgement": "Hasil",
        "history_image": "Gambar",
        "history_program": "Program",
        "history_correct": "Koreksi",
        "history_mark_ok": "Tandai OK",
        "history_mark_ng": "Tandai NG",
        "history_rebuild": "Rebuild Model",
        "history_rollback": "Rollback ke v{version}",
        "history_no_data": "Belum ada riwayat",

        # Settings page
        "settings_title": "Pengaturan",
        "settings_camera": "Kamera",
        "settings_roi": "ROI (Region of Interest)",
        "settings_plc": "PLC (Serial)",
        "settings_model": "Model AI",
        "settings_history": "Riwayat & Retensi",
        "settings_flask": "Flask API Internal",
        "settings_language": "Bahasa",
        "settings_save": "Simpan Pengaturan",
        "settings_saved": "Pengaturan tersimpan",
        "settings_restart_required": "Beberapa perubahan perlu restart aplikasi",

        # Diagnostics page
        "diagnostics_title": "Diagnostik",
        "diagnostics_logs": "Log Live",
        "diagnostics_performance": "Performa",
        "diagnostics_ram_usage": "RAM: {mb} MB",
        "diagnostics_cpu_usage": "CPU: {percent}%",
        "diagnostics_fps": "FPS Kamera: {fps}",
        "diagnostics_inference_latency": "Latensi Inferensi: {ms} ms (rata-rata)",
        "diagnostics_latency_p95": "Latensi P95: {ms} ms",
        "diagnostics_threads": "Status Thread",
        "diagnostics_plc_test": "Tes PLC",
        "diagnostics_plc_send_test": "Kirim Frame Uji",
        "diagnostics_plc_test_sent": "Frame terkirim: {frame}",

        # PLC
        "plc_rs232": "RS232",
        "plc_rs485": "RS485",
        "plc_modbus": "Modbus RTU",
        "plc_ascii": "ASCII Sederhana",
        "plc_auto_direction": "Auto Direction",
        "plc_rts_controlled": "RTS-Controlled",

        # Errors
        "error_camera_open": "Gagal membuka kamera: {error}",
        "error_inference": "Error inferensi: {error}",
        "error_training": "Error training: {error}",
        "error_serial": "Error serial: {error}",
        "error_program": "Error program: {error}",
        "error_generic": "Terjadi kesalahan: {error}",
    }

    # ==================== EN = English ====================
    _en: dict[str, str] = {
        "app_name": "VisionInspect",
        "app_title": "VisionInspect — Industrial Visual Inspection System",
        "ok": "OK",
        "ng": "NG",
        "warning": "Warning",
        "error": "Error",
        "save": "Save",
        "cancel": "Cancel",
        "delete": "Delete",
        "confirm": "Confirm",
        "loading": "Loading...",
        "ready": "Ready",
        "running": "Running",
        "stopped": "Stopped",
        "connected": "Connected",
        "disconnected": "Disconnected",
        "nav_run": "RUN",
        "nav_teach": "TEACH",
        "nav_history": "HISTORY",
        "nav_settings": "SETTINGS",
        "nav_diagnostics": "DIAGNOSTICS",
        "run_title": "Inspection Mode",
        "run_status": "Status",
        "run_score": "Anomaly Score",
        "run_latency": "Latency",
        "run_fps": "FPS",
        "run_judgement": "Result",
        "run_counter_total": "Total Inspections",
        "run_counter_ok": "OK",
        "run_counter_ng": "NG",
        "run_plc_status": "PLC Status",
        "run_trigger_mode": "Trigger Mode",
        "run_trigger_continuous": "Continuous",
        "run_trigger_plc": "PLC",
        "run_trigger_manual": "Manual",
        "run_trigger_now": "Trigger Now",
        "run_no_camera": "No camera",
        "run_no_model": "No trained model yet",
        "teach_title": "Teaching & Training",
        "teach_capture_ok": "Capture OK",
        "teach_capture_ng": "Capture NG",
        "teach_import": "Import File",
        "teach_gallery_ok": "OK Gallery",
        "teach_gallery_ng": "NG Gallery",
        "teach_count_ok": "OK: {count} images",
        "teach_count_ng": "NG: {count} images",
        "teach_train": "TRAIN",
        "teach_rebuild": "REBUILD MODEL",
        "teach_training_progress": "Training: {percent}%",
        "teach_training_done": "Training complete!",
        "teach_training_failed": "Training failed: {error}",
        "teach_threshold": "Threshold",
        "teach_threshold_adaptive": "Adaptive",
        "teach_threshold_manual": "Manual",
        "teach_histogram": "Score Distribution",
        "teach_model_version": "Model Version: v{version}",
        "teach_min_samples": "Minimum {n} OK images required. Current: {count}",
        "teach_accuracy_warning": "Accuracy is better with 10-30 OK images",
        "history_title": "Inspection History",
        "history_filter": "Filter",
        "history_all": "All",
        "history_filter_ok": "OK Only",
        "history_filter_ng": "NG Only",
        "history_date": "Date",
        "history_score": "Score",
        "history_judgement": "Result",
        "history_image": "Image",
        "history_program": "Program",
        "history_correct": "Correction",
        "history_mark_ok": "Mark as OK",
        "history_mark_ng": "Mark as NG",
        "history_rebuild": "Rebuild Model",
        "history_rollback": "Rollback to v{version}",
        "history_no_data": "No history yet",
        "settings_title": "Settings",
        "settings_camera": "Camera",
        "settings_roi": "ROI (Region of Interest)",
        "settings_plc": "PLC (Serial)",
        "settings_model": "AI Model",
        "settings_history": "History & Retention",
        "settings_flask": "Flask Internal API",
        "settings_language": "Language",
        "settings_save": "Save Settings",
        "settings_saved": "Settings saved",
        "settings_restart_required": "Some changes require restart",
        "diagnostics_title": "Diagnostics",
        "diagnostics_logs": "Live Logs",
        "diagnostics_performance": "Performance",
        "diagnostics_ram_usage": "RAM: {mb} MB",
        "diagnostics_cpu_usage": "CPU: {percent}%",
        "diagnostics_fps": "Camera FPS: {fps}",
        "diagnostics_inference_latency": "Inference Latency: {ms} ms (avg)",
        "diagnostics_latency_p95": "P95 Latency: {ms} ms",
        "diagnostics_threads": "Thread Status",
        "diagnostics_plc_test": "PLC Test",
        "diagnostics_plc_send_test": "Send Test Frame",
        "diagnostics_plc_test_sent": "Frame sent: {frame}",
        "plc_rs232": "RS232",
        "plc_rs485": "RS485",
        "plc_modbus": "Modbus RTU",
        "plc_ascii": "Simple ASCII",
        "plc_auto_direction": "Auto Direction",
        "plc_rts_controlled": "RTS-Controlled",
        "error_camera_open": "Failed to open camera: {error}",
        "error_inference": "Inference error: {error}",
        "error_training": "Training error: {error}",
        "error_serial": "Serial error: {error}",
        "error_program": "Program error: {error}",
        "error_generic": "An error occurred: {error}",
    }

    def __init__(self, language: str = "id"):
        self._strings = {"id": self._id, "en": self._en}
        self._language = language if language in self._strings else "id"

    @property
    def language(self) -> str:
        return self._language

    @language.setter
    def language(self, lang: str) -> None:
        if lang in self._strings:
            self._language = lang

    def tr(self, key: str, **kwargs) -> str:
        """
        Translate a key. Falls back to key name if not found.
        Supports format placeholders: tr("hello_{name}", name="World")
        """
        lang_dict = self._strings.get(self._language, self._strings["id"])
        text = lang_dict.get(key, self._strings["id"].get(key, key))
        if kwargs:
            try:
                return text.format(**kwargs)
            except KeyError:
                return text
        return text

    def available_languages(self) -> list[tuple[str, str]]:
        """Return list of (code, name) tuples."""
        return [("id", "Bahasa Indonesia"), ("en", "English")]
