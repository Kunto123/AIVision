"""
VisionInspect - Flask API Internal (Opsional)
REST API lokal di 127.0.0.1 untuk integasi eksternal.
Hanya aktif jika di-setting di config.
Bind HANYA ke localhost.
"""

import json
import threading
import time
from typing import Any, Callable, Optional

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("api")

try:
    from flask import Flask, jsonify, request, make_response
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    logger.warning("Flask not installed")


class FlaskAPI:
    """
    Internal REST API.
    Berjalan di thread terpisah, bind ke 127.0.0.1 saja.
    """

    def __init__(
        self,
        port: int = 5000,
        api_key: str = "",
        get_status_fn: Optional[Callable] = None,
        get_last_result_fn: Optional[Callable] = None,
        trigger_inspection_fn: Optional[Callable] = None,
        get_history_fn: Optional[Callable] = None,
        activate_program_fn: Optional[Callable] = None,
    ):
        self._port = port
        self._api_key = api_key or self._generate_key()
        self._app: Any = None
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self._get_status = get_status_fn
        self._get_last_result = get_last_result_fn
        self._trigger_inspection = trigger_inspection_fn
        self._get_history = get_history_fn
        self._activate_program = activate_program_fn

        if HAS_FLASK:
            self._init_app()

    def _generate_key(self) -> str:
        import hashlib
        import uuid
        return hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:32]

    def _init_app(self):
        """Initialize Flask app with routes."""
        if not HAS_FLASK:
            return

        self._app = Flask("VisionInspect-API")

        # === Middleware: Auth ===
        @self._app.before_request
        def check_auth():
            if self._api_key:
                auth = request.headers.get("X-API-Key", "")
                if auth != self._api_key:
                    return jsonify({"error": "Unauthorized"}), 401

        # === Routes ===

        @self._app.route("/status", methods=["GET"])
        def status():
            if self._get_status:
                return jsonify(self._get_status())
            return jsonify({"status": "unknown"})

        @self._app.route("/last_result", methods=["GET"])
        def last_result():
            if self._get_last_result:
                return jsonify(self._get_last_result())
            return jsonify({"error": "no data"})

        @self._app.route("/trigger", methods=["POST"])
        def trigger():
            if self._trigger_inspection:
                self._trigger_inspection()
                return jsonify({"status": "triggered"})
            return jsonify({"error": "not available"}), 503

        @self._app.route("/history", methods=["GET"])
        def history():
            limit = request.args.get("limit", 100, type=int)
            if self._get_history:
                entries = self._get_history(limit=limit)
                return jsonify({"entries": entries, "count": len(entries)})
            return jsonify({"entries": [], "count": 0})

        @self._app.route("/program/<name>/activate", methods=["POST"])
        def activate_program(name):
            if self._activate_program:
                try:
                    self._activate_program(name)
                    return jsonify({"status": "activated", "program": name})
                except Exception as e:
                    return jsonify({"error": str(e)}), 400
            return jsonify({"error": "not available"}), 503

        @self._app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})

        logger.info("Flask API initialized (port=%d)", self._port)

    # ---- Lifecycle ----

    def start(self) -> None:
        """Start Flask in background thread."""
        if not self._app:
            logger.warning("Flask not available, API not started")
            return

        if self._thread and self._thread.is_alive():
            logger.warning("Flask API already running")
            return

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="flask-api",
        )
        self._thread.start()
        logger.info("Flask API started on 127.0.0.1:%d", self._port)

    def _run(self):
        try:
            self._app.run(
                host="127.0.0.1",
                port=self._port,
                debug=False,
                use_reloader=False,
            )
        except Exception as e:
            logger.error("Flask API error: %s", e)

    def stop(self) -> None:
        """Stop Flask API."""
        # Flask's built-in server doesn't support clean shutdown easily.
        # In production, use a proper WSGI server.
        logger.info("Flask API stop requested")

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
