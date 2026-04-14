"""
Minimal full-stack web application for the Smart Irrigation System.
"""

from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.ml_service import (
    CROP_WATER_REQUIREMENTS,
    CROPS,
    IRRIGATION_TYPES,
    generate_dataset,
    get_dashboard_stats,
    model_ready,
    predict,
    train_model,
)

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
OUTPUTS_DIR = BASE_DIR / "outputs"


class SmartIrrigationHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "model_ready": model_ready(),
                    "supported_crops": CROPS,
                    "irrigation_types": IRRIGATION_TYPES,
                    "water_requirements": CROP_WATER_REQUIREMENTS,
                }
            )
            return

        if parsed.path == "/api/stats":
            self._send_json(get_dashboard_stats())
            return

        if parsed.path.startswith("/outputs/"):
            file_path = OUTPUTS_DIR / parsed.path.removeprefix("/outputs/")
            if file_path.exists() and file_path.is_file():
                content = file_path.read_bytes()
                mime_type, _ = mimetypes.guess_type(str(file_path))
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime_type or "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            self._send_json({"error": "File not found."}, HTTPStatus.NOT_FOUND)
            return

        if parsed.path in {"/", "/index.html"}:
            self.path = "/index.html"

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        try:
            if parsed.path == "/api/predict":
                payload = self._read_json_body()
                self._send_json(predict(payload))
                return

            if parsed.path == "/api/train":
                metadata = train_model()
                self._send_json({"message": "Model trained successfully.", "model": metadata})
                return

            if parsed.path == "/api/generate-dataset":
                body = self._read_json_body()
                sample_count = int(body.get("samples", 5000))
                dataset = generate_dataset(sample_count)
                self._send_json(
                    {
                        "message": "Dataset generated successfully.",
                        "rows": int(len(dataset)),
                    }
                )
                return

            self._send_json({"error": "Endpoint not found."}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json(
                {"error": f"Internal server error: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), SmartIrrigationHandler)
    print("=" * 72)
    print("Smart Irrigation web app is running")
    print(f"Open http://{host}:{port} in your browser")
    print("=" * 72)
    server.serve_forever()


if __name__ == "__main__":
    run()
