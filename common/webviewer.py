"""Lightweight MJPEG web viewer for live annotated frames."""

from __future__ import annotations

import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Optional

import cv2

LOGGER = logging.getLogger(__name__)


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class WebViewer:
    """Serve a simple MJPEG stream with a basic HTML page."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        *,
        title: str = "Scotty Viewer",
        jpeg_quality: int = 80,
    ) -> None:
        self._frame: Optional[bytes] = None
        self._cond = threading.Condition()
        self._title = title
        self._jpeg_quality = int(jpeg_quality)
        self._running = True
        self._server = _ThreadedHTTPServer((host, port), self._make_handler())
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        LOGGER.info("Web viewer serving at http://%s:%d/", host, port)

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except Exception:  # pragma: no cover - background server
            LOGGER.exception("Web viewer server crashed.")

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "ScottyWebViewer/0.1"

            def log_message(self, format: str, *args) -> None:  # noqa: D401 - Base override
                """Silence default stdout logging; forward to LOGGER."""

                LOGGER.debug("[web] %s - %s", self.client_address[0], format % args)

            def do_GET(self) -> None:  # noqa: D401 - Base override
                """Serve the index page or MJPEG stream."""

                if self.path in ("/", "/index.html"):
                    self._serve_index()
                elif self.path.startswith("/stream"):
                    self._serve_stream()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

            def _serve_index(self) -> None:
                html = f"""
                    <!doctype html>
                    <html lang=\"en\">
                      <head>
                        <meta charset=\"utf-8\" />
                        <title>{viewer._title}</title>
                        <style>
                          body {{
                            margin: 0;
                            background: #111;
                            color: #eee;
                            font-family: system-ui, sans-serif;
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                          }}
                          header {{
                            padding: 1rem;
                            font-size: 1.2rem;
                          }}
                          img {{
                            width: 100%;
                            height: auto;
                            max-width: 960px;
                            background: #000;
                          }}
                        </style>
                      </head>
                      <body>
                        <header>{viewer._title}</header>
                        <img src=\"/stream\" alt=\"Live stream\" />
                      </body>
                    </html>
                """.strip().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)

            def _serve_stream(self) -> None:
                boundary = "frame"
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type", f"multipart/x-mixed-replace; boundary={boundary}"
                )
                self.end_headers()

                try:
                    for chunk in viewer._frame_generator():
                        self.wfile.write(b"--" + boundary.encode("ascii") + b"\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(chunk)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(chunk)
                        self.wfile.write(b"\r\n")
                except BrokenPipeError:  # pragma: no cover - network interruption
                    LOGGER.debug("Client disconnected from stream.")

        return Handler

    def _frame_generator(self):
        while True:
            with self._cond:
                while self._frame is None and self._running:
                    self._cond.wait()
                if not self._running:
                    return
                frame = self._frame
            yield frame

    def publish(self, frame) -> None:
        if not self._running:
            return
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        )
        if not ok:
            LOGGER.warning("Failed to encode frame for web viewer.")
            return
        data = encoded.tobytes()
        with self._cond:
            self._frame = data
            self._cond.notify_all()

    def close(self) -> None:
        self._running = False
        self._server.shutdown()
        self._server.server_close()
        with self._cond:
            self._frame = None
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

