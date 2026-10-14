"""Loopback-only local application server. Public account hosting is not enabled."""
import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
from .orchestrator import Orchestrator

LOG = logging.getLogger("mind1")


class MindServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address=("127.0.0.1", 8765), orchestrator=None, web_root=None):
        if address[0] not in {"127.0.0.1", "localhost"}:
            raise ValueError("Public binding unavailable; use a reviewed authenticated hosting layer")
        self.orchestrator = orchestrator or Orchestrator()
        self.web_root = Path(web_root or Path(__file__).resolve().parents[3] / "mind1_web").resolve()
        self.active = {}
        self.guard = threading.Lock()
        self.rate = defaultdict(deque)
        self.slots = threading.BoundedSemaphore(4)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "Mind1"
    def log_message(self, format, *args):
        # Do not log URLs, user text, provider responses, keys or conversation contents.
        LOG.info(json.dumps({"event": "http_request", "method": self.command}))

    def _headers(self, status, mime, size):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()

    def _json(self, status, data):
        raw = json.dumps(data, allow_nan=False).encode()
        self._headers(status, "application/json; charset=utf-8", len(raw))
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _trusted(self):
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{h}" for h in hosts}:
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            return False
        return True

    def do_GET(self):
        self.connection.settimeout(10)
        if not self._trusted():
            return self._json(403, {"error": "Untrusted host or origin"})
        path = urlsplit(self.path).path
        if path == "/api/capabilities":
            return self._json(200, self.server.orchestrator.capabilities())
        if path == "/api/health":
            return self._json(200, {"status": "ok", "product": "Mind 1", "public_hosting": False})
        if path == "/api/models":
            return self._json(200, {"external_inference": False, "native": self.server.orchestrator.capabilities()['native']})
        files = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css", "/math.js": "math.js", "/science.js": "science.js"}
        if path not in files:
            return self._json(404, {"error": "Not found"})
        file = self.server.web_root / files[path]
        if not file.is_file():
            return self._json(404, {"error": "Frontend not installed"})
        raw = file.read_bytes()
        self._headers(200, (mimetypes.guess_type(str(file))[0] or "text/plain") + "; charset=utf-8", len(raw))
        self.wfile.write(raw)

    def do_POST(self):
        self.connection.settimeout(10)
        if not self._trusted():
            return self._json(403, {"error": "Untrusted host or origin"})
        if self.headers.get_content_type() != "application/json":
            return self._json(415, {"error": "application/json required"})
        if self.headers.get("Transfer-Encoding"):
            return self._json(400, {"error": "Transfer encoding unsupported"})
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 < length <= 96000:
                return self._json(413, {"error": "Request size outside 1–96000 bytes"})
            data = json.loads(self.rfile.read(length), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if not isinstance(data, dict):
                raise ValueError()
        except (ValueError, OSError):
            return self._json(400, {"error": "Invalid JSON request"})
        route = urlsplit(self.path).path
        request_id = data.get("request_id", "")
        if route == "/api/cancel":
            with self.server.guard:
                event = self.server.active.get(request_id) if isinstance(request_id, str) else None
                if event:
                    event.set()
            return self._json(200, {"cancelled": bool(event)})
        if route != "/api/chat":
            return self._json(404, {"error": "Not found"})
        if not isinstance(request_id, str) or (request_id and not re.fullmatch(r'[a-zA-Z0-9_-]{8,80}', request_id)):
            return self._json(400, {"error": "Invalid request id"})
        with self.server.guard:
            bucket = self.server.rate[self.client_address[0]]
            now = time.monotonic()
            while bucket and now-bucket[0] >= 60:
                bucket.popleft()
            if len(bucket) >= 30:
                return self._json(429, {"error": "Local rate limit exceeded; retry in a minute"})
            bucket.append(now)
        if not self.server.slots.acquire(blocking=False):
            return self._json(429, {"error": "Concurrent request limit reached"})
        cancel = threading.Event()
        key = request_id or f"internal-{id(cancel)}"
        with self.server.guard:
            if key in self.server.active:
                self.server.slots.release()
                return self._json(409, {"error": "Request id already active"})
            self.server.active[key] = cancel
        try:
            # Isolate the HTTP deadline from potentially slow provider I/O. Tools terminate
            # on cancellation. An in-flight provider call stops at its socket timeout.
            with ThreadPoolExecutor(max_workers=1) as executor:
                task = executor.submit(self.server.orchestrator.chat, data, cancel)
                deadline = time.monotonic() + 125
                while not task.done() and not cancel.is_set() and time.monotonic() < deadline:
                    time.sleep(.05)
                if not task.done():
                    cancel.set()
                    self._json(200, {"answer": "Request cancelled or time ceiling reached.", "status": "cancelled", "trace": [], "tools": [], "limitations": ["An in-flight provider call may complete until its socket timeout"], "provenance": {}})
                    return
                self._json(200, task.result())
        except InterruptedError:
            self._json(200, {"answer": "Request cancelled.", "status": "cancelled", "trace": [], "tools": [], "provenance": {}, "limitations": []})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception:
            LOG.error("Request failed; exception contents redacted")
            self._json(500, {"error": "Internal request failure; no result verified"})
        finally:
            cancel.set()
            with self.server.guard:
                self.server.active.pop(key, None)
            self.server.slots.release()


def main():
    parser = argparse.ArgumentParser(description="Mind 1 loopback scientific assistant")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    server = MindServer(("127.0.0.1", args.port))
    print(f"Mind 1: http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
