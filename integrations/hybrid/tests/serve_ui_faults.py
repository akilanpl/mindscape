"""Explicit local-only UI fault fixture; never used by production startup.

Run from integrations/hybrid: PYTHONPATH=. python tests/serve_ui_faults.py
First chat returns 429; retry uses the real bounded tools (no model loaded).
"""
from pathlib import Path
from mindscape_delivery.server import MindServer, Handler
from mindscape_delivery.assistant import Assistant

class FaultHandler(Handler):
    def do_POST(self):
        if self.path == '/api/chat' and self.server.first_busy:
            self.server.first_busy = False
            self.rfile.read(int(self.headers.get('Content-Length', '0')))
            return self._json(429, {'error': 'Explicit test fixture: local model busy'})
        return super().do_POST()

if __name__ == '__main__':
    server = MindServer(('127.0.0.1', 8773), orchestrator=Assistant(),
                        web_root=Path(__file__).resolve().parents[1] / 'web')
    server.RequestHandlerClass = FaultHandler
    server.first_busy = True
    print('UI fault fixture only: http://127.0.0.1:8773', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
