"""Manual offline startup/asset/API proof with installed resources, zero fitting.

From integrations/hybrid: PYTHONPATH=. python tests/offline_startup.py /path/to/model
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.request
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
connect = socket.socket.connect

def local_only(sock, address):
    if isinstance(address, tuple) and address[0] not in {'127.0.0.1', 'localhost', '::1'}:
        raise RuntimeError('External network blocked for offline proof')
    return connect(sock, address)

socket.socket.connect = local_only
from mindscape_delivery.neural import NeuralBackend
from mindscape_delivery.assistant import Assistant
from mindscape_delivery.server import MindServer

if __name__ == '__main__':
    started = time.perf_counter()
    neural = NeuralBackend(model_path=sys.argv[1], device='cpu').load()
    server = MindServer(('127.0.0.1', 0), orchestrator=Assistant(neural),
                        web_root=Path(__file__).resolve().parents[1] / 'web')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f'http://127.0.0.1:{server.server_address[1]}'
        assets = {}
        for route in ('/', '/style.css', '/app.js', '/math.js', '/science.js'):
            with urllib.request.urlopen(base + route) as response:
                data = response.read()
                assert response.status == 200
                assets[route] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        request = urllib.request.Request(base + '/api/chat', data=json.dumps({
            'message': 'Differentiate x^3 + sin(x) with respect to x',
            'inference_mode': 'hybrid'}).encode(), headers={'Content-Type': 'application/json'})
        response = json.load(urllib.request.urlopen(request, timeout=90))
        assert response['status'] == 'verified'
        assert response['solver_result']['answer'] == '3*x**2 + cos(x)'
        assert response['provenance']['pretrained_used'] is True
        assert all(not p.requires_grad and p.grad is None for p in neural.model.parameters())
        print(json.dumps({'external_connections_blocked': True, 'assets': assets,
                          'response': response, 'optimizer_updates': 0,
                          'total_seconds': time.perf_counter() - started}, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
