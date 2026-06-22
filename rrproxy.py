"""Minimal async round-robin reverse-proxy for OpenAI-compatible backends.
Usage: python3 rrproxy.py <listen_port> <backend1> <backend2> ...
Each request is forwarded to the next backend in round-robin order.
"""
import asyncio, itertools, sys, urllib.parse
from http.server import BaseHTTPRequestHandler
import http.client, socket, threading

BACKENDS: list[str] = []
_cycle = None

def next_backend() -> str:
    return next(_cycle)

class ProxyHandler(BaseHTTPRequestHandler):
    log_message = lambda *a: None  # silence access log

    def _forward(self):
        backend = next_backend()
        parsed = urllib.parse.urlparse(backend)
        host, port = parsed.hostname, parsed.port or 80
        conn = http.client.HTTPConnection(host, port, timeout=360)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        hdrs = {k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length", "transfer-encoding")}
        conn.request(self.command, self.path, body=body or None, headers=hdrs)
        resp = conn.getresponse()
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in ("transfer-encoding",):
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(resp.read())
        conn.close()

    def do_GET(self):  self._forward()
    def do_POST(self): self._forward()
    def do_OPTIONS(self): self._forward()

def main():
    global _cycle
    listen_port = int(sys.argv[1])
    BACKENDS.extend(sys.argv[2:])
    _cycle = itertools.cycle(BACKENDS)
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("0.0.0.0", listen_port), ProxyHandler)
    print(f"rrproxy :{listen_port} -> {BACKENDS}", flush=True)
    srv.serve_forever()

if __name__ == "__main__":
    main()
