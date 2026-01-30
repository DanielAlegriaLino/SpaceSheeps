"""
Local proxy server for the Satellite Proximity Alert app.
Serves static files and proxies N2YO API requests to bypass CORS.

Usage:
    python landing/server.py
    Then open http://localhost:8080
"""

import http.server
import urllib.request
import json
import os

PORT = 8080
LANDING_DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=LANDING_DIR, **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/"):
            self.proxy_n2yo()
        else:
            super().do_GET()

    def proxy_n2yo(self):
        # /api/<rest of n2yo path> -> https://api.n2yo.com/rest/v1/satellite/<rest>
        api_path = self.path[len("/api/"):]
        url = f"https://api.n2yo.com/rest/v1/satellite/{api_path}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


if __name__ == "__main__":
    print(f"Serving on http://localhost:{PORT}")
    http.server.HTTPServer(("", PORT), Handler).serve_forever()
