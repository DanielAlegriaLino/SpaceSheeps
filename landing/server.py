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
PROJECT_DIR = os.path.dirname(LANDING_DIR)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=LANDING_DIR, **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/"):
            self.proxy_n2yo()
        elif self.path.startswith("/videos/"):
            self.serve_video()
        else:
            super().do_GET()

    def serve_video(self):
        filename = os.path.basename(self.path)
        filepath = os.path.join(PROJECT_DIR, filename)
        if not os.path.isfile(filepath) or not filename.endswith(".mp4"):
            self.send_error(404, "Video not found")
            return
        size = os.path.getsize(filepath)
        range_header = self.headers.get("Range")

        if range_header:
            # Parse Range: bytes=start-end
            range_spec = range_header.replace("bytes=", "")
            parts = range_spec.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else size - 1
            end = min(end, size - 1)
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(filepath, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(filepath, "rb") as f:
                self.wfile.write(f.read())

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
