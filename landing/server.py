"""
Local proxy server for the Satellite Proximity Alert app.
Serves static files, proxies N2YO API requests, and provides live camera AI detection.

Usage:
    python landing/server.py
    Then open http://localhost:8080

For AI camera detection:
    python landing/server.py --camera 0
    Then go to Camera tab and click "Connect to Camera"

The camera is only active when a client is connected to /camera_feed.
"""

import http.server
import urllib.request
import json
import os
import argparse
import threading
import time

PORT = 8080
LANDING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(LANDING_DIR)

# Global variables for camera processing
camera_source = None  # Set from command line
model = None
class_names = None
confidence_threshold = 0.25
active_clients = 0
clients_lock = threading.Lock()

BOX_COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]


def load_model(model_path):
    """Load the YOLO model."""
    global model, class_names
    try:
        from ultralytics import YOLO
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)
        class_names = model.names
        print(f"Model loaded! Classes: {class_names}")
        return True
    except ImportError:
        print("WARNING: ultralytics not installed. Camera AI detection disabled.")
        print("Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=LANDING_DIR, **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/"):
            self.proxy_n2yo()
        elif self.path.startswith("/videos/"):
            self.serve_video()
        elif self.path.startswith("/models/"):
            self.serve_model()
        elif self.path == "/camera_feed" or self.path.startswith("/camera_feed?"):
            self.serve_camera_feed()
        elif self.path == "/camera_status":
            self.serve_camera_status()
        else:
            super().do_GET()

    def serve_model(self):
        """Serve model files from the project directory."""
        filename = os.path.basename(self.path)
        filepath = os.path.join(PROJECT_DIR, "custom_model", filename)
        if not os.path.isfile(filepath):
            filepath = os.path.join(PROJECT_DIR, filename)
        if not os.path.isfile(filepath):
            self.send_error(404, "Model not found")
            return
        
        size = os.path.getsize(filepath)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        with open(filepath, "rb") as f:
            self.wfile.write(f.read())

    def serve_video(self):
        filename = os.path.basename(self.path)
        filepath = os.path.join(PROJECT_DIR, filename)
        if not os.path.isfile(filepath) or not filename.endswith(".mp4"):
            self.send_error(404, "Video not found")
            return
        size = os.path.getsize(filepath)
        range_header = self.headers.get("Range")

        if range_header:
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

    def serve_camera_feed(self):
        """Serve MJPEG camera stream with AI detection. Opens camera on-demand."""
        global camera_source, model, class_names, active_clients
        
        if camera_source is None:
            self.send_error(503, "Camera not enabled. Start server with --camera option.")
            return
        
        try:
            import cv2
        except ImportError:
            self.send_error(500, "OpenCV not installed")
            return
        
        # Track client connection
        with clients_lock:
            active_clients += 1
            print(f"Client connected. Active clients: {active_clients}")
        
        # Open camera for this connection
        print(f"Opening camera: {camera_source}")
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            with clients_lock:
                active_clients -= 1
            self.send_error(503, f"Could not open camera {camera_source}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {width}x{height}")
        
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Run YOLO inference if model is loaded
                if model is not None:
                    results = model(frame, conf=confidence_threshold, verbose=False)
                    result = results[0]
                    
                    num_detections = len(result.boxes)
                    
                    for box in result.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        color = BOX_COLORS[class_id % len(BOX_COLORS)]
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{class_names[class_id]} {confidence:.2f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # FPS counter
                    frame_count += 1
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    # Draw overlay
                    overlay_text = f"FPS: {fps:.1f} | Detections: {num_detections}"
                    cv2.putText(frame, overlay_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    cv2.putText(frame, "SpaceSheeps AI", (10, height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 210, 255), 2, cv2.LINE_AA)
                
                # Encode and send frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                self.wfile.write(frame_bytes)
                self.wfile.write(b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
                
        except (BrokenPipeError, ConnectionResetError):
            print("Client disconnected")
        except Exception as e:
            print(f"Stream error: {e}")
        finally:
            # Release camera and decrement client count
            cap.release()
            with clients_lock:
                active_clients -= 1
                print(f"Camera released. Active clients: {active_clients}")

    def serve_camera_status(self):
        """Return camera status as JSON."""
        global camera_source, active_clients
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        status = {
            "enabled": camera_source is not None,
            "model_loaded": model is not None,
            "active_clients": active_clients
        }
        self.wfile.write(json.dumps(status).encode())

    def proxy_n2yo(self):
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


# Threaded HTTP server for handling multiple connections
class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SpaceSheeps Landing Page Server')
    parser.add_argument('--camera', type=str, default=None,
                        help='Camera index (0 for webcam) or IP camera URL to enable AI detection')
    parser.add_argument('--port', type=int, default=8080,
                        help='Server port (default: 8080)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold (default: 0.25)')
    parser.add_argument('--model', type=str, default='custom_model/best.pt',
                        help='Path to YOLO model file')
    
    args = parser.parse_args()
    PORT = args.port
    confidence_threshold = args.conf
    
    print("=" * 60)
    print("SpaceSheeps Landing Page Server")
    print("=" * 60)
    
    # Set camera source (camera is NOT opened until client connects)
    if args.camera is not None:
        camera_source = int(args.camera) if args.camera.isdigit() else args.camera
        model_path = os.path.join(PROJECT_DIR, args.model)
        if load_model(model_path):
            print(f"Camera AI detection: READY (camera {camera_source})")
            print("Camera will be opened when a client connects to /camera_feed")
        else:
            print("Camera AI detection: DISABLED (model failed to load)")
            camera_source = None
    else:
        print("Camera AI detection: DISABLED (use --camera to enable)")
    
    print("=" * 60)
    print(f"Serving on http://localhost:{PORT}")
    print("=" * 60)
    
    try:
        ThreadedHTTPServer(("", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
