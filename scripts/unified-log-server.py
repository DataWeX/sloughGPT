#!/usr/bin/env python3
"""
Unified Logging Server for SloughGPT
Receives logs from API, Web, and Core Engine via HTTP POST.
Displays them in real-time in the terminal.
"""

import http.server
import socketserver
import json
import threading
import sys
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("/tmp/sloughgpt-unified.log")
LOG_FILE.unlink(missing_ok=True)


class LogHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_POST(self):
        if self.path == "/log":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                entry = json.loads(body)
                timestamp = entry.get("timestamp", datetime.now().isoformat())
                source = entry.get("source", "unknown")
                level = entry.get("level", "INFO")
                message = entry.get("message", "")

                # Color codes for terminal
                colors = {
                    "DEBUG": "\033[36m",  # Cyan
                    "INFO": "\033[32m",  # Green
                    "WARNING": "\033[33m",  # Yellow
                    "ERROR": "\033[31m",  # Red
                    "CRITICAL": "\033[35m",  # Magenta
                    "api": "\033[34m",  # Blue
                    "web": "\033[35m",  # Magenta
                    "engine": "\033[33m",  # Yellow
                }
                reset = "\033[0m"
                color = colors.get(source, colors.get(level, "\033[0m"))

                # Format for terminal
                ts_short = timestamp.split("T")[1][:12] if "T" in timestamp else timestamp[:12]
                line = f"{color}[{ts_short}]{reset} {color}[{source:8s}]{reset} {message}"

                # Print to stdout
                print(line)

                # Write to log file
                with open(LOG_FILE, "a") as f:
                    f.write(f"[{timestamp}] [{source}] {message}\n")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = 9999):
    with socketserver.TCPServer(("", port), LogHandler) as httpd:
        print(f"🪵 Unified log server listening on port {port}")
        print(f"   Send logs via: POST /log with JSON body")
        print(f"   View logs: tail -f {LOG_FILE}")
        print("-" * 60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Log server stopped")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    run_server(port)
