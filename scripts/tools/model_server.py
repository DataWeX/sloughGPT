#!/usr/bin/env python3
"""
SloughGPT Model Server
Serves GGUF models for Aria app download.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Configuration
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8001  # Different from main sloughgpt server
MODEL_DIR = Path(__file__).parent.parent / 'dist' / 'models'
METADATA_FILE = MODEL_DIR / 'model_metadata.json'

def ensure_dirs():
    """Create necessary directories."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

def get_file_hash(filepath):
    """Get SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def load_metadata():
    """Load or create model metadata."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return None

def save_metadata(metadata):
    """Save model metadata."""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def get_model_list():
    """Get list of available models."""
    models = []
    for gguf_file in MODEL_DIR.glob('*.gguf'):
        size = gguf_file.stat().st_size
        metadata = load_metadata()
        
        model_info = {
            'filename': gguf_file.name,
            'size': size,
            'size_mb': round(size / 1024 / 1024, 2),
        }
        
        if metadata and metadata.get('filename') == gguf_file.name:
            model_info.update(metadata)
        else:
            model_info.update({
                'name': gguf_file.stem,
                'version': '1.0.0',
                'format': 'gguf',
                'sha256': get_file_hash(gguf_file),
                'updated_at': datetime.now().isoformat(),
            })
        
        models.append(model_info)
    
    return models

def main():
    import http.server
    import socketserver
    
    ensure_dirs()
    
    class ModelHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(MODEL_DIR), **kwargs)
        
        def do_GET(self):
            if self.path == '/' or self.path == '/models':
                self.send_json({
                    'service': 'SloughGPT Model Server',
                    'version': '1.0.0',
                    'models': get_model_list(),
                })
            
            elif self.path == '/health':
                self.send_json({'status': 'healthy'})
            
            elif self.path.startswith('/models/'):
                model_name = self.path[8:]  # Remove '/models/'
                
                if model_name == '':
                    self.send_json({'models': get_model_list()})
                    return
                
                if model_name == 'list':
                    self.send_json({'models': get_model_list()})
                    return
                
                # Check for specific model info or download
                gguf_path = MODEL_DIR / model_name
                gguf_path_meta = MODEL_DIR / f"{model_name}.gguf"
                
                if model_name.endswith('/version'):
                    model_id = model_name.replace('/version', '')
                    info = self._get_model_info(model_id)
                    self.send_json(info or {'error': 'Model not found'})
                
                elif model_name.endswith('/download'):
                    model_id = model_name.replace('/download', '')
                    self._serve_download(model_id)
                
                elif model_name.endswith('.gguf'):
                    # Direct GGUF download
                    self._serve_file(gguf_path)
                
                else:
                    # Model info endpoint
                    info = self._get_model_info(model_name)
                    if info:
                        self.send_json(info)
                    else:
                        self.send_error(404, 'Model not found')
            
            else:
                self.send_error(404, 'Not found')
        
        def _get_model_info(self, model_id):
            """Get model info by ID."""
            models = get_model_list()
            for m in models:
                if m['filename'].replace('.gguf', '') == model_id or m.get('name') == model_id:
                    return {
                        'name': m.get('name', model_id),
                        'filename': m['filename'],
                        'version': m.get('version', '1.0.0'),
                        'size': m['size'],
                        'size_mb': m.get('size_mb', 0),
                        'format': m.get('format', 'gguf'),
                        'quantization': m.get('quantization', 'Q4_K_M'),
                        'parameters': m.get('parameters', 135000000),
                        'sha256': m.get('sha256', ''),
                        'download_url': f"/models/{m['filename']}/download",
                        'updated_at': m.get('updated_at', ''),
                    }
            return None
        
        def _serve_download(self, model_id):
            """Serve model file for download."""
            models = get_model_list()
            for m in models:
                if m['filename'].replace('.gguf', '') == model_id or m.get('name') == model_id:
                    gguf_path = MODEL_DIR / m['filename']
                    if gguf_path.exists():
                        self._serve_file(gguf_path)
                        return
            self.send_error(404, 'Model not found')
        
        def _serve_file(self, filepath):
            """Serve a file."""
            if not filepath.exists():
                self.send_error(404, 'File not found')
                return
            
            size = filepath.stat().st_size
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Disposition', f'attachment; filename="{filepath.name}"')
            self.send_header('Content-Length', str(size))
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
        
        def send_json(self, data):
            """Send JSON response."""
            json_str = json.dumps(data, indent=2)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(json_str)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json_str.encode())
        
        def log_message(self, format, *args):
            print(f"[ModelServer] {args[0]}")
    
    with socketserver.TCPServer((SERVER_HOST, SERVER_PORT), ModelHandler) as httpd:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║             SloughGPT Model Server                          ║
╠══════════════════════════════════════════════════════════════╣
║  Serving GGUF models from: {MODEL_DIR}
║  Running on: http://{SERVER_HOST}:{SERVER_PORT}
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                ║
║    GET /                          - Server info            ║
║    GET /health                   - Health check           ║
║    GET /models                   - List all models       ║
║    GET /models/<name>            - Model metadata         ║
║    GET /models/<name>/version    - Version info           ║
║    GET /models/<name>/download   - Download model         ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            httpd.shutdown()

if __name__ == '__main__':
    main()
