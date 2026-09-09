"""
Web interface for testing the phoneme encoder.

Flask-based web app that provides a simple UI for encoding text to phonemes.
"""

from __future__ import annotations

import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from flask import Flask, render_template_string, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from domains.multimodal.phoneme_encoder import PhonemeEncoder
from domains.multimodal.german_phoneme_encoder import GermanPhonemeEncoder
from domains.multimodal.unified_phoneme_encoder import UnifiedPhonemeEncoder


app = Flask(__name__)
encoder = UnifiedPhonemeEncoder()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phoneme Encoder</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 {
            color: #e94560;
            text-align: center;
        }
        .input-group {
            margin: 20px 0;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #e94560;
        }
        input, select, textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #333;
            border-radius: 5px;
            background: #16213e;
            color: #eee;
            font-size: 16px;
        }
        button {
            background: #e94560;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #c73e54;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background: #16213e;
            border-radius: 5px;
            border: 1px solid #333;
        }
        .result h3 {
            color: #e94560;
            margin-top: 0;
        }
        .phoneme-list {
            font-family: monospace;
            font-size: 18px;
            color: #0f3460;
            background: #eee;
            padding: 10px;
            border-radius: 3px;
        }
        .id-list {
            font-family: monospace;
            color: #53a8b6;
        }
        .error {
            color: #ff6b6b;
            background: #2d1b1b;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>Phoneme Encoder</h1>

    <div class="input-group">
        <label for="text">Text:</label>
        <input type="text" id="text" placeholder="Enter text to encode..." value="hello world">
    </div>

    <div class="input-group">
        <label for="language">Language:</label>
        <select id="language">
            <option value="auto">Auto-detect</option>
            <option value="en">English</option>
            <option value="de">German</option>
            <option value="fr">French</option>
            <option value="es">Spanish</option>
            <option value="it">Italian</option>
            <option value="pt">Portuguese</option>
        </select>
    </div>

    <button onclick="encodeText()">Encode</button>

    <div id="result" class="result" style="display: none;">
        <h3>Result:</h3>
        <p><strong>Phonemes:</strong> <span id="phonemes" class="phoneme-list"></span></p>
        <p><strong>IDs:</strong> <span id="ids" class="id-list"></span></p>
        <p><strong>Decoded:</strong> <span id="decoded"></span></p>
    </div>

    <div id="error" class="error" style="display: none;"></div>

    <script>
        async function encodeText() {
            const text = document.getElementById('text').value;
            const language = document.getElementById('language').value;

            try {
                const response = await fetch('/api/encode', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text, language }),
                });

                const data = await response.json();

                if (data.error) {
                    document.getElementById('error').textContent = data.error;
                    document.getElementById('error').style.display = 'block';
                    document.getElementById('result').style.display = 'none';
                } else {
                    document.getElementById('phonemes').textContent = data.phonemes.join(' ');
                    document.getElementById('ids').textContent = JSON.stringify(data.ids);
                    document.getElementById('decoded').textContent = data.decoded;
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('error').style.display = 'none';
                }
            } catch (error) {
                document.getElementById('error').textContent = error.message;
                document.getElementById('error').style.display = 'block';
                document.getElementById('result').style.display = 'none';
            }
        }

        // Encode on Enter key
        document.getElementById('text').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                encodeText();
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/encode', methods=['POST'])
def api_encode():
    """Encode text to phonemes."""
    data = request.get_json()
    text = data.get('text', '')
    language = data.get('language', 'auto')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        if language == 'auto':
            ids = encoder.encode(text)
            lang = encoder.current_language
        else:
            ids = encoder.encode(text, language=language)
            lang = language

        phonemes = encoder.decode_phonemes(ids, language=lang)
        decoded = encoder.decode(ids, language=lang)

        return jsonify({
            'phonemes': phonemes,
            'ids': ids.flatten().tolist(),
            'decoded': decoded,
            'language': lang,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_server(host: str = '0.0.0.0', port: int = 5000):
    """Run the web server."""
    if not FLASK_AVAILABLE:
        print("Flask not installed. Install with: pip install flask")
        sys.exit(1)

    print(f"Starting phoneme encoder web interface at http://{host}:{port}")
    app.run(host=host, port=port, debug=True)


if __name__ == '__main__':
    run_server()
