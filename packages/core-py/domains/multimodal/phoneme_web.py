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

    <div id="live-preview" style="margin-top: 10px; padding: 10px; background: #1a1a2e; border-radius: 5px; display: none;">
        <h4 style="margin: 0 0 5px 0; color: #e94560;">Live Preview:</h4>
        <p style="margin: 0; font-family: monospace; color: #53a8b6;" id="live-phonemes"></p>
    </div>

    <div id="result" class="result" style="display: none;">
        <h3>Result:</h3>
        <p><strong>Phonemes:</strong> <span id="phonemes" class="phoneme-list"></span></p>
        <p><strong>IDs:</strong> <span id="ids" class="id-list"></span></p>
        <p><strong>Decoded:</strong> <span id="decoded"></span></p>
    </div>

    <div id="error" class="error" style="display: none;"></div>

    <h2>TTS Synthesis</h2>

    <div class="input-group">
        <label for="tts-text">Text to synthesize:</label>
        <input type="text" id="tts-text" placeholder="Enter text to synthesize..." value="hello world">
    </div>

    <div class="input-group">
        <label>
            <input type="checkbox" id="tts-ssml" onchange="toggleSSML()"> Use SSML
        </label>
    </div>

    <div id="ssml-help" style="display: none; margin-bottom: 10px; padding: 10px; background: #f0f0f0; border-radius: 4px; font-size: 12px;">
        <strong>SSML Tags:</strong><br>
        &lt;break time="500ms"/&gt; - pause<br>
        &lt;prosody rate="slow" pitch="low"&gt;text&lt;/prosody&gt; - prosody<br>
        &lt;emphasis level="strong"&gt;text&lt;/emphasis&gt; - emphasis
    </div>

    <div class="input-group">
        <label for="tts-speed">Speed (0.5-2.0):</label>
        <input type="number" id="tts-speed" value="1.0" min="0.5" max="2.0" step="0.1">
    </div>

    <div class="input-group">
        <label for="tts-pitch">Pitch shift (semitones):</label>
        <input type="number" id="tts-pitch" value="0" min="-12" max="12" step="1">
    </div>

    <button onclick="synthesizeSpeech()">Synthesize</button>

    <div id="tts-result" class="result" style="display: none;">
        <h3>TTS Result:</h3>
        <p><strong>Duration:</strong> <span id="tts-duration"></span> seconds</p>
        <p><strong>Sample Rate:</strong> <span id="tts-sample-rate"></span> Hz</p>
        <audio id="tts-audio" controls style="width: 100%; margin-top: 10px;"></audio>
        <canvas id="spectrogram-canvas" width="800" height="200" style="width: 100%; margin-top: 10px; border: 1px solid #ccc;"></canvas>
        <p style="font-size: 12px; color: #666;">Spectrogram visualization (generated from waveform)</p>
    </div>

    <div id="tts-error" class="error" style="display: none;"></div>

    <h2>Pronunciation Scoring</h2>

    <div class="input-group">
        <label for="target-text">Target word:</label>
        <input type="text" id="target-text" placeholder="Enter target word..." value="hello">
    </div>

    <div class="input-group">
        <label for="spoken-text">Spoken word:</label>
        <input type="text" id="spoken-text" placeholder="Enter spoken word..." value="helo">
    </div>

    <button onclick="scorePronunciation()">Score</button>
    <button onclick="document.getElementById('score-audio-upload').click()">Upload Audio</button>
    <input type="file" id="score-audio-upload" accept="audio/*" style="display: none;" onchange="handleScoreAudioUpload(event)">

    <div id="score-result" class="result" style="display: none;">
        <h3>Score Result:</h3>
        <p><strong>Score:</strong> <span id="pronunciation-score"></span></p>
        <p><strong>Precision:</strong> <span id="pronunciation-precision"></span></p>
        <p><strong>Recall:</strong> <span id="pronunciation-recall"></span></p>
        <p><strong>Target Phonemes:</strong> <span id="target-phonemes" class="phoneme-list"></span></p>
        <p><strong>Spoken Phonemes:</strong> <span id="spoken-phonemes" class="phoneme-list"></span></p>
    </div>

    <div id="score-error" class="error" style="display: none;"></div>

    <h2>Pronunciation History</h2>
    <div id="history-container" style="margin-top: 10px;">
        <button onclick="clearHistory()" style="margin-bottom: 10px;">Clear History</button>
        <button onclick="exportHistory()" style="margin-bottom: 10px;">Export History</button>
        <button onclick="document.getElementById('import-history').click()" style="margin-bottom: 10px;">Import History</button>
        <input type="file" id="import-history" accept=".json" style="display: none;" onchange="importHistory(event)">
        <div id="history-list" style="max-height: 300px; overflow-y: auto;"></div>
    </div>

    <h2>Language Detection</h2>

    <div class="input-group">
        <label for="detect-text">Text to detect:</label>
        <input type="text" id="detect-text" placeholder="Enter text to detect language..." value="hello world">
    </div>

    <button onclick="detectLanguage()">Detect</button>

    <div id="detect-result" class="result" style="display: none;">
        <h3>Detected Language:</h3>
        <p><strong>Language:</strong> <span id="detected-language"></span></p>
    </div>

    <div id="detect-error" class="error" style="display: none;"></div>

    <h2>Pronunciation Training</h2>

    <div class="input-group">
        <label for="training-word">Word to practice:</label>
        <input type="text" id="training-word" placeholder="Enter word to practice..." value="hello">
    </div>

    <button onclick="getTrainingFeedback()">Get Feedback</button>

    <div id="training-result" class="result" style="display: none;">
        <h3>Training Feedback:</h3>
        <p><strong>Word:</strong> <span id="training-word-display"></span></p>
        <p><strong>Target Phonemes:</strong> <span id="training-target-phonemes" class="phoneme-list"></span></p>
        <p><strong>Pronunciation Tips:</strong></p>
        <ul id="training-tips"></ul>
        <p><strong>Practice Suggestions:</strong></p>
        <ul id="training-suggestions"></ul>
    </div>

    <div id="training-error" class="error" style="display: none;"></div>

    <h2>Pronunciation Practice</h2>

    <div class="input-group">
        <label for="practice-word">Word to practice:</label>
        <input type="text" id="practice-word" placeholder="Enter word to practice..." value="hello">
    </div>

    <button onclick="startPractice()">Start Practice</button>

    <div id="practice-result" class="result" style="display: none;">
        <h3>Practice Mode:</h3>
        <p><strong>Word:</strong> <span id="practice-word-display"></span></p>
        <p><strong>Target Phonemes:</strong> <span id="practice-target-phonemes" class="phoneme-list"></span></p>
        <p><strong>Your Attempt:</strong></p>
        <input type="text" id="practice-attempt" placeholder="Type how you would say it...">
        <button onclick="checkPractice()">Check</button>
        <button id="record-btn" onclick="toggleRecording()">Record</button>
        <button id="upload-btn" onclick="document.getElementById('audio-upload').click()">Upload Audio</button>
        <input type="file" id="audio-upload" accept="audio/*" style="display: none;" onchange="handleAudioUpload(event)">
        <div id="recording-status" style="display: none; margin: 10px 0;">
            <span id="recording-indicator" style="color: red;">● Recording...</span>
            <span id="recording-time"></span>
        </div>
        <audio id="practice-audio" controls style="width: 100%; margin-top: 10px; display: none;"></audio>
        <div id="practice-feedback" style="display: none;"></div>
    </div>

    <div id="practice-error" class="error" style="display: none;"></div>

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

        // Real-time phoneme preview
        let previewTimeout = null;
        document.getElementById('text').addEventListener('input', function(e) {
            const text = e.target.value;
            if (text.length === 0) {
                document.getElementById('live-preview').style.display = 'none';
                return;
            }

            // Debounce API calls
            clearTimeout(previewTimeout);
            previewTimeout = setTimeout(async () => {
                try {
                    const language = document.getElementById('language').value;
                    const response = await fetch('/api/encode', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text, language }),
                    });

                    const data = await response.json();

                    if (!data.error) {
                        document.getElementById('live-phonemes').textContent = data.phonemes.join(' ');
                        document.getElementById('live-preview').style.display = 'block';
                    }
                } catch (error) {
                    // Silently ignore errors for live preview
                }
            }, 150);
        });

        // Encode on Enter key
        document.getElementById('text').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                encodeText();
            }
        });

        function toggleSSML() {
            const ssmlHelp = document.getElementById('ssml-help');
            const ssmlCheckbox = document.getElementById('tts-ssml');
            ssmlHelp.style.display = ssmlCheckbox.checked ? 'block' : 'none';
        }

        async function synthesizeSpeech() {
            const text = document.getElementById('tts-text').value;
            const speed = parseFloat(document.getElementById('tts-speed').value);
            const pitch = parseInt(document.getElementById('tts-pitch').value);
            const useSSML = document.getElementById('tts-ssml').checked;

            try {
                const response = await fetch('/api/synthesize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text, speed, pitch, use_ssml: useSSML }),
                });

                const data = await response.json();

                if (data.error) {
                    document.getElementById('tts-error').textContent = data.error;
                    document.getElementById('tts-error').style.display = 'block';
                    document.getElementById('tts-result').style.display = 'none';
                } else {
                    document.getElementById('tts-duration').textContent = data.duration.toFixed(2);
                    document.getElementById('tts-sample-rate').textContent = data.sample_rate;

                    // Create audio from waveform
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const audioBuffer = audioContext.createBuffer(1, data.waveform.length, data.sample_rate);
                    audioBuffer.getChannelData(0).set(new Float32Array(data.waveform));

                    const source = audioContext.createBufferSource();
                    source.buffer = audioBuffer;

                    // For playback, we need to convert to a playable format
                    // Create a WAV file in memory
                    const wavBlob = createWavBlob(data.waveform, data.sample_rate);
                    const audioUrl = URL.createObjectURL(wavBlob);
                    document.getElementById('tts-audio').src = audioUrl;

                    document.getElementById('tts-result').style.display = 'block';
                    document.getElementById('tts-error').style.display = 'none';

                    // Draw spectrogram
                    drawSpectrogram(data.waveform, data.sample_rate);
                }
            } catch (error) {
                document.getElementById('tts-error').textContent = error.message;
                document.getElementById('tts-error').style.display = 'block';
                document.getElementById('tts-result').style.display = 'none';
            }
        }

        function createWavBlob(samples, sampleRate) {
            const buffer = new ArrayBuffer(44 + samples.length * 2);
            const view = new DataView(buffer);

            // WAV header
            writeString(view, 0, 'RIFF');
            view.setUint32(4, 36 + samples.length * 2, true);
            writeString(view, 8, 'WAVE');
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, 1, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true);
            writeString(view, 36, 'data');
            view.setUint32(40, samples.length * 2, true);

            // Convert samples to 16-bit PCM
            for (let i = 0; i < samples.length; i++) {
                const sample = Math.max(-1, Math.min(1, samples[i]));
                view.setInt16(44 + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            }

            return new Blob([buffer], { type: 'audio/wav' });
        }

        function writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        function drawSpectrogram(waveform, sampleRate) {
            const canvas = document.getElementById('spectrogram-canvas');
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;

            // Clear canvas
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, width, height);

            // Simple spectrogram using FFT
            const fftSize = 256;
            const hopSize = 128;
            const numFrames = Math.floor((waveform.length - fftSize) / hopSize);

            if (numFrames <= 0) return;

            for (let frame = 0; frame < numFrames; frame++) {
                const start = frame * hopSize;
                const frameData = waveform.slice(start, start + fftSize);

                // Apply Hamming window
                const windowed = new Float32Array(fftSize);
                for (let i = 0; i < fftSize; i++) {
                    windowed[i] = frameData[i] * (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (fftSize - 1)));
                }

                // Simple FFT (magnitude spectrum)
                const spectrum = new Float32Array(fftSize / 2);
                for (let k = 0; k < fftSize / 2; k++) {
                    let real = 0;
                    let imag = 0;
                    for (let n = 0; n < fftSize; n++) {
                        const angle = -2 * Math.PI * k * n / fftSize;
                        real += windowed[n] * Math.cos(angle);
                        imag += windowed[n] * Math.sin(angle);
                    }
                    spectrum[k] = Math.sqrt(real * real + imag * imag);
                }

                // Draw frame
                const x = (frame / numFrames) * width;
                const binHeight = height / (fftSize / 2);

                for (let k = 0; k < fftSize / 2; k++) {
                    const magnitude = Math.min(1, spectrum[k] * 10);
                    const y = height - (k * binHeight);

                    // Color mapping (simple heatmap)
                    const r = Math.floor(magnitude * 255);
                    const g = Math.floor(magnitude * 100);
                    const b = Math.floor((1 - magnitude) * 100);

                    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                    ctx.fillRect(x, y - binHeight, Math.ceil(width / numFrames), Math.ceil(binHeight));
                }
            }

            // Draw axis labels
            ctx.fillStyle = '#fff';
            ctx.font = '10px Arial';
            ctx.fillText('Time →', width - 50, height - 5);
            ctx.save();
            ctx.translate(10, height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Frequency ↑', 0, 0);
            ctx.restore();
        }

        // Pronunciation History
        let pronunciationHistory = JSON.parse(localStorage.getItem('pronunciationHistory') || '[]');

        function addToHistory(target, spoken, score, phonemes) {
            const entry = {
                timestamp: new Date().toISOString(),
                target,
                spoken,
                score,
                phonemes
            };
            pronunciationHistory.unshift(entry);
            if (pronunciationHistory.length > 50) {
                pronunciationHistory = pronunciationHistory.slice(0, 50);
            }
            localStorage.setItem('pronunciationHistory', JSON.stringify(pronunciationHistory));
            renderHistory();
        }

        function renderHistory() {
            const container = document.getElementById('history-list');
            if (pronunciationHistory.length === 0) {
                container.innerHTML = '<p style="color: #666;">No pronunciation attempts yet.</p>';
                return;
            }

            let html = '';
            pronunciationHistory.forEach((entry, index) => {
                const scorePercent = (entry.score * 100).toFixed(1);
                const scoreColor = entry.score >= 0.8 ? '#4caf50' : entry.score >= 0.5 ? '#ff9800' : '#f44336';
                const time = new Date(entry.timestamp).toLocaleTimeString();
                html += `
                    <div style="padding: 8px; margin: 5px 0; background: #1a1a2e; border-radius: 4px; border-left: 3px solid ${scoreColor};">
                        <div style="display: flex; justify-content: space-between;">
                            <span><strong>${entry.target}</strong> → ${entry.spoken}</span>
                            <span style="color: ${scoreColor};">${scorePercent}%</span>
                        </div>
                        <div style="font-size: 12px; color: #666; margin-top: 4px;">${time}</div>
                    </div>
                `;
            });
            container.innerHTML = html;
        }

        function clearHistory() {
            pronunciationHistory = [];
            localStorage.removeItem('pronunciationHistory');
            renderHistory();
        }

        function exportHistory() {
            const data = JSON.stringify(pronunciationHistory, null, 2);
            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `pronunciation-history-${new Date().toISOString().slice(0, 10)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        function importHistory(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const imported = JSON.parse(e.target.result);
                    if (Array.isArray(imported)) {
                        pronunciationHistory = [...imported, ...pronunciationHistory];
                        if (pronunciationHistory.length > 50) {
                            pronunciationHistory = pronunciationHistory.slice(0, 50);
                        }
                        localStorage.setItem('pronunciationHistory', JSON.stringify(pronunciationHistory));
                        renderHistory();
                        alert(`Imported ${imported.length} entries.`);
                    } else {
                        alert('Invalid file format.');
                    }
                } catch (error) {
                    alert('Error reading file: ' + error.message);
                }
            };
            reader.readAsText(file);
            event.target.value = '';
        }

        // Initialize history on page load
        renderHistory();

        // Synthesize on Enter key
        document.getElementById('tts-text').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                synthesizeSpeech();
            }
        });

        async function scorePronunciation() {
            const target = document.getElementById('target-text').value;
            const spoken = document.getElementById('spoken-text').value;

            try {
                const response = await fetch('/api/score', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ target, spoken }),
                });

                const data = await response.json();

                if (data.error) {
                    document.getElementById('score-error').textContent = data.error;
                    document.getElementById('score-error').style.display = 'block';
                    document.getElementById('score-result').style.display = 'none';
                } else {
                    document.getElementById('pronunciation-score').textContent = (data.score * 100).toFixed(1) + '%';
                    document.getElementById('pronunciation-precision').textContent = (data.precision * 100).toFixed(1) + '%';
                    document.getElementById('pronunciation-recall').textContent = (data.recall * 100).toFixed(1) + '%';
                    document.getElementById('target-phonemes').textContent = data.target_phonemes.join(' ');
                    document.getElementById('spoken-phonemes').textContent = data.spoken_phonemes.join(' ');
                    document.getElementById('score-result').style.display = 'block';
                    document.getElementById('score-error').style.display = 'none';

                    // Add to history
                    addToHistory(target, spoken, data.score, data.target_phonemes);
                }
            } catch (error) {
                document.getElementById('score-error').textContent = error.message;
                document.getElementById('score-error').style.display = 'block';
                document.getElementById('score-result').style.display = 'none';
            }
        }

        // Score on Enter key
        document.getElementById('spoken-text').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                scorePronunciation();
            }
        });

        async function handleScoreAudioUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            // Display the uploaded audio
            const audioElement = document.createElement('audio');
            audioElement.controls = true;
            audioElement.src = URL.createObjectURL(file);
            audioElement.style.width = '100%';
            audioElement.style.marginTop = '10px';

            const resultDiv = document.getElementById('score-result');
            const uploadedAudio = document.getElementById('uploaded-score-audio');
            if (uploadedAudio) {
                uploadedAudio.remove();
            }
            audioElement.id = 'uploaded-score-audio';
            resultDiv.parentNode.insertBefore(audioElement, resultDiv);

            // Show file info
            document.getElementById('pronunciation-score').textContent = 'Processing...';
            document.getElementById('pronunciation-precision').textContent = '-';
            document.getElementById('pronunciation-recall').textContent = '-';
            document.getElementById('target-phonemes').textContent = '-';
            document.getElementById('spoken-phonemes').textContent = '-';
            resultDiv.style.display = 'block';

            // Note: In a real implementation, you would send the audio to a speech-to-text service
            // For now, we'll show a message indicating the limitation
            setTimeout(() => {
                document.getElementById('pronunciation-score').textContent = 'N/A (requires STT)';
                document.getElementById('pronunciation-precision').textContent = '-';
                document.getElementById('pronunciation-recall').textContent = '-';
                document.getElementById('target-phonemes').textContent = 'Upload received: ' + file.name;
                document.getElementById('spoken-phonemes').textContent = 'Size: ' + (file.size / 1024).toFixed(1) + ' KB';
            }, 500);
        }

        async function detectLanguage() {
            const text = document.getElementById('detect-text').value;

            try {
                const response = await fetch('/api/detect', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text }),
                });

                const data = await response.json();

                if (data.error) {
                    document.getElementById('detect-error').textContent = data.error;
                    document.getElementById('detect-error').style.display = 'block';
                    document.getElementById('detect-result').style.display = 'none';
                } else {
                    document.getElementById('detected-language').textContent = data.language;
                    document.getElementById('detect-result').style.display = 'block';
                    document.getElementById('detect-error').style.display = 'none';
                }
            } catch (error) {
                document.getElementById('detect-error').textContent = error.message;
                document.getElementById('detect-error').style.display = 'block';
                document.getElementById('detect-result').style.display = 'none';
            }
        }

        // Detect on Enter key
        document.getElementById('detect-text').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                detectLanguage();
            }
        });

        async function getTrainingFeedback() {
            const word = document.getElementById('training-word').value;

            try {
                const response = await fetch('/api/training', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ word }),
                });

                const data = await response.json();

                if (data.error) {
                    document.getElementById('training-error').textContent = data.error;
                    document.getElementById('training-error').style.display = 'block';
                    document.getElementById('training-result').style.display = 'none';
                } else {
                    document.getElementById('training-word-display').textContent = data.word;
                    document.getElementById('training-target-phonemes').textContent = data.target_phonemes.join(' ');

                    // Add tips
                    const tipsList = document.getElementById('training-tips');
                    tipsList.innerHTML = '';
                    data.tips.forEach(tip => {
                        const li = document.createElement('li');
                        li.textContent = tip;
                        tipsList.appendChild(li);
                    });

                    // Add suggestions
                    const suggestionsList = document.getElementById('training-suggestions');
                    suggestionsList.innerHTML = '';
                    data.suggestions.forEach(suggestion => {
                        const li = document.createElement('li');
                        li.textContent = suggestion;
                        suggestionsList.appendChild(li);
                    });

                    document.getElementById('training-result').style.display = 'block';
                    document.getElementById('training-error').style.display = 'none';
                }
            } catch (error) {
                document.getElementById('training-error').textContent = error.message;
                document.getElementById('training-error').style.display = 'block';
                document.getElementById('training-result').style.display = 'none';
            }
        }

        // Training on Enter key
        document.getElementById('training-word').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                getTrainingFeedback();
            }
        });

        let currentPracticeWord = '';

        async function startPractice() {
            const word = document.getElementById('practice-word').value;
            currentPracticeWord = word;

            try {
                const response = await fetch('/api/training', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ word }),
                });

                const data = await response.json();

                if (data.error) {
                    document.getElementById('practice-error').textContent = data.error;
                    document.getElementById('practice-error').style.display = 'block';
                    document.getElementById('practice-result').style.display = 'none';
                } else {
                    document.getElementById('practice-word-display').textContent = data.word;
                    document.getElementById('practice-target-phonemes').textContent = data.target_phonemes.join(' ');
                    document.getElementById('practice-attempt').value = '';
                    document.getElementById('practice-feedback').style.display = 'none';
                    document.getElementById('practice-result').style.display = 'block';
                    document.getElementById('practice-error').style.display = 'none';
                }
            } catch (error) {
                document.getElementById('practice-error').textContent = error.message;
                document.getElementById('practice-error').style.display = 'block';
                document.getElementById('practice-result').style.display = 'none';
            }
        }

        async function checkPractice() {
            const attempt = document.getElementById('practice-attempt').value;

            try {
                const response = await fetch('/api/score', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ target: currentPracticeWord, spoken: attempt }),
                });

                const data = await response.json();

                if (data.error) {
                    document.getElementById('practice-error').textContent = data.error;
                    document.getElementById('practice-error').style.display = 'block';
                } else {
                    const feedback = document.getElementById('practice-feedback');
                    feedback.innerHTML = `
                        <p><strong>Score:</strong> ${(data.score * 100).toFixed(1)}%</p>
                        <p><strong>Precision:</strong> ${(data.precision * 100).toFixed(1)}%</p>
                        <p><strong>Recall:</strong> ${(data.recall * 100).toFixed(1)}%</p>
                        <p><strong>Your Phonemes:</strong> ${data.spoken_phonemes.join(' ')}</p>
                    `;
                    feedback.style.display = 'block';
                }
            } catch (error) {
                document.getElementById('practice-error').textContent = error.message;
                document.getElementById('practice-error').style.display = 'block';
            }
        }

        // Practice on Enter key
        document.getElementById('practice-attempt').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                checkPractice();
            }
        });

        let mediaRecorder = null;
        let audioChunks = [];
        let recordingStartTime = null;
        let recordingTimer = null;

        async function toggleRecording() {
            const recordBtn = document.getElementById('record-btn');
            const recordingStatus = document.getElementById('recording-status');

            if (mediaRecorder && mediaRecorder.state === 'recording') {
                // Stop recording
                mediaRecorder.stop();
                recordBtn.textContent = 'Record';
                recordingStatus.style.display = 'none';
                clearInterval(recordingTimer);
            } else {
                // Start recording
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];

                    mediaRecorder.ondataavailable = (event) => {
                        audioChunks.push(event.data);
                    };

                    mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const audioElement = document.getElementById('practice-audio');
                        audioElement.src = audioUrl;
                        audioElement.style.display = 'block';

                        // Stop all tracks to release microphone
                        stream.getTracks().forEach(track => track.stop());
                    };

                    mediaRecorder.start();
                    recordBtn.textContent = 'Stop';
                    recordingStatus.style.display = 'block';
                    recordingStartTime = Date.now();

                    // Update recording time
                    recordingTimer = setInterval(() => {
                        const elapsed = ((Date.now() - recordingStartTime) / 1000).toFixed(1);
                        document.getElementById('recording-time').textContent = `${elapsed}s`;
                    }, 100);
                } catch (error) {
                    document.getElementById('practice-error').textContent = 'Microphone access denied: ' + error.message;
                    document.getElementById('practice-error').style.display = 'block';
                }
            }
        }

        async function handleAudioUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            const audioElement = document.getElementById('practice-audio');
            audioElement.src = URL.createObjectURL(file);
            audioElement.style.display = 'block';

            // For now, we just display the uploaded audio
            // In a real implementation, you would send it to the server for analysis
            document.getElementById('practice-feedback').innerHTML = `
                <p><strong>Uploaded:</strong> ${file.name}</p>
                <p><strong>Size:</strong> ${(file.size / 1024).toFixed(1)} KB</p>
                <p><em>Note: Audio analysis requires speech-to-text integration.</em></p>
            `;
            document.getElementById('practice-feedback').style.display = 'block';
        }
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


@app.route('/api/synthesize', methods=['POST'])
def api_synthesize():
    """Synthesize speech from text."""
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        from domains.multimodal.tts import TTSEngine
        import numpy as np

        engine = TTSEngine()
        speed = data.get('speed', 1.0)
        pitch = data.get('pitch', 0)
        use_ssml = data.get('use_ssml', False)

        if use_ssml:
            waveform = engine.ssml_to_waveform(text, max_frames=200)
        else:
            waveform = engine.text_to_waveform(text, speed=speed, pitch_shift=pitch)

        return jsonify({
            'waveform': waveform.tolist(),
            'sample_rate': engine.sample_rate,
            'duration': len(waveform) / engine.sample_rate,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-synthesize', methods=['POST'])
def api_batch_synthesize():
    """Synthesize speech from multiple texts."""
    data = request.get_json()
    texts = data.get('texts', [])

    if not texts:
        return jsonify({'error': 'No texts provided'}), 400

    try:
        from domains.multimodal.tts import TTSEngine
        import numpy as np

        engine = TTSEngine()
        speed = data.get('speed', 1.0)
        pitch = data.get('pitch', 0)

        results = []
        for text in texts:
            waveform = engine.text_to_waveform(text, speed=speed, pitch_shift=pitch)
            results.append({
                'text': text,
                'waveform': waveform.tolist(),
                'sample_rate': engine.sample_rate,
                'duration': len(waveform) / engine.sample_rate,
            })

        return jsonify({
            'results': results,
            'count': len(results),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/score', methods=['POST'])
def api_score():
    """Score pronunciation accuracy."""
    data = request.get_json()
    target = data.get('target', '')
    spoken = data.get('spoken', '')

    if not target or not spoken:
        return jsonify({'error': 'Both target and spoken text required'}), 400

    try:
        result = encoder.score_pronunciation(target, spoken)

        return jsonify({
            'target': target,
            'spoken': spoken,
            'score': result['score'],
            'precision': result['precision'],
            'recall': result['recall'],
            'target_phonemes': result['target_phonemes'],
            'spoken_phonemes': result['spoken_phonemes'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """Detect language from text."""
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        language = detect_language(text)

        return jsonify({
            'text': text,
            'language': language,
            'supported_languages': encoder.supported_languages,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/training', methods=['POST'])
def api_training():
    """Get pronunciation training feedback."""
    data = request.get_json()
    word = data.get('word', '')

    if not word:
        return jsonify({'error': 'No word provided'}), 400

    try:
        # Encode the word
        ids = encoder.encode(word)
        phonemes = encoder.decode_phonemes(ids)

        # Generate tips based on phonemes
        tips = []
        suggestions = []

        # Basic pronunciation tips based on phoneme patterns
        for i, phoneme in enumerate(phonemes):
            if phoneme in ['TH', 'DH']:  # English-specific
                tips.append(f"The '{phoneme}' sound is produced by placing your tongue between your teeth")
            elif phoneme in ['SH', 'ZH']:
                tips.append(f"The '{phoneme}' sound requires rounded lips")
            elif phoneme in ['R', 'L']:
                tips.append(f"The '{phoneme}' sound involves tongue movement along the roof of your mouth")
            elif phoneme in ['P', 'B', 'T', 'D', 'K', 'G']:
                tips.append(f"The '{phoneme}' sound is a stop consonant - stop airflow briefly then release")

        # Add general suggestions
        suggestions.append(f"Practice saying '{word}' slowly, focusing on each sound")
        suggestions.append(f"Record yourself saying '{word}' and compare to the phoneme breakdown")
        suggestions.append(f"Break '{word}' into syllables: {'-'.join(phonemes)}")

        return jsonify({
            'word': word,
            'target_phonemes': phonemes,
            'ids': ids.flatten().tolist(),
            'tips': tips,
            'suggestions': suggestions,
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
