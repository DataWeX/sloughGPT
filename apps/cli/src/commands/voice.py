"""
Voice commands - Text-to-speech and speech-to-text via the API.
"""
import sys
import os
from pathlib import Path

from domains.logging import get_global

log = get_global()


def cmd_voice_tts(args):
    """Convert text to speech and save/play the audio."""
    import requests
    base_url = f"http://{args.host}:{args.port}"
    try:
        resp = requests.post(
            f"{base_url}/voice/tts",
            json={"text": args.text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("data", resp.json())
        if not data.get("audio"):
            log.warning("TTS returned empty audio")
            return
        import base64
        audio_bytes = base64.b64decode(data["audio"])
        out_path = args.output or "tts_output.wav"
        Path(out_path).write_bytes(audio_bytes)
        log.success(f"Saved {len(audio_bytes)} bytes to {out_path}")
        if args.play:
            _play_audio(out_path)
    except requests.ConnectionError:
        log.error(f"Cannot connect to {base_url} — is the server running?")
        sys.exit(1)
    except Exception as e:
        log.error(f"TTS failed: {e}")
        sys.exit(1)


def cmd_voice_stt(args):
    """Transcribe an audio file to text."""
    import requests
    base_url = f"http://{args.host}:{args.port}"
    audio_path = Path(args.file)
    if not audio_path.is_file():
        log.error(f"File not found: {args.file}")
        sys.exit(1)
    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{base_url}/voice/stt",
                files={"audio": (audio_path.name, f, "audio/wav")},
                data={"language": args.language},
                timeout=60,
            )
        resp.raise_for_status()
        data = resp.json().get("data", resp.json())
        text = data.get("text", "")
        if text:
            print(text)
        else:
            log.warning("No speech detected in audio")
        if args.verbose:
            log.info(f"Confidence: {data.get('confidence', 0):.2f}")
            log.info(f"Language: {data.get('language', 'unknown')}")
            log.info(f"Valid: {data.get('is_valid', False)}")
    except requests.ConnectionError:
        log.error(f"Cannot connect to {base_url} — is the server running?")
        sys.exit(1)
    except Exception as e:
        log.error(f"STT failed: {e}")
        sys.exit(1)


def _play_audio(path: str):
    """Best-effort audio playback."""
    import subprocess
    for cmd in (["aplay", path], ["ffplay", "-nodisp", "-autoexit", path], ["afplay", path]):
        try:
            subprocess.run(cmd, capture_output=True, timeout=30)
            return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    log.warning("No audio player found (tried aplay, ffplay, afplay)")
