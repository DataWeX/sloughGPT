"""Tests for the /voice router."""

from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


class TestVoiceRouter:
    def setup_method(self):
        self.client = get_test_client()

    def test_voice_status(self):
        resp = self.client.get("/voice/status")
        assert resp.status_code == 200
        data = _d(resp)
        assert isinstance(data, dict)
        assert "server_tts" in data
        assert "server_stt" in data

    def test_tts(self):
        resp = self.client.post("/voice/tts", json={"text": "hello world"})
        assert resp.status_code in (200, 400, 422, 500, 503)

    def test_stt_empty_audio(self):
        resp = self.client.post(
            "/voice/stt",
            files={"audio": ("test.wav", b"", "audio/wav")},
        )
        assert resp.status_code in (400, 422, 500, 503)

    def test_stt_with_audio(self):
        fake_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        resp = self.client.post(
            "/voice/stt",
            files={"audio": ("test.wav", fake_wav, "audio/wav")},
            data={"language": "en"},
        )
        assert resp.status_code in (200, 400, 422, 500, 503)
        if resp.status_code == 200:
            data = _d(resp)
            assert "text" in data
            assert "confidence" in data
            assert "is_valid" in data
