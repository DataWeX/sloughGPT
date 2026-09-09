"""Tests for phoneme_web.py — Flask web application endpoints."""

import json
import pytest

# Skip all tests if Flask is not installed
pytest.importorskip("flask")

from domains.multimodal.phoneme_web import app, encoder


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestPhonemeWebEncode:
    def test_encode_success(self, client):
        response = client.post("/api/encode", json={"text": "hello", "language": "en"})
        assert response.status_code == 200
        data = response.get_json()
        assert "phonemes" in data
        assert "ids" in data
        assert "decoded" in data
        assert "language" in data
        assert data["language"] == "en"

    def test_encode_auto_detect(self, client):
        response = client.post("/api/encode", json={"text": "hello"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["language"] == "en"

    def test_encode_empty_text(self, client):
        response = client.post("/api/encode", json={"text": ""})
        assert response.status_code == 400

    def test_encode_german(self, client):
        response = client.post("/api/encode", json={"text": "ich", "language": "de"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["language"] == "de"

    def test_encode_french(self, client):
        response = client.post("/api/encode", json={"text": "je", "language": "fr"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["language"] == "fr"

    def test_encode_spanish(self, client):
        response = client.post("/api/encode", json={"text": "hola", "language": "es"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["language"] == "es"

    def test_encode_italian(self, client):
        response = client.post("/api/encode", json={"text": "ciao", "language": "it"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["language"] == "it"

    def test_encode_portuguese(self, client):
        response = client.post("/api/encode", json={"text": "ola", "language": "pt"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["language"] == "pt"


class TestPhonemeWebScore:
    def test_score_success(self, client):
        response = client.post("/api/score", json={"target": "hello", "spoken": "hello"})
        assert response.status_code == 200
        data = response.get_json()
        assert "score" in data
        assert "precision" in data
        assert "recall" in data
        assert "target_phonemes" in data
        assert "spoken_phonemes" in data
        assert data["score"] == 1.0

    def test_score_imperfect(self, client):
        response = client.post("/api/score", json={"target": "hello", "spoken": "helo"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["score"] < 1.0

    def test_score_empty_target(self, client):
        response = client.post("/api/score", json={"target": "", "spoken": "hello"})
        assert response.status_code == 400

    def test_score_empty_spoken(self, client):
        response = client.post("/api/score", json={"target": "hello", "spoken": ""})
        assert response.status_code == 400


class TestPhonemeWebDetect:
    def test_detect_success(self, client):
        response = client.post("/api/detect", json={"text": "hello"})
        assert response.status_code == 200
        data = response.get_json()
        assert "language" in data
        assert data["language"] == "en"

    def test_detect_german(self, client):
        response = client.post("/api/detect", json={"text": "ich"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["language"] == "de"

    def test_detect_empty_text(self, client):
        response = client.post("/api/detect", json={"text": ""})
        assert response.status_code == 400


class TestPhonemeWebIndex:
    def test_index_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"Phoneme Encoder" in response.data

    def test_index_contains_forms(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"encodeText" in response.data
        assert b"synthesizeSpeech" in response.data
        assert b"scorePronunciation" in response.data

    def test_index_contains_quiz(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"quiz" in response.data.lower()

    def test_index_contains_compare(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"compare" in response.data.lower()

    def test_index_contains_wotd(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"Word of the Day" in response.data


class TestPhonemeWebQuiz:
    def test_quiz_endpoint(self, client):
        response = client.post("/api/quiz", json={"language": "en"})
        assert response.status_code == 200
        data = response.get_json()
        assert "word" in data
        assert "language" in data
        assert "phonemes" in data

    def test_quiz_german(self, client):
        response = client.post("/api/quiz", json={"language": "de"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["language"] == "de"

    def test_quiz_invalid_language(self, client):
        response = client.post("/api/quiz", json={"language": "xyz"})
        assert response.status_code == 400


class TestPhonemeWebCompare:
    def test_compare_same_words(self, client):
        response = client.post("/api/encode", json={"text": "hello", "language": "en"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["phonemes"]

    def test_compare_different_words(self, client):
        res1 = client.post("/api/encode", json={"text": "hello", "language": "en"})
        res2 = client.post("/api/encode", json={"text": "world", "language": "en"})
        assert res1.status_code == 200
        assert res2.status_code == 200
        data1 = res1.get_json()
        data2 = res2.get_json()
        assert data1["phonemes"] != data2["phonemes"]
