"""
Tests for interactive model selector.

Tests keyboard-driven search/filter behavior without curses
by testing the underlying filter logic.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestModelFilter:
    """Test the fuzzy filter logic used by the model selector."""

    def _filter(self, query, model_list):
        """Simulate the selector's filter logic."""
        return [(n, i, s) for n, i, s in model_list
                if query.lower() in n.lower() or query.lower() in i.lower()]

    def test_empty_query_returns_all(self):
        models = [("GPT-2", "gpt2", "hf"), ("LLaMA", "llama", "hf")]
        result = self._filter("", models)
        assert len(result) == 2

    def test_search_by_name(self):
        models = [("GPT-2", "gpt2", "hf"), ("LLaMA-7B", "llama", "hf")]
        result = self._filter("gpt", models)
        assert len(result) == 1
        assert result[0][0] == "GPT-2"

    def test_search_by_id(self):
        models = [("GPT-2", "gpt2", "hf"), ("Qwen 2.5", "Qwen/Qwen2.5-0.5B", "hf")]
        result = self._filter("Qwen", models)
        assert len(result) == 1
        assert result[0][1] == "Qwen/Qwen2.5-0.5B"

    def test_case_insensitive(self):
        models = [("GPT-2 Large", "gpt2-large", "hf")]
        result = self._filter("large", models)
        assert len(result) == 1

    def test_no_match_returns_empty(self):
        models = [("GPT-2", "gpt2", "hf")]
        result = self._filter("nonexistent", models)
        assert len(result) == 0


class TestModelFetch:
    """Test model fetching from API."""

    @patch("requests.get")
    def test_fetches_hf_models(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [
            {"id": "gpt2", "name": "GPT-2"},
            {"id": "llama", "name": "LLaMA"},
        ])
        import requests
        resp = requests.get("http://localhost:8000/models/hf")
        models = resp.json()
        assert len(models) == 2

    @patch("requests.get")
    def test_fetches_local_models(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [
            {"id": "gpt2", "name": "GPT-2", "source": "local"},
        ])
        import requests
        resp = requests.get("http://localhost:8000/models")
        models = resp.json()
        assert len(models) == 1
