"""Coverage for remaining sloughgpt_sdk.models branches."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "sdk-py"))

from sloughgpt_sdk.models import (  # noqa: E402
    BatchRequest,
    BatchResult,
    ChatMessage,
    ChatRequest,
    GenerateRequest,
)


class TestGenerateRequestFlags:
    def test_early_stopping_and_personality(self):
        d = GenerateRequest("p", early_stopping=True, personality="warm").to_dict()
        assert d["early_stopping"] is True
        assert d["personality"] == "warm"


class TestChatMessageName:
    def test_to_dict_includes_name(self):
        assert ChatMessage("user", "hi", name="alice").to_dict() == {
            "role": "user",
            "content": "hi",
            "name": "alice",
        }

    def test_assistant_classmethod(self):
        assert ChatMessage.assistant("a").to_dict() == {
            "role": "assistant",
            "content": "a",
        }


class TestBatchRequest:
    def test_to_dict(self):
        d = BatchRequest(["a", "b"], max_new_tokens=50, temperature=0.2).to_dict()
        assert d["prompts"] == ["a", "b"]
        assert d["max_new_tokens"] == 50
        assert d["temperature"] == 0.2


class TestChatRequestAdditional:
    def test_to_dict_with_model_and_stream_flag_ignored(self):
        d = ChatRequest(
            messages=[ChatMessage.user("q")],
            model="gpt2",
            temperature=0.3,
            max_new_tokens=10,
            top_p=0.5,
            top_k=20,
        ).to_dict()
        assert d["model"] == "gpt2"
        assert d["temperature"] == 0.3
        assert d["max_new_tokens"] == 10
        assert d["top_p"] == 0.5
        assert d["top_k"] == 20


class TestBatchResultFromResponse:
    def test_responses_branch(self):
        result = BatchResult.from_response(
            {"responses": [{"text": "a"}, {"text": "b"}], "total_time_ms": 5.0},
            ["p1", "p2"],
        )
        assert result.total_prompts == 2
        assert result.successful == 2
        assert result.failed == 0
        assert result.total_time_ms == 5.0
        assert [r.generated_text for r in result.results] == ["a", "b"]
        assert result.results[0].prompt == "p1"
        assert result.results[1].prompt == "p2"