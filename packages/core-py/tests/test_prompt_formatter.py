"""Tests for domains/inference/prompt_formatter.py — message formatting + output cleaning."""

import pytest
from domains.inference.prompt_formatter import PromptFormatter


class TestBaseFormat:
    def test_single_user_message(self):
        fmt = PromptFormatter()
        result = fmt.messages_to_prompt([{"role": "user", "content": "Hello"}])
        assert "User: Hello" in result
        assert result.endswith("Assistant:")

    def test_user_assistant_turns(self):
        fmt = PromptFormatter()
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hey!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = fmt.messages_to_prompt(msgs)
        assert "User: Hi" in result
        assert "Assistant: Hey!" in result
        assert "User: How are you?" in result

    def test_system_message_skipped_for_base(self):
        fmt = PromptFormatter()
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = fmt.messages_to_prompt(msgs)
        assert "Be helpful" not in result
        assert "User: Hello" in result

    def test_empty_messages(self):
        fmt = PromptFormatter()
        result = fmt.messages_to_prompt([])
        assert result == "Assistant:"

    def test_custom_prefixes(self):
        fmt = PromptFormatter(user_prefix="Human", assistant_prefix="AI")
        result = fmt.messages_to_prompt([{"role": "user", "content": "Hi"}])
        assert "Human: Hi" in result
        assert result.endswith("AI:")

    def test_custom_format_fn(self):
        custom = lambda msgs: "CUSTOM:" + msgs[0]["content"]
        fmt = PromptFormatter(format_fn=custom)
        result = fmt.messages_to_prompt([{"role": "user", "content": "Hi"}])
        assert result == "CUSTOM:Hi"


class TestChatTemplate:
    def test_uses_chat_template_when_available(self):
        class FakeTokenizer:
            chat_template = "template exists"

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "TEMPLATE_OUTPUT"

        fmt = PromptFormatter(tokenizer=FakeTokenizer())
        result = fmt.messages_to_prompt([{"role": "user", "content": "Hi"}])
        assert result == "TEMPLATE_OUTPUT"

    def test_falls_back_when_template_raises(self):
        class BrokenTokenizer:
            chat_template = "broken"

            def apply_chat_template(self, *a, **kw):
                raise ValueError("nope")

        fmt = PromptFormatter(tokenizer=BrokenTokenizer())
        result = fmt.messages_to_prompt([{"role": "user", "content": "Hi"}])
        assert "User: Hi" in result


class TestCleanChunk:
    def test_strips_im_start_tokens(self):
        fmt = PromptFormatter()
        result = fmt.clean_chunk("<|im_start|>assistant\nHello")
        assert "<|im_start|>" not in result
        assert "Hello" in result

    def test_strips_im_end_tokens(self):
        fmt = PromptFormatter()
        result = fmt.clean_chunk("Hello<|im_end|>")
        assert "<|im_end|>" not in result
        assert "Hello" in result

    def test_strips_eot_id(self):
        fmt = PromptFormatter()
        result = fmt.clean_chunk("Hello<|eot_id|>")
        assert "<|eot_id|>" not in result

    def test_first_chunk_strips_leading_newlines(self):
        fmt = PromptFormatter()
        result = fmt.clean_chunk("\n\n\nHello", first=True)
        assert result == "Hello"

    def test_first_chunk_strips_assistant_prefix(self):
        fmt = PromptFormatter()
        result = fmt.clean_chunk("Assistant: Hello", first=True)
        assert result == "Hello"

    def test_first_chunk_strips_user_prefix(self):
        fmt = PromptFormatter()
        result = fmt.clean_chunk("User: Hello", first=True)
        assert result == "Hello"

    def test_non_first_chunk_keeps_content(self):
        fmt = PromptFormatter()
        result = fmt.clean_chunk(" some text ", first=False)
        assert "some text" in result

    def test_strips_knowledge_blocks(self):
        fmt = PromptFormatter()
        result = fmt.clean_chunk("[KNOWLEDGE]fact[/KNOWLEDGE] response")
        assert "KNOWLEDGE" not in result
        assert "response" in result

    def test_strips_personality_instructions(self):
        fmt = PromptFormatter()
        result = fmt.clean_chunk("[PERSONALITY INSTRUCTIONS]be nice\nAssistant: Hello")
        assert "PERSONALITY" not in result
        assert "Hello" in result


class TestCleanResponse:
    def test_strips_special_tokens(self):
        fmt = PromptFormatter()
        result = fmt.clean_response("<|im_start|>assistant\nHello<|im_end|>")
        assert "<|" not in result
        assert "Hello" in result

    def test_strips_trailing_role(self):
        fmt = PromptFormatter()
        result = fmt.clean_response("Hello\n\nAssistant:")
        assert "Assistant:" not in result
        assert "Hello" in result

    def test_strips_trailing_user_role(self):
        fmt = PromptFormatter()
        result = fmt.clean_response("Hello\n\nUser:")
        assert "User:" not in result

    def test_strips_knowledge(self):
        fmt = PromptFormatter()
        result = fmt.clean_response("[KNOWLEDGE]info[/KNOWLEDGE] response")
        assert "KNOWLEDGE" not in result
        assert "response" in result

    def test_strips_leading_trailing_whitespace(self):
        fmt = PromptFormatter()
        result = fmt.clean_response("  Hello  ")
        assert result == "Hello"
