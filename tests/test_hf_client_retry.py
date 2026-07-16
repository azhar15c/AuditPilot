"""Tests for models.hf_client's Groq 429 retry/backoff behavior.

Concurrent per-employee LLM calls (the multi-agent fan-out's whole point)
can burst past Groq's per-minute token limit even on unremarkable total
usage, since the old sequential pipeline spread the same calls out over the
run's full wall-clock time and the new one doesn't. _call_with_retry keeps
the concurrency benefit by retrying with the wait time Groq itself reports,
instead of serializing everything or capping fan-out width.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from groq import RateLimitError

from models.hf_client import HFClient, _retry_after_seconds


def _rate_limit_error(retry_after_header: str | None = None, message: str = "rate_limit_exceeded") -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"retry-after": retry_after_header} if retry_after_header else {}
    response = httpx.Response(429, headers=headers, request=request)
    return RateLimitError(message, response=response, body={"error": {"message": message}})


class TestRetryAfterSeconds:
    def test_prefers_retry_after_header(self):
        exc = _rate_limit_error(retry_after_header="12")
        assert _retry_after_seconds(exc) == 12.0

    def test_falls_back_to_message_hint(self):
        exc = _rate_limit_error(message="Please try again in 17.205s. Need more tokens?")
        assert _retry_after_seconds(exc) == pytest.approx(17.205)

    def test_falls_back_to_default_when_nothing_parseable(self):
        exc = _rate_limit_error(message="rate limited, no timing info")
        assert _retry_after_seconds(exc) == 2.0


class TestChatWithToolsRetriesOnRateLimit:
    def test_retries_once_then_succeeds(self):
        client = HFClient.__new__(HFClient)  # bypass __init__ — no real Groq/HF client needed
        client._gen = MagicMock()
        success_response = MagicMock()
        client._gen.chat.completions.create.side_effect = [
            _rate_limit_error(retry_after_header="0.01"),
            success_response,
        ]

        with patch("models.hf_client.time.sleep") as mock_sleep:
            result = client.chat_with_tools([{"role": "user", "content": "hi"}], [], tool_choice="required")

        assert result is success_response
        assert client._gen.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once()

    def test_exhausts_retries_and_reraises(self):
        client = HFClient.__new__(HFClient)
        client._gen = MagicMock()
        client._gen.chat.completions.create.side_effect = _rate_limit_error(retry_after_header="0.01")

        with patch("models.hf_client.time.sleep"):
            with pytest.raises(RateLimitError):
                client.chat_with_tools([{"role": "user", "content": "hi"}], [], tool_choice="required")

        # 1 initial attempt + _MAX_RATE_LIMIT_RETRIES retries
        assert client._gen.chat.completions.create.call_count == 4

    def test_non_rate_limit_errors_are_not_retried(self):
        client = HFClient.__new__(HFClient)
        client._gen = MagicMock()
        client._gen.chat.completions.create.side_effect = ValueError("some other failure")

        with pytest.raises(ValueError):
            client.chat_with_tools([{"role": "user", "content": "hi"}], [], tool_choice="required")

        assert client._gen.chat.completions.create.call_count == 1


class TestGenerateRetriesOnRateLimit:
    def test_retries_then_succeeds(self):
        client = HFClient.__new__(HFClient)
        client._gen = MagicMock()
        success_response = MagicMock()
        success_response.choices = [MagicMock(message=MagicMock(content="generated text"))]
        client._gen.chat.completions.create.side_effect = [
            _rate_limit_error(retry_after_header="0.01"),
            success_response,
        ]

        with patch("models.hf_client.time.sleep"):
            result = client.generate(prompt="p", system="s")

        assert result == "generated text"
        assert client._gen.chat.completions.create.call_count == 2
