"""Tests for models.hf_client's Groq 429 retry/backoff behavior.

Concurrent per-employee LLM calls (the multi-agent fan-out's whole point)
can burst past Groq's per-minute token limit even on unremarkable total
usage, since the old sequential pipeline spread the same calls out over the
run's full wall-clock time and the new one doesn't. _call_with_retry keeps
the concurrency benefit by retrying with the wait time Groq itself reports,
instead of serializing everything or capping fan-out width.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
from groq import RateLimitError

import models.hf_client as hf_client_module
from models.hf_client import HFClient, _retry_after_seconds


def _rate_limit_error(retry_after_header: str | None = None, message: str = "rate_limit_exceeded") -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"retry-after": retry_after_header} if retry_after_header else {}
    response = httpx.Response(429, headers=headers, request=request)
    return RateLimitError(message, response=response, body={"error": {"message": message}})


class TestRetryAfterSeconds:
    """Waits are jittered +/- _JITTER_FRACTION so concurrent callers rate-limited
    in the same instant don't all retry in lockstep — assertions check the
    jittered range around the base value, not an exact match."""

    def test_prefers_retry_after_header(self):
        exc = _rate_limit_error(retry_after_header="12")
        assert _retry_after_seconds(exc) == pytest.approx(12.0, rel=0.31)

    def test_falls_back_to_message_hint(self):
        exc = _rate_limit_error(message="Please try again in 17.205s. Need more tokens?")
        assert _retry_after_seconds(exc) == pytest.approx(17.205, rel=0.31)

    def test_falls_back_to_default_when_nothing_parseable(self):
        exc = _rate_limit_error(message="rate limited, no timing info")
        assert _retry_after_seconds(exc) == pytest.approx(2.0, rel=0.31)

    def test_jitter_actually_varies_the_wait(self):
        exc = _rate_limit_error(retry_after_header="100")  # large base makes jitter unmistakable
        samples = {_retry_after_seconds(exc) for _ in range(20)}
        assert len(samples) > 1, "expected jittered waits to vary across repeated calls"
        assert all(70.0 <= s <= 130.0 for s in samples)


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


class TestConcurrencyIsBounded:
    """Without a cap, N employees' concurrent calls all fire at once, all get
    429'd together, and retry in lockstep — recreating the same burst on
    every round. _MAX_CONCURRENT_GROQ_CALLS converts that into calls queuing
    locally and executing at a sustainable rate."""

    def test_concurrent_calls_never_exceed_the_cap(self):
        client = HFClient.__new__(HFClient)
        client._gen = MagicMock()

        in_flight = 0
        max_observed = 0
        lock = threading.Lock()

        def _slow_call(*args, **kwargs):
            nonlocal in_flight, max_observed
            with lock:
                in_flight += 1
                max_observed = max(max_observed, in_flight)
            time.sleep(0.05)
            with lock:
                in_flight -= 1
            response = MagicMock()
            response.choices = [MagicMock(message=MagicMock(content="ok"))]
            return response

        client._gen.chat.completions.create.side_effect = _slow_call

        threads = [
            threading.Thread(target=client.generate, kwargs={"prompt": "p", "system": "s"})
            for _ in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_observed <= hf_client_module._MAX_CONCURRENT_GROQ_CALLS, (
            f"observed {max_observed} concurrent Groq calls, cap is "
            f"{hf_client_module._MAX_CONCURRENT_GROQ_CALLS}"
        )


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
