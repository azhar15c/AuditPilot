"""Tests for agents.nodes.retrieval_agent — Workstream B.

retrieval_agent runs an iterative retrieve -> evaluate_sufficiency loop
(up to _MAX_ITERATIONS = 4) against the NCCI RAG store, forcing a tool call
named "evaluate_retrieval" (args: sufficient, reasoning, refined_query) on
every iteration via _client.chat_with_tools(...).

Guarded with importorskip since agents/nodes/retrieval_agent.py does not
exist yet in this worktree — Workstream B builds it in a separate worktree
and merges later.

Fake Groq response objects are built as small local classes with explicit
attribute assignment rather than relying on MagicMock auto-attributes —
per the project's own postmortem, an unset `.tool_calls` on a bare MagicMock
is truthy and silently hides bugs.
"""

import json

import pytest

pytest.importorskip("agents.nodes.retrieval_agent")

from unittest.mock import patch  # noqa: E402

from agents.nodes.retrieval_agent import retrieval_agent  # noqa: E402


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name: str, arguments: str):
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, tool_calls, content=None):
        self.tool_calls = tool_calls
        self.content = content


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


def _evaluate_response(sufficient: bool, reasoning: str = "", refined_query: str = "") -> _FakeResponse:
    args = json.dumps({
        "sufficient": sufficient,
        "reasoning": reasoning,
        "refined_query": refined_query,
    })
    tc = _FakeToolCall("evaluate_retrieval", args)
    return _FakeResponse(_FakeMessage(tool_calls=[tc]))


_SAMPLE_CITATIONS = [
    {"chunk_text": "ROOFING - ALL KINDS & DRIVERS. Code 5551.", "distance": 0.2, "metadata": {"source": "manual"}},
]


@pytest.mark.workstream_b
@pytest.mark.retrieval_convergence
class TestRetrievalAgentConvergence:
    def test_sufficiency_met_on_first_try(self, base_employee_task_state):
        with patch("agents.nodes.retrieval_agent._retriever") as mock_retriever, \
             patch("agents.nodes.retrieval_agent._client") as mock_client:
            mock_retriever.query.return_value = _SAMPLE_CITATIONS
            mock_client.chat_with_tools.return_value = _evaluate_response(
                sufficient=True, reasoning="Excerpt explicitly names roofer.", refined_query="",
            )

            result = retrieval_agent(base_employee_task_state)

        output = result["retrieval_output"]
        assert output["iterations_run"] == 1
        assert output["sufficiency_met"] is True

    def test_engineered_unanswerable_stops_at_max_iterations(self, base_employee_task_state):
        with patch("agents.nodes.retrieval_agent._retriever") as mock_retriever, \
             patch("agents.nodes.retrieval_agent._client") as mock_client:
            mock_retriever.query.return_value = []
            mock_client.chat_with_tools.return_value = _evaluate_response(
                sufficient=False, reasoning="No matching excerpt found.", refined_query="broader search",
            )

            result = retrieval_agent(base_employee_task_state)

        output = result["retrieval_output"]
        assert output["iterations_run"] == 4
        assert output["sufficiency_met"] is False
        assert output["sufficiency_reasoning"] != ""

    def test_reformulates_query_between_iterations(self, base_employee_task_state):
        responses = [
            _evaluate_response(sufficient=False, reasoning="too vague", refined_query="roofing subcontractor shingles"),
            _evaluate_response(sufficient=True, reasoning="found it", refined_query=""),
        ]
        with patch("agents.nodes.retrieval_agent._retriever") as mock_retriever, \
             patch("agents.nodes.retrieval_agent._client") as mock_client:
            mock_retriever.query.return_value = _SAMPLE_CITATIONS
            mock_client.chat_with_tools.side_effect = responses

            result = retrieval_agent(base_employee_task_state)

        output = result["retrieval_output"]
        assert output["iterations_run"] == 2
        assert len(output["search_queries_used"]) >= 2
        assert output["search_queries_used"][1] != output["search_queries_used"][0]
