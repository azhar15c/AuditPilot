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
    {"id": "chunk-001", "chunk_text": "ROOFING - ALL KINDS & DRIVERS. Code 5551.", "distance": 0.2, "metadata": {"source": "manual"}},
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


@pytest.mark.workstream_b
@pytest.mark.retrieval_convergence
class TestRetrievalAgentContextEngineering:
    """Covers the accumulate/dedupe/cap/compact redesign: results from an
    earlier round must not be lost just because a later round searched
    something else, duplicate chunks across rounds must not double-count,
    the final citation set must be capped by relevance, and later rounds'
    prompts must carry a compact scratchpad rather than replaying every raw
    chunk ever seen."""

    def test_citations_accumulate_across_rounds_not_overwritten(self, base_employee_task_state):
        round_1_chunk = {"id": "chunk-A", "chunk_text": "Roofing excerpt A.", "distance": 0.3, "metadata": {"source": "manual"}}
        round_2_chunk = {"id": "chunk-B", "chunk_text": "Drivers carve-out excerpt B.", "distance": 0.4, "metadata": {"source": "manual"}}
        responses = [
            _evaluate_response(sufficient=False, reasoning="missing drivers clause", refined_query="drivers carve-out"),
            _evaluate_response(sufficient=True, reasoning="now complete", refined_query=""),
        ]
        with patch("agents.nodes.retrieval_agent._retriever") as mock_retriever, \
             patch("agents.nodes.retrieval_agent._client") as mock_client:
            mock_retriever.query.side_effect = [[round_1_chunk], [round_2_chunk]]
            mock_client.chat_with_tools.side_effect = responses

            result = retrieval_agent(base_employee_task_state)

        citation_ids = {c["id"] for c in result["retrieval_output"]["policy_citations"]}
        assert citation_ids == {"chunk-A", "chunk-B"}, (
            "round 1's chunk must survive into the final output even though "
            "round 2 searched for something else — it must not be overwritten"
        )

    def test_duplicate_chunk_across_rounds_is_not_double_counted(self, base_employee_task_state):
        same_chunk = {"id": "chunk-X", "chunk_text": "Roofing excerpt.", "distance": 0.25, "metadata": {"source": "manual"}}
        responses = [
            _evaluate_response(sufficient=False, reasoning="want a second opinion", refined_query="roofing residential"),
            _evaluate_response(sufficient=True, reasoning="confirmed", refined_query=""),
        ]
        with patch("agents.nodes.retrieval_agent._retriever") as mock_retriever, \
             patch("agents.nodes.retrieval_agent._client") as mock_client:
            mock_retriever.query.return_value = [same_chunk]  # identical id both rounds
            mock_client.chat_with_tools.side_effect = responses

            result = retrieval_agent(base_employee_task_state)

        citations = result["retrieval_output"]["policy_citations"]
        assert len(citations) == 1, "the same chunk id found in two rounds must be deduped, not duplicated"

    def test_final_citations_capped_and_sorted_by_relevance(self, base_employee_task_state):
        # 10 unique chunks across two rounds — more than _MAX_CITATIONS_RETURNED (8)
        round_1 = [
            {"id": f"chunk-{i}", "chunk_text": f"Excerpt {i}", "distance": 0.1 * i, "metadata": {}}
            for i in range(5)
        ]
        round_2 = [
            {"id": f"chunk-{i}", "chunk_text": f"Excerpt {i}", "distance": 0.1 * i, "metadata": {}}
            for i in range(5, 10)
        ]
        responses = [
            _evaluate_response(sufficient=False, reasoning="need more", refined_query="broader search"),
            _evaluate_response(sufficient=True, reasoning="enough now", refined_query=""),
        ]
        with patch("agents.nodes.retrieval_agent._retriever") as mock_retriever, \
             patch("agents.nodes.retrieval_agent._client") as mock_client:
            mock_retriever.query.side_effect = [round_1, round_2]
            mock_client.chat_with_tools.side_effect = responses

            result = retrieval_agent(base_employee_task_state)

        citations = result["retrieval_output"]["policy_citations"]
        assert len(citations) == 8, "final citation set must be capped at _MAX_CITATIONS_RETURNED"
        distances = [c["distance"] for c in citations]
        assert distances == sorted(distances), "kept citations must be the closest by distance, in order"
        assert distances[-1] < 0.8, "the cap must keep the closest chunks, not an arbitrary/late-found subset"

    def test_later_round_prompt_uses_compact_scratchpad_not_raw_replay(self, base_employee_task_state):
        """The prompt sent on round 2 must reference round 1's distilled reasoning
        (compaction) but must NOT re-paste round 1's raw chunk_text (that would be
        replaying full history instead of curating it)."""
        round_1_chunk = {"id": "chunk-A", "chunk_text": "MARKER_ROUND_1_RAW_TEXT_UNIQUE", "distance": 0.3, "metadata": {"source": "manual"}}
        round_2_chunk = {"id": "chunk-B", "chunk_text": "Second round excerpt.", "distance": 0.4, "metadata": {"source": "manual"}}

        captured_prompts = []

        def _capture(messages, tools, tool_choice):
            captured_prompts.append(messages[-1]["content"])
            if len(captured_prompts) == 1:
                return _evaluate_response(sufficient=False, reasoning="MARKER_ROUND_1_REASONING_UNIQUE", refined_query="second query")
            return _evaluate_response(sufficient=True, reasoning="done", refined_query="")

        with patch("agents.nodes.retrieval_agent._retriever") as mock_retriever, \
             patch("agents.nodes.retrieval_agent._client") as mock_client:
            mock_retriever.query.side_effect = [[round_1_chunk], [round_2_chunk]]
            mock_client.chat_with_tools.side_effect = _capture

            retrieval_agent(base_employee_task_state)

        assert len(captured_prompts) == 2
        round_2_prompt = captured_prompts[1]
        assert "MARKER_ROUND_1_REASONING_UNIQUE" in round_2_prompt, (
            "round 2's prompt must carry forward round 1's distilled reasoning (compaction)"
        )
        assert "MARKER_ROUND_1_RAW_TEXT_UNIQUE" not in round_2_prompt, (
            "round 2's prompt must NOT replay round 1's raw chunk text — only new chunks "
            "plus the compact scratchpad should be shown each round"
        )
