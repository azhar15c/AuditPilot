"""Tests for agents.nodes.classification_agent — Workstream B.

classification_agent takes an EmployeeTaskState (with retrieval_output already
populated by the retrieval stage) and forces a "finalize_classification" tool
call via _client.chat_with_tools(...), same pattern as today's classify_node.

Guarded with importorskip since agents/nodes/classification_agent.py does not
exist yet in this worktree — Workstream B builds it in a separate worktree
and merges later.
"""

import json

import pytest

pytest.importorskip("agents.nodes.classification_agent")

from unittest.mock import patch  # noqa: E402

from agents.nodes.classification_agent import classification_agent  # noqa: E402


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


def _finalize_response(ncci_code="5551", classification="ROOFING - ALL KINDS & DRIVERS",
                        rationale="Residential roofing contractor.", confidence="HIGH") -> _FakeResponse:
    args = json.dumps({
        "ncci_code": ncci_code,
        "classification": classification,
        "rationale": rationale,
        "confidence": confidence,
    })
    tc = _FakeToolCall("finalize_classification", args)
    return _FakeResponse(_FakeMessage(tool_calls=[tc]))


def _with_retrieval_output(state: dict) -> dict:
    return {
        **state,
        "retrieval_output": {
            "employee_id": state["employee_record"]["employee_id"],
            "policy_citations": [
                {"chunk_text": "ROOFING - ALL KINDS & DRIVERS. Code 5551.", "distance": 0.2, "metadata": {}},
            ],
            "policycenter_fields": {},
            "search_queries_used": ["roofer"],
            "iterations_run": 1,
            "sufficiency_met": True,
            "sufficiency_reasoning": "Excerpt explicitly names roofer.",
        },
    }


@pytest.mark.workstream_b
class TestClassificationAgent:
    def test_classification_output_shape(self, base_employee_task_state):
        state = _with_retrieval_output(base_employee_task_state)
        with patch("agents.nodes.classification_agent._client") as mock_client:
            mock_client.chat_with_tools.return_value = _finalize_response()
            result = classification_agent(state)

        output = result["classification_output"]
        assert output["employee_id"] == base_employee_task_state["employee_record"]["employee_id"]
        assert output["ncci_code"] == "5551"
        assert output["classification"] == "ROOFING - ALL KINDS & DRIVERS"
        assert output["confidence"] == "HIGH"
        assert output["rationale"] != ""
        assert output["employee"] == base_employee_task_state["employee_record"]["name"]

    def test_audit_trail_entry_appended(self, base_employee_task_state):
        state = _with_retrieval_output(base_employee_task_state)
        with patch("agents.nodes.classification_agent._client") as mock_client:
            mock_client.chat_with_tools.return_value = _finalize_response()
            result = classification_agent(state)

        assert len(result["audit_trail"]) >= 1
        assert any(step.get("agent") == "classification" for step in result["audit_trail"])

    def test_low_confidence_when_llm_flags_low(self, base_employee_task_state):
        state = _with_retrieval_output(base_employee_task_state)
        with patch("agents.nodes.classification_agent._client") as mock_client:
            mock_client.chat_with_tools.return_value = _finalize_response(
                ncci_code="8810", classification="CLERICAL OFFICE EMPLOYEES NOC",
                rationale="No exact match found.", confidence="LOW",
            )
            result = classification_agent(state)

        assert result["classification_output"]["confidence"] == "LOW"
