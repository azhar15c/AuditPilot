"""Least-privilege enforcement tests — Workstream B/C boundary.

retrieval_agent may reach the NCCI RAG store but must NEVER call PolicyCenter
directly (that data is fetched once, up front, by the Supervisor and handed
down via EmployeeTaskState.prior_classifications).

classification_agent must NEVER reach RAG or PolicyCenter directly — it only
reasons over the RetrievalOutput already assembled by the retrieval stage.

Each check is two-layered: structural (the forbidden symbol isn't even
imported into the module's namespace) and behavioral (patch the real
dispatch points and assert they're never invoked when the agent runs).

Each target module is importorskip'd individually so a partial merge (e.g.
retrieval_agent landed, classification_agent hasn't) doesn't block the whole
file.
"""

import json
from unittest.mock import patch

import pytest


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


def _evaluate_sufficient_response() -> _FakeResponse:
    args = json.dumps({"sufficient": True, "reasoning": "ok", "refined_query": ""})
    return _FakeResponse(_FakeMessage(tool_calls=[_FakeToolCall("evaluate_retrieval", args)]))


def _finalize_classification_response() -> _FakeResponse:
    args = json.dumps({
        "ncci_code": "5551",
        "classification": "ROOFING - ALL KINDS & DRIVERS",
        "rationale": "Residential roofing.",
        "confidence": "HIGH",
    })
    return _FakeResponse(_FakeMessage(tool_calls=[_FakeToolCall("finalize_classification", args)]))


@pytest.mark.tool_boundary
class TestRetrievalAgentToolBoundary:
    def test_retrieval_agent_cannot_reach_policycenter(self, base_employee_task_state):
        pytest.importorskip("agents.nodes.retrieval_agent")
        import agents.nodes.retrieval_agent as ra

        assert "dispatch_tool" not in dir(ra), "retrieval_agent must not import PolicyCenter's dispatch_tool"

        with patch("mcp_servers.policycenter.dispatch_tool") as mock_dispatch, \
             patch.object(ra, "_retriever") as mock_retriever, \
             patch.object(ra, "_client") as mock_client:
            mock_retriever.query.return_value = []
            mock_client.chat_with_tools.return_value = _evaluate_sufficient_response()

            ra.retrieval_agent(base_employee_task_state)

        mock_dispatch.assert_not_called()


@pytest.mark.tool_boundary
class TestClassificationAgentToolBoundary:
    def test_classification_agent_cannot_reach_rag_or_policycenter(self, base_employee_task_state):
        pytest.importorskip("agents.nodes.classification_agent")
        import agents.nodes.classification_agent as ca

        assert "dispatch_tool" not in dir(ca), "classification_agent must not import PolicyCenter's dispatch_tool"
        assert not hasattr(ca, "NCCIRetriever"), "classification_agent must not import NCCIRetriever"

        state = {
            **base_employee_task_state,
            "retrieval_output": {
                "employee_id": base_employee_task_state["employee_record"]["employee_id"],
                "policy_citations": [],
                "policycenter_fields": {},
                "search_queries_used": [],
                "iterations_run": 1,
                "sufficiency_met": True,
                "sufficiency_reasoning": "ok",
            },
        }

        with patch("mcp_servers.policycenter.dispatch_tool") as mock_dispatch, \
             patch("rag.retriever.NCCIRetriever.query") as mock_query, \
             patch.object(ca, "_client") as mock_client:
            mock_client.chat_with_tools.return_value = _finalize_classification_response()

            ca.classification_agent(state)

        mock_dispatch.assert_not_called()
        mock_query.assert_not_called()
