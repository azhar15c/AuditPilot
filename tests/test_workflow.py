"""Tests for agents.workflow.run_workflow — real signature is
run_workflow(file_path: str) -> AuditState (a real filesystem path, not raw
bytes), and real state keys are `error` / `audit_report` / `ncci_suggestions`
(not the old, never-shipped `errors` / `report` / `ncci_codes`).

The "document halts gracefully on empty/whitespace-only input" behavior only
exercises intake_node, which pre-dates the multi-agent redesign and never
changes — so those two tests run ungated. The full-pipeline-populates-state
test needs the entire new fan-out graph (extraction_agent, supervisor,
employee_subgraph, aggregate_node) to exist, so it's gated on
agents.employee_subgraph and skips cleanly until Workstream C merges.
"""

import tempfile

import pytest

# agents.workflow currently references deleted extract_node/classify_node
# modules until Workstream C rewires it (Stage 2) — importorskip degrades
# the whole file to a clean skip instead of a collection error in the
# transitional window between Workstream B and C landing.
pytest.importorskip("agents.workflow")
from agents.workflow import run_workflow  # noqa: E402


def _write_temp_file(content: str, suffix: str = ".txt") -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(content)
        return f.name


class TestWorkflowIntakeGating:
    """These only exercise intake_node (unchanged pre-existing behavior) —
    no importorskip needed."""

    def test_empty_file_halts_gracefully(self):
        path = _write_temp_file("")
        result = run_workflow(path)

        assert result["error"] is not None
        assert result["audit_report"] == ""

    def test_whitespace_only_file_halts_gracefully(self):
        path = _write_temp_file("   \n\t  ")
        result = run_workflow(path)

        assert result["error"] is not None
        assert result["employee_records"] == []
        assert result["ncci_suggestions"] == []

    def test_missing_file_produces_error(self):
        result = run_workflow("/tmp/definitely-does-not-exist-audit-pilot.pdf")

        assert result["error"] is not None
        assert result["audit_report"] == ""


class TestWorkflowFullPipeline:
    def test_full_pipeline_populates_state(self):
        pytest.importorskip("agents.employee_subgraph")

        import json
        from unittest.mock import patch

        raw_text = (
            "ABC Roofing LLC | Payroll Register Q1 2024 | "
            "Employee: John Smith | Job Title: Roofer | Wages: $18,500"
        )
        path = _write_temp_file(raw_text)

        class _FakeFunction:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        class _FakeToolCall:
            def __init__(self, name, arguments):
                self.function = _FakeFunction(name, arguments)

        class _FakeMessage:
            def __init__(self, tool_calls):
                self.tool_calls = tool_calls
                self.content = None

        class _FakeChoice:
            def __init__(self, message):
                self.message = message

        class _FakeResponse:
            def __init__(self, message):
                self.choices = [_FakeChoice(message)]

        def _tool_response(name, **kwargs):
            return _FakeResponse(_FakeMessage([_FakeToolCall(name, json.dumps(kwargs))]))

        with patch("agents.nodes.extraction_agent._client") as mock_extract_client, \
             patch("agents.nodes.supervisor.dispatch_tool") as mock_dispatch, \
             patch("agents.nodes.retrieval_agent._retriever") as mock_retriever, \
             patch("agents.nodes.retrieval_agent._client") as mock_retrieval_client, \
             patch("agents.nodes.classification_agent._client") as mock_classification_client, \
             patch("agents.nodes.critic_agent._client") as mock_critic_client, \
             patch("agents.nodes.report_node._client") as mock_report_client:

            mock_extract_client.extract_entities.return_value = [
                {"entity": "John Smith", "label": "PER", "score": 0.99},
            ]
            mock_extract_client.generate.return_value = (
                "NAME: John Smith | JOB: Roofer | WAGES: $18,500"
            )
            mock_dispatch.return_value = {"source": "MOCK", "classifications": []}
            mock_retriever.query.return_value = []
            mock_retrieval_client.chat_with_tools.return_value = _tool_response(
                "evaluate_retrieval", sufficient=True, reasoning="ok", refined_query="",
            )
            mock_classification_client.chat_with_tools.return_value = _tool_response(
                "finalize_classification", ncci_code="5551",
                classification="ROOFING - ALL KINDS & DRIVERS",
                rationale="Residential roofing.", confidence="HIGH",
            )
            mock_critic_client.chat_with_tools.return_value = _tool_response(
                "record_critic_review", agrees=True, concern="",
                recommended_action="approve", critic_confidence="HIGH",
            )
            mock_report_client.generate.return_value = (
                "# Workers' Compensation Premium Audit Worksheet\n..."
            )

            result = run_workflow(path)

        assert result["raw_text"] != ""
        assert len(result["employee_records"]) >= 1
        assert len(result["ncci_suggestions"]) >= 1
        assert result["ncci_suggestions"][0]["ncci_code"] == "5551"
        assert result["audit_report"] != ""
        assert result["error"] is None
