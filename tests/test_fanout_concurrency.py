"""Concurrency + correctness-parity test for the full fan-out graph.

Guarded on agents.employee_subgraph specifically (not just agents.workflow,
which already exists pre-redesign) — the whole point of this file only
becomes meaningful once Workstream C has wired intake -> extraction_agent ->
supervisor ->(Send fan-out)-> employee_pipeline (retrieval_agent ->
classification_agent -> critic_agent) -> aggregate_node -> report_node.
Until then this file skips cleanly.

Proof of genuine concurrency: each of the 4 employee branches sleeps 0.2s
inside the (mocked) RAG query. If the fan-out actually runs branches in
parallel, total wall time stays well under 4 * 0.2s = 0.8s (serial). If some
future change accidentally serializes the branches again, this test catches
it via the elapsed-time budget.
"""

import json
import tempfile
import time
from unittest.mock import patch

import pytest

pytest.importorskip("agents.employee_subgraph")

from agents.workflow import run_workflow  # noqa: E402

_N_EMPLOYEES = 4
_SLEEP_SECONDS = 0.2


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


def _finalize_response() -> _FakeResponse:
    args = json.dumps({
        "ncci_code": "5551",
        "classification": "ROOFING - ALL KINDS & DRIVERS",
        "rationale": "Residential roofing contractor.",
        "confidence": "HIGH",
    })
    return _FakeResponse(_FakeMessage(tool_calls=[_FakeToolCall("finalize_classification", args)]))


def _critic_response() -> _FakeResponse:
    args = json.dumps({
        "agrees": True,
        "concern": "",
        "recommended_action": "approve",
        "critic_confidence": "HIGH",
    })
    return _FakeResponse(_FakeMessage(tool_calls=[_FakeToolCall("record_critic_review", args)]))


def _make_extraction_response(n: int) -> str:
    return "\n".join(
        f"NAME: Employee {i} | JOB: Roofer | WAGES: $1,{i:03d}" for i in range(n)
    )


@pytest.mark.fanout_concurrency
class TestFanoutConcurrency:
    def test_employee_branches_run_concurrently_and_correctly(self):
        raw_text = (
            "Acme Roofing LLC | Payroll Register Q1 2024\n"
            + "\n".join(f"Employee {i}: Roofer, $1,{i:03d}" for i in range(_N_EMPLOYEES))
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(raw_text)
            tmp_path = f.name

        def _slow_query(*args, **kwargs):
            time.sleep(_SLEEP_SECONDS)
            return []

        with patch("agents.nodes.extraction_agent._client") as mock_extract_client, \
             patch("agents.nodes.supervisor.dispatch_tool") as mock_dispatch, \
             patch("agents.nodes.retrieval_agent._retriever") as mock_retriever, \
             patch("agents.nodes.retrieval_agent._client") as mock_retrieval_client, \
             patch("agents.nodes.classification_agent._client") as mock_classification_client, \
             patch("agents.nodes.critic_agent._client") as mock_critic_client, \
             patch("agents.nodes.report_node._client") as mock_report_client:

            mock_extract_client.extract_entities.return_value = []
            mock_extract_client.generate.return_value = _make_extraction_response(_N_EMPLOYEES)

            mock_dispatch.return_value = {"source": "MOCK", "classifications": []}

            mock_retriever.query.side_effect = _slow_query
            mock_retrieval_client.chat_with_tools.return_value = _evaluate_sufficient_response()

            mock_classification_client.chat_with_tools.return_value = _finalize_response()
            mock_critic_client.chat_with_tools.return_value = _critic_response()

            mock_report_client.generate.return_value = "# Audit Worksheet\n..."

            start = time.monotonic()
            result = run_workflow(tmp_path)
            elapsed = time.monotonic() - start

        # Serial execution would take >= 4 * 0.2s = 0.8s. A genuinely
        # concurrent fan-out should finish well under that.
        assert elapsed < _SLEEP_SECONDS * _N_EMPLOYEES * 0.6, (
            f"elapsed={elapsed:.3f}s suggests employee branches ran serially, not concurrently"
        )

        assert len(result["ncci_suggestions"]) == _N_EMPLOYEES

        # Correctness parity: regardless of completion order, the SET of
        # (employee_id, ncci_code) pairs must be exactly what's expected —
        # list-order equality would be flaky under real concurrency.
        pairs = {(s["employee_id"], s["ncci_code"]) for s in result["ncci_suggestions"]}
        expected_ids = {f"EMP-{i:03d}" for i in range(_N_EMPLOYEES)}
        assert {eid for eid, _ in pairs} == expected_ids
        assert all(code == "5551" for _, code in pairs)
