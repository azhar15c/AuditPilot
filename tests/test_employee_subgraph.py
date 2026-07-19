"""Tests for agents.employee_subgraph.employee_pipeline_node's fault-isolation
boundary.

Confirmed by direct reproduction (not assumed): before this boundary existed,
an unhandled exception from any node inside the per-employee subgraph (e.g. a
Groq call that exhausted its own retries) propagated all the way up through
LangGraph's executor and crashed the entire audit run — including employees
whose branches had already succeeded. employee_pipeline_node now catches that
exception at the subgraph-invocation boundary so one employee's failure
contributes nothing instead of taking everyone else down with it.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("agents.employee_subgraph")

from agents.employee_subgraph import employee_pipeline_node


def _task_state(employee_id: str = "EMP-000", name: str = "Alice") -> dict:
    return {
        "employee_record": {"employee_id": employee_id, "name": name, "job_description": ["Roofer"], "org": "Acme"},
        "policy_number": "WC-TEST-001",
        "prior_classifications": {},
        "retrieval_output": None,
        "classification_output": None,
        "critic_output": None,
        "audit_trail": [],
    }


class TestEmployeePipelineNodeFaultIsolation:
    def test_subgraph_exception_does_not_propagate(self):
        """The whole point of the boundary: a raised exception must be caught
        here, not bubble up to the caller (which would crash the parent
        graph's .invoke() for the entire audit, per the direct reproduction
        that motivated this fix)."""
        with patch("agents.employee_subgraph._employee_subgraph") as mock_subgraph:
            mock_subgraph.invoke.side_effect = RuntimeError("exhausted retries")
            result = employee_pipeline_node(_task_state())  # must not raise

        assert "retrieval_outputs" not in result
        assert "ncci_suggestions" not in result
        assert "critic_assessments" not in result

    def test_failure_is_logged_as_an_error_status_audit_step(self):
        with patch("agents.employee_subgraph._employee_subgraph") as mock_subgraph:
            mock_subgraph.invoke.side_effect = RuntimeError("exhausted retries")
            result = employee_pipeline_node(_task_state(employee_id="EMP-002", name="Cara"))

        assert len(result["audit_trail"]) == 1
        step = result["audit_trail"][0]
        assert step["agent"] == "employee_pipeline"
        assert step["employee_id"] == "EMP-002"
        assert step["status"] == "error"
        assert "exhausted retries" in step["output_summary"], (
            "the real exception message must be visible for debugging, not swallowed"
        )

    def test_successful_run_contributes_normally_and_status_defaults_to_ok(self):
        subgraph_result = {
            "retrieval_output": {"employee_id": "EMP-000", "policy_citations": []},
            "classification_output": {"employee_id": "EMP-000", "ncci_code": "5551"},
            "critic_output": {"employee_id": "EMP-000", "agrees": True},
            "audit_trail": [
                {"agent": "retrieval", "employee_id": "EMP-000", "input_summary": "", "output_summary": "",
                 "timestamp": "t", "duration_ms": 1.0, "status": "ok"},
            ],
        }
        with patch("agents.employee_subgraph._employee_subgraph") as mock_subgraph:
            mock_subgraph.invoke.return_value = subgraph_result
            result = employee_pipeline_node(_task_state())

        assert result["retrieval_outputs"] == [subgraph_result["retrieval_output"]]
        assert result["ncci_suggestions"] == [subgraph_result["classification_output"]]
        assert result["critic_assessments"] == [subgraph_result["critic_output"]]
        assert result["audit_trail"][0]["status"] == "ok"
