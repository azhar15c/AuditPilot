"""Tests for agents.nodes.supervisor — Workstream A.

The Supervisor fetches PolicyCenter prior-classification data ONCE per audit
run (not once per employee — that was the bug driving this redesign) and then
fans out one Send per employee via route_to_employee_fanout, targeting the
"employee_pipeline" subgraph node with an EmployeeTaskState-shaped .arg.

Guarded with importorskip since agents/nodes/supervisor.py does not exist yet
in this worktree — Workstream A builds it in a separate worktree and merges
later.
"""

import pytest

pytest.importorskip("agents.nodes.supervisor")

from unittest.mock import patch  # noqa: E402

from agents.nodes.supervisor import route_to_employee_fanout, supervisor_node  # noqa: E402

_EMPLOYEE_TASK_STATE_KEYS = {
    "employee_record",
    "policy_number",
    "prior_classifications",
    "retrieval_output",
    "classification_output",
    "critic_output",
    "audit_trail",
}


def _record(i: int) -> dict:
    return {
        "name": f"Employee {i}",
        "employee_id": f"EMP-{i:03d}",
        "job_description": ["roofer"],
        "org": "Acme Roofing",
        "wages": "$1,000",
        "dates": [],
    }


def _state(n: int) -> dict:
    return {
        "uploaded_file_path": "/tmp/test.pdf",
        "raw_text": "text",
        "entities": [],
        "employee_records": [_record(i) for i in range(n)],
        "policy_number": "WC-TEST-001",
        "prior_classifications": {},
        "retrieval_outputs": [],
        "ncci_suggestions": [],
        "critic_assessments": [],
        "audit_trail": [],
        "completeness_flags": {},
        "audit_report": "",
        "error": None,
        "current_step": "extraction",
    }


_MOCK_PRIOR = {
    "source": "MOCK",
    "policy_number": "WC-TEST-001",
    "classifications": [],
    "total_audited_payroll": 0.0,
}


@pytest.mark.workstream_a
class TestSupervisorDispatchesOnce:
    def test_single_employee_calls_dispatch_tool_once(self):
        with patch("agents.nodes.supervisor.dispatch_tool") as mock_dispatch:
            mock_dispatch.return_value = _MOCK_PRIOR
            supervisor_node(_state(1))

        mock_dispatch.assert_called_once()

    def test_five_employees_still_calls_dispatch_tool_once(self):
        with patch("agents.nodes.supervisor.dispatch_tool") as mock_dispatch:
            mock_dispatch.return_value = _MOCK_PRIOR
            supervisor_node(_state(5))

        mock_dispatch.assert_called_once()


@pytest.mark.workstream_a
class TestRouteToEmployeeFanout:
    def test_one_send_per_employee_targeting_employee_pipeline(self):
        state = _state(3)
        state["prior_classifications"] = _MOCK_PRIOR

        sends = route_to_employee_fanout(state)

        assert len(sends) == 3
        for send in sends:
            assert send.node == "employee_pipeline"

    def test_send_arg_is_employee_task_state_shaped(self):
        state = _state(2)
        state["prior_classifications"] = _MOCK_PRIOR

        sends = route_to_employee_fanout(state)

        for send in sends:
            assert set(send.arg.keys()) == _EMPLOYEE_TASK_STATE_KEYS
            assert send.arg["policy_number"] == "WC-TEST-001"
            assert send.arg["retrieval_output"] is None
            assert send.arg["classification_output"] is None
            assert send.arg["critic_output"] is None
            assert send.arg["audit_trail"] == []

    def test_each_send_carries_its_own_employee_record(self):
        state = _state(2)
        state["prior_classifications"] = _MOCK_PRIOR

        sends = route_to_employee_fanout(state)
        employee_ids = {send.arg["employee_record"]["employee_id"] for send in sends}

        assert employee_ids == {"EMP-000", "EMP-001"}
