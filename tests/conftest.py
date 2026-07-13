"""Shared fixtures for the multi-agent AuditPilot test suite.

Markers used across this suite are registered in pytest.ini (repo root):
workstream_a, workstream_b, workstream_c, workstream_d, fanout_concurrency,
tool_boundary, critic_disagreement, retrieval_convergence.
"""

import pytest


def _base_employee_record() -> dict:
    return {
        "name": "Alice Smith",
        "employee_id": "EMP-000",
        "job_description": ["roofer"],
        "org": "Acme Roofing",
        "wages": "$1,200",
        "dates": [],
    }


@pytest.fixture
def base_employee_record() -> dict:
    """A single, well-formed employee record shaped like extraction_agent output."""
    return _base_employee_record()


def _base_employee_task_state(record: dict | None = None) -> dict:
    record = record if record is not None else _base_employee_record()
    return {
        "employee_record": record,
        "policy_number": "WC-TEST-001",
        "prior_classifications": {},
        "retrieval_output": None,
        "classification_output": None,
        "critic_output": None,
        "audit_trail": [],
    }


@pytest.fixture
def base_employee_task_state() -> dict:
    """An EmployeeTaskState-shaped dict built from base_employee_record(), ready to
    feed into retrieval_agent / classification_agent / critic_agent."""
    return _base_employee_task_state()
