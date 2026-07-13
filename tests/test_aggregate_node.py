"""Tests for agents.nodes.aggregate_node — graceful degradation when one
employee's branch fails to produce a classification (e.g. a Groq API error
mid-run), and correctness of the delta-only reducer-field returns."""

import pytest

pytest.importorskip("agents.nodes.aggregate_node")

from agents.nodes.aggregate_node import aggregate_node


def _record(employee_id: str, name: str) -> dict:
    return {"employee_id": employee_id, "name": name, "job_description": [], "org": None, "wages": None}


def _suggestion(employee_id: str) -> dict:
    return {
        "employee_id": employee_id, "employee": "x", "ncci_code": "5551",
        "classification": "ROOFING", "rationale": "r", "confidence": "HIGH",
    }


class TestAggregateNodeDegradesGracefully:
    def test_all_employees_classified_no_changes_needed(self):
        state = {
            "error": None,
            "employee_records": [_record("EMP-000", "Alice"), _record("EMP-001", "Bob")],
            "ncci_suggestions": [_suggestion("EMP-000"), _suggestion("EMP-001")],
        }
        result = aggregate_node(state)

        assert "ncci_suggestions" not in result, "should not contribute a delta when nothing is missing"
        assert "audit_trail" not in result
        assert result["current_step"] == "aggregate"

    def test_one_missing_employee_gets_placeholder_and_audit_run_continues(self):
        state = {
            "error": None,
            "employee_records": [_record("EMP-000", "Alice"), _record("EMP-001", "Bob")],
            "ncci_suggestions": [_suggestion("EMP-000")],  # Bob's branch failed
        }
        result = aggregate_node(state)

        assert len(result["ncci_suggestions"]) == 1, "delta should contain only the missing employee"
        placeholder = result["ncci_suggestions"][0]
        assert placeholder["employee_id"] == "EMP-001"
        assert placeholder["classification"] == "NOT CLASSIFIED — retry required"
        assert placeholder["ncci_code"] is None

        assert len(result["audit_trail"]) == 1
        assert result["audit_trail"][0]["agent"] == "aggregate"
        assert result["audit_trail"][0]["employee_id"] == "EMP-001"

    def test_never_fails_the_whole_run(self):
        state = {
            "error": None,
            "employee_records": [_record("EMP-000", "Alice")],
            "ncci_suggestions": [],  # every branch failed
        }
        result = aggregate_node(state)

        assert "error" not in result or result.get("error") is None
        assert len(result["ncci_suggestions"]) == 1

    def test_existing_error_short_circuits_without_touching_reducer_fields(self):
        state = {"error": "upstream failure", "employee_records": [], "ncci_suggestions": []}
        result = aggregate_node(state)

        assert result == {}
