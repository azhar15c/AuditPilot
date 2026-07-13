"""Tests for agents.nodes.extraction_agent — Workstream B.

extraction_agent replaces the old extract_node with the same overall shape
(BERT NER via _client.extract_entities, LLM structured extraction via
_client.generate) but every returned employee record must also carry a
unique, zero-padded "employee_id" (e.g. "EMP-000", "EMP-001", ...) so the
Supervisor can fan out one Send per employee downstream.

Guarded with importorskip since agents/nodes/extraction_agent.py does not
exist yet in this worktree — Workstream B builds it in a separate worktree
and merges later. This file should skip cleanly until then.
"""

import re
from unittest.mock import patch

import pytest

pytest.importorskip("agents.nodes.extraction_agent")

from agents.nodes.extraction_agent import extraction_agent  # noqa: E402

BASE_STATE = {
    "uploaded_file_path": "/tmp/test.pdf",
    "raw_text": (
        "ABC Roofing LLC | Payroll Register Q1 2024 | "
        "Employee: John Smith | Job Title: Roofer | Wages: $18,500"
    ),
    "entities": [],
    "employee_records": [],
    "policy_number": None,
    "prior_classifications": {},
    "retrieval_outputs": [],
    "ncci_suggestions": [],
    "critic_assessments": [],
    "audit_trail": [],
    "completeness_flags": {"has_payroll_data": True, "has_employee_names": True, "has_dates": True},
    "audit_report": "",
    "error": None,
    "current_step": "intake",
}

SAMPLE_NER = [
    {"entity": "John Smith", "label": "PER", "score": 0.99},
    {"entity": "ABC Roofing LLC", "label": "ORG", "score": 0.97},
    {"entity": "Roofer", "label": "MISC", "score": 0.85},
]

_LLM_MULTI_EMPLOYEE_RESPONSE = (
    "NAME: Alice Brown | JOB: Roofer | WAGES: $22,000\n"
    "NAME: Bob Lee | JOB: Office Manager | WAGES: $15,000\n"
    "NAME: Carla Diaz | JOB: Electrician | WAGES: $30,000\n"
)

_EMPLOYEE_ID_RE = re.compile(r"^EMP-\d+$")


class TestExtractionAgent:
    def test_entities_stored_in_state(self):
        with patch("agents.nodes.extraction_agent._client") as mock_client:
            mock_client.extract_entities.return_value = SAMPLE_NER
            mock_client.generate.return_value = (
                "NAME: John Smith | JOB: Roofer | WAGES: $18,500\n"
            )
            result = extraction_agent(BASE_STATE)

        assert len(result["entities"]) == 3
        labels = {e["label"] for e in result["entities"]}
        assert {"PER", "ORG", "MISC"} == labels

    def test_every_record_gets_unique_zero_padded_employee_id(self):
        with patch("agents.nodes.extraction_agent._client") as mock_client:
            mock_client.extract_entities.return_value = SAMPLE_NER
            mock_client.generate.return_value = _LLM_MULTI_EMPLOYEE_RESPONSE
            result = extraction_agent(BASE_STATE)

        records = result["employee_records"]
        assert len(records) == 3

        ids = [r["employee_id"] for r in records]
        assert len(set(ids)) == len(ids), "employee_id values must be unique"
        for eid in ids:
            assert _EMPLOYEE_ID_RE.match(eid), f"expected zero-padded EMP-### id, got {eid!r}"

    def test_employee_id_stable_order_matches_record_order(self):
        with patch("agents.nodes.extraction_agent._client") as mock_client:
            mock_client.extract_entities.return_value = SAMPLE_NER
            mock_client.generate.return_value = _LLM_MULTI_EMPLOYEE_RESPONSE
            result = extraction_agent(BASE_STATE)

        records = result["employee_records"]
        names = [r["name"] for r in records]
        assert names == ["Alice Brown", "Bob Lee", "Carla Diaz"]
        # ids should be increasing / positionally assigned, not shuffled
        ids = [r["employee_id"] for r in records]
        assert ids == sorted(ids)

    def test_passes_through_on_existing_error(self):
        state = {**BASE_STATE, "error": "upstream failure"}
        with patch("agents.nodes.extraction_agent._client") as mock_client:
            result = extraction_agent(state)

        mock_client.extract_entities.assert_not_called()
        assert result["error"] == "upstream failure"
        assert result["employee_records"] == []

    def test_no_records_extracted_produces_empty_list(self):
        with patch("agents.nodes.extraction_agent._client") as mock_client:
            mock_client.extract_entities.return_value = []
            mock_client.generate.return_value = ""
            result = extraction_agent(BASE_STATE)

        assert result["employee_records"] == []
