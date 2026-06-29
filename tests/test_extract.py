from unittest.mock import patch
import pytest

from agents.nodes.extract_node import extract_node

BASE_STATE = {
    "uploaded_file_path": "/tmp/test.pdf",
    "raw_text": "ABC Roofing LLC | Payroll Register Q1 2024 | Employee: John Smith | Job Title: Roofer | Wages: $18,500",
    "entities": [],
    "employee_records": [],
    "ncci_suggestions": [],
    "completeness_flags": {"has_payroll_data": True, "has_employee_names": True, "has_dates": True},
    "audit_report": "",
    "error": None,
    "current_step": "intake",
}

SAMPLE_NER = [
    {"entity": "John Smith",       "label": "PER",   "score": 0.99},
    {"entity": "ABC Roofing LLC",  "label": "ORG",   "score": 0.97},
    {"entity": "$18,500",          "label": "MONEY", "score": 0.91},
    {"entity": "Roofer",           "label": "MISC",  "score": 0.85},
    {"entity": "Q1 2024",          "label": "DATE",  "score": 0.88},
]


class TestExtractNode:

    def test_entities_stored_in_state(self):
        with patch("agents.nodes.extract_node._client") as mock_client:
            mock_client.extract_entities.return_value = SAMPLE_NER
            result = extract_node(BASE_STATE)

        assert len(result["entities"]) == 5
        labels = {e["label"] for e in result["entities"]}
        assert {"PER", "ORG", "MONEY", "MISC", "DATE"} == labels

    def test_per_and_money_grouped_into_employee_record(self):
        with patch("agents.nodes.extract_node._client") as mock_client:
            mock_client.extract_entities.return_value = SAMPLE_NER
            result = extract_node(BASE_STATE)

        assert len(result["employee_records"]) == 1
        record = result["employee_records"][0]
        assert record["name"] == "John Smith"
        assert record["wages"] == "$18,500"
        assert record["org"] == "ABC Roofing LLC"
        assert "Roofer" in record["job_description"]
        assert "Q1 2024" in record["dates"]

    def test_no_money_entities_wages_is_none(self):
        ner_no_money = [
            {"entity": "Jane Doe",         "label": "PER", "score": 0.98},
            {"entity": "Best Electric LLC", "label": "ORG", "score": 0.94},
        ]
        with patch("agents.nodes.extract_node._client") as mock_client:
            mock_client.extract_entities.return_value = ner_no_money
            result = extract_node(BASE_STATE)

        assert len(result["employee_records"]) == 1
        assert result["employee_records"][0]["wages"] is None

    def test_multiple_employees_matched_positionally_to_wages(self):
        ner_multi = [
            {"entity": "Alice Brown", "label": "PER",   "score": 0.99},
            {"entity": "Bob Lee",     "label": "PER",   "score": 0.97},
            {"entity": "$22,000",     "label": "MONEY", "score": 0.92},
            {"entity": "$15,000",     "label": "MONEY", "score": 0.90},
        ]
        with patch("agents.nodes.extract_node._client") as mock_client:
            mock_client.extract_entities.return_value = ner_multi
            result = extract_node(BASE_STATE)

        records = result["employee_records"]
        assert records[0]["name"] == "Alice Brown"
        assert records[0]["wages"] == "$22,000"
        assert records[1]["name"] == "Bob Lee"
        assert records[1]["wages"] == "$15,000"

    def test_more_employees_than_wages_remaining_get_none(self):
        ner_unequal = [
            {"entity": "Alice Brown", "label": "PER",   "score": 0.99},
            {"entity": "Bob Lee",     "label": "PER",   "score": 0.97},
            {"entity": "$22,000",     "label": "MONEY", "score": 0.92},
            # only one wage for two employees
        ]
        with patch("agents.nodes.extract_node._client") as mock_client:
            mock_client.extract_entities.return_value = ner_unequal
            result = extract_node(BASE_STATE)

        assert result["employee_records"][0]["wages"] == "$22,000"
        assert result["employee_records"][1]["wages"] is None

    def test_no_per_entities_produces_empty_records(self):
        ner_no_per = [
            {"entity": "ABC Roofing LLC", "label": "ORG",   "score": 0.97},
            {"entity": "$18,500",         "label": "MONEY", "score": 0.91},
        ]
        with patch("agents.nodes.extract_node._client") as mock_client:
            mock_client.extract_entities.return_value = ner_no_per
            result = extract_node(BASE_STATE)

        assert result["employee_records"] == []

    def test_passes_through_on_existing_error(self):
        state = {**BASE_STATE, "error": "upstream failure"}
        with patch("agents.nodes.extract_node._client") as mock_client:
            result = extract_node(state)

        mock_client.extract_entities.assert_not_called()
        assert result["error"] == "upstream failure"
        assert result["employee_records"] == []
