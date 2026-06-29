from unittest.mock import patch
import pytest

from agents.nodes.classify_node import classify_node

BASE_STATE = {
    "uploaded_file_path": "/tmp/test.pdf",
    "raw_text": "John Smith is a roofer at ABC Roofing LLC. Q1 2024 wages: $18,500.",
    "entities": [
        {"entity": "John Smith",      "label": "PER",   "score": 0.99},
        {"entity": "ABC Roofing LLC", "label": "ORG",   "score": 0.97},
        {"entity": "$18,500",         "label": "MONEY", "score": 0.91},
    ],
    "employee_records": [
        {
            "name": "John Smith",
            "wages": "$18,500",
            "org": "ABC Roofing LLC",
            "dates": ["Q1 2024"],
            "job_description": ["Roofer"],
        }
    ],
    "ncci_suggestions": [],
    "completeness_flags": {"has_payroll_data": True, "has_employee_names": True, "has_dates": True},
    "audit_report": "",
    "error": None,
    "current_step": "extract",
}

_VALID_RESPONSE = (
    "NCCI CODE: 5551\n"
    "RATIONALE: Residential roofing contractor performing exterior roof installation.\n"
    "CONFIDENCE: HIGH"
)


class TestClassifyNode:

    def test_suggestion_has_all_required_fields(self):
        with patch("agents.nodes.classify_node._client") as mock_client, \
             patch("agents.nodes.classify_node._retriever") as mock_retriever:
            mock_retriever.query.return_value = []
            mock_retriever.format_context.return_value = "CLASS CODE: 5551 - Roofing"
            mock_client.generate.return_value = _VALID_RESPONSE

            result = classify_node(BASE_STATE)

        assert len(result["ncci_suggestions"]) == 1
        s = result["ncci_suggestions"][0]
        assert s["employee"] == "John Smith"
        assert s["ncci_code"] == "5551"
        assert s["confidence"] == "HIGH"
        assert "roofing" in s["rationale"].lower()

    def test_each_employee_gets_own_llm_call(self):
        state = {
            **BASE_STATE,
            "employee_records": [
                {"name": "John Smith", "wages": "$18,500", "org": "ABC Roofing", "dates": [], "job_description": ["Roofer"]},
                {"name": "Jane Doe",   "wages": "$32,000", "org": "ABC Roofing", "dates": [], "job_description": ["Office Manager"]},
            ],
        }
        with patch("agents.nodes.classify_node._client") as mock_client, \
             patch("agents.nodes.classify_node._retriever") as mock_retriever:
            mock_retriever.query.return_value = []
            mock_retriever.format_context.return_value = ""
            mock_client.generate.return_value = _VALID_RESPONSE

            result = classify_node(state)

        assert len(result["ncci_suggestions"]) == 2
        assert mock_client.generate.call_count == 2

    def test_retriever_queried_per_employee(self):
        with patch("agents.nodes.classify_node._client") as mock_client, \
             patch("agents.nodes.classify_node._retriever") as mock_retriever:
            mock_retriever.query.return_value = []
            mock_retriever.format_context.return_value = ""
            mock_client.generate.return_value = _VALID_RESPONSE

            classify_node(BASE_STATE)

        mock_retriever.query.assert_called_once()
        mock_retriever.format_context.assert_called_once()

    def test_no_employee_records_sets_error(self):
        state = {**BASE_STATE, "employee_records": []}
        with patch("agents.nodes.classify_node._client") as mock_client, \
             patch("agents.nodes.classify_node._retriever"):
            result = classify_node(state)

        mock_client.generate.assert_not_called()
        assert result["error"] is not None
        assert result["ncci_suggestions"] == []

    def test_passes_through_on_existing_error(self):
        state = {**BASE_STATE, "error": "upstream failure"}
        with patch("agents.nodes.classify_node._client") as mock_client, \
             patch("agents.nodes.classify_node._retriever") as mock_retriever:
            result = classify_node(state)

        mock_client.generate.assert_not_called()
        mock_retriever.query.assert_not_called()
        assert result["error"] == "upstream failure"

    def test_malformed_llm_response_handled_gracefully(self):
        with patch("agents.nodes.classify_node._client") as mock_client, \
             patch("agents.nodes.classify_node._retriever") as mock_retriever:
            mock_retriever.query.return_value = []
            mock_retriever.format_context.return_value = ""
            mock_client.generate.return_value = "I'm unable to determine the applicable code."

            result = classify_node(BASE_STATE)

        s = result["ncci_suggestions"][0]
        assert s["ncci_code"] is None
        assert s["confidence"] == "LOW"
        assert s["employee"] == "John Smith"
