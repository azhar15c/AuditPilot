from unittest.mock import patch, MagicMock

from agents.workflow import run_workflow

SAMPLE_TEXT = b"ABC Roofing LLC | Payroll Register Q1 2024 | Employee: John Smith | Job Title: Roofer | Wages: $18,500"


class TestWorkflow:
    def test_full_pipeline_populates_state(self):
        mock_ner = [{"word": "John Smith", "entity_group": "PER", "score": 0.99}]
        mock_classify_response = "5183"
        mock_report_response = "AUDIT WORKSHEET\nPolicyholder: ABC Roofing LLC\nEmployee: John Smith (Roofer)\nCode: 5183\nAllocated Payroll: $18,500\nFindings: Classification consistent with submitted payroll register."

        with patch("agents.nodes.extract_node._client") as mock_extract_client, \
             patch("agents.nodes.classify_node._client") as mock_classify_client, \
             patch("agents.nodes.classify_node._retriever") as mock_retriever, \
             patch("agents.nodes.report_node._client") as mock_report_client:

            mock_extract_client.ner.return_value = mock_ner
            mock_retriever.query.return_value = "CLASS CODE: 5183 - Plumbing"
            mock_classify_client.classify.return_value = mock_classify_response
            mock_report_client.generate_report.return_value = mock_report_response

            result = run_workflow(SAMPLE_TEXT)

        assert result["document_text"] != ""
        assert len(result["entities"]) > 0
        assert "5183" in result["ncci_codes"]
        assert result["report"] != ""
        assert result["errors"] == []

    def test_empty_file_halts_gracefully(self):
        result = run_workflow(b"")
        assert result["errors"] != []
        assert result["report"] == ""

    def test_error_propagates_through_pipeline(self):
        result = run_workflow(b"   ")
        assert result["errors"] != []
        assert result["entities"] == []
        assert result["ncci_codes"] == []
