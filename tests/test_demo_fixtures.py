"""Tests for demo_fixtures.py — the demo-safe mode used when live Groq
availability can't be relied on (see README's "Demo-Safe Mode" section).
Confirms the canned responses drive the real graph to a complete, correct,
zero-error result for the actual sample document, with no live API calls."""

import pytest

pytest.importorskip("demo_fixtures")

import demo_fixtures
from agents.workflow import run_workflow

SAMPLE_PATH = "data/sample_payroll_register.pdf"


class TestDemoModePatches:
    def test_full_pipeline_completes_with_zero_errors(self):
        with demo_fixtures.demo_mode_patches():
            state = run_workflow(SAMPLE_PATH)

        assert state["error"] is None
        assert len(state["employee_records"]) == 3
        assert len(state["ncci_suggestions"]) == 3
        assert len(state["critic_assessments"]) == 3
        error_steps = [s for s in state["audit_trail"] if s["status"] == "error"]
        assert error_steps == [], f"demo mode must never produce a failed branch: {error_steps}"

    def test_each_employee_gets_the_expected_classification(self):
        with demo_fixtures.demo_mode_patches():
            state = run_workflow(SAMPLE_PATH)

        by_name = {s["employee"]: s for s in state["ncci_suggestions"]}
        assert by_name["Michael A. Torres"]["ncci_code"] == "5551"
        assert by_name["Sarah L. Chen"]["ncci_code"] == "8810"
        assert by_name["James K. Wright"]["ncci_code"] == "5190"
        for s in state["ncci_suggestions"]:
            assert s["confidence"] == "HIGH"

    def test_critic_approves_all_three(self):
        with demo_fixtures.demo_mode_patches():
            state = run_workflow(SAMPLE_PATH)

        assert all(c["agrees"] is True for c in state["critic_assessments"])
        assert all(c["recommended_action"] == "approve" for c in state["critic_assessments"])

    def test_report_is_generated(self):
        with demo_fixtures.demo_mode_patches():
            state = run_workflow(SAMPLE_PATH)

        assert state["audit_report"]
        assert "SUMMIT BUILDERS" in state["audit_report"]

    def test_patches_are_removed_after_the_context_exits(self):
        """The patches must not leak into subsequent calls outside the with block."""
        import agents.nodes.extraction_agent as extraction_module

        original_client = extraction_module._client
        with demo_fixtures.demo_mode_patches():
            assert extraction_module._client is not original_client
        assert extraction_module._client is original_client
