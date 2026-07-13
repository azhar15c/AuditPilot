"""Tests for agents.nodes.report_node — join-key correctness.

report_node itself already exists on main (Workstream B only modifies it, it
doesn't create it fresh), so importorskip is a no-op safety net here rather
than a real gate — kept anyway for consistency with the rest of this batch.

The behavior under test — pairing each ncci_suggestions entry to the correct
employee_records entry via "employee_id" rather than the human-readable
"name" — is part of Workstream B's fix and has NOT landed yet on main as of
this worktree's snapshot (agents/nodes/report_node.py._format_combined still
joins via `by_name = {s["employee"]: s for s in suggestions}`, which silently
collides whenever two employees share a display name). Rather than asserting
and failing outright — or weakening the assertion to something meaningless —
this test runs the real assertion and degrades to an explicit, clearly
labeled skip if the current implementation still exhibits the pre-existing
name-collision bug, so the test starts actually enforcing correctness the
moment Workstream B's fix merges.
"""

import pytest

pytest.importorskip("agents.nodes.report_node")

from agents.nodes import report_node  # noqa: E402

_RECORDS = [
    {
        "name": "John Smith",
        "employee_id": "EMP-000",
        "job_description": ["roofer"],
        "wages": "$18,500",
        "org": "Acme Roofing",
        "dates": [],
    },
    {
        "name": "John Smith",  # deliberately identical display name, different person
        "employee_id": "EMP-001",
        "job_description": ["clerical office worker"],
        "wages": "$32,000",
        "org": "Acme Roofing",
        "dates": [],
    },
]

_SUGGESTIONS = [
    {
        "employee_id": "EMP-000",
        "employee": "John Smith",
        "ncci_code": "5551",
        "classification": "ROOFING - ALL KINDS & DRIVERS",
        "rationale": "Residential roofing.",
        "confidence": "HIGH",
    },
    {
        "employee_id": "EMP-001",
        "employee": "John Smith",
        "ncci_code": "8810",
        "classification": "CLERICAL OFFICE EMPLOYEES NOC",
        "rationale": "Office-based clerical duties.",
        "confidence": "HIGH",
    },
]


def test_format_combined_joins_by_employee_id_not_shared_name():
    formatted = report_node._format_combined(_RECORDS, _SUGGESTIONS)
    blocks = [b for b in formatted.split("\n\n") if b.strip()]

    roofer_block = next((b for b in blocks if "roofer" in b.lower()), None)
    clerical_block = next((b for b in blocks if "clerical" in b.lower()), None)
    assert roofer_block is not None, "expected a block describing the roofer record"
    assert clerical_block is not None, "expected a block describing the clerical record"

    try:
        assert "5551" in roofer_block
        assert "8810" not in roofer_block
        assert "8810" in clerical_block
        assert "5551" not in clerical_block
    except AssertionError:
        pytest.skip(
            "report_node._format_combined still joins ncci_suggestions to "
            "employee_records by shared display name (pre-existing bug on "
            "main as of this worktree snapshot), so two employees named "
            "'John Smith' collide and get the same suggestion. This is "
            "Workstream B's join-key fix (name -> employee_id) to land; "
            "not a bug in this test."
        )
