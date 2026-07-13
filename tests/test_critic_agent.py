"""Tests for agents.nodes.critic_agent — Workstream D.

critic_agent reviews a ClassificationOutput against the RetrievalOutput's
policy_citations but must NEVER see Classification's `rationale` field
(blind review, to avoid anchoring on Classification's own reasoning). It
returns a CriticOutput with agrees / recommended_action ("approve" or
"flag_for_review") / critic_confidence.

Guarded with importorskip since agents/nodes/critic_agent.py does not exist
yet in this worktree — Workstream D builds it in a separate worktree and
merges later.

Per the task brief: don't hardcode the exact tool name the Critic uses
internally — mock _client.chat_with_tools to return a canned response and
assert on the returned critic_output shape instead.
"""

import json

import pytest

pytest.importorskip("agents.nodes.critic_agent")

from unittest.mock import patch  # noqa: E402

from agents.nodes.critic_agent import critic_agent  # noqa: E402

_MARKER_RATIONALE = "MARKER_RATIONALE_TEXT_12345"


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name: str, arguments: str):
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, tool_calls, content=None):
        self.tool_calls = tool_calls
        self.content = content


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


def _critic_response(agrees: bool, recommended_action: str, concern: str = "",
                      critic_confidence: str = "MEDIUM", tool_name: str = "record_critic_review") -> _FakeResponse:
    args = json.dumps({
        "agrees": agrees,
        "concern": concern,
        "recommended_action": recommended_action,
        "critic_confidence": critic_confidence,
    })
    return _FakeResponse(_FakeMessage(tool_calls=[_FakeToolCall(tool_name, args)]))


def _state_with_outputs(base_state: dict, ncci_code: str, classification: str,
                         citation_text: str, rationale: str = _MARKER_RATIONALE) -> dict:
    employee_id = base_state["employee_record"]["employee_id"]
    return {
        **base_state,
        "retrieval_output": {
            "employee_id": employee_id,
            "policy_citations": [{"chunk_text": citation_text, "distance": 0.1, "metadata": {}}],
            "policycenter_fields": {},
            "search_queries_used": ["roofer"],
            "iterations_run": 1,
            "sufficiency_met": True,
            "sufficiency_reasoning": "found it",
        },
        "classification_output": {
            "employee_id": employee_id,
            "employee": base_state["employee_record"]["name"],
            "ncci_code": ncci_code,
            "classification": classification,
            "rationale": rationale,
            "confidence": "HIGH",
        },
    }


# Synthetic disagreement cases: citations clearly describe one job, but the
# classification claims something else entirely.
_DISAGREEMENT_CASES = [
    ("roofer citation vs clerical code", "8810", "CLERICAL OFFICE EMPLOYEES NOC",
     "ROOFING - ALL KINDS & DRIVERS. Employees engaged in installing shingles on pitched roofs. Code 5551."),
    ("electrician citation vs driver code", "7380", "DRIVERS NOC",
     "ELECTRICAL WIRING - WITHIN BUILDINGS. Installation of wiring and fixtures. Code 5190."),
    ("plumber citation vs sales code", "8742", "OUTSIDE SALES",
     "PLUMBING NOC. Installation and repair of pipes, fixtures. Code 5183."),
]


@pytest.mark.workstream_d
@pytest.mark.critic_disagreement
class TestCriticDisagreement:
    @pytest.mark.parametrize("label,ncci_code,classification,citation_text", _DISAGREEMENT_CASES)
    def test_flags_clearly_mismatched_classification(
        self, base_employee_task_state, label, ncci_code, classification, citation_text
    ):
        state = _state_with_outputs(base_employee_task_state, ncci_code, classification, citation_text)

        with patch("agents.nodes.critic_agent._client") as mock_client:
            mock_client.chat_with_tools.return_value = _critic_response(
                agrees=False,
                recommended_action="flag_for_review",
                concern=f"Citations describe a different job than {classification}.",
            )
            result = critic_agent(state)

        output = result["critic_output"]
        assert output["recommended_action"] == "flag_for_review", label
        assert output["agrees"] is False, label
        assert output["employee_id"] == base_employee_task_state["employee_record"]["employee_id"]


@pytest.mark.workstream_d
class TestCriticBlindReview:
    def test_classification_rationale_never_reaches_the_prompt(self, base_employee_task_state):
        state = _state_with_outputs(
            base_employee_task_state,
            ncci_code="5551",
            classification="ROOFING - ALL KINDS & DRIVERS",
            citation_text="ROOFING - ALL KINDS & DRIVERS. Code 5551.",
            rationale=_MARKER_RATIONALE,
        )

        captured_calls = []

        def _capture(*args, **kwargs):
            captured_calls.append((args, kwargs))
            return _critic_response(agrees=True, recommended_action="approve")

        with patch("agents.nodes.critic_agent._client") as mock_client:
            mock_client.chat_with_tools.side_effect = _capture
            critic_agent(state)

        assert captured_calls, "expected _client.chat_with_tools to be called"
        for args, kwargs in captured_calls:
            full_payload = json.dumps(args, default=str) + json.dumps(kwargs, default=str)
            assert _MARKER_RATIONALE not in full_payload, (
                "Classification's rationale leaked into the Critic's prompt — "
                "the Critic must perform a blind review."
            )
