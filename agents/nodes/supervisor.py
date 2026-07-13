"""Supervisor node — runs once after extraction, before the per-employee fan-out.

Today's classify_node.py calls dispatch_tool("get_prior_classifications", ...)
once per employee even though policy_number (and therefore the prior-period
data) is constant for the whole audit run. The Supervisor fixes that: it
fetches PolicyCenter's prior classifications exactly once, then hands the
same shared result to every employee branch via the Send payload.
"""

import time
from typing import Optional

from langgraph.types import Send

from agents.schemas import make_agent_step
from agents.state import AuditState
from mcp_servers.policycenter import dispatch_tool

_DEFAULT_POLICY_NUMBER = "WC-DEMO-001"


def supervisor_node(state: AuditState) -> AuditState:
    """Runs once, after extraction, before fan-out. Fetches PolicyCenter's prior
    classifications exactly once (not per-employee — that was the old node's waste)."""
    if state.get("error"):
        return state

    start = time.monotonic()
    policy_number = state.get("policy_number") or _DEFAULT_POLICY_NUMBER
    prior_data = dispatch_tool("get_prior_classifications", {"policy_number": policy_number})
    duration_ms = (time.monotonic() - start) * 1000

    step = make_agent_step(
        agent="supervisor",
        employee_id=None,
        input_summary=f"policy_number={policy_number}",
        output_summary=f"fetched {len(prior_data.get('classifications', []))} prior classification(s)",
        duration_ms=duration_ms,
    )
    return {
        **state,
        "prior_classifications": prior_data,
        "policy_number": policy_number,
        "audit_trail": [step],
        "current_step": "supervisor",
    }


def route_to_employee_fanout(state: AuditState) -> list[Send]:
    """Conditional-edge function: fans out one Send per employee record, targeting
    the "employee_pipeline" node (a per-employee subgraph wired by a different
    workstream — this function is only the caller, not the builder, of that node).

    Returns an empty list when there are no employee records; LangGraph handles
    fan-out to zero branches without any special-casing needed here.
    """
    records = state.get("employee_records", [])
    policy_number = state.get("policy_number") or _DEFAULT_POLICY_NUMBER
    prior_classifications = state.get("prior_classifications", {})
    return [
        Send(
            "employee_pipeline",
            {
                "employee_record": record,
                "policy_number": policy_number,
                "prior_classifications": prior_classifications,
                "retrieval_output": None,
                "classification_output": None,
                "critic_output": None,
                "audit_trail": [],
            },
        )
        for record in records
    ]
