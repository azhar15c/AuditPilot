"""Typed handoff contracts between AuditPilot's agents.

Every agent-to-agent handoff in the multi-agent graph (Supervisor, Extraction,
Retrieval, Classification, Critic) is shaped by one of these TypedDicts rather
than passed as free-form dicts. Plain TypedDict — not Pydantic — to match
agents/state.py's existing convention; no new dependency, no runtime
validation (malformed handoffs surface as KeyError at the point of use).
"""

import operator
from datetime import datetime, timezone
from typing import Annotated, Optional, TypedDict


class AgentStep(TypedDict):
    agent: str  # "extraction" | "retrieval" | "classification" | "critic" | "supervisor" | "report" | "aggregate"
    employee_id: Optional[str]
    input_summary: str
    output_summary: str
    timestamp: str
    duration_ms: float


class CaseInput(TypedDict):
    uploaded_file_path: str
    policy_number: Optional[str]


class ExtractionOutput(TypedDict):
    employee_records: list[dict]  # each record includes "employee_id": str
    entities: list[dict]


class RetrievalOutput(TypedDict):
    employee_id: str
    policy_citations: list[dict]  # raw NCCIRetriever.query() results, best-found across all iterations
    policycenter_fields: dict  # passthrough of prior_classifications relevant to this employee
    search_queries_used: list[str]  # every query tried, in order
    iterations_run: int  # 1..max_iterations
    sufficiency_met: bool  # False if capped out without confidence
    sufficiency_reasoning: str  # why the loop stopped (found enough / gave up)


class ClassificationOutput(TypedDict):
    employee_id: str
    employee: str  # name, kept for display
    ncci_code: Optional[str]
    classification: str
    rationale: str
    confidence: str  # HIGH | MEDIUM | LOW


class CriticInput(TypedDict):
    """Deliberately excludes ClassificationOutput.rationale — blind review to reduce anchoring bias."""

    employee_id: str
    employee: str
    final_classification: dict  # {ncci_code, classification, confidence} — no rationale
    policy_citations: list[dict]  # from RetrievalOutput directly, not filtered through Classification


class CriticOutput(TypedDict):
    employee_id: str
    agrees: bool
    concern: Optional[str]
    recommended_action: str  # "approve" | "flag_for_review"
    critic_confidence: str  # HIGH | MEDIUM | LOW


class FinalPackage(TypedDict):
    employee_records: list[dict]
    ncci_suggestions: list[dict]
    critic_assessments: list[dict]
    audit_report: str
    completeness_flags: dict
    full_audit_trail: list[AgentStep]


def make_agent_step(
    agent: str,
    employee_id: Optional[str],
    input_summary: str,
    output_summary: str,
    duration_ms: float,
) -> AgentStep:
    """Shared factory for AgentStep entries so every node formats the audit trail identically."""
    return {
        "agent": agent,
        "employee_id": employee_id,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms,
    }


class EmployeeTaskState(TypedDict):
    """Local state schema for the per-employee subgraph (agents/employee_subgraph.py).

    A plain TypedDict with NO Annotated/reducer fields — each subgraph invocation
    (one per employee, spawned via Send from the Supervisor) runs fully isolated,
    so there is never more than one writer per field within a single run. The
    reducers live one level up, on AuditState, where the parent graph concatenates
    each isolated subgraph invocation's single-item contribution.
    """

    employee_record: dict  # set once, at Send time; read by every stage
    policy_number: str
    prior_classifications: dict  # PolicyCenter get_prior_classifications() result, fetched once by Supervisor
    retrieval_output: Optional[dict]  # RetrievalOutput shape; set by retrieval stage
    classification_output: Optional[dict]  # ClassificationOutput shape; set by classification stage
    critic_output: Optional[dict]  # CriticOutput shape; set by critic stage
    # Annotated/operator.add so each subgraph node's single-item-list return
    # (matching the "return only the delta" convention used throughout this
    # codebase) appends rather than overwrites as retrieval -> classification
    # -> critic run in sequence within one isolated subgraph invocation. Safe
    # here specifically because each subgraph invocation only ever has one
    # writer per step — this is not exposed to the cross-branch double-count
    # risk that applies to AuditState's reducers one level up.
    audit_trail: Annotated[list[dict], operator.add]
