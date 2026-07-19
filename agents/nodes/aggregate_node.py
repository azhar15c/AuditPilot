"""Aggregate node — runs once after the per-employee fan-out reconverges.

Validates that every employee got classified; for any that didn't (a branch
failure, e.g. a Groq API error mid-classification), degrades gracefully
rather than failing the whole run: adds a placeholder "NOT CLASSIFIED —
retry required" entry for that employee and logs a warning AgentStep, but
lets the rest of the audit complete.
"""

from agents.schemas import make_agent_step
from agents.state import AuditState


def aggregate_node(state: AuditState) -> dict:
    """Runs once after the per-employee fan-out reconverges. Validates
    completeness and degrades gracefully (never fails the whole run) if any
    employee's branch didn't produce a classification.

    Never spreads **state: retrieval_outputs/ncci_suggestions/
    critic_assessments/audit_trail are Annotated[list[dict], operator.add]
    fields on AuditState, and by the time this node runs they already hold
    the fan-out's fully-merged contents. Echoing an already-accumulated list
    back through the reducer would duplicate every item, so this node only
    ever returns delta (new-items-only) lists for those fields, and omits a
    reducer field's key entirely when it has nothing new to contribute.
    """
    if state.get("error"):
        return {}

    employee_records = state.get("employee_records", [])
    ncci_suggestions = state.get("ncci_suggestions", [])
    audit_trail = state.get("audit_trail", [])
    classified_ids = {s["employee_id"] for s in ncci_suggestions}

    # employee_pipeline_node logs an "error"-status step (agent="employee_pipeline")
    # when it catches a branch failure — look those up so the placeholder below can
    # show the real cause instead of a generic "didn't complete" message.
    failure_reasons = {
        step["employee_id"]: step["output_summary"]
        for step in audit_trail
        if step.get("agent") == "employee_pipeline" and step.get("status") == "error" and step.get("employee_id")
    }

    missing_suggestions = []
    warning_steps = []
    for record in employee_records:
        employee_id = record["employee_id"]
        if employee_id not in classified_ids:
            reason = failure_reasons.get(
                employee_id, "This employee's classification branch did not complete.",
            )
            missing_suggestions.append({
                "employee_id": employee_id,
                "employee": record.get("name", ""),
                "ncci_code": None,
                "classification": "NOT CLASSIFIED — retry required",
                "rationale": reason,
                "confidence": "LOW",
            })
            warning_steps.append(make_agent_step(
                agent="aggregate",
                employee_id=employee_id,
                input_summary=f"expected classification for {employee_id}, none found",
                output_summary=f"marked NOT CLASSIFIED — retry required ({reason}); audit continues for other employees",
                duration_ms=0.0,
                status="error",
            ))

    result: dict = {"current_step": "aggregate"}
    if missing_suggestions:
        result["ncci_suggestions"] = missing_suggestions   # delta only — operator.add appends these
    if warning_steps:
        result["audit_trail"] = warning_steps               # delta only
    return result
