"""Per-employee subgraph — the Send() target for the Supervisor's fan-out.

A Send()-spawned branch that is chained through top-level graph nodes via
normal add_edge calls loses per-item identity: downstream nodes run once
globally after all branches converge, not once per item, and any field not
declared in the parent schema vanishes between hops. The fix verified against
the installed LangGraph 1.2.7: wrap the per-employee steps in their own,
separately-compiled StateGraph(EmployeeTaskState), and make the *subgraph
itself* — not its individual nodes — the thing Send() targets from the
parent graph. That gives genuine per-employee isolation and genuine
concurrency.
"""

import time

from langgraph.graph import StateGraph, END

from agents.schemas import EmployeeTaskState, make_agent_step
from agents.nodes.retrieval_agent import retrieval_agent
from agents.nodes.classification_agent import classification_agent
from agents.nodes.critic_agent import critic_agent


def _build_employee_subgraph():
    graph = StateGraph(EmployeeTaskState)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("classification_agent", classification_agent)
    graph.add_node("critic_agent", critic_agent)
    graph.set_entry_point("retrieval_agent")
    graph.add_edge("retrieval_agent", "classification_agent")
    graph.add_edge("classification_agent", "critic_agent")
    graph.add_edge("critic_agent", END)
    return graph.compile()


_employee_subgraph = _build_employee_subgraph()


def employee_pipeline_node(state: EmployeeTaskState) -> dict:
    """The Send() target from supervisor.route_to_employee_fanout. Runs the
    per-employee subgraph (retrieval -> classification -> critic) to
    completion in full isolation, then reshapes its final local state into
    AuditState's single-item-list contributions so the parent graph's
    operator.add reducers concatenate one entry per employee.

    Fault isolation boundary: an unhandled exception from ANY node inside the
    subgraph (e.g. a Groq call that exhausted its own retries) used to
    propagate all the way up through LangGraph's executor and crash the
    entire audit — including employees whose branches had already succeeded
    (confirmed by direct reproduction, not assumed). Catching it here means
    one employee's failure contributes nothing instead of taking everyone
    else down with it; aggregate_node's existing "employee missing from
    ncci_suggestions" logic then marks just that employee for retry.
    """
    employee_record = state["employee_record"]
    employee_id = employee_record["employee_id"]
    start = time.monotonic()

    try:
        result = _employee_subgraph.invoke(state)
    except Exception as exc:
        duration_ms = (time.monotonic() - start) * 1000
        failure_step = make_agent_step(
            agent="employee_pipeline",
            employee_id=employee_id,
            input_summary=f"employee={employee_record.get('name', '')!r}",
            output_summary=f"branch failed: {exc}",
            duration_ms=duration_ms,
            status="error",
        )
        return {"audit_trail": [failure_step]}

    contributions: dict = {"audit_trail": result.get("audit_trail", [])}
    if result.get("retrieval_output"):
        contributions["retrieval_outputs"] = [result["retrieval_output"]]
    if result.get("classification_output"):
        contributions["ncci_suggestions"] = [result["classification_output"]]
    if result.get("critic_output"):
        contributions["critic_assessments"] = [result["critic_output"]]
    return contributions
