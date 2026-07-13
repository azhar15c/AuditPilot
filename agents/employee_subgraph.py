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

from langgraph.graph import StateGraph, END

from agents.schemas import EmployeeTaskState
from agents.nodes.retrieval_agent import retrieval_agent
from agents.nodes.classification_agent import classification_agent


def _build_employee_subgraph():
    graph = StateGraph(EmployeeTaskState)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("classification_agent", classification_agent)
    graph.set_entry_point("retrieval_agent")
    graph.add_edge("retrieval_agent", "classification_agent")
    graph.add_edge("classification_agent", END)
    return graph.compile()


_employee_subgraph = _build_employee_subgraph()


def employee_pipeline_node(state: EmployeeTaskState) -> dict:
    """The Send() target from supervisor.route_to_employee_fanout. Runs the
    per-employee subgraph to completion in full isolation, then reshapes its
    final local state into AuditState's single-item-list contributions so the
    parent graph's operator.add reducers concatenate one entry per employee.

    critic_output is always None right now since Critic isn't wired into the
    subgraph yet (deferred to a later integration step); the `if` guard below
    means critic_assessments simply won't appear in the returned dict when
    there's nothing to add, which is correct and harmless.
    """
    result = _employee_subgraph.invoke(state)

    contributions: dict = {"audit_trail": result.get("audit_trail", [])}
    if result.get("retrieval_output"):
        contributions["retrieval_outputs"] = [result["retrieval_output"]]
    if result.get("classification_output"):
        contributions["ncci_suggestions"] = [result["classification_output"]]
    if result.get("critic_output"):
        contributions["critic_assessments"] = [result["critic_output"]]
    return contributions
