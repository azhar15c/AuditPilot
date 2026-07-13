from langgraph.graph import StateGraph, END

from agents.state import AuditState
from agents.nodes.intake_node import intake_node
from agents.nodes.extraction_agent import extraction_agent
from agents.nodes.supervisor import supervisor_node, route_to_employee_fanout
from agents.employee_subgraph import employee_pipeline_node
from agents.nodes.aggregate_node import aggregate_node
from agents.nodes.report_node import report_node


def _route_after_intake(state: AuditState) -> str:
    """After intake: abort if the document is critically incomplete, otherwise proceed."""
    if state.get("error"):
        return END
    if state.get("completeness_flags", {}).get("critical_missing"):
        return END
    return "extraction"


def _build_graph():
    graph = StateGraph(AuditState)

    graph.add_node("intake", intake_node)
    graph.add_node("extraction", extraction_agent)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("employee_pipeline", employee_pipeline_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("intake")

    graph.add_conditional_edges(
        "intake",
        _route_after_intake,
        {"extraction": "extraction", END: END},
    )
    graph.add_edge("extraction", "supervisor")
    graph.add_conditional_edges("supervisor", route_to_employee_fanout, ["employee_pipeline"])
    graph.add_edge("employee_pipeline", "aggregate")
    graph.add_edge("aggregate", "report")
    graph.add_edge("report", END)

    return graph.compile()


_app = _build_graph()


def run_workflow(file_path: str) -> AuditState:
    initial_state: AuditState = {
        "uploaded_file_path": file_path,
        "raw_text": "",
        "entities": [],
        "employee_records": [],
        "policy_number": None,
        "prior_classifications": {},
        "retrieval_outputs": [],
        "ncci_suggestions": [],
        "critic_assessments": [],
        "audit_trail": [],
        "completeness_flags": {},
        "audit_report": "",
        "error": None,
        "current_step": "start",
    }
    return _app.invoke(initial_state)
