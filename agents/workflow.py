from langgraph.graph import StateGraph, END

from agents.state import AuditState
from agents.nodes.intake_node import intake_node
from agents.nodes.extract_node import extract_node
from agents.nodes.classify_node import classify_node
from agents.nodes.report_node import report_node


def _route_after_intake(state: AuditState) -> str:
    """After intake: abort if the document is critically incomplete, otherwise proceed."""
    if state.get("error"):
        return END
    if state.get("completeness_flags", {}).get("critical_missing"):
        return END
    return "extract"


def _build_graph():
    graph = StateGraph(AuditState)

    graph.add_node("intake", intake_node)
    graph.add_node("extract", extract_node)
    graph.add_node("classify", classify_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("intake")

    graph.add_conditional_edges(
        "intake",
        _route_after_intake,
        {"extract": "extract", END: END},
    )
    graph.add_edge("extract", "classify")
    graph.add_edge("classify", "report")
    graph.add_edge("report", END)

    return graph.compile()


_app = _build_graph()


def run_workflow(file_path: str) -> AuditState:
    initial_state: AuditState = {
        "uploaded_file_path": file_path,
        "raw_text": "",
        "entities": [],
        "employee_records": [],
        "ncci_suggestions": [],
        "completeness_flags": {},
        "audit_report": "",
        "error": None,
        "current_step": "start",
    }
    return _app.invoke(initial_state)
