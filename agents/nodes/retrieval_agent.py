import json
import time
from agents.schemas import EmployeeTaskState, make_agent_step
from models.hf_client import HFClient
from rag.retriever import NCCIRetriever

_client    = HFClient()
_retriever = NCCIRetriever()

_MAX_ITERATIONS = 4

_SUFFICIENCY_SYSTEM = """You are a workers' compensation premium audit research assistant.

You are judging whether a batch of NCCI manual excerpts, retrieved via semantic
search for one specific employee, contains enough information to confidently
assign that employee an NCCI class code.

Call evaluate_retrieval with your judgment. Be strict: if the excerpts are
generic, off-topic, or missing a cross-referenced clause the job clearly needs
(e.g. a "drivers" or "clerical" carve-out), mark not sufficient and propose a
sharper, more specific search query targeting exactly what is missing.
Always call evaluate_retrieval — do not respond in plain text."""

_SUFFICIENCY_TOOL = {
    "type": "function",
    "function": {
        "name": "evaluate_retrieval",
        "description": "Judge whether the retrieved NCCI excerpts are sufficient to classify this employee, and if not, propose a better search query.",
        "parameters": {
            "type": "object",
            "properties": {
                "sufficient": {"type": "boolean", "description": "True if these excerpts contain enough to confidently classify this employee"},
                "reasoning": {"type": "string", "description": "Why sufficient, or specifically why not (wrong section? too vague? missing a cross-referenced clause?)"},
                "refined_query": {"type": "string", "description": "If not sufficient, a reformulated search query targeting the specific gap. Empty string if sufficient."},
            },
            "required": ["sufficient", "reasoning", "refined_query"],
        },
    },
}


def retrieval_agent(state: EmployeeTaskState) -> dict:
    """Node in the per-employee subgraph. Iteratively searches ChromaDB, judging
    after each attempt whether the results are sufficient to answer this employee's
    classification question; if not, reformulates the query based on *why* it fell
    short, up to _MAX_ITERATIONS attempts."""
    record      = state["employee_record"]
    employee_id = record["employee_id"]
    prior_data  = state.get("prior_classifications", {})

    start = time.monotonic()

    query          = _build_query(record)
    queries_used: list[str] = []
    results: list[dict]     = []
    sufficiency_met = False
    reasoning       = ""
    iterations_run  = 0

    for _ in range(_MAX_ITERATIONS):
        iterations_run += 1
        queries_used.append(query)

        results = _retriever.query(query, top_k=5)
        context = _retriever.format_context(results)

        sufficient, reasoning, refined_query = _evaluate_sufficiency(record, query, context, prior_data)

        if sufficient:
            sufficiency_met = True
            break

        query = refined_query if refined_query else query

    duration_ms = (time.monotonic() - start) * 1000

    audit_step = make_agent_step(
        agent="retrieval",
        employee_id=employee_id,
        input_summary=f"job={record.get('job_description')!r}, initial_query={queries_used[0]!r}",
        output_summary=(
            f"iterations={iterations_run}, sufficient={sufficiency_met}, "
            f"citations={len(results)}, last_query={queries_used[-1]!r}"
        ),
        duration_ms=duration_ms,
    )

    return {
        "retrieval_output": {
            "employee_id": employee_id,
            "policy_citations": results,
            "policycenter_fields": prior_data,
            "search_queries_used": queries_used,
            "iterations_run": iterations_run,
            "sufficiency_met": sufficiency_met,
            "sufficiency_reasoning": reasoning,
        },
        "audit_trail": [audit_step],
    }


def _build_query(record: dict) -> str:
    parts = list(record.get("job_description") or [])
    if record.get("org"):
        parts.append(record["org"])
    return " ".join(parts) if parts else record.get("name", "employee")


def _build_eval_prompt(record: dict, query: str, context: str, prior_data: dict) -> str:
    job_desc = ", ".join(record.get("job_description") or []) or "Not specified"

    prior_section = ""
    if prior_data.get("classifications"):
        lines = "\n".join(
            f"  - {c['ncci_code']}: {c['classification']}"
            for c in prior_data["classifications"]
        )
        prior_section = f"\nPrior policy period classifications for this employer (context only):\n{lines}\n"

    return (
        f"Employee:       {record.get('name', 'Unknown')}\n"
        f"Job title/role: {job_desc}\n"
        f"Employer:       {record.get('org') or 'Unknown'}\n"
        f"{prior_section}\n"
        f"Search query used: {query!r}\n\n"
        f"Retrieved NCCI Manual Excerpts:\n{context}\n\n"
        "Call evaluate_retrieval with your judgment on whether these excerpts are sufficient."
    )


def _evaluate_sufficiency(record: dict, query: str, context: str, prior_data: dict) -> tuple[bool, str, str]:
    messages = [
        {"role": "system", "content": _SUFFICIENCY_SYSTEM},
        {"role": "user",   "content": _build_eval_prompt(record, query, context, prior_data)},
    ]
    response = _client.chat_with_tools(messages, [_SUFFICIENCY_TOOL], tool_choice="required")
    return _extract_sufficiency(response)


def _extract_sufficiency(response) -> tuple[bool, str, str]:
    msg = response.choices[0].message

    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name == "evaluate_retrieval":
                args = json.loads(tc.function.arguments)
                return (
                    bool(args.get("sufficient", False)),
                    args.get("reasoning", ""),
                    args.get("refined_query") or "",
                )

    # Model somehow didn't call the tool despite tool_choice="required" — don't crash,
    # treat as insufficient so the loop either retries or caps out honestly.
    return False, "Model did not return a structured evaluation; treating as insufficient.", ""
