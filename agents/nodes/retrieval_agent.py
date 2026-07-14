import json
import time
from agents.schemas import EmployeeTaskState, make_agent_step
from models.hf_client import HFClient
from rag.retriever import NCCIRetriever

_client    = HFClient()
_retriever = NCCIRetriever()

_MAX_ITERATIONS = 4
_MAX_CITATIONS_RETURNED = 8  # cap after accumulation, before handoff to classification_agent

_SUFFICIENCY_SYSTEM = """You are a workers' compensation premium audit research assistant.

You are iteratively searching NCCI manual excerpts for one specific employee.
Each round shows you a compact summary of prior rounds' findings (not the raw
excerpts already reviewed — those have already been judged) plus any NEW
excerpts just found. Judge sufficiency against everything collected so far,
not just this round's new excerpts.

Call evaluate_retrieval with your judgment. Be strict: if the excerpts
collected so far are generic, off-topic, or missing a cross-referenced clause
the job clearly needs (e.g. a "drivers" or "clerical" carve-out), mark not
sufficient and propose a sharper, more specific search query targeting
exactly what is missing. Always call evaluate_retrieval — do not respond in
plain text."""

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
    after each attempt whether the results collected *so far* are sufficient to
    answer this employee's classification question; if not, reformulates the
    query based on *why* it fell short, up to _MAX_ITERATIONS attempts.

    Context engineering, not just "loop and pass more": results are
    accumulated and deduped across rounds (a chunk found in round 1 is not
    lost just because round 2 searched something else), but each round's
    prompt only shows *new* chunks plus a compact scratchpad distilled from
    prior rounds' own reasoning — not a replay of every raw chunk ever seen.
    That keeps prompt size roughly flat across iterations instead of growing
    with them. The final accumulated set is capped to the top
    _MAX_CITATIONS_RETURNED by relevance before handoff to
    classification_agent, so a long search doesn't balloon the next agent's
    prompt either.
    """
    record      = state["employee_record"]
    employee_id = record["employee_id"]
    prior_data  = state.get("prior_classifications", {})

    start = time.monotonic()

    query          = _build_query(record)
    queries_used: list[str] = []
    collected: dict[str, dict] = {}  # chunk id -> chunk, accumulated + deduped across rounds
    scratchpad: list[str] = []       # compact per-round notes, carried forward instead of raw chunk replay
    sufficiency_met = False
    reasoning       = ""
    iterations_run  = 0

    for round_num in range(1, _MAX_ITERATIONS + 1):
        iterations_run = round_num
        queries_used.append(query)

        round_results = _retriever.query(query, top_k=5)
        new_chunks = [c for c in round_results if c["id"] not in collected]
        collected.update({c["id"]: c for c in round_results})

        new_context = _format_new_context(round_results, new_chunks)

        sufficient, reasoning, refined_query = _evaluate_sufficiency(
            record, query, new_context, scratchpad, len(collected), prior_data,
        )
        scratchpad.append(f"Round {round_num} (query={query!r}): {reasoning}")

        if sufficient:
            sufficiency_met = True
            break

        query = refined_query if refined_query else query

    duration_ms = (time.monotonic() - start) * 1000

    final_citations = sorted(collected.values(), key=lambda c: c["distance"])[:_MAX_CITATIONS_RETURNED]

    audit_step = make_agent_step(
        agent="retrieval",
        employee_id=employee_id,
        input_summary=f"job={record.get('job_description')!r}, initial_query={queries_used[0]!r}",
        output_summary=(
            f"iterations={iterations_run}, sufficient={sufficiency_met}, "
            f"collected={len(collected)}, returned={len(final_citations)}, last_query={queries_used[-1]!r}"
        ),
        duration_ms=duration_ms,
    )

    return {
        "retrieval_output": {
            "employee_id": employee_id,
            "policy_citations": final_citations,
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


def _format_new_context(round_results: list[dict], new_chunks: list[dict]) -> str:
    if not round_results:
        return "No relevant NCCI reference material found for this query."
    if not new_chunks:
        return "(query returned only already-seen excerpts — no new information this round)"
    return _retriever.format_context(new_chunks)


def _build_eval_prompt(
    record: dict, query: str, new_context: str, scratchpad: list[str],
    collected_count: int, prior_data: dict,
) -> str:
    job_desc = ", ".join(record.get("job_description") or []) or "Not specified"

    prior_section = ""
    if prior_data.get("classifications"):
        lines = "\n".join(
            f"  - {c['ncci_code']}: {c['classification']}"
            for c in prior_data["classifications"]
        )
        prior_section = f"\nPrior policy period classifications for this employer (context only):\n{lines}\n"

    scratchpad_section = ""
    if scratchpad:
        notes = "\n".join(f"  - {note}" for note in scratchpad)
        scratchpad_section = (
            f"\nProgress so far ({collected_count} unique excerpt(s) collected across all rounds):\n{notes}\n"
        )

    return (
        f"Employee:       {record.get('name', 'Unknown')}\n"
        f"Job title/role: {job_desc}\n"
        f"Employer:       {record.get('org') or 'Unknown'}\n"
        f"{prior_section}"
        f"{scratchpad_section}\n"
        f"Search query used this round: {query!r}\n\n"
        f"New excerpts found this round:\n{new_context}\n\n"
        "Considering everything collected so far — not just this round — call "
        "evaluate_retrieval with your judgment on whether it's now sufficient."
    )


def _evaluate_sufficiency(
    record: dict, query: str, new_context: str, scratchpad: list[str],
    collected_count: int, prior_data: dict,
) -> tuple[bool, str, str]:
    messages = [
        {"role": "system", "content": _SUFFICIENCY_SYSTEM},
        {"role": "user",   "content": _build_eval_prompt(record, query, new_context, scratchpad, collected_count, prior_data)},
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
