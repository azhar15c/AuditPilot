import json
import re
import time
from agents.schemas import EmployeeTaskState, make_agent_step
from models.hf_client import HFClient

_client = HFClient()

_CRITIC_SYSTEM = """You are an independent quality-control reviewer for workers'
compensation premium audit classifications.

You are shown a proposed NCCI classification for one employee — the class code,
its official title, and the confidence level assigned — along with the raw NCCI
manual excerpts that were retrieved for this employee. You are NOT shown the
classifying agent's reasoning. Your job is to independently judge, from the raw
excerpts alone, whether the proposed code is actually supported.

Call finalize_critic_review with your verdict. Be genuinely independent: if the
excerpts don't clearly support the stated code, or contradict it, or are too
generic to justify HIGH confidence, say so — do not assume the proposed code is
correct just because it was proposed. Always call finalize_critic_review — do
not respond in plain text."""

_CRITIC_TOOL = {
    "type": "function",
    "function": {
        "name": "finalize_critic_review",
        "description": "Record your independent verdict on a proposed NCCI classification.",
        "parameters": {
            "type": "object",
            "properties": {
                "agrees": {"type": "boolean", "description": "True if the raw excerpts support the proposed code"},
                "concern": {"type": "string", "description": "If not agreeing (or agreeing with reservations), what specifically is unsupported or contradicted. Empty string if no concern."},
                "recommended_action": {"type": "string", "enum": ["approve", "flag_for_review"], "description": "approve if the excerpts clearly support the code; flag_for_review otherwise"},
                "critic_confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"], "description": "Your confidence in this verdict"},
            },
            "required": ["agrees", "concern", "recommended_action", "critic_confidence"],
        },
    },
}


def critic_agent(state: EmployeeTaskState) -> dict:
    """Node in the per-employee subgraph, runs after classification_agent.
    Independently reviews the proposed classification against Retrieval's raw
    citations — deliberately never sees Classification's own rationale."""
    record = state["employee_record"]
    employee_id = record["employee_id"]

    classification_output = state.get("classification_output") or {}
    retrieval_output = state.get("retrieval_output") or {}
    citations = retrieval_output.get("policy_citations", [])

    # Blind review: only ncci_code/classification/confidence — rationale is
    # intentionally excluded from this dict and therefore cannot leak into the prompt.
    final_classification = {
        "ncci_code": classification_output.get("ncci_code"),
        "classification": classification_output.get("classification"),
        "confidence": classification_output.get("confidence"),
    }

    start = time.monotonic()
    messages = [
        {"role": "system", "content": _CRITIC_SYSTEM},
        {"role": "user", "content": _build_prompt(record, final_classification, citations)},
    ]
    response = _client.chat_with_tools(messages, [_CRITIC_TOOL], tool_choice="required")
    result = _extract_result(response)
    duration_ms = (time.monotonic() - start) * 1000

    audit_step = make_agent_step(
        agent="critic",
        employee_id=employee_id,
        input_summary=f"reviewing ncci_code={final_classification.get('ncci_code')!r} against {len(citations)} citation(s)",
        output_summary=f"agrees={result['agrees']}, action={result['recommended_action']}",
        duration_ms=duration_ms,
    )

    return {
        "critic_output": {
            "employee_id": employee_id,
            "agrees": result["agrees"],
            "concern": result.get("concern") or None,
            "recommended_action": result["recommended_action"],
            "critic_confidence": result["critic_confidence"],
        },
        "audit_trail": [audit_step],
    }


def _build_prompt(record: dict, final_classification: dict, citations: list[dict]) -> str:
    """Deliberately does not accept or reference a "rationale" parameter anywhere —
    only the employee record, the proposed code/title/confidence, and Retrieval's
    raw citations are ever surfaced to the Critic."""
    job_desc = ", ".join(record.get("job_description") or []) or "Not specified"
    rag_context = _format_citations(citations)

    return (
        f"Employee:       {record.get('name', '')}\n"
        f"Job title/role: {job_desc}\n"
        f"Employer:       {record.get('org') or 'Unknown'}\n\n"
        f"Proposed classification (under review):\n"
        f"  NCCI code:      {final_classification.get('ncci_code')}\n"
        f"  Classification: {final_classification.get('classification')}\n"
        f"  Confidence:     {final_classification.get('confidence')}\n\n"
        f"Raw NCCI Manual Excerpts (the only evidence you should rely on):\n{rag_context}\n\n"
        "Independently judge whether these excerpts actually support the proposed "
        "code above, then call finalize_critic_review with your verdict."
    )


def _format_citations(citations: list[dict]) -> str:
    """Format retrieval_output's policy_citations into prompt-ready text.
    Mirrors classification_agent.py's _format_citations without importing
    NCCIRetriever (Critic must not reach ChromaDB directly)."""
    if not citations:
        return "No relevant NCCI reference material found."
    sections = []
    for i, r in enumerate(citations, start=1):
        source = (r.get("metadata") or {}).get("source", "unknown")
        sections.append(
            f"[Ref {i} | {source} | distance: {r.get('distance')}]\n{r.get('chunk_text', '')}"
        )
    return "\n\n---\n\n".join(sections)


# ── Result extraction ─────────────────────────────────────────────────────────

def _extract_result(response) -> dict:
    msg = response.choices[0].message

    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name == "finalize_critic_review":
                args = json.loads(tc.function.arguments)
                return {
                    "agrees":            bool(args.get("agrees", False)),
                    "concern":           args.get("concern", ""),
                    "recommended_action": args.get("recommended_action", "flag_for_review"),
                    "critic_confidence": args.get("critic_confidence", "LOW").upper(),
                }

    # Fallback: model somehow didn't call the tool despite tool_choice="required",
    # or called a differently-named tool. A parsing failure must never be treated
    # as a silent approval — default to the conservative, human-escalating verdict.
    content = msg.content or ""
    concern_match = re.search(r'CONCERN:\s*(.+)', content, re.IGNORECASE)
    return {
        "agrees": False,
        "concern": concern_match.group(1).strip() if concern_match else (
            "Critic did not return a structured verdict; flagged for human review as a precaution."
        ),
        "recommended_action": "flag_for_review",
        "critic_confidence": "LOW",
    }
