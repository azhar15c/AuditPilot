import json
import re
import time
from agents.schemas import EmployeeTaskState, make_agent_step
from models.hf_client import HFClient

_client = HFClient()

_AGENT_SYSTEM = """You are a workers' compensation premium audit specialist.

You have been given:
- An employee's name, job title, and reported wages
- Relevant NCCI manual excerpts already retrieved via semantic search
- NCCI class codes used for this employer in the prior policy period (for consistency)

Your task: call finalize_classification with the best-fit NCCI class code.

Rules:
- Base your code on the NCCI manual excerpts — do not invent codes not present in the material
- Use prior year classifications as a consistency signal, not a constraint — reclassify if the job changed
- For clerical or administrative roles, distinguish between 8810 (office staff) and 8742 (outside sales)
- Mark confidence HIGH only if an excerpt explicitly names this job type
- Always call finalize_classification — do not respond in plain text"""

_FINALIZE_TOOL = {
    "type": "function",
    "function": {
        "name": "finalize_classification",
        "description": "Record your final NCCI classification decision for this employee.",
        "parameters": {
            "type": "object",
            "properties": {
                "ncci_code": {
                    "type": "string",
                    "description": "4-digit NCCI class code, e.g. '5551'",
                },
                "classification": {
                    "type": "string",
                    "description": "Official NCCI class code title, e.g. 'ROOFING - ALL KINDS & DRIVERS'",
                },
                "rationale": {
                    "type": "string",
                    "description": "One sentence explaining why this code applies to this employee",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                    "description": "HIGH if an NCCI excerpt explicitly names the job type; LOW if no close match",
                },
            },
            "required": ["ncci_code", "classification", "rationale", "confidence"],
        },
    },
}


def classification_agent(state: EmployeeTaskState) -> dict:
    """Node in the per-employee subgraph. Forced-tool-call classification using
    ONLY the citations Retrieval already gathered — no direct RAG/PolicyCenter access."""
    record      = state["employee_record"]
    employee_id = record["employee_id"]
    employee    = record.get("name", "")

    retrieval_output = state.get("retrieval_output") or {}
    citations        = retrieval_output.get("policy_citations", [])
    rag_context      = _format_citations(citations)
    prior_data       = state.get("prior_classifications", {})

    start = time.monotonic()

    messages = [
        {"role": "system", "content": _AGENT_SYSTEM},
        {"role": "user",   "content": _build_prompt(record, rag_context, prior_data)},
    ]
    response = _client.chat_with_tools(messages, [_FINALIZE_TOOL], tool_choice="required")
    result   = _extract_result(response)

    duration_ms = (time.monotonic() - start) * 1000

    audit_step = make_agent_step(
        agent="classification",
        employee_id=employee_id,
        input_summary=f"employee={employee!r}, citations_used={len(citations)}",
        output_summary=f"ncci_code={result.get('ncci_code')}, confidence={result.get('confidence')}",
        duration_ms=duration_ms,
    )

    return {
        "classification_output": {
            "employee_id":    employee_id,
            "employee":       employee,
            "ncci_code":      result.get("ncci_code"),
            "classification": result.get("classification", ""),
            "rationale":      result.get("rationale", ""),
            "confidence":     result.get("confidence", "LOW"),
        },
        "audit_trail": [audit_step],
    }


def _format_citations(citations: list[dict]) -> str:
    """Format retrieval_output's policy_citations into prompt-ready text.
    Mirrors NCCIRetriever.format_context without importing that class
    (Classification must not reach ChromaDB directly)."""
    if not citations:
        return "No relevant NCCI reference material found."
    sections = []
    for i, r in enumerate(citations, start=1):
        source = (r.get("metadata") or {}).get("source", "unknown")
        sections.append(
            f"[Ref {i} | {source} | distance: {r.get('distance')}]\n{r.get('chunk_text', '')}"
        )
    return "\n\n---\n\n".join(sections)


def _build_prompt(record: dict, rag_context: str, prior_data: dict) -> str:
    job_desc = ", ".join(record.get("job_description") or []) or "Not specified"

    prior_section = ""
    if prior_data.get("classifications"):
        lines = "\n".join(
            f"  - {c['ncci_code']}: {c['classification']}  (prior payroll: ${c.get('audited_payroll', 0):,.0f})"
            for c in prior_data["classifications"]
        )
        prior_section = f"\nPrior policy period classifications for this employer:\n{lines}\n"

    return (
        f"Employee:       {record.get('name', '')}\n"
        f"Job title/role: {job_desc}\n"
        f"Employer:       {record.get('org') or 'Unknown'}\n"
        f"Reported wages: {record.get('wages') or 'Not specified'}\n"
        f"{prior_section}\n"
        f"NCCI Manual Excerpts:\n{rag_context}\n\n"
        "Call finalize_classification with the most applicable NCCI code."
    )


# ── Result extraction ─────────────────────────────────────────────────────────

def _extract_result(response) -> dict:
    msg = response.choices[0].message

    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name == "finalize_classification":
                args = json.loads(tc.function.arguments)
                return {
                    "ncci_code":      args.get("ncci_code"),
                    "classification": args.get("classification", ""),
                    "rationale":      args.get("rationale", ""),
                    "confidence":     args.get("confidence", "LOW").upper(),
                }

    # Fallback: model responded in plain text despite tool_choice="required"
    content        = msg.content or ""
    code           = re.search(r'\b(\d{4})\b', content)
    classification = re.search(r'CLASSIFICATION:\s*(.+)', content, re.IGNORECASE)
    rationale      = re.search(r'RATIONALE:\s*(.+)',      content, re.IGNORECASE)
    confidence     = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', content, re.IGNORECASE)
    return {
        "ncci_code":      code.group(1) if code else None,
        "classification": classification.group(1).strip() if classification else "",
        "rationale":      rationale.group(1).strip() if rationale else content.strip()[:200],
        "confidence":     confidence.group(1).upper() if confidence else "LOW",
    }
