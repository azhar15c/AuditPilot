import json
import re
from agents.state import AuditState
from models.hf_client import HFClient
from rag.retriever import NCCIRetriever
from mcp_servers.policycenter import dispatch_tool

_client    = HFClient()
_retriever = NCCIRetriever()

_DEFAULT_POLICY_NUMBER    = "WC-DEMO-001"
_POOR_MATCH_THRESHOLD     = 1.2   # cosine distance above this triggers a second RAG query

_AGENT_SYSTEM = """You are a workers' compensation premium audit specialist.

You have been given:
- An employee's name, job title, and reported wages
- Relevant NCCI manual excerpts retrieved via semantic search
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


def classify_node(state: AuditState) -> AuditState:
    if state.get("error"):
        return state

    records = state.get("employee_records", [])
    if not records:
        return {**state, "error": "No employee records to classify.", "current_step": "classify"}

    policy_number = state.get("policy_number") or _DEFAULT_POLICY_NUMBER
    suggestions   = [_classify_employee(record, policy_number) for record in records]
    return {**state, "ncci_suggestions": suggestions, "current_step": "classify"}


# ── Data fetching (Python-controlled) ─────────────────────────────────────────

def _classify_employee(record: dict, policy_number: str) -> dict:
    """Fetch all context deterministically, then let the LLM reason and classify."""

    # RAG: primary query from job title
    primary_query = _build_query(record)
    rag_results   = _retriever.query(primary_query, top_k=5)

    # RAG: if first query is a poor match, try a narrower alt query and combine
    if rag_results and rag_results[0]["distance"] > _POOR_MATCH_THRESHOLD:
        alt_query = _build_alt_query(record)
        if alt_query != primary_query:
            alt_results = _retriever.query(alt_query, top_k=3)
            rag_results = rag_results[:3] + alt_results

    rag_context = _retriever.format_context(rag_results)

    # PolicyCenter MCP: prior-period classifications (mock until GW tenant is set)
    prior_data = dispatch_tool("get_prior_classifications", {"policy_number": policy_number})

    # LLM: classify with all context assembled, forced to call finalize_classification
    messages = [
        {"role": "system", "content": _AGENT_SYSTEM},
        {"role": "user",   "content": _build_prompt(record, rag_context, prior_data)},
    ]
    response = _client.chat_with_tools(messages, [_FINALIZE_TOOL], tool_choice="required")
    return _extract_result(response, record["name"])


def _build_query(record: dict) -> str:
    parts = list(record.get("job_description") or [])
    if record.get("org"):
        parts.append(record["org"])
    return " ".join(parts) if parts else record.get("name", "employee")


def _build_alt_query(record: dict) -> str:
    """Job description words only, without the employer name."""
    parts = list(record.get("job_description") or [])
    return " ".join(parts) if parts else _build_query(record)


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
        f"Employee:       {record['name']}\n"
        f"Job title/role: {job_desc}\n"
        f"Employer:       {record.get('org') or 'Unknown'}\n"
        f"Reported wages: {record.get('wages') or 'Not specified'}\n"
        f"{prior_section}\n"
        f"NCCI Manual Excerpts:\n{rag_context}\n\n"
        "Call finalize_classification with the most applicable NCCI code."
    )


# ── Result extraction ─────────────────────────────────────────────────────────

def _extract_result(response, employee: str) -> dict:
    msg = response.choices[0].message

    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name == "finalize_classification":
                args = json.loads(tc.function.arguments)
                return {
                    "employee":       employee,
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
        "employee":       employee,
        "ncci_code":      code.group(1) if code else None,
        "classification": classification.group(1).strip() if classification else "",
        "rationale":      rationale.group(1).strip() if rationale else content.strip()[:200],
        "confidence":     confidence.group(1).upper() if confidence else "LOW",
    }
