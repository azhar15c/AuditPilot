import re
from agents.state import AuditState
from models.hf_client import HFClient

_client = HFClient()

_EXTRACT_SYSTEM = (
    "You are a payroll data extraction specialist. "
    "Extract all employee records from the payroll document provided. "
    "Respond with one employee per line in this exact format:\n"
    "NAME: <full name> | JOB: <job title> | WAGES: <total gross pay amount>\n"
    "Include only employees — not subcontractors, vendors, or totals rows."
)


def extraction_agent(state: AuditState) -> AuditState:
    if state.get("error"):
        return state

    raw_text = state["raw_text"]

    # BERT NER — entity list for supplementary pipeline context
    entities = _client.extract_entities(raw_text)

    # LLM extraction — structured records with wages (BERT has no MONEY label)
    employee_records = _llm_extract_records(raw_text)

    # Fallback: NER-based records if LLM returns nothing
    if not employee_records:
        employee_records = _records_from_ner(_group_by_label(entities))

    return {
        **state,
        "entities": entities,
        "employee_records": employee_records,
        "current_step": "extraction",
    }


def _llm_extract_records(raw_text: str) -> list[dict]:
    prompt = (
        "Extract all employee records from this payroll document. "
        "List only employees, not subcontractors or totals:\n\n"
        f"{raw_text[:4000]}"
    )
    response = _client.generate(prompt=prompt, system=_EXTRACT_SYSTEM)
    records = []
    for line in response.strip().split("\n"):
        name_m = re.search(r'NAME:\s*(.+?)\s*\|', line, re.IGNORECASE)
        job_m  = re.search(r'JOB:\s*(.+?)\s*(?:\||$)', line, re.IGNORECASE)
        wage_m = re.search(r'WAGES:\s*(.+?)(?:\s*\||$)', line, re.IGNORECASE)
        if name_m:
            records.append({
                "employee_id":     f"EMP-{len(records):03d}",
                "name":            name_m.group(1).strip(),
                "wages":           wage_m.group(1).strip() if wage_m else None,
                "org":             None,
                "dates":           [],
                "job_description": [job_m.group(1).strip()] if job_m else [],
            })
    return records


def _group_by_label(entities: list[dict]) -> dict:
    groups: dict = {"PER": [], "ORG": [], "DATE": [], "MISC": []}
    for e in entities:
        label = e.get("label", "MISC")
        groups.setdefault(label, []).append(e["entity"])
    return groups


def _records_from_ner(grouped: dict) -> list[dict]:
    """Fallback when LLM extraction fails — wages will be None."""
    persons = grouped.get("PER", [])
    orgs    = grouped.get("ORG", [])
    dates   = grouped.get("DATE", [])
    misc    = grouped.get("MISC", [])
    return [
        {
            "employee_id":     f"EMP-{i:03d}",
            "name":            name,
            "wages":           None,
            "org":             orgs[0] if orgs else None,
            "dates":           dates,
            "job_description": misc,
        }
        for i, name in enumerate(persons)
    ]
