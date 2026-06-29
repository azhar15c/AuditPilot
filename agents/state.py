from typing import Optional, TypedDict


class AuditState(TypedDict):
    uploaded_file_path: str
    raw_text: str
    entities: list[dict]
    employee_records: list[dict]
    ncci_suggestions: list[dict]
    completeness_flags: dict
    audit_report: str
    error: Optional[str]
    current_step: str
    policy_number: Optional[str]  # WC policy number; used by classify agent for prior-classification lookup
