import operator
from typing import Annotated, Optional, TypedDict


class AuditState(TypedDict):
    uploaded_file_path: str
    raw_text: str
    entities: list[dict]
    employee_records: list[dict]  # written once by extraction_agent, pre-fan-out
    policy_number: Optional[str]  # WC policy number; used by supervisor for prior-classification lookup
    prior_classifications: dict  # written once by supervisor, pre-fan-out; shared across all employee branches
    retrieval_outputs: Annotated[list[dict], operator.add]  # one item appended per parallel employee branch
    ncci_suggestions: Annotated[list[dict], operator.add]  # one item appended per parallel employee branch
    critic_assessments: Annotated[list[dict], operator.add]  # one item appended per parallel employee branch
    audit_trail: Annotated[list[dict], operator.add]  # AgentStep entries, appended by every node
    completeness_flags: dict
    audit_report: str
    error: Optional[str]
    current_step: str
