"""Demo-safe mode: replaces live Groq/HF calls with realistic canned
responses so the sample audit works reliably regardless of external API
quota state (Groq's free tier daily/per-minute limits proved too fragile
to depend on for a live interview demo — see README's "Groq Rate Limits"
section for the full story).

Activated by setting AUDITPILOT_DEMO_MODE=1 in the environment (see .env).
Only patches the model-calling clients — the actual pipeline logic
(extraction parsing, the retrieval agent's loop, classification/critic
tool-call handling, aggregate's fault isolation, report assembly) all runs
for real. Only the external network calls are replaced.

The fixture content below is not invented — it's built from real captured
output of successful live runs against data/sample_payroll_register.pdf
(exact employee records, exact wages, exact classification rationale text
for Michael Torres and James Wright). Sarah Chen's classification never
completed live (repeatedly hit Groq's rate limit before a result came
back), so hers is constructed from domain knowledge: NCCI 8810 (Clerical
Office Employees NOC) is the standard code for an internal office-manager
role — see classification_agent.py's own system prompt, which already
distinguishes 8810 (office staff) from 8742 (outside sales) for exactly
this kind of judgment call.
"""

import json
import os
from contextlib import ExitStack
from unittest.mock import patch

DEMO_MODE = os.getenv("AUDITPILOT_DEMO_MODE", "").strip().lower() in ("1", "true", "yes")


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name, arguments):
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, tool_calls=None, content=None):
        self.tool_calls = tool_calls
        self.content = content


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


def _tool_response(name: str, **kwargs) -> _FakeResponse:
    return _FakeResponse(_FakeMessage(tool_calls=[_FakeToolCall(name, json.dumps(kwargs))]))


def _text_response(content: str) -> _FakeResponse:
    return _FakeResponse(_FakeMessage(content=content))


# ── Real extraction output (captured from a real run against the sample PDF) ──

_EXTRACTION_TEXT = (
    "NAME: Michael A. Torres | JOB: Roofer | WAGES: $18,500.00\n"
    "NAME: Sarah L. Chen | JOB: Office Manager | WAGES: $12,000.00\n"
    "NAME: James K. Wright | JOB: Electrician | WAGES: $21,750.00"
)

# ── Per-employee RAG citations, styled like real NCCIRetriever.query() output ──

_CITATIONS = {
    "Roofer": [
        {"id": "demo-roof-1", "chunk_text": "5551 ROOFING - ALL KINDS & DRIVERS. Applies to employees engaged in the installation, repair, or replacement of roofing materials on pitched or flat roofs, including shingles, tile, metal, and built-up roofing systems.", "distance": 0.18, "metadata": {"source": "tx_wc_basic_manual.pdf"}},
        {"id": "demo-roof-2", "chunk_text": "Code 5551 includes drivers of vehicles used to transport roofing materials and crew between job sites. Does not apply to roofing material sales or estimating performed away from the job site.", "distance": 0.24, "metadata": {"source": "tx_wc_basic_manual.pdf"}},
    ],
    "Office Manager": [
        {"id": "demo-clerical-1", "chunk_text": "8810 CLERICAL OFFICE EMPLOYEES NOC. Applies to employees whose duties are confined to keeping books or accounts, general office work, correspondence, and record-keeping performed in office areas physically separated from operative hazards.", "distance": 0.21, "metadata": {"source": "tx_wc_basic_manual.pdf"}},
        {"id": "demo-clerical-2", "chunk_text": "8742 OUTSIDE SALES PERSONNEL, COLLECTORS. Applies to employees principally engaged in sales or collection duties away from the employer's premises. Code 8810 governs when duties are confined to an office setting with no regular outside sales component.", "distance": 0.29, "metadata": {"source": "tx_wc_alpha_index.pdf"}},
    ],
    "Electrician": [
        {"id": "demo-elec-1", "chunk_text": "5190 ELECTRICAL WIRING - WITHIN BUILDINGS. Applies to employees engaged in the installation, maintenance, or repair of electrical wiring, fixtures, and equipment within buildings or structures.", "distance": 0.16, "metadata": {"source": "tx_wc_basic_manual.pdf"}},
        {"id": "demo-elec-2", "chunk_text": "Code 5190 covers work performed entirely within building envelopes. Outdoor electrical line work is classified separately under utility-specific codes.", "distance": 0.27, "metadata": {"source": "tx_wc_basic_manual.pdf"}},
    ],
}

# ── Real classification results (Roofer + Electrician captured from a real
#    successful run; Office Manager constructed per the module docstring) ──

_CLASSIFICATIONS = {
    "Roofer": {
        "ncci_code": "5551", "classification": "ROOFING - ALL KINDS & DRIVERS",
        "rationale": "The employee's job title is Roofer, and the NCCI manual excerpts explicitly mention 'ROOFING - ALL KINDS & DRIVERS' with the code 5551.",
        "confidence": "HIGH",
    },
    "Office Manager": {
        "ncci_code": "8810", "classification": "CLERICAL OFFICE EMPLOYEES NOC",
        "rationale": "The employee's job title is Office Manager and duties are confined to general office work; the excerpts distinguish 8810 (office-confined clerical work) from 8742 (outside sales), and nothing indicates an outside-sales component.",
        "confidence": "HIGH",
    },
    "Electrician": {
        "ncci_code": "5190", "classification": "ELECTRICAL WIRING - WITHIN BLDGS",
        "rationale": "The job title of the employee is Electrician, which directly matches the NCCI class code 5190 for ELECTRICAL WIRING - WITHIN BLDGS as described in the provided NCCI manual excerpts.",
        "confidence": "HIGH",
    },
}

_REPORT_TEXT = """# Workers' Compensation Premium Audit Worksheet
**DRAFT — Requires Auditor Review Before Submission**

---

## 1. Policyholder Information
| Field | Details |
|---|---|
| Company | SUMMIT BUILDERS LLC |
| Audit Period | January 1, 2024 – March 31, 2024 |
| Policy Number | WC-2024-08841 |
| Report Prepared | [Current Date] |
| Total Audited Payroll | $52,250.00 |

---

## 2. Employee Classification Summary
| Employee | Job Title | NCCI Code | Classification | Payroll | Confidence |
|---|---|---|---|---|---|
| Michael A. Torres | Roofer | 5551 | ROOFING - ALL KINDS & DRIVERS | $18,500.00 | HIGH |
| Sarah L. Chen | Office Manager | 8810 | CLERICAL OFFICE EMPLOYEES NOC | $12,000.00 | HIGH |
| James K. Wright | Electrician | 5190 | ELECTRICAL WIRING - WITHIN BLDGS | $21,750.00 | HIGH |

---

## 3. Classification Notes
* Michael A. Torres: Job title Roofer matches NCCI code 5551 (ROOFING - ALL KINDS & DRIVERS) directly.
* Sarah L. Chen: Job title Office Manager, duties confined to general office work — classified 8810 (Clerical Office Employees NOC), distinguished from 8742 (outside sales) since no outside-sales duties are indicated.
* James K. Wright: Job title Electrician matches NCCI code 5190 (ELECTRICAL WIRING - WITHIN BLDGS) directly.

---

## 4. Subcontractor & COI Review
| Subcontractor Name | EIN | Amount Paid | COI On File | COI Expiry | Audit Notes |
|---|---|---|---|---|---|
| Lone Star Drywall Inc. | 74-1823045 | $8,400.00 | YES | 12/31/2024 | COI verified on file |
| Rio Grande Plumbing LLC | 74-2945188 | $5,200.00 | NO ⚠ | — | Payments subject to reclassification as insured payroll |

---

## 5. Auditor Action Items
1. Verify all three NCCI codes against the current NCCI Basic Manual for Texas prior to submission.
2. Review and confirm the COI status for Rio Grande Plumbing LLC to determine if their payments should be reclassified as insured payroll.
3. Confirm the total audited payroll of $52,250.00 is accurate and complete for the audit period.

---
*Draft generated by AuditPilot AI · Auditor of record must review and approve before submission.*
"""


def _classify_response(job: str) -> _FakeResponse:
    c = _CLASSIFICATIONS[job]
    return _tool_response("finalize_classification", **c)


def _critic_response_for(job: str) -> _FakeResponse:
    return _tool_response(
        "finalize_critic_review", agrees=True, concern="",
        recommended_action="approve", critic_confidence="HIGH",
    )


def _job_for_prompt(prompt_text: str) -> str:
    """The mocked clients receive the real prompt text built by each node —
    figure out which employee/job it's for so the right fixture is returned,
    same technique used throughout this project's own test suite."""
    for job in _CITATIONS:
        if job in prompt_text:
            return job
    return "Roofer"  # should not happen against the real sample document


def demo_mode_patches() -> ExitStack:
    """Returns an ExitStack of active patches; caller enters it as a context
    manager around a run_workflow() call. Only patches the model-calling
    clients — nothing about the graph, the nodes, or the fault-isolation
    logic is bypassed."""
    stack = ExitStack()

    mock_extract = stack.enter_context(patch("agents.nodes.extraction_agent._client"))
    mock_extract.extract_entities.return_value = []
    mock_extract.generate.return_value = _EXTRACTION_TEXT

    mock_retriever = stack.enter_context(patch("agents.nodes.retrieval_agent._retriever"))
    mock_retrieval_client = stack.enter_context(patch("agents.nodes.retrieval_agent._client"))

    def _query(query_text, top_k=5):
        job = _job_for_prompt(query_text)
        return _CITATIONS[job]

    def _format_context(results):
        if not results:
            return "No relevant NCCI reference material found."
        return "\n\n---\n\n".join(
            f"[Ref {i} | {r['metadata'].get('source', 'unknown')} | distance: {r['distance']}]\n{r['chunk_text']}"
            for i, r in enumerate(results, start=1)
        )

    mock_retriever.query.side_effect = _query
    mock_retriever.format_context.side_effect = _format_context

    def _sufficiency(messages, tools, tool_choice):
        job = _job_for_prompt(messages[-1]["content"])
        return _tool_response("evaluate_retrieval", sufficient=True, reasoning=f"Excerpts explicitly name the {job} classification.", refined_query="")

    mock_retrieval_client.chat_with_tools.side_effect = _sufficiency

    mock_classify_client = stack.enter_context(patch("agents.nodes.classification_agent._client"))

    def _classify(messages, tools, tool_choice):
        job = _job_for_prompt(messages[-1]["content"])
        return _classify_response(job)

    mock_classify_client.chat_with_tools.side_effect = _classify

    mock_critic_client = stack.enter_context(patch("agents.nodes.critic_agent._client"))

    def _critic(messages, tools, tool_choice):
        job = _job_for_prompt(messages[-1]["content"])
        return _critic_response_for(job)

    mock_critic_client.chat_with_tools.side_effect = _critic

    mock_report_client = stack.enter_context(patch("agents.nodes.report_node._client"))
    mock_report_client.generate.return_value = _REPORT_TEXT

    return stack
