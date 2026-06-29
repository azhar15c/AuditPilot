"""
PolicyCenter MCP Server — Guidewire PolicyCenter integration for AuditPilot.

Architecture
------------
This module exposes three tools that Groq (via function calling) can invoke
during the audit workflow:

  get_policy_details(policy_number)     → employer info, coverage period, estimated payroll
  get_prior_classifications(policy_number) → NCCI codes used in previous audits
  push_audit_results(policy_number, worksheet) → write completed worksheet back to PC

How it connects
---------------
Groq supports OpenAI-compatible tool/function calling. Each tool below is
defined as a JSON schema in POLICYCENTER_TOOLS and called via:

    client.chat.completions.create(model=..., messages=..., tools=POLICYCENTER_TOOLS)

The LLM decides when to call a tool; the result is fed back as a tool message.

Current state
-------------
All three tools return MOCK data. To connect a real Guidewire instance:

  1. Set these variables in .env:
       GW_PC_BASE_URL=https://your-tenant.guidewire.net/pc/rest/v1
       GW_PC_USERNAME=your_api_user
       GW_PC_PASSWORD=your_api_password

  2. Replace the _mock_* functions with the real REST calls marked below.
     Guidewire PolicyCenter REST API reference:
     https://docs.guidewire.com/cloud/pc/202310/cloudapibf/CloudAPIBF/index.html

Guidewire sandbox access
------------------------
Options for getting a sandbox environment:
  • Guidewire Education tenant (ACE / training access)
  • Guidewire Technology Partner program (marketplace.guidewire.com)
  • Your employer's non-production PolicyCenter environment
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

_PC_BASE_URL = os.getenv("GW_PC_BASE_URL", "")
_PC_USERNAME = os.getenv("GW_PC_USERNAME", "")
_PC_PASSWORD = os.getenv("GW_PC_PASSWORD", "")

_IS_MOCK = not all([_PC_BASE_URL, _PC_USERNAME, _PC_PASSWORD])


# ── Tool schemas (passed to Groq as the `tools` parameter) ───────────────────

POLICYCENTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_policy_details",
            "description": (
                "Retrieve policyholder details from Guidewire PolicyCenter for a given "
                "policy number. Returns employer name, FEIN, address, policy period, "
                "estimated payroll, and workers' compensation coverage state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_number": {
                        "type": "string",
                        "description": "The WC policy number, e.g. WC-2024-08841",
                    }
                },
                "required": ["policy_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_prior_classifications",
            "description": (
                "Retrieve the NCCI class codes used in the prior policy period's audit "
                "for this policyholder. Useful for detecting reclassification changes "
                "and consistency checking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_number": {
                        "type": "string",
                        "description": "The WC policy number",
                    }
                },
                "required": ["policy_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "push_audit_results",
            "description": (
                "Write the completed audit worksheet back to the PolicyCenter policy "
                "record as an activity note. Also updates the audited payroll figures "
                "and NCCI codes on the policy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_number": {
                        "type": "string",
                        "description": "The WC policy number",
                    },
                    "audited_payroll": {
                        "type": "number",
                        "description": "Total audited payroll amount in dollars",
                    },
                    "classifications": {
                        "type": "array",
                        "description": "List of employee classification results",
                        "items": {
                            "type": "object",
                            "properties": {
                                "employee_name": {"type": "string"},
                                "ncci_code":     {"type": "string"},
                                "classification":{"type": "string"},
                                "payroll":       {"type": "string"},
                                "confidence":    {"type": "string"},
                            },
                        },
                    },
                    "worksheet_text": {
                        "type": "string",
                        "description": "Full audit worksheet narrative to attach as an activity note",
                    },
                },
                "required": ["policy_number", "audited_payroll", "classifications", "worksheet_text"],
            },
        },
    },
]


# ── Tool dispatcher ───────────────────────────────────────────────────────────

def dispatch_tool(tool_name: str, arguments: dict) -> dict:
    """
    Called by the Groq tool-calling loop when the LLM invokes a tool.
    Routes to mock or real implementation based on whether GW credentials are set.
    """
    if tool_name == "get_policy_details":
        return (
            _real_get_policy_details(**arguments)
            if not _IS_MOCK
            else _mock_get_policy_details(**arguments)
        )
    if tool_name == "get_prior_classifications":
        return (
            _real_get_prior_classifications(**arguments)
            if not _IS_MOCK
            else _mock_get_prior_classifications(**arguments)
        )
    if tool_name == "push_audit_results":
        return (
            _real_push_audit_results(**arguments)
            if not _IS_MOCK
            else _mock_push_audit_results(**arguments)
        )
    return {"error": f"Unknown tool: {tool_name}"}


# ── Mock implementations (active when GW credentials are not set) ─────────────

def _mock_get_policy_details(policy_number: str) -> dict:
    return {
        "source": "MOCK — connect GW_PC_BASE_URL in .env for live data",
        "policy_number": policy_number,
        "employer_name": "Summit Builders LLC",
        "fein": "45-2837164",
        "address": "4821 Oak Creek Blvd, Suite 100, Austin, TX 78744",
        "policy_period": "01/01/2024 – 12/31/2024",
        "estimated_payroll": 209000.00,
        "coverage_state": "TX",
        "carrier": "Texas Mutual Insurance Co.",
        "status": "In Force",
    }


def _mock_get_prior_classifications(policy_number: str) -> dict:
    return {
        "source": "MOCK",
        "policy_number": policy_number,
        "prior_period": "01/01/2023 – 12/31/2023",
        "classifications": [
            {"ncci_code": "5551", "classification": "ROOFING - ALL KINDS & DRIVERS",     "audited_payroll": 71200.00},
            {"ncci_code": "5190", "classification": "ELECTRICAL WIRING - WITHIN BLDGS",  "audited_payroll": 84500.00},
            {"ncci_code": "8810", "classification": "CLERICAL OFFICE EMPLOYEES NOC",      "audited_payroll": 46800.00},
        ],
        "total_audited_payroll": 202500.00,
        "experience_mod": 0.94,
    }


def _mock_push_audit_results(
    policy_number: str,
    audited_payroll: float,
    classifications: list,
    worksheet_text: str,
) -> dict:
    return {
        "source": "MOCK",
        "status": "success",
        "policy_number": policy_number,
        "activity_id": "ACT-MOCK-20240401-001",
        "message": (
            f"MOCK: Audit results for {policy_number} would be written to PolicyCenter. "
            f"Audited payroll: ${audited_payroll:,.2f}. "
            f"{len(classifications)} classification(s) recorded."
        ),
    }


# ── Real implementations (active when GW_PC_BASE_URL is set) ─────────────────
# Replace the _mock_* functions above with these once you have a Guidewire tenant.
# Guidewire REST API docs: https://docs.guidewire.com/cloud/pc/202310/cloudapibf/

def _gw_client() -> httpx.Client:
    return httpx.Client(
        base_url=_PC_BASE_URL,
        auth=(_PC_USERNAME, _PC_PASSWORD),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=30.0,
    )


def _real_get_policy_details(policy_number: str) -> dict:
    """
    GET /policies?filter=policyNumber={policy_number}
    Returns the first matching policy's AccountHolder + Period details.
    """
    with _gw_client() as client:
        resp = client.get("/policies", params={"filter": f"policyNumber={policy_number}"})
        resp.raise_for_status()
        data = resp.json()
    # TODO: map Guidewire response fields to the dict structure above
    return data


def _real_get_prior_classifications(policy_number: str) -> dict:
    """
    GET /policies/{id}/policyperiods — find the prior period and its WC lines.
    """
    with _gw_client() as client:
        # Step 1: resolve policy ID
        resp = client.get("/policies", params={"filter": f"policyNumber={policy_number}"})
        resp.raise_for_status()
        policy_id = resp.json()["data"][0]["id"]
        # Step 2: fetch periods
        resp = client.get(f"/policies/{policy_id}/policyperiods")
        resp.raise_for_status()
        data = resp.json()
    # TODO: extract WC class codes from the prior period's WorkersComp line
    return data


def _real_push_audit_results(
    policy_number: str,
    audited_payroll: float,
    classifications: list,
    worksheet_text: str,
) -> dict:
    """
    POST /policies/{id}/activities — attach worksheet as an activity note.
    PATCH /policies/{id}/policyperiods/{period_id}/wclines — update class codes + payroll.
    """
    with _gw_client() as client:
        resp = client.get("/policies", params={"filter": f"policyNumber={policy_number}"})
        resp.raise_for_status()
        policy_id = resp.json()["data"][0]["id"]

        activity_payload = {
            "data": {
                "attributes": {
                    "subject":   f"AuditPilot — Premium Audit Worksheet ({policy_number})",
                    "body":      worksheet_text,
                    "activityPattern": {"id": "audit_complete"},
                }
            }
        }
        resp = client.post(f"/policies/{policy_id}/activities", json=activity_payload)
        resp.raise_for_status()
        activity_id = resp.json()["data"]["id"]

    # TODO: PATCH WC line class codes + audited payroll figures
    return {"status": "success", "policy_number": policy_number, "activity_id": activity_id}
