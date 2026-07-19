# AuditPilot — Claude Code Project Instructions

## What This Project Does
AuditPilot is an AI-assisted workers' compensation **premium audit** platform. It automates the manual steps auditors perform when reviewing policyholder documents — ingesting 941 forms, payroll registers, and certificates of insurance (COIs), extracting structured fields, suggesting NCCI employee classification codes via RAG, and drafting an audit worksheet.

This is **not** a claims processing tool. The users are WC premium auditors (often third-party contractors), not claims adjusters.

## Entry Points
- `app.py` — starts the combined FastAPI + Gradio server (`python app.py`, localhost:7860)
- Gradio is mounted inside FastAPI via `mount_gradio_app` — there is only one process, one port
- `api/main.py` — FastAPI routes (accessible at localhost:7860/docs, not 8000)

## Pipeline Order
```
intake_node → extract_node → classify_node → report_node
```
Each node reads from and writes to `AuditState` (defined in `agents/state.py`).

`classify_node` runs as an **agent** (see below) — the others are standard nodes.

## Key Files

| File | Purpose |
|------|---------|
| `agents/state.py` | `AuditState` TypedDict — all fields including `policy_number: Optional[str]` |
| `agents/workflow.py` | LangGraph `StateGraph` wiring; exports `run_workflow(file_path)` |
| `agents/nodes/intake_node.py` | Parse uploaded audit documents via pdfplumber |
| `agents/nodes/extract_node.py` | LLM-based structured extraction: name, job title, wages per employee |
| `agents/nodes/classify_node.py` | **Agent**: dual RAG + PolicyCenter MCP + `finalize_classification` tool |
| `agents/nodes/report_node.py` | Draft audit worksheet via Groq LLM |
| `models/hf_client.py` | `HFClient`: BERT NER, BGE embeddings, Groq generation + `chat_with_tools()` |
| `rag/ingest.py` | Chunk + embed NCCI manuals into ChromaDB |
| `rag/retriever.py` | `NCCIRetriever.query()` — top-k NCCI context chunks |
| `mcp_servers/policycenter.py` | PolicyCenter MCP tools: mock + real Guidewire REST stubs |
| `app.py` | Gradio UI (Blocks layout, two tabs, CSS theme) + FastAPI mount |

## Models and Clients

All model IDs and client setup are in `models/hf_client.py`:

```
NER_MODEL      = "dslim/bert-base-NER"          # HF Serverless token-classification
EMBED_MODEL    = "BAAI/bge-large-en-v1.5"       # HF Serverless feature-extraction
GENERATE_MODEL = "llama-3.3-70b-versatile"       # Groq chat completions
```

`HFClient` methods:
- `extract_entities(text)` → BERT NER via HF Serverless
- `embed_text(texts)` → BGE embeddings via HF Serverless
- `generate(prompt, system)` → Groq chat, returns plain string
- `chat_with_tools(messages, tools, tool_choice="auto")` → Groq chat with tool schemas, returns raw response object

**Do not** add separate model ID constants (`CLASSIFY_MODEL`, `REPORT_MODEL`) — all three generation tasks (extract, classify, report) use the same `GENERATE_MODEL` via Groq.

## classify_node — Agent Pattern

This node does NOT use the full ReAct multi-turn tool-calling loop. `llama-3.3-70b-versatile` generates malformed tool call syntax during multi-turn loops (`[]` artifact in function names). The working pattern:

1. **Python fetches data** (deterministic, no LLM involvement):
   - RAG primary query (top-5 chunks from job title)
   - RAG fallback query (job description only) if first match distance > 1.2 threshold — results merged
   - PolicyCenter `get_prior_classifications` via `dispatch_tool()` in `mcp_servers/policycenter.py`

2. **LLM classifies** with all context in one prompt. Single tool available: `finalize_classification`. Called with `tool_choice="required"` — forces structured output, no plain text response.

Output shape (appended to `state["ncci_suggestions"]`):
```python
{"employee": str, "ncci_code": str, "classification": str, "rationale": str, "confidence": "HIGH"|"MEDIUM"|"LOW"}
```

## MCP Server — PolicyCenter

`mcp_servers/policycenter.py` provides:
- `POLICYCENTER_TOOLS` — list of three Groq-compatible tool schemas
- `dispatch_tool(tool_name, arguments)` — routes to mock or real implementation

**Mock mode** (default, no env vars needed): Returns Summit Builders LLC fixture data.
**Live mode**: Set `GW_PC_BASE_URL`, `GW_PC_USERNAME`, `GW_PC_PASSWORD` in `.env`.

Real REST implementations are stubbed in `_real_*` functions with `TODO` comments for field mapping.

## AuditState Fields

```python
class AuditState(TypedDict):
    uploaded_file_path: str
    raw_text: str
    entities: list[dict]           # BERT NER output
    employee_records: list[dict]   # LLM-extracted: name, job_description, wages, org
    ncci_suggestions: list[dict]   # classify_node output: per-employee code + rationale
    completeness_flags: dict
    audit_report: str
    error: Optional[str]
    current_step: str
    policy_number: Optional[str]   # WC policy number for PolicyCenter lookups; defaults to "WC-DEMO-001"
```

## Environment Variables

```
HF_TOKEN=...           # Required — HF Serverless NER + embeddings
GROQ_API_KEY=...       # Required — Groq LLM generation
GW_PC_BASE_URL=...     # Optional — Guidewire PolicyCenter base URL
GW_PC_USERNAME=...     # Optional — PolicyCenter API user
GW_PC_PASSWORD=...     # Optional — PolicyCenter API password
```

Never commit `.env`. It is listed in `.gitignore`.

## RAG Knowledge Base

Collection name: `"ncci_codes"` (persisted to `./chroma_db/`).

Source PDFs in `data/`: `tx_wc_basic_manual.pdf` (353 chunks) + `tx_wc_alpha_index.pdf` (82 chunks) = 435 vectors total.

Rebuild: `python -m rag.ingest --source data/<file>.pdf`

Do not reference collection name `"ncci_docs"` — that is the old name, no longer used.

## UI Structure (app.py)

Three-tab Gradio Blocks layout (on `feature/multi-agent-redesign`; `main` still has the original two-tab v1 layout until that branch merges):
- **Tab 1 — Audit Workspace**: file upload, sample button, Employee Classification Table (`gr.DataFrame`), Export CSV, Generate Audit Report button
- **Tab 2 — Audit Report**: `gr.Markdown` with `.report-panel` CSS class (white background, forced dark text), Download Report PDF button, Push to PolicyCenter button (placeholder — shows `gr.Info` toast until GW credentials wired)
- **Tab 3 — Audit Trail**: `gr.DataFrame` rendering `state["audit_trail"]` — one row per agent invocation (status, agent, employee, duration, summary). `⚠ ERROR` rows are per-employee branch failures caught at the `employee_pipeline_node` boundary (`agents/employee_subgraph.py`) — the rest of the audit completes around them instead of the whole run crashing.

**Note:** this file otherwise still describes the v1 pipeline (`extract_node`/`classify_node`, the two-tab layout, etc.). The multi-agent redesign on `feature/multi-agent-redesign` supersedes most of it — see `docs/IMPLEMENTATION_PLAN.md` and the README's "v2 — Multi-agent architecture" section for the current, as-built shape. A full rewrite of this file to match is a separate task, not done as part of this change.

CSS class `.report-panel` sets `color: #1a202c` explicitly on all child elements — required because `gr.themes.Soft` with `neutral_hue="slate"` otherwise makes text nearly invisible on white.

## Domain Context
- **NCCI class codes** — 4-digit codes used to classify employee job functions for WC rating purposes. The RAG knowledge base contains NCCI manual descriptions to support accurate suggestions.
- **Audit documents** — 941 (employer quarterly tax return), payroll registers, COIs (certificates of insurance for subcontractors).
- **COI verification** — Expired or missing COIs for subcontractors can cause their payments to be reclassified as the policyholder's payroll, raising premiums.
- **Premium audit** — end-of-policy review comparing actual payroll to the estimated payroll used to set the initial premium.

## Tests
```bash
pytest tests/
```
Tests use `unittest.mock` — no real HF API calls, Groq calls, or ChromaDB needed.
