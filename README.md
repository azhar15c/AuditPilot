---
title: AuditPilot
emoji: 📋
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# AuditPilot — AI-Powered Workers' Compensation Premium Audit

**Demo:** [huggingface.co/spaces/azhar15c/AuditPilot](https://huggingface.co/spaces/azhar15c/AuditPilot) | **Code:** [github.com/azhar15c/AuditPilot](https://github.com/azhar15c/AuditPilot)

> Automating the most manual, error-prone step in workers' compensation insurance — turning days of payroll document review into a minutes-long AI-assisted workflow.

---

## ✅ Architecture Redesign — Complete on this branch, pending merge to `main`

This branch (`feature/multi-agent-redesign`) migrates AuditPilot from a fixed, linear LangGraph pipeline into a **Supervisor-orchestrated, partially-parallel multi-agent system** with a dedicated Critic/Verifier step. Full rationale, target architecture, data contracts, and rollout plan are in **[docs/MULTI_AGENT_REDESIGN_SPEC.md](docs/MULTI_AGENT_REDESIGN_SPEC.md)**; the concrete staged build plan that was executed against that spec (file-level scope per workstream, merge order, verification commands, and the LangGraph `Send()`-per-branch-identity correction found mid-build) is in **[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)**.

`main` — and the live HF Space demo — stay on the original pipeline (tagged [`v1-pipeline`](../../releases/tag/v1-pipeline)) until this branch is explicitly merged, which is a separate, later, human decision — not something this build performs on its own.

**Why:** the old pipeline (`intake → extract → classify → report`) fused RAG retrieval, a PolicyCenter lookup, and LLM classification into one function, called serially once per employee — no independent verification step before results reached the auditor, and no parallelism despite each employee's classification being logically independent. See the spec's [Background and Motivation](docs/MULTI_AGENT_REDESIGN_SPEC.md#1-background-and-motivation) for the full case.

**Progress:**

| Stage | Workstream | Status |
|---|---|---|
| 0 | Typed agent-handoff schemas (`agents/schemas.py`) + parallel-fanout state reducers (`agents/state.py`) | ✅ Done |
| 1 | **A** — Supervisor node (single-fetch PolicyCenter lookup, per-employee fan-out) | ✅ Done |
| 1 | **B** — Split `classify_node` into `retrieval_agent` (iterative self-evaluating RAG loop) + `classification_agent`; renamed `extract_node` → `extraction_agent` | ✅ Done |
| 1 | **E** — New test harness (`pytest.ini`, tool-boundary tests, fan-out concurrency tests) + fixed 10 pre-existing stale tests | ✅ Done |
| 2 | **C** — Per-employee subgraph wiring (`agents/employee_subgraph.py`) + `aggregate_node` + `workflow.py` rewire (`intake → extraction → supervisor → Send fan-out → employee_pipeline → aggregate → report`) | ✅ Done |
| 2 | **D** — Critic/Verifier agent (blind review — structurally never reads Classification's `rationale`) | ✅ Done |
| 3 | Final integration: Critic spliced into the subgraph, `api/main.py` response fields, `app.py` Critic status column | ✅ Done |

`pytest tests/ -q` currently: **36 passed, 0 skipped, 0 failed.** Verified end-to-end through the real FastAPI `/audit/run` HTTP layer (not just the pipeline function directly) with a real file upload: 2 employees in → 2 classifications → 2 critic assessments (one `✓ Approved`, one `⚠ Flagged: ...` in a deliberately-engineered disagreement case) → 7 audit-trail entries → a rendered report, all with zero duplication. A live run against the real HF/Groq APIs was also attempted and correctly hit HF's free-tier rate limit — confirming the pre-existing error-handling path still works correctly under the new graph, though it means this particular check didn't get a real-API pass in this session. Post-completion, the retrieval loop was further reworked to apply context engineering (accumulate/dedupe/compact/cap instead of overwrite-and-replay) — see [Retrieval, Classification & Critic Agents](#retrieval-classification--critic-agents-v2) below.

---

## What It Does

Every workers' compensation policy requires an annual premium audit. An auditor manually requests payroll records, 941 forms, and certificates of insurance; extracts employee names, wages, and job roles from unstructured PDFs; maps each employee to an NCCI classification code; and produces a draft worksheet for underwriter review.

AuditPilot automates this process. Upload a payroll register PDF and the pipeline:

1. Extracts raw text from the document
2. Uses an LLM to identify employee names, job titles, and gross payroll
3. Retrieves relevant NCCI manual excerpts via semantic search (RAG) — with automatic fallback to a second query if the first match is poor
4. Fetches prior-period NCCI classifications from PolicyCenter (via MCP, mock by default)
5. Suggests a 4-digit NCCI class code + official title per employee using structured tool output
6. Generates a formatted draft audit worksheet
7. Exports a PDF worksheet ready for auditor review and sign-off
8. Pushes results to Guidewire PolicyCenter (scaffold ready — connects when GW credentials are set)

---

## Architecture

### v1 — Original linear pipeline (tagged `v1-pipeline`, live on `main`)

```
┌────────────────────────────────────────────────────────────┐
│                Browser  (localhost:7860)                   │
│    Tab 1: Audit Workspace  │  Tab 2: Audit Report          │
└──────────────────────────┬─────────────────────────────────┘
                           │  POST /audit/run
┌──────────────────────────▼─────────────────────────────────┐
│        FastAPI + Gradio  (single server, app.py)            │
│    Gradio mounted inside FastAPI via mount_gradio_app       │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────--
│             LangGraph StateGraph  (agents/)                 │
│                                                             │
│   intake_node → extract_node → classify_node → report_node  │
│                                    │                        │
│                             [Agent pattern]                 │
│                          dual RAG lookup +                  │
│                          PolicyCenter MCP +                 │
│                          finalize_classification tool       │
└──────┬──────────────────────┬──────────────────────────────-
       │                      │
  ┌────▼────┐          ┌──────▼────────────────────────────┐
  │   HF    │          │  Groq  (llama-3.3-70b-versatile)  │
  │Serverless│          │                                   │
  │         │          │  • extract_node: employee record  │
  │ BERT NER│          │    extraction (name, job, wages)  │
  │  (NER)  │          │  • classify_node: structured      │
  │         │          │    classification via tool call   │
  │   BGE   │          │  • report_node: draft audit       │
  │(embeddings)        │    worksheet in Markdown          │
  └────┬────┘          └───────────────────────────────────┘
       │
  ChromaDB                  ┌──────────────────────────────┐
  (ncci_codes)              │  MCP Layer (mcp_servers/)    │
  435 vectors from          │                              │
  TX WC manuals             │  PolicyCenter tools:         │
                            │  • get_policy_details        │
                            │  • get_prior_classifications │
                            │  • push_audit_results        │
                            │                              │
                            │  Mock mode (default)         │
                            │  Live mode (set GW_PC_*      │
                            │  in .env)                    │
                            └──────────────────────────────┘
```

### v2 — Multi-agent architecture (complete on this branch)

Full detail in [docs/MULTI_AGENT_REDESIGN_SPEC.md § 2](docs/MULTI_AGENT_REDESIGN_SPEC.md#2-target-architecture).

```
                intake_node → extraction_agent → supervisor_node
                                                      │
                                        dispatch_tool("get_prior_classifications")
                                        called ONCE per run (was once per employee)
                                                      │
                                          route_to_employee_fanout()
                                          Send() — one per employee, parallel
                                                      │
                      ┌───────────────────────────────┼───────────────────────────────┐
                      ▼                                ▼                                ▼
           employee_pipeline (subgraph)      employee_pipeline (subgraph)     employee_pipeline (subgraph)
           ┌─────────────────────────┐       ┌─────────────────────────┐      ┌─────────────────────────┐
           │ retrieval_agent          │       │ retrieval_agent          │      │ retrieval_agent          │
           │  iterative sufficiency   │       │  loop, ChromaDB search   │      │  ...                     │
           │  loop (max 4 rounds)     │       │  ↓                       │      │                          │
           │ classification_agent     │       │ classification_agent     │      │ classification_agent     │
           │  forced tool call,       │       │  ...                     │      │                          │
           │  no direct RAG/MCP       │       │                          │      │                          │
           │ critic_agent              │       │ critic_agent              │      │ critic_agent              │
           │  blind review — never    │       │                          │      │                          │
           │  sees the rationale      │       │                          │      │                          │
           └─────────────────────────┘       └─────────────────────────┘      └─────────────────────────┘
                      └───────────────────────────────┬───────────────────────────────┘
                                                        ▼
                                          aggregate_node  →  report_node  →  END
                                    validates fan-out completeness, degrades
                                    gracefully on a single employee's failure
```

Each employee's `retrieval_agent → classification_agent` chain (🆕 `critic_agent` joining in Stage 3) runs as an isolated per-employee **subgraph** invocation (not three top-level graph nodes) — a deliberate fix for a LangGraph mechanic confirmed during this build: a plain `Send()` fan-out only replicates the *immediately targeted* node, so downstream nodes reached via normal edges would otherwise run once globally instead of once per employee. Verified end-to-end: a 2-employee mocked run produces exactly 2 `ncci_suggestions` and 2 `retrieval_outputs` (not 4 — a reducer-doubling bug this build also caught and fixed along the way, see `agents/nodes/report_node.py`'s commit history) and calls `dispatch_tool("get_prior_classifications")` exactly once.

This is the live graph shape on `main` after this branch merges — every box above is wired and tested.

---

## AI Components

| | Model | Provider | Purpose |
|---|---|---|---|
| **NER** | `dslim/bert-base-NER` | HF Serverless | Named entity extraction — persons, orgs, dates |
| **Embeddings** | `BAAI/bge-large-en-v1.5` | HF Serverless | NCCI manual chunk retrieval via ChromaDB |
| **Generation** | `llama-3.3-70b-versatile` | Groq (direct) | Employee record extraction, NCCI classification, report drafting |

**Why RAG over fine-tuning:** NCCI classification rules update annually by state. The knowledge base can be refreshed with `python -m rag.ingest --source new_manual.pdf` — no retraining required.

**Why BERT + LLM:** BERT handles fast entity extraction without burning Groq quota. The 70B model is reserved for reasoning tasks: structured extraction from tabular payroll formats, classification with rationale, and report generation.

**Why the classify agent uses tool calling for output:** Structured output via `finalize_classification` with `tool_choice="required"` gives typed fields (code, title, rationale, confidence) without regex parsing. The LLM cannot hallucinate format — the tool schema enforces it.

---

## Classify Node — Agent Pattern

> **v1 (main / `v1-pipeline`)** — described below. **On this branch**, this single node has been split into `retrieval_agent` + `classification_agent` + `critic_agent` — see [Retrieval, Classification & Critic Agents](#retrieval-classification--critic-agents-v2) below.

`classify_node` uses an agent pattern rather than a fixed pipeline call. For each employee:

1. **RAG — primary query** from job title/description → top-5 NCCI manual chunks
2. **RAG — fallback query** (job description only) if first match distance exceeds threshold — results merged
3. **PolicyCenter MCP** — `get_prior_classifications` fetches codes from the prior audit period for consistency checking
4. **LLM classification** — all context assembled in one prompt; model calls `finalize_classification` (structured tool output)

Python controls the data-fetching sequence. The LLM handles reasoning — it does not decide what to look up. This separation avoids multi-turn tool-calling loops which are unreliable on `llama-3.3-70b-versatile`.

---

## Retrieval, Classification & Critic Agents (v2)

On this branch, the single `classify_node` above is split into three narrowly-scoped agents, each with least-privilege tool access (verified by tool-boundary tests in `tests/test_tool_boundaries.py`):

1. **`retrieval_agent`** (`agents/nodes/retrieval_agent.py`) — replaces the old static primary+fallback RAG query with an **iterative, self-evaluating loop**: query ChromaDB, ask the LLM via one forced tool call (`evaluate_retrieval`) whether the citations collected *so far* answer this employee's classification question, and if not, reformulate the query based on *why* it fell short — up to 4 rounds. If still insufficient at the cap, returns the best citations found with `sufficiency_met: False` rather than silently passing off a weak result as confident. No access to `dispatch_tool`/PolicyCenter.

   **Context engineering, not just "loop and pass more":** the naive version of this loop would either lose earlier rounds' findings (overwrite instead of accumulate) or bloat the prompt with every raw chunk ever seen (replay instead of curate) — this implementation does neither. Results are accumulated and deduped across rounds by stable chunk id (`rag/retriever.py`'s `NCCIRetriever.query()` surfaces it specifically for this); each round's prompt shows only *new* chunks plus a compact scratchpad distilled from prior rounds' own reasoning, not a replay of raw history (*compaction*); and the final accumulated set is capped to the 8 most relevant citations by distance before handoff to `classification_agent`, so a long search doesn't balloon the next agent's prompt either (*boundary curation*). See `tests/test_retrieval_agent.py::TestRetrievalAgentContextEngineering` for the behavior this guarantees.
2. **`classification_agent`** (`agents/nodes/classification_agent.py`) — the forced-tool-call `finalize_classification` step, largely unchanged from v1, except it consumes citations `retrieval_agent` already gathered rather than querying ChromaDB or PolicyCenter itself.
3. **`critic_agent`** (`agents/nodes/critic_agent.py`) — an independent verifier that checks Classification's decision against Retrieval's raw citations *before* the result reaches the human auditor. Deliberately blind-reviewed: `state["classification_output"]["rationale"]` is never read anywhere in the file, only the final decision — reducing the risk of simply agreeing with whatever justification it's shown. If the LLM fails to return a structured verdict, the fallback defaults to `flag_for_review`, never a silent approval.

A **`supervisor_node`** (`agents/nodes/supervisor.py`) runs once per audit, before these three fan out — it fetches PolicyCenter's prior classifications exactly once and shares the result across every employee, fixing a v1 inefficiency where that same lookup ran once per employee.

Full data contracts (`RetrievalOutput`, `ClassificationOutput`, `CriticOutput`, etc.) are in `agents/schemas.py`.

---

## Knowledge Base

The RAG knowledge base is built from two publicly available Texas Department of Insurance documents:

| File | Description | Chunks |
|---|---|---|
| `data/tx_wc_basic_manual.pdf` | Texas Basic Manual of Rules, Classifications & Experience Rating | 353 |
| `data/tx_wc_alpha_index.pdf` | Texas WC Alphabetical Index of Classifications | 82 |

**Total: 435 vectors** stored in ChromaDB collection `ncci_codes`.

---

## Setup

### Prerequisites

- Python 3.9+
- Free [HuggingFace](https://huggingface.co/join) account and API token
- Free [Groq](https://console.groq.com) account and API key

### Install

```bash
git clone https://github.com/yourusername/auditpilot.git
cd auditpilot

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` and fill in the required keys:

```
# Required
HF_TOKEN=hf_your_token_here
GROQ_API_KEY=gsk_your_key_here

# Optional — PolicyCenter integration (leave blank to use mock mode)
GW_PC_BASE_URL=https://your-tenant.guidewire.net/pc/rest/v1
GW_PC_USERNAME=your_api_user
GW_PC_PASSWORD=your_api_password
```

HuggingFace token: **Settings → Access Tokens**. Groq key: **console.groq.com → API Keys**.

When `GW_PC_*` vars are blank, all PolicyCenter calls return realistic mock data — the app is fully functional without a Guidewire tenant.

### Build the NCCI knowledge base

```bash
python -m rag.ingest --source data/tx_wc_basic_manual.pdf
python -m rag.ingest --source data/tx_wc_alpha_index.pdf
```

Run once on first setup, and again whenever you add new documents.

---

## Run

Single command starts the combined FastAPI + Gradio server:

```bash
python app.py
```

| URL | Purpose |
|---|---|
| `http://localhost:7860` | Gradio UI |
| `http://localhost:7860/docs` | FastAPI interactive API docs |
| `http://localhost:7860/audit/health` | Health check endpoint |

---

## Using the App

**Tab 1 — Audit Workspace**

- Upload a payroll register, IRS 941, or COI (PDF or TXT)
- Or click **Run Sample Audit** to try the built-in demo (`sample_payroll_register.pdf` — Summit Builders LLC, Q1 2024)
- Results populate the **Employee Classification Table** with: employee name, job title, suggested NCCI code, official classification title, payroll amount, confidence level, and rationale
- Export the table as CSV

**Tab 2 — Audit Report**

- Click **Generate Audit Report** on Tab 1 to switch here
- The draft worksheet renders as formatted Markdown with: policyholder info, classification summary table, classification notes, subcontractor COI review, and auditor action items
- Download as a formatted PDF with **Download Report PDF**
- **Push to PolicyCenter** — scaffold is wired; connects automatically when `GW_PC_*` credentials are set in `.env`

---

## Sample Document

`data/sample_payroll_register.pdf` is a synthetic ADP-style payroll register included for demonstration:

- **Company:** Summit Builders LLC, Austin TX
- **Period:** Q1 2024 (Jan–Mar)
- **Employees:** Michael Torres (Roofer, $18,500), Sarah Chen (Office Manager, $12,000), James Wright (Electrician, $21,750)
- **Subcontractors:** Lone Star Drywall Inc. (COI valid), Rio Grande Plumbing LLC (**COI missing** — triggers reclassification risk)

This file is synthetic demo data. Not a real employer payroll document.

---

## Tests

```bash
pytest tests/
```

Tests use `unittest.mock` — no real API calls or ChromaDB required.

---

## Project Structure

**This tree reflects the current state of `feature/multi-agent-redesign`** (🆕 new since `v1-pipeline`, ❌ removed — superseded). See `git tag v1-pipeline` for the original file layout.

```
auditpilot/
├── app.py                        # Gradio UI + FastAPI server (single entry point)
├── api/
│   └── main.py                   # FastAPI routes + Gradio mount
├── agents/
│   ├── state.py                  # AuditState TypedDict — now with operator.add reducers for fan-out fields
│   ├── schemas.py                # 🆕 typed agent-handoff contracts (RetrievalOutput, ClassificationOutput, etc.)
│   ├── workflow.py                # 🆕 rewired: intake→extraction→supervisor→Send fan-out→aggregate→report
│   ├── employee_subgraph.py      # 🆕 per-employee retrieval→classification→critic subgraph
│   └── nodes/
│       ├── intake_node.py        # PDF/text extraction via pdfplumber (unchanged)
│       ├── extraction_agent.py   # 🆕 renamed from extract_node.py — now assigns stable employee_id
│       ├── supervisor.py         # 🆕 single-fetch PolicyCenter lookup + per-employee Send() fan-out
│       ├── retrieval_agent.py    # 🆕 iterative self-evaluating RAG loop (replaces static fallback query)
│       ├── classification_agent.py  # 🆕 forced-tool-call classification, no direct RAG/MCP access
│       ├── critic_agent.py       # 🆕 blind-review verifier, wired as the subgraph's third node
│       ├── aggregate_node.py     # 🆕 post-fan-out validation, graceful partial-failure handling
│       └── report_node.py        # Draft audit worksheet generation — join key fixed to employee_id, no longer double-counts reducer fields
├── models/
│   └── hf_client.py              # HFClient: BERT NER, BGE embeddings, Groq generation + tool calling
├── mcp_servers/
│   └── policycenter.py           # PolicyCenter MCP tools (mock + real REST stubs)
├── rag/
│   ├── ingest.py                 # Chunk + embed PDFs into ChromaDB
│   └── retriever.py              # NCCIRetriever.query() — top-k NCCI context
├── data/
│   ├── sample_payroll_register.pdf   # Synthetic demo payroll register
│   ├── tx_wc_basic_manual.pdf        # Texas WC Basic Manual (RAG source)
│   └── tx_wc_alpha_index.pdf         # Texas WC Alphabetical Index (RAG source)
├── docs/
│   ├── MULTI_AGENT_REDESIGN_SPEC.md  # 🆕 full redesign spec — background, target architecture, rollout plan
│   └── IMPLEMENTATION_PLAN.md        # 🆕 the staged build plan executed against that spec
├── tests/                        # 🆕 pytest.ini + tool-boundary/fan-out/retrieval-convergence test suites added
├── .env.example                  # Key template — copy to .env and fill in
├── .gitignore                    # Excludes .env, venv/, chroma_db/
└── requirements.txt
```

`extract_node.py` and `classify_node.py` (v1) are ❌ removed on this branch — their logic now lives in `extraction_agent.py` and `retrieval_agent.py` + `classification_agent.py` respectively.

---

## PolicyCenter MCP Integration

`mcp_servers/policycenter.py` exposes three tools callable by the classify agent and the Push to PolicyCenter button:

| Tool | Description |
|---|---|
| `get_policy_details` | Employer name, FEIN, address, coverage period, estimated payroll |
| `get_prior_classifications` | NCCI codes + audited payroll from the prior policy period |
| `push_audit_results` | Write completed worksheet + class codes back to the PC policy record |

**Mock mode (default):** All three return realistic Summit Builders LLC data with no external calls.

**Live mode:** Set `GW_PC_BASE_URL`, `GW_PC_USERNAME`, `GW_PC_PASSWORD` in `.env`. The real REST implementations use the [Guidewire PolicyCenter Cloud API](https://docs.guidewire.com/cloud/pc/202310/cloudapibf/CloudAPIBF/index.html).

**Getting a sandbox:**
- Guidewire Education platform (ACE/training access)
- Guidewire Technology Partner program (marketplace.guidewire.com)
- Your employer's non-production PolicyCenter environment

---

## Limitations

- **Text-layer PDFs only** — `pdfplumber` reads embedded text. Scanned or photographed PDFs require OCR pre-processing before upload.
- **English only** — NER model and NCCI knowledge base are English-language.
- **Texas classification base** — The included knowledge base covers Texas WC rules. Add your state's manual PDFs and re-run `rag/ingest.py` for other states.
- **Draft only** — All AI outputs require auditor review before submission. Classification suggestions are decision support, not final determinations.
- **PolicyCenter push is placeholder** — The button is wired; real write-back requires a GW tenant and completing the `_real_push_audit_results` field mapping in `mcp_servers/policycenter.py`.

---

## Roadmap

- **PolicyCenter live write-back** — Complete `_real_push_audit_results` field mapping once GW sandbox is available; wire policy number from document header into `AuditState.policy_number`
- **Multi-state support** — Ingest additional state WC manuals into ChromaDB with state-tagged metadata for jurisdiction-specific retrieval
- **COI date extraction** — Automated expiry parsing from uploaded COI documents to flag lapsed certificates
- **Subcontractor reclassification calculator** — Premium impact estimate when a COI is missing
- **OCR pipeline** — Tesseract or a document-question-answering model for scanned PDFs
- **Audit dispute workflow** — Human-in-the-loop queue for LOW-confidence classifications requiring auditor override
- **report_node as agent** — Once PolicyCenter is live, report_node should call `get_policy_details` to pre-populate employer info rather than parsing it from raw text

---

## Domain Context

- **Premium audit** — mandatory end-of-policy review verifying actual payroll matches the estimate used to set the initial premium. If actual > estimated, policyholder owes additional premium; if actual < estimated, the carrier issues a return premium.
- **NCCI class codes** — 4-digit codes assigned to employee job duties that carry different loss cost rates based on occupational risk. Misclassification is the most common source of audit disputes.
- **COI (Certificate of Insurance)** — proof that a subcontractor carries their own WC coverage. Without a valid COI, the carrier may treat subcontractor payments as the policyholder's own payroll, significantly increasing premium.

---

## About

Built by **Azhar** — Guidewire ACE certified developer (PolicyCenter, ClaimCenter), IBM AI certified, 15 years P&C insurance systems experience, transitioning into AI engineering with a focus on insurance domain applications.

**How this was built:** Domain knowledge of WC audit workflows came directly from production insurance systems experience. AI engineering skills — LangGraph, RAG pipelines, MCP servers, HuggingFace Inference, Groq tool calling — developed through hands-on implementation using [Claude Code](https://claude.ai/code) as a pair programming tool.

---

## License

MIT — see [LICENSE](LICENSE) for details.

> **Note on data:** This repository contains no real policyholder data. All sample documents are synthetic and created for demonstration purposes only. Never commit real payroll records or PII to a public repository.
