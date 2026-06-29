# AuditPilot — AI-Powered Workers' Compensation Premium Audit

[![Python](https://img.shields.io/badge/python-3.9+-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-agentic%20workflow-green)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.3%2070B-orange)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> Automating the most manual, error-prone step in workers' compensation insurance — turning days of payroll document review into a minutes-long AI-assisted workflow.

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

`classify_node` uses an agent pattern rather than a fixed pipeline call. For each employee:

1. **RAG — primary query** from job title/description → top-5 NCCI manual chunks
2. **RAG — fallback query** (job description only) if first match distance exceeds threshold — results merged
3. **PolicyCenter MCP** — `get_prior_classifications` fetches codes from the prior audit period for consistency checking
4. **LLM classification** — all context assembled in one prompt; model calls `finalize_classification` (structured tool output)

Python controls the data-fetching sequence. The LLM handles reasoning — it does not decide what to look up. This separation avoids multi-turn tool-calling loops which are unreliable on `llama-3.3-70b-versatile`.

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

```
auditpilot/
├── app.py                    # Gradio UI + FastAPI server (single entry point)
├── api/
│   └── main.py               # FastAPI routes + Gradio mount
├── agents/
│   ├── state.py              # AuditState TypedDict (incl. policy_number field)
│   ├── workflow.py           # LangGraph StateGraph wiring
│   └── nodes/
│       ├── intake_node.py    # PDF/text extraction via pdfplumber
│       ├── extract_node.py   # LLM-based employee record extraction
│       ├── classify_node.py  # Agent: dual RAG + MCP + finalize_classification tool
│       └── report_node.py    # Draft audit worksheet generation
├── models/
│   └── hf_client.py          # HFClient: BERT NER, BGE embeddings, Groq generation + tool calling
├── mcp_servers/
│   └── policycenter.py       # PolicyCenter MCP tools (mock + real REST stubs)
├── rag/
│   ├── ingest.py             # Chunk + embed PDFs into ChromaDB
│   └── retriever.py          # NCCIRetriever.query() — top-k NCCI context
├── data/
│   ├── sample_payroll_register.pdf   # Synthetic demo payroll register
│   ├── tx_wc_basic_manual.pdf        # Texas WC Basic Manual (RAG source)
│   └── tx_wc_alpha_index.pdf         # Texas WC Alphabetical Index (RAG source)
├── tests/
├── .env.example              # Key template — copy to .env and fill in
├── .gitignore                # Excludes .env, venv/, chroma_db/
└── requirements.txt
```

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
