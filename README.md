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

`pytest tests/ -q` currently: **62 passed, 0 skipped, 0 failed.** Verified end-to-end through the real FastAPI `/audit/run` HTTP layer (not just the pipeline function directly) with a real file upload: 2 employees in → 2 classifications → 2 critic assessments (one `✓ Approved`, one `⚠ Flagged: ...` in a deliberately-engineered disagreement case) → 7 audit-trail entries → a rendered report, all with zero duplication.

Post-completion, two things were found running the sample audit live against the real APIs, both since fixed:
- The retrieval loop was reworked to apply context engineering (accumulate/dedupe/compact/cap instead of overwrite-and-replay) — see [Retrieval, Classification & Critic Agents](#retrieval-classification--critic-agents-v2) below.
- A real 3-employee run hit Groq's free-tier per-minute token limit — a direct consequence of the fan-out's own concurrency win (see [Groq rate limits](#groq-rate-limits-and-retrybackoff) below) — now handled with automatic retry/backoff rather than failing the audit.

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

## Groq Rate Limits and Retry/Backoff

Groq's free/on-demand tier enforces **two separate limits**, confirmed directly against the API (`x-ratelimit-*` response headers): 12,000 tokens **per minute**, resetting in single-digit seconds, and 100,000 tokens **per day**, on a **rolling 24-hour window**. Running the sample audit live surfaced both, for different reasons:

- **The per-minute limit** is a direct consequence of the multi-agent redesign's own concurrency win: the v1 pipeline classified employees one at a time in a Python loop, so its Groq calls were naturally spread across the whole run's wall-clock time. The v2 pipeline fans every employee's `retrieval_agent → classification_agent → critic_agent` chain out **concurrently** via `Send()` — that's the point, it's faster — but it also means every employee's LLM calls can land in the same narrow time window instead of being spread out, and without jitter, calls that get rate-limited together tend to retry together, re-triggering the same burst on every round (a "thundering herd").
- **The per-day limit** is unrelated to concurrency — it's a small, absolute budget (100k tokens) that ordinary same-day development and testing (extraction, retrieval's up-to-4-round searches, classification, critic, and report generation, repeated across many manual runs) can exhaust outright. Because the window is rolling, not a fixed midnight reset, the wait Groq reports reflects how long until *earlier that same day's* usage ages out of the last 24 hours — on a day with heavy testing, that observably escalated past an hour, not the few minutes a first read of `Retry-After` might suggest. No retry strategy shortens this; Groq's own reported wait is the real number, and repeatedly retrying against a still-saturated rolling window just keeps reporting a later one.

`models/hf_client.py`'s `HFClient.generate()` and `.chat_with_tools()` — the two methods every node in the graph calls into Groq through — route through `_call_with_retry`, which addresses the per-minute case on two fronts: a `threading.Semaphore` caps how many Groq requests this process has in flight at once (`_MAX_CONCURRENT_GROQ_CALLS`, currently `1` — serializing the actual HTTP calls, though the graph itself still runs employees concurrently; tuned down from `2` after observing live that even 2 concurrent calls produced enough contention on a tightly-budgeted org for some employees to exhaust retries), converting "every employee fires at once and mostly fails" into "calls queue locally and execute at a sustainable rate"; and on a `groq.RateLimitError`, the wait before retrying is jittered ±30% (preferring the response's `Retry-After` header, falling back to parsing Groq's own "try again in Xs" message, then a fixed 2s default) so concurrent callers rate-limited in the same instant desynchronize instead of retrying in lockstep, up to `_MAX_RATE_LIMIT_RETRIES` (currently `6`, raised from `3` for the same reason) before re-raising. For a daily-quota exhaustion, the same mechanism still waits out and retries whatever `Retry-After` says — it's just that the wait is Groq's real, unavoidable reset time rather than something pacing can shorten. See `tests/test_hf_client_retry.py` for the retry-precedence, jitter, concurrency-cap, exhaustion, and non-rate-limit-errors-pass-through behavior this guarantees.

**Even with all of this, live Groq availability turned out to be too fragile to depend on for a real-time demo** — see "Demo-Safe Mode" below for how the sample audit is made to work reliably regardless of live quota state.

**Three more bugs surfaced live-debugging a report that a real audit run "wasn't working" (no error shown, just never finished)** — reproduced directly against the real API rather than mocks, since none of these manifest with mocked responses:
1. **The Groq SDK has its own hidden `max_retries=2`**, with its own uncoordinated backoff, running *inside* every `.create()` call — completely invisible to and stacked underneath `_call_with_retry`. A single rate-limited call could silently retry twice inside the SDK, then three more times in our own wrapper, with none of the SDK's half logged anywhere. Fixed by constructing the client with `max_retries=0`, making `_call_with_retry` the single, fully-visible retry layer.
2. **llama-3.3-70b sometimes emits a stringified `"false"`/`"true"` instead of a JSON boolean** in tool-call arguments (confirmed directly: `evaluate_retrieval`'s `sufficient` and `finalize_critic_review`'s `agrees`). This compounded two ways — Groq's strict schema validation 400s the whole request outright when the declared type is strictly `boolean` and the generation doesn't match, and even on the rare occasion it got through, `bool("false")` is `True` in Python (any non-empty string is truthy), silently inverting the model's actual answer. Fixed: both tool schemas now accept `[boolean, string]`, and a `_coerce_bool` helper in `retrieval_agent.py` and `critic_agent.py` parses the string form correctly.
3. **`max_tokens=1024` on `chat_with_tools` was truncating longer tool-call generations mid-JSON** — confirmed via the raw `failed_generation` field, cut off before the closing brace and `</function>` tag. Groq's parser rejects a truncated tool call as a 400 "Failed to call a function," which isn't a `RateLimitError`, so it was never retried — and retrying wouldn't have helped anyway, since it's the identical malformed request every time. Fixed by raising the budget to 2048.

---

## Demo-Safe Mode

Groq's free tier proved too fragile to depend on for a live demo — not from a code bug, but from the resource itself: the daily quota is scoped to the **organization**, not the API key (a new key from the same account inherits the same exhausted budget), and even a genuinely fresh organization's small per-minute budget got contended enough under this project's own concurrent fan-out that some employees exhausted retries. Rather than keep tuning against a moving, externally-controlled target, `demo_fixtures.py` adds an escape hatch: set `AUDITPILOT_DEMO_MODE=1` in `.env` and every model-calling client (`extraction_agent`, `retrieval_agent`, `classification_agent`, `critic_agent`, `report_node`) gets patched with realistic canned responses before `run_workflow()` is invoked — both in the Gradio UI (`app.py::_run_pipeline`) and the raw API (`api/main.py`'s `/audit/run`).

**What's real and what's canned:** the entire graph — Supervisor's fan-out, the per-employee subgraph, the retrieval agent's loop, classification's and critic's tool-call parsing, `aggregate_node`, report assembly — runs exactly as it does live. Only the network calls to Groq/HF are replaced. The fixture data itself isn't invented: Michael Torres's and James Wright's classifications (5551 Roofing, 5190 Electrical) and rationale text are copied verbatim from a real successful run against the sample document; Sarah Chen's (8810 Clerical Office Employees NOC) never completed live before quota ran out, so it's constructed from the same 8810-vs-8742 distinction `classification_agent.py`'s own system prompt already makes.

**How to tell it's active:** there's no UI badge — check `AUDITPILOT_DEMO_MODE` in `.env`, or look for `status: "ok"` audit-trail rows with suspiciously round durations and the exact fixture rationale text quoted above. Leave `AUDITPILOT_DEMO_MODE` unset (the default) for genuine live runs.

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

**Tab 3 — Audit Trail**

- One row per agent invocation from the run — Supervisor's single PolicyCenter fetch, then each employee's Retrieval, Classification, and Critic steps, with agent name, employee, duration, and a summary
- `⚠ ERROR` rows are agent failures caught for that one employee — `agents/employee_subgraph.py`'s `employee_pipeline_node` catches any unhandled exception from a branch (e.g. a Groq call that exhausted its retries) at the subgraph-invocation boundary, so it contributes nothing instead of crashing the whole run; `aggregate_node` then marks that employee `"NOT CLASSIFIED — retry required"` with the real failure reason surfaced in both the classification table's rationale and this tab's summary column, and the rest of the audit completes normally around it
- This closes a real gap found by direct testing, not by inspection: before this fix, one employee's unhandled exception took down the entire run, including employees whose branches had already succeeded — confirmed by reproduction, then fixed, then re-confirmed with the same reproduction. See `tests/test_employee_subgraph.py` and `tests/test_aggregate_node.py::test_missing_employee_placeholder_surfaces_the_real_failure_reason`

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
├── demo_fixtures.py              # 🆕 AUDITPILOT_DEMO_MODE — canned model responses for demo reliability
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
│   └── hf_client.py              # HFClient: BERT NER, BGE embeddings, Groq generation + tool calling, 🆕 429 retry/backoff
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
│   ├── MULTI_AGENT_REDESIGN_SPEC.md   # 🆕 full redesign spec — background, target architecture, rollout plan (annotated with a correction note — see below)
│   ├── IMPLEMENTATION_PLAN.md         # 🆕 the staged build plan executed against that spec
│   └── INTERVIEW_TALKING_POINTS.md    # 🆕 interview-prep writeup — architecture walkthrough, design tradeoffs, failure handling, Q&A
├── tests/                        # 🆕 pytest.ini + fan-out/tool-boundary/retrieval-convergence/fault-isolation/demo-mode suites (62 passing)
│   ├── test_employee_subgraph.py # 🆕 per-employee fault isolation — a branch exception never crashes the whole run
│   ├── test_aggregate_node.py    # 🆕 partial-failure placeholder surfaces the real per-employee failure reason
│   └── test_demo_fixtures.py     # 🆕 AUDITPILOT_DEMO_MODE — canned responses drive the real graph to a correct, zero-error result
├── .env.example                  # Key template — copy to .env and fill in (🆕 includes AUDITPILOT_DEMO_MODE)
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
- **Groq free/on-demand tier has a 100,000-tokens-per-day cap on a rolling 24-hour window** — separate from (and far more restrictive than) the 12,000-tokens-per-minute limit `_call_with_retry` handles automatically (see [Groq Rate Limits and Retry/Backoff](#groq-rate-limits-and-retrybackoff)). Heavy same-day testing (extraction + retrieval's up-to-4-round searches + classification + critic + report generation, per audit) can exhaust the daily quota outright, and because the window is rolling rather than a fixed reset time, the wait Groq reports can observably escalate to an hour or more on a heavily-tested day rather than clearing quickly — confirmed directly: repeated `Retry-After` values climbed from ~11 minutes to ~103 minutes across one debugging session as the pipeline's own calls kept the rolling window saturated. `HFClient.generate()`/`.chat_with_tools()` do correctly wait out and retry short per-minute bursts; a daily-quota exhaustion just means waiting for the rolling window to genuinely clear (practically, often the next day), or upgrading to a paid tier for sustained development use.

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
