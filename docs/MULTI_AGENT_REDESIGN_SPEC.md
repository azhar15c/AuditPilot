# AuditPilot Multi-Agent Redesign — Specification

**Author:** Azhar Chaus
**Status:** Draft for implementation — **partially superseded during implementation, see note below**
**Purpose:** Redesign AuditPilot from a linear (Pipeline-pattern) LangGraph chain into a Supervisor-orchestrated, partially-parallel multi-agent system with a dedicated verification step — while using this build itself as a deliberate exercise in agentic coding (parallel Claude Code / Codex sessions, spec-first development).

> **As-built correction:** Section 2.1's diagram and the 2.2 table below describe Extraction and Retrieval as case-level parallel agents ("Parallel with Retrieval" / "Parallel with Extraction"). That's **not** what was actually built. During implementation it became clear Retrieval needs job-specific details (job title, employer) that only exist *after* Extraction produces the employee list — a roofer and a bookkeeper need different search queries — so literal Extraction‖Retrieval parallelism isn't achievable for this domain. The as-built design instead runs Extraction once (document-level), then fans out **per employee**: N employees' `Retrieval → Classification → Critic` chains run concurrently against *each other*, not against Extraction. This is preserved here as the original spec/historical record; see `docs/IMPLEMENTATION_PLAN.md` and the README's "v2 — Multi-agent architecture" diagram for the accurate as-built shape.

---

## 1. Background and Motivation

### 1.1 Current architecture

AuditPilot today runs a fixed, sequential LangGraph pipeline:

```
intake → extract → classify → report → human sign-off
```

Each stage is a single agent node with broad tool access, handing off to the next stage unconditionally. Extraction (BERT NER + LLM hybrid) and rules retrieval (RAG over ChromaDB + MCP to Guidewire PolicyCenter) currently happen inside the same stage rather than as independent, parallelizable units. There is no dedicated step that checks one agent's output against another before it reaches the human auditor.

In LangGraph/multi-agent terminology, this is a **Pipeline pattern** — the simplest of the five patterns used in production systems (Pipeline, Fan-out/parallel, Supervisor, Swarm, Debate).

### 1.2 Why redesign

1. **Architectural depth.** A fixed pipeline doesn't demonstrate dynamic routing, delegation, or verification — increasingly the baseline expectation for "agentic AI engineer" roles.
2. **Latency.** Extraction and rules retrieval are largely independent given a document; running them sequentially wastes wall-clock time that a fan-out step would recover.
3. **Trust.** A single classification agent's output goes straight to the human auditor today. A dedicated Critic/Verifier agent that checks the classification against retrieved policy language — and flags disagreement — is a stronger trust story than "one model said so."
4. **Reusability.** Splitting one large "extract+classify" agent into narrower, single-responsibility agents with scoped tool access is safer (least-privilege) and easier to test in isolation.

### 1.3 Non-goals

- Not changing the underlying LLM provider (stays Groq-hosted Llama 3.3 70B with `tool_choice="required"`).
- Not replacing ChromaDB as the vector store — only how the Retrieval Agent decides when and how many times to query it (see 2.2.1).
- Not removing the mandatory human-in-the-loop sign-off — the Critic agent supplements it, does not replace it.
- Not a full rewrite — existing extraction/retrieval logic is refactored into new agent boundaries, not discarded.

---

## 2. Target Architecture

### 2.1 Pattern

**Supervisor pattern** as the top-level control flow, with a **fan-out** sub-step for independent work, and a **debate-style** verification step before human sign-off.

> ⚠️ The diagram and table immediately below show Extraction and Retrieval as case-level parallel agents. **This is the original design, not the as-built one** — see the correction note at the top of this document. As built, the fan-out is per-employee (Extraction runs once, then N employees' Retrieval→Classification→Critic chains run concurrently).

```
                         ┌─────────────────────┐
                         │   Supervisor Agent   │
                         │  (Orchestrator)      │
                         └──────────┬───────────┘
                                    │ routes case, decides which
                                    │ specialists to invoke
                    ┌───────────────┼───────────────┐
                    │ (parallel fan-out where independent)
          ┌─────────▼─────────┐          ┌──────────▼──────────┐
          │ Extraction Agent  │          │  Retrieval Agent     │
          │ (BERT NER + LLM)  │          │  (RAG + MCP/PC)      │
          └─────────┬─────────┘          └──────────┬──────────┘
                    │                                │
                    └───────────────┬────────────────┘
                                     ▼
                         ┌─────────────────────┐
                         │ Classification Agent │
                         └──────────┬───────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  Critic / Verifier    │
                         │  Agent (debate step)  │
                         └──────────┬───────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  Human-in-the-Loop    │
                         │  Sign-off (existing)  │
                         └──────────┬───────────┘
                                    ▼
                              Final Report
```

### 2.2 Agent responsibilities

| Agent | Owns | Tool access (least-privilege) | Runs |
|---|---|---|---|
| **Supervisor** | Case intake, routing decisions, invoking specialists, synthesizing final handoff to Critic | Read-only case metadata; agent-invocation tool only (no data/document tools) | Always, first |
| **Extraction Agent** | Document text extraction, BERT NER, duplicate detection | Document store (read-only), NER model, LLM (extraction prompt only) | Parallel with Retrieval |
| **Retrieval Agent** | Iterative, self-evaluating search over ChromaDB + MCP calls to Guidewire PolicyCenter (see 2.2.1) | ChromaDB (read-only) as a callable **search tool**, MCP PolicyCenter tools (read-only), LLM (query-refinement + sufficiency-judgment prompt only) | Parallel with Extraction |
| **Classification Agent** | Risk/audit classification decision, using Extraction + Retrieval outputs as context | LLM (classification prompt only); **no direct document or PolicyCenter access** — must use upstream agents' outputs | After Extraction + Retrieval both return |
| **Critic / Verifier Agent** | Independently checks Classification's decision against the retrieved policy language; flags disagreement with a reason | LLM (verification prompt only), read-only access to Retrieval Agent's citations — **not** the Classification Agent's reasoning, to avoid anchoring bias | After Classification |
| **Human-in-the-Loop** | Final sign-off; sees Classification's decision, Critic's assessment, and any disagreement flag | N/A (human) | After Critic |

Key design decision: the Critic agent is deliberately **not** shown the Classification agent's reasoning chain, only its final decision plus the Retrieval agent's raw citations — a version of "blind" review to reduce the risk of the Critic simply agreeing with whatever justification it's shown.

### 2.2.1 Retrieval Agent — iterative search loop (upgrade from static/dual-query RAG)

**Motivation.** The original design used a bounded, static retrieval step: query ChromaDB once, and if that returned nothing useful, try one reformulated fallback query, then hand off to Classification regardless of whether the citations actually answer the case — a fixed number of attempts with no judgment about sufficiency. This mirrors the "traditional RAG" pattern Anthropic contrasts against in their multi-agent research system writeup: fetch some chunks, generate, done. The upgrade replaces that fixed step with a loop where the agent itself judges whether it has enough to proceed, and if not, decides how to search again — the same "dynamically finds relevant information, adapts to new findings" pattern.

**Loop design:**

```
1. Formulate initial search query from the case's extracted entities / policy number.
2. Call the ChromaDB search tool → get candidate policy citations.
3. Sufficiency check (LLM reasoning step): "Given the case's classification question,
   do these citations actually contain what's needed to answer it?"
     - If YES → return RetrievalOutput, done.
     - If NO  → reason about *why* (wrong policy section? too vague? missing a
       cross-referenced clause?), formulate a refined follow-up query targeting
       that specific gap, and go to step 2.
4. Hard stop: max_iterations (default 4) — if sufficiency still fails at the cap,
   return the best citations found so far, with `sufficiency_met: false` flagged
   explicitly in the output (never silently pass off a known-insufficient result).
```

This is a genuine architectural upgrade, not just more retries: each subsequent query is *informed by why the previous one fell short*, rather than a fixed, pre-written fallback string. The `sufficiency_met: false` escape hatch matters — the Critic and the human auditor should be able to see explicitly when retrieval gave up rather than have a weak result look identical to a confident one.

**Cost control.** Every iteration is an additional LLM reasoning call plus a vector-store query, so this is not free — see the new risk logged in Section 5.

### 2.3 Data contracts (handoff schemas)

Each agent-to-agent handoff should be a typed, validated payload — not free text. Sketch (pseudo-schema, refine in implementation):

```
CaseInput
  case_id: str
  document_refs: list[str]
  policy_number: str | None

ExtractionOutput
  case_id: str
  extracted_entities: list[Entity]     # BERT NER results
  extracted_text_segments: list[TextSegment]
  duplicate_flags: list[str]
  confidence: float

RetrievalOutput
  case_id: str
  policy_citations: list[Citation]      # source + text + relevance score
  policycenter_fields: dict             # via MCP
  search_queries_used: list[str]        # every query tried, in order
  iterations_run: int                   # 1..max_iterations
  sufficiency_met: bool                 # False if capped out without confidence
  sufficiency_reasoning: str            # why the loop stopped (found enough / gave up)

ClassificationInput = ExtractionOutput + RetrievalOutput

ClassificationOutput
  case_id: str
  decision: str
  rationale: str
  cited_sources: list[str]
  confidence: float

CriticInput
  case_id: str
  decision: str                         # from ClassificationOutput, reasoning withheld
  policy_citations: list[Citation]      # from RetrievalOutput directly, not via Classification

CriticOutput
  case_id: str
  agrees: bool
  disagreement_reason: str | None
  confidence: float

FinalPackage (to human)
  case_id: str
  classification: ClassificationOutput
  critic_assessment: CriticOutput
  full_audit_trail: list[AgentStep]     # every agent call, input, output — for observability
```

### 2.4 Observability

Every agent invocation (Supervisor's routing decision, each specialist call, the Critic's verdict) should be logged as a structured `AgentStep` and attached to the case's audit trail — not just the final decision. This is both a genuine engineering improvement (debuggability) and a direct answer to "how do you trace what happened" in an interview.

---

## 3. Evaluation Strategy

Extend the existing `pytest` / `unittest.mock` harness rather than replacing it:

1. **Regression suite** — every existing test case must still pass with the same expected classification outcomes, run against the new graph.
2. **New: Critic disagreement cases** — construct at least 3–5 synthetic cases where the Classification agent's expected output should *not* match the retrieved policy language, and assert the Critic flags disagreement. This is the test suite that proves the new step earns its complexity.
3. **New: Parallel fan-out correctness** — assert Extraction and Retrieval agents produce identical output whether invoked in parallel or sequentially (no shared-state race conditions).
4. **New: Tool-boundary tests** — assert each agent's mocked tool client rejects calls outside its declared scope (e.g., Classification agent cannot call the PolicyCenter MCP tool directly).
5. **New: Iterative retrieval convergence** — construct cases where the first search query is deliberately insufficient (e.g., ambiguous policy reference) and assert: (a) the loop actually reformulates the query rather than repeating it verbatim, (b) it terminates within `max_iterations` on all test cases, (c) `sufficiency_met` is correctly `False` on a case engineered to be unanswerable within the cap, and (d) recall on the insufficient-first-query cases improves versus the old static/dual-query implementation (regression comparison against the `v1-pipeline` tag).

Human-in-the-loop sign-off remains the final gate; nothing in this redesign auto-finalizes a case.

---

## 4. Implementation Plan (Agentic-Coding Workflow)

This section is deliberately about *how* to build it, not just what to build — the process itself is the FDE-interview-relevant part.

### 4.1 Repository and branching strategy

This redesign happens **inside the existing `azhar15c/AuditPilot` repo, on branches — not as a new, separate project.** The GitHub URL and the Hugging Face Spaces demo link are already circulated across every active job application (Level, PAR, both Apple roles, Kalepa, Drata, Wipfli, Sixfold); a separate repo would orphan those links and split one flagship project's history into two disconnected ones. Concretely:

1. **Tag the current state before touching anything:** `git tag v1-pipeline` on `main`. This is the named rollback point and the baseline the new eval comparisons in Section 3 (item 5) are measured against.
2. **Create one top-level feature branch off `main`:** `feature/multi-agent-redesign` (or `v2-supervisor-architecture`). All redesign work lives under this branch, not directly on `main`.
3. **Workstream branches (A–E, Section 4.2) branch off the feature branch, not off `main` directly** — e.g., `feature/multi-agent-redesign/workstream-b-agent-refactor`. Each is its own git worktree so parallel Claude Code/Codex sessions never collide on a working directory.
4. **`main` stays untouched and deployed for the entire build.** The live Hugging Face Space keeps pointing at `main`, so the demo link already sitting in sent applications keeps working exactly as-is throughout.
5. **Merge workstream branches into the feature branch** as each clears its acceptance criteria (Section 4.4), running the eval suite after each merge — same merge order as Section 4.3, just landing on the feature branch instead of `main`.
6. **Merge the feature branch into `main` only after the full regression + new eval suite (Section 3) passes end-to-end on it.** That eval run is the merge gate, not a formality.
7. **Only then update the live Hugging Face Space** to redeploy from `main`.

**Optional, if you need to demo work-in-progress before merging** (e.g., a Sixfold technical interview lands mid-build): stand up a second, temporary Space — `AuditPilot-v2-preview` — pointed at the feature branch. This keeps any risk of breaking the live, already-shared demo link at zero while still giving you something to show. Tear it down after merging to `main`.

### 4.2 Decompose into independent workstreams

| Workstream | Scope | Depends on |
|---|---|---|
| A — Supervisor/router | New orchestrator node, routing logic, invocation of specialists | Schemas (2.3) |
| B — Agent boundary refactor | Split current extract/classify stage into Extraction, Retrieval, Classification agents with scoped tools | Schemas (2.3) |
| C — Fan-out wiring | Parallel invocation of Extraction + Retrieval, join before Classification | B |
| D — Critic agent | New agent, blind-review design, disagreement flagging | B (needs Retrieval + Classification outputs) |
| E — Eval harness extension | New test cases per Section 3 | A, B, C, D (can be written test-first against schemas) |

Note E can start immediately (test-first) using the schemas in 2.3 before A–D are implemented.

### 4.3 Parallel build process

1. Write this spec (done) and get the schemas in 2.3 locked before writing implementation code.
2. Open one git worktree/branch per workstream (A–E), each off the feature branch per Section 4.1.
3. Run one Claude Code (or Codex) session per worktree concurrently, each scoped to its workstream section of this spec — not the whole document.
4. Each session works independently; use the schemas as the contract so branches don't need to coordinate mid-build.
5. Review each branch individually against its workstream's acceptance criteria (Section 4.4) before merging.
6. Merge order: E (tests, can merge early as failing/pending) → B → A → C → D, running the full eval suite after each merge, landing on the feature branch (not `main`).
7. Final integration pass: run the complete regression + new eval suite end-to-end on the feature branch; fix any cross-workstream contract mismatches surfaced only at integration; then follow Section 4.1 steps 6–7 to merge to `main` and redeploy.

### 4.4 Acceptance criteria per workstream

- **A (Supervisor):** Given a case, correctly invokes B/C's agents in the right order; unit-testable with mocked specialist agents.
- **B (Agent refactor):** Each of the three agents passes its existing single-responsibility tests with no access to out-of-scope tools (verified by tool-boundary tests).
- **C (Fan-out):** Extraction and Retrieval measurably run concurrently (timing assertion) and produce output identical to sequential execution on the same input.
- **D (Critic):** Flags disagreement on all synthetic adversarial cases (Section 3.2) without false-flagging the existing regression suite's known-good cases.
- **E (Eval harness):** All new test categories (regression, disagreement, fan-out correctness, tool-boundary) pass in CI.

### 4.5 What to capture along the way

- Before/after architecture diagrams (Section 2.1 vs. the original pipeline).
- A short written log of the build: what was spec'd, how work was split across parallel agent sessions, and at least one specific instance where a session's first-pass output was wrong and how it was caught in review.
- Updated eval report showing the Critic step catching a case the old linear pipeline would have passed through unchecked.

---

## 5. Open Questions / Risks

- **Latency vs. accuracy tradeoff of the Critic step** — adds an extra LLM call per case; measure actual latency impact before deciding if it runs on every case or only below a Classification-confidence threshold.
- **Supervisor overhead** — per the Supervisor-vs-Swarm tradeoff, a routing LLM call adds cost/latency that's only justified if routing decisions actually vary case-to-case. If every case follows the identical Extraction→Retrieval→Classification→Critic sequence with no real branching, a lighter-weight fixed orchestrator (not a full LLM-reasoning Supervisor) may be more appropriate — revisit after the first batch of real cases.
- **Blind-review design for the Critic** — withholding Classification's rationale reduces anchoring but may cause the Critic to flag stylistic differences rather than substantive disagreements; monitor false-positive rate in the new eval cases.
- **Iterative retrieval loop cost/latency** — each additional search round (Section 2.2.1) adds one LLM sufficiency-check call plus one vector-store query. `max_iterations = 4` is a starting guess, not a measured value; instrument `iterations_run` in production and tune the cap against real cost/latency data once live. Watch for diminishing returns — most cases likely converge in 1–2 iterations, with the cap mainly protecting against pathological cases rather than being hit routinely.
