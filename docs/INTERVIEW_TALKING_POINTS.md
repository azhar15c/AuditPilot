# AuditPilot — "Tell Me About Your Project" Interview Prep

---

## The 60-Second Version

AuditPilot automates the most manual part of a workers' compensation premium audit: an auditor gets a payroll register, has to identify every employee, figure out what NCCI classification code applies to their job, and draft a worksheet — normally a multi-day, error-prone process. I built an AI pipeline that does extraction, classification, and report drafting, backed by a RAG knowledge base of the actual NCCI manuals so the model is grounded in real classification rules instead of just guessing from training data.

I started with a working linear pipeline, then redesigned it into a Supervisor-orchestrated multi-agent system: extraction runs once, then every employee's classification runs as an isolated, concurrent unit — retrieval, classification, and an independent Critic agent that reviews the classification before it reaches the human auditor. The auditor still signs off on everything; this is decision support, not automation of the actual audit judgment.

The interesting part wasn't wiring agents together — it was the things that broke along the way and what fixing them taught me: a real LangGraph mechanic where naive fan-out loses per-branch identity, a state-reducer bug that silently doubled results, and three separate bugs I only found by testing against the real Groq API instead of trusting mocks.

---

## The Problem, In Domain Terms

Every workers' comp policy gets an annual premium audit: compare actual payroll to the estimate used to set the initial premium. The auditor has to:
- Pull every employee out of a payroll register, 941 form, or COI
- Assign each one a 4-digit NCCI class code based on their actual job duties (misclassification is the single most common source of audit disputes)
- Check subcontractors have valid Certificates of Insurance — an expired or missing COI means that subcontractor's payments get reclassified as the policyholder's own payroll, which raises the premium
- Draft a worksheet justifying every code, ready for underwriter review

This is exactly the kind of work I understand from the inside — I spent 15 years in P&C insurance systems, Guidewire ACE certified on PolicyCenter and ClaimCenter, so I know what a real auditor's workflow looks like and what "good enough to submit" actually means. AuditPilot isn't a toy demo of AI-does-insurance; the domain judgment behind what fields matter, why COI status matters, why NCCI consistency across audit periods matters, came from that background.

---

## What I Built — Two Versions

**v1 (the working baseline):** A linear LangGraph pipeline — `intake → extract → classify → report`. The `classify` step was already agent-like: it did a RAG lookup against the NCCI manuals, a PolicyCenter MCP call for the employer's prior-period classifications, then forced a structured tool call (`finalize_classification`) so the output was typed fields, not free text I'd have to regex-parse.

**v2 (the redesign, what I want to talk about):** I rebuilt it into a Supervisor-orchestrated, partially-parallel multi-agent system with a dedicated verification step. That's the version worth walking through in detail.

---

## Architecture Walkthrough

```
intake → extraction_agent → supervisor_node → (fan-out, one per employee) →
    employee_pipeline (subgraph: retrieval_agent → classification_agent → critic_agent)
    → aggregate_node → report_node
```

**Extraction runs once, document-level.** BERT NER plus an LLM structured-extraction pass pull every employee, job title, and wage out of the raw document text.

**Supervisor runs once, right after extraction.** Its whole job is fetching the employer's prior-period NCCI classifications from PolicyCenter — and it does this *exactly once* per audit, not once per employee. That sounds small, but the old code was calling the same PolicyCenter endpoint once per employee even though the result never changes within one audit — pure waste I caught while splitting the old monolithic classify step apart.

**Then it fans out per employee.** Each employee becomes an isolated, concurrent unit running `retrieval_agent → classification_agent → critic_agent`. Retrieval does an iterative, self-evaluating search over the NCCI manual embeddings — I'll come back to this, it's the part I'd most want to talk through in a deep-dive. Classification forces a structured tool call for the final code, title, rationale, and confidence. Critic is the new piece: an independent agent that reviews the classification against the retrieved citations *before* a human ever sees it — and it's deliberately blind-reviewed. It never sees Classification's own reasoning, only the final decision plus the raw citations, specifically so it can't just rubber-stamp whatever justification it's shown. That's enforced structurally, not by prompt instruction — the data shape handed to the Critic literally excludes the rationale field.

**Aggregate validates and degrades gracefully.** If one employee's branch fails for any reason, the audit doesn't die — that employee gets marked "not classified, retry required" and everyone else's results still make it into the report. Human-in-the-loop sign-off is unchanged; the auditor still reviews and approves everything before it goes anywhere.

---

## The Decisions I'd Actually Want to Talk About

Anyone can describe a multi-agent diagram. What I think is worth an interviewer's time is *why* it looks like this and what I found out building it.

### 1. I deviated from my own spec, and that's the right call, not a failure

My original design (written spec-first, before any code) called for Extraction and Retrieval running in literal case-level parallel — a fairly standard-looking diagram. Building it, I realized Retrieval genuinely can't start until Extraction finishes: it needs the specific job title and employer to form a good search query — a roofer and a bookkeeper need completely different NCCI manual sections. So I moved the parallelism to where the real independence actually is: Extraction runs once, then N employees run concurrently against *each other*. That's a bigger latency win than the original design anyway, since the actual bottleneck in the old code was serial per-employee classification, not Extraction racing Retrieval.

### 2. Context engineering in the retrieval loop — not just "loop and pass more"

The naive version of iterative retrieval is: search, ask the model if it's enough, if not search again, repeat. I built that first, then realized it was silently *losing* information — every round overwrote the previous round's results instead of accumulating them, so if round 1 found something useful and round 2 searched for something else, round 1's finding just vanished. The fix does three things together: accumulate and dedupe citations across rounds by a stable chunk ID; keep each round's prompt to only *new* chunks plus a compact summary distilled from prior rounds' own reasoning, instead of replaying every raw chunk ever seen (which would just trade "losing information" for "degrading the model's attention with prompt bloat"); and cap the final accumulated set to the most relevant handful before it crosses into the next agent's prompt, so a long search doesn't balloon what Classification has to read. That's the actual definition of context engineering — curating what's in the window, not maximizing what's in it — and I can point to the specific commit and tests where I fixed the naive version.

### 3. A real LangGraph mechanic I had to discover empirically

LangGraph's `Send()` API is how you fan out to N parallel branches. What I learned the hard way: `Send()` only replicates the *node it directly targets* — if that node has a normal edge to a second node, the second node runs once, globally, after all branches converge, not once per branch. I confirmed this by writing a minimal reproduction script before trusting any assumption about it, then fixed it by wrapping the per-employee stages in their own separately-compiled subgraph, with the subgraph itself as the `Send()` target. I verified the fix the same way — a timing test showing 4 parallel branches with a deliberate 0.2s delay each completed in ~0.21s total, not ~0.8s, proving genuine concurrency with correct per-employee isolation.

### 4. A silent correctness bug in the reducer design

LangGraph state fields can be declared with a reducer (`operator.add`) so parallel branches can each contribute to a shared list without clobbering each other. What I didn't initially realize: that reducer fires on *every* node return that touches the field, not just at the fan-out/fan-in boundary. A downstream node that echoed back the accumulated state (a pattern used everywhere else in the codebase) would re-trigger the reducer and silently double every result. I caught it with a direct reproduction — a 2-employee test run that should have produced 2 classifications produced 4 — before it could reach a real audit.

### 5. Three bugs I only found by testing against the real API

All of my automated tests use mocks, and they all passed. But when I actually ran the pipeline against live Groq calls, it hung. Debugging that surfaced three separate, real issues, none of which show up in mocked tests because mocks never generate the kind of malformed output that triggers them:

- The Groq SDK has its own hidden `max_retries` default, running its own uncoordinated retry loop *inside* every API call, completely invisible to and stacked underneath the retry logic I'd written myself.
- `llama-3.3-70b` sometimes returns a stringified `"false"` instead of a real JSON boolean in structured tool-call output — which both fails Groq's strict schema validation outright, and, worse, `bool("false")` evaluates to `True` in Python, so a naive cast would have silently inverted the model's actual answer if it had ever gotten through.
- A `max_tokens` budget that was fine for short outputs was truncating longer tool-call generations mid-JSON, which Groq's parser then rejects as unparseable — not a rate-limit error, so my retry logic never even engaged for it.

I fixed all three with targeted, tested changes, and I made a point of reproducing each one directly against the real API before believing it was fixed — a green mocked test suite had already told me everything was fine once, and it was wrong.

### 6. Rate-limit resilience, and knowing the difference between two failure modes

Groq's free tier has two separate limits: a small per-minute token budget that resets in seconds, and a much smaller-feeling daily budget on a rolling 24-hour window. My multi-agent fan-out is a direct contributor to hitting the per-minute one — N employees firing concurrent LLM calls bursts token usage into a narrower window than the old sequential pipeline ever did. I addressed that with a semaphore bounding concurrent in-flight requests plus jittered backoff, so retries desynchronize instead of re-triggering the same burst in lockstep. But I also had to learn to tell that apart from the daily quota, which no amount of retry logic fixes — I watched reported wait times *escalate* from minutes to over an hour across one session because the rolling window doesn't clear until earlier usage ages out. Knowing which failure mode you're looking at, and not throwing more retry logic at a problem retries can't solve, was its own lesson.

---

## Failure Handling: What Happens When One Agent Fails?

I want to answer this the way I actually found and fixed it, not just describe the end state — the process is the more interesting part.

**What genuinely works today:**
- `intake_node` failures (a bad or empty file) are a clean, hard stop — the conditional edge after intake routes straight to `END` with a clear error message. That's correct behavior; there's nothing to recover from without extractable text.
- Every Groq call goes through a retry wrapper that handles transient rate-limit failures with jittered backoff, which reduces how often an employee's branch ever reaches a hard failure in the first place.
- `aggregate_node` has real, tested logic for "an employee is missing from the results" — it marks them `"NOT CLASSIFIED — retry required"`, logs a warning into the audit trail, and lets the rest of the report complete around them.

**The gap I found by testing it directly, not by reading my own code and assuming:** `aggregate_node`'s degradation logic only fired if a branch *quietly returned without producing a result*. That's not how failures actually manifest. I ran a controlled test — two employees, one whose retrieval call raises after retries are exhausted, one that succeeds normally — and the entire `run_workflow()` call raised. The employee that had already succeeded lost their result too; nothing partial came back. LangGraph's executor propagates an unhandled exception from any single branch all the way up through the subgraph, through the parent graph's task, and out of the top-level `.invoke()` call. `aggregate_node`'s graceful-degradation path was real, already-tested code — it was just unreachable from the failure mode that actually happens.

**The fix I shipped:** a try/except around the per-employee subgraph invocation itself, at the `Send()` branch boundary (`employee_pipeline_node` in `agents/employee_subgraph.py`). One employee's exception is caught right there, logs an `AgentStep` with `status="error"` and the real exception message, and contributes nothing to that employee's results instead of crashing everything — at which point `aggregate_node`'s existing "missing from results" logic finally runs the way it was designed to, now additionally looking up the real failure reason from the audit trail so the placeholder's rationale says *why*, not just *that* it failed. I reran the exact same two-employee reproduction after the fix: the run completed successfully, the employee who succeeded kept their result, and the one who failed showed up as `"NOT CLASSIFIED — retry required"` with the actual error message attached. Same test, before and after — that's what "verified," not just "should work," looks like. It's also now visibly surfaced in the product itself: a new "Audit Trail" tab in the UI renders every agent step with an ⚠ ERROR flag on exactly the rows this fix produces, so a failure is debuggable by the auditor, not just by someone reading logs.

## What's Generally Hard About Multi-Agent Systems — And What I Ran Into Here

Most of what's hard about multi-agent systems isn't the diagram — it's a set of specific, recurring failure classes. Here's the honest mapping between the general problem and what actually happened building this one:

| General problem | What it looks like | What happened here / how I addressed it |
|---|---|---|
| **Fault isolation** | One sub-agent's failure takes down the whole system instead of degrading gracefully | Found the gap by direct reproduction, fixed it with a try/except at the `Send()` branch boundary, re-verified with the same reproduction — see "Failure Handling" above |
| **Hidden data dependencies** | Agents assumed independent turn out not to be; naive parallelism breaks or produces wrong results | Extraction→Retrieval — Retrieval needs job-specific details Extraction produces, so I moved concurrency to the actual independent unit (per-employee) instead of case-level |
| **Context propagation across framework boundaries** | Orchestration frameworks have sharp, non-obvious edges in how state moves between parallel branches | LangGraph's `Send()` only replicating the *immediately targeted* node — found by writing a minimal reproduction before trusting the assumption, fixed with subgraph isolation |
| **Silent correctness bugs in shared state** | Reducer/merge logic that's correct at the boundary can still double-fire on every intermediate write | A downstream node echoing back accumulated state re-triggered `operator.add` and doubled every result — caught with a deliberate exact-count check, not just "did it run" |
| **Context rot / prompt bloat** | The more agents hand off to each other, the more context accumulates unless someone actively curates it | The naive retrieval loop was losing information *and* would have bloated prompts if fixed carelessly — solved with accumulate+dedupe+compact+cap together |
| **Correlated errors / anchoring** | A "verifier" agent shown the same reasoning as the agent it's checking tends to just agree with it — not real independent signal | Critic is structurally blind to Classification's rationale — enforced by what the data contract contains, not by prompt instruction |
| **Resource contention from parallel LLM calls** | N concurrent agents means N concurrent API calls, which can burst past provider rate limits even when total usage is unremarkable | Groq's per-minute limit — a semaphore bounding concurrent calls plus jittered retry backoff |
| **Structured-output unreliability** | LLMs don't always perfectly conform to a declared schema, especially under concurrent/forced tool-calling | Stringified booleans in tool args, truncated JSON from too-tight token budgets — both only appeared against the real model, fixed with defensive coercion and budget adjustments |
| **Observability across distributed agent calls** | When something goes wrong, it's hard to know which agent, on which input, did what | Every agent invocation writes a structured step into a shared audit trail, uniform regardless of which agent or how many ran |
| **Testing multi-agent systems reliably** | Mocks can't reproduce model-specific quirks — a fully green mocked suite can still hide real bugs | All three live-API bugs above were invisible to 53 passing mocked tests; only found by testing against the real model directly, in small doses |

The honest throughline, if asked to summarize this in one sentence: most of these weren't solved by anticipating them upfront — they were found by testing rigorously against reality instead of trusting design intent, and then fixed with a targeted, verified change.

---

## Process — How I Actually Built This

I treated the redesign itself as a deliberate exercise in spec-first, agentic development, not just as a feature to ship. I wrote the full architecture spec before any implementation code, broke it into independent workstreams with explicit file-level ownership boundaries, and built it in staged checkpoints — schemas and state design first, then parallel work on the Supervisor, the agent split, and the test harness, then the fan-out wiring and the Critic agent, then final integration — running the full test suite as a gate after every merge, never moving forward on red. `main` and the live demo stayed untouched and deployed the entire time; the whole redesign happened on a feature branch with an explicit rollback tag, and merging to `main` is still a deliberate, separate decision I haven't made yet.

---

## Where It Stands

53 automated tests, covering everything from the iterative retrieval loop's convergence behavior to tool-boundary enforcement (verifying, for instance, that the Classification agent structurally cannot reach the vector store or PolicyCenter directly) to the rate-limit retry precedence. I've verified the full pipeline end-to-end both with mocked model calls and, separately, against the real Groq and PolicyCenter APIs.

**What I'd say if asked what's missing:** live PolicyCenter write-back is stubbed but not wired to a real Guidewire tenant; the knowledge base currently covers one state's NCCI manual; and the Supervisor today is a fixed dispatcher, not a reasoning router — there's no case-by-case branching logic because every audit currently needs the identical sequence of steps. If real case volume showed genuine variance worth deciding between — skip the Critic on high-confidence cases to save a call, for instance — that's exactly where I'd add real LLM-driven routing instead of a fixed fan-out.

---

## Likely Follow-Ups, and How I'd Answer Them

**"Is your Supervisor agent actually doing orchestration?"**
Honestly, no — not in the reasoning-router sense. It's a fixed dispatcher: one deterministic data fetch, then an unconditional fan-out. There's no branching logic because every employee currently needs the identical three steps. I'd rather say that precisely than oversell it — I know exactly where the line is and what would need to change to cross it.

**"Why multi-agent instead of one big prompt with all the context?"**
Least-privilege and observability, mainly. Classification structurally cannot reach the vector store or PolicyCenter — it can only use what Retrieval already gathered. The Critic structurally cannot see Classification's rationale. Those are real safety properties enforced by what data crosses each boundary, not by hoping one giant prompt behaves. And every agent's invocation gets logged into a structured audit trail, which a monolithic prompt can't give you for free.

**"What was the hardest bug?"**
The reducer-doubling one, because it was silent — it would have shipped a report with every employee's classification duplicated, and nothing would have crashed or logged an error. I only caught it because I ran a deliberate end-to-end check with an exact expected count instead of just checking "did it run."

**"What would you do differently next time?"**
Test against the real API earlier and more often, even in small, cheap doses, rather than trusting a fully green mocked suite as sufficient proof. Every one of the three live-API bugs would have shipped invisibly otherwise.

**"What happens if one agent in your fan-out fails?"**
That employee gets marked "not classified, retry required" with the actual error message attached, everyone else's results complete normally, and it's visible on an Audit Trail tab as a flagged row — but I want to be honest about how I got there, because it's a better story than "I designed it that way from the start." I originally assumed `aggregate_node`'s degrade-gracefully logic covered this, then tested it directly and found it didn't: an unhandled exception from one employee's branch propagated all the way up through LangGraph's executor and crashed the entire run, including employees who'd already succeeded. I fixed it with a try/except at the `Send()` branch boundary, then reran the exact same reproduction to confirm — same test, before and after, not just "should work now."
