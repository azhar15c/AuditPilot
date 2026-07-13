# AuditPilot Multi-Agent Redesign — Implementation Plan

## Context

AuditPilot currently runs a fixed, linear LangGraph pipeline (`intake → extract → classify → report`) where `classify_node` does the real work of a multi-agent system — RAG retrieval, PolicyCenter MCP lookup, and forced-tool-call LLM classification — all fused into one function, called serially once per employee. There's no independent verification step before results reach the human auditor, and per-employee classification wastes wall-clock time running sequentially when each employee's classification is logically independent.

The user authored a full redesign spec (`AuditPilot_MultiAgent_Redesign_Spec.md`) moving this to a **Supervisor-orchestrated, partially-parallel multi-agent system** with a dedicated Critic/Verifier step — both a genuine architectural improvement (latency, trust, least-privilege tool access) and a deliberate exercise in agentic/spec-first coding practice. This plan turns that spec into concrete file-level steps grounded in the actual code (confirmed via direct reads of every node, `rag/retriever.py`, `mcp_servers/policycenter.py`, `models/hf_client.py`, and by running the test suite).

**Two facts discovered during exploration, not in the original spec, that shape this plan:**
1. **10 of 16 existing tests fail on `main` today** (`pytest tests/ -q` → 10 failed, 6 passed) — `test_workflow.py`, most of `test_classify.py`, and all of `test_extract.py` were written against an old API shape (`.generate()`/`.classify()`/`.ner()` methods that don't exist, `bytes` input to `run_workflow`, wrong state keys). These are folded into Workstream E rather than fixed separately.
2. **The employee `name`-string join is fragile in two places, not one** — `report_node.py::_format_combined` **and** `app.py::_build_dataframe` (~line 308) both join `employee_records` to `ncci_suggestions` by exact-string `name`. Both get fixed together via a new stable `employee_id`.

**Decisions locked with the user before this plan:**
- Stale test fixes fold into Workstream E's scope.
- Execution model: parallel **worktree-isolated subagents** per workstream (mirrors the spec's Section 4.3 multi-session vision), reconciled and merged by this orchestrating session.
- Full redesign (Workstreams A–E), executed in **staged, reviewable checkpoints** — pause for review between stages, not one giant diff.
- **Fan-out is per-employee**, not literal case-level Extraction‖Retrieval: Extraction runs once (document-level, produces the employee list), then the Supervisor fans out per employee via LangGraph's `Send` API — each employee gets its own concurrent `Retrieval → Classification → Critic` chain. This also fixes the current redundant per-employee PolicyCenter refetch (today `get_prior_classifications` is called once per employee despite `policy_number` being constant for the run).
- `max_iterations = 4` for the retrieval loop (spec default, not lowered).
- Schemas are plain `TypedDict`s, matching `agents/state.py`'s existing convention — no new Pydantic dependency.
- Partial branch failure degrades gracefully: `aggregate_node` marks a failed employee `"NOT CLASSIFIED — retry required"` and the rest of the audit still completes, rather than short-circuiting the whole run.
- Critic's `flag_for_review` verdict gets minimal UI surfacing — a column/flag added to the existing Employee Classification DataFrame in `app.py`.
- `main` is never touched, pushed, or redeployed by this plan. `push_audit_results` (unused today) stays unwired — out of scope.

LangGraph 1.2.7 is already installed with the `Send` API available (confirmed via import) — no dependency bump needed.

---

## Stage 0 — Schemas & state (done directly, not delegated)

1. `git tag v1-pipeline` on current `main` HEAD — named rollback point, touches nothing else.
2. Create `feature/multi-agent-redesign` off `main`. All subsequent work happens here; `main` is never checked out for writes again in this plan.
3. **New file `agents/schemas.py`** — `TypedDict`s for every agent handoff: `AgentStep`, `CaseInput`, `ExtractionOutput`, `RetrievalOutput` (includes `policy_citations`, `policycenter_fields`, `search_queries_used`, `iterations_run`, `sufficiency_met`, `sufficiency_reasoning`), `ClassificationOutput`, `CriticInput` (explicitly excludes classification `rationale` — structural enforcement of blind review), `CriticOutput`, `FinalPackage`.
4. **Modify `agents/state.py`** — add `prior_classifications: dict` (written once by Supervisor, pre-fan-out) and four `Annotated[list[dict], operator.add]` fields written by parallel branches: `retrieval_outputs`, `ncci_suggestions` (changed from plain `list[dict]`), `critic_assessments`, `audit_trail`.

**Reducer design rationale:** LangGraph's Pregel executor synchronizes parallel `Send` branches at the next shared node and merges each branch's partial-state return via the field's reducer. Without `operator.add`, N concurrent writes to `ncci_suggestions` would use default last-write-wins TypedDict semantics — silently dropping N-1 employees' results. `operator.add` on `list` fields is the minimal, standard LangGraph idiom for "N branches each contribute one item, concatenate."

**Stable employee_id:** assigned in `extraction_agent.py` at record-construction time (`f"EMP-{i:03d}"`, sequential, scoped to one run). Threaded through `RetrievalOutput`, `ClassificationOutput`, `CriticOutput`. `report_node.py::_format_combined` and `app.py::_build_dataframe` both switch their join key from `name` to `employee_id`.

**Checkpoint 0 — pause for review** of `agents/schemas.py` + `agents/state.py` diff before spawning any workstream agent (everything downstream is built against this contract).

---

## Stage 1 — Workstreams A, B, E (parallel worktree agents)

Branch/worktree naming: `feature/multi-agent-redesign/ws-a-supervisor`, `.../ws-b-agents`, `.../ws-e-tests`, each its own `git worktree`, all forked from the tip of `feature/multi-agent-redesign`. File ownership is partitioned so none of the three touch the same file.

### Workstream A — owns `agents/nodes/supervisor.py` only
New file: `agents/nodes/supervisor.py`. `supervisor_node(state)` runs once after extraction: calls `dispatch_tool("get_prior_classifications", {"policy_number": ...})` from `mcp_servers/policycenter.py` **exactly once** (not per-employee), stores into `state["prior_classifications"]`, appends an `AgentStep`. `_DEFAULT_POLICY_NUMBER = "WC-DEMO-001"` moves here from `classify_node.py`. Plus `route_to_employee_fanout(state) -> list[Send]` building one `Send("employee_pipeline", {"employee_record": r, "policy_number": ..., "prior_classifications": ..., "retrieval_output": None, "classification_output": None, "critic_output": None, "audit_trail": []})` per employee record (shape = `EmployeeTaskState` from `agents/schemas.py`) — this is the graph's conditional-edge function. `"employee_pipeline"` is the node Workstream C wires up in Stage 2; A only needs to target that name, not build it.

### Workstream B — owns `agents/nodes/*` (except `supervisor.py`/`critic_agent.py`) plus `report_node.py`
- **New `agents/nodes/extraction_agent.py`** (renamed from `extract_node.py`, deleted): same dual NER+LLM logic, adds `employee_id` assignment per record.
- **New `agents/nodes/retrieval_agent.py`**: `retrieval_agent(state: EmployeeTaskState) -> dict`, a node in the per-employee subgraph (wired by Workstream C in Stage 2, not by B). The iterative self-evaluating loop replacing today's single-fallback (`_POOR_MATCH_THRESHOLD = 1.2`) logic entirely. One `chat_with_tools` call per iteration combines the sufficiency judgment + reformulated query into a single forced tool call (`evaluate_retrieval`: `sufficient`, `reasoning`, `refined_query`) — not two separate LLM calls per iteration. Loop runs up to `_MAX_ITERATIONS = 4`; on cap-out, returns best-found citations with `sufficiency_met: False` and non-empty `sufficiency_reasoning`. Returns `{"retrieval_output": {...RetrievalOutput...}, "audit_trail": [...]}`. No `dispatch_tool` import (tool-boundary: Retrieval cannot reach PolicyCenter).
- **New `agents/nodes/classification_agent.py`**: `classification_agent(state: EmployeeTaskState) -> dict`, the next node in the same subgraph, reads `state["retrieval_output"]` (set by the prior subgraph node) + `state["prior_classifications"]` (passthrough from the Send payload). Extracted from today's `_classify_employee` steps 5–8 (prompt assembly, `finalize_classification` forced tool call, result parsing). Returns `{"classification_output": {...ClassificationOutput...}, "audit_trail": [...]}`. No `NCCIRetriever` or `dispatch_tool` import (tool-boundary: Classification cannot query ChromaDB or PolicyCenter directly — must receive citations from Retrieval).
- **Delete `agents/nodes/classify_node.py`** — superseded by the two files above.
- **Modify `agents/nodes/report_node.py`**: `_format_combined` joins by `employee_id` instead of `name`.

Note: B builds `retrieval_agent.py`/`classification_agent.py` as standalone functions taking/returning `EmployeeTaskState`-shaped dicts, unit-tested by direct function call (Workstream E's tests call them directly, not through any graph). B does **not** need the subgraph itself to exist to build or test these — that wiring is Workstream C's job in Stage 2.

### Workstream E — owns `tests/*` exclusively (never touches `agents/` or `mcp_servers/`)
- New `tests/conftest.py` (shared fixtures) and `pytest.ini` (registers markers: `workstream_a/b/c/d`, `fanout_concurrency`, `tool_boundary`, `critic_disagreement`, `retrieval_convergence` — repo has neither config file today).
- **Delete** stale `tests/test_extract.py`, `tests/test_classify.py`; replace with `tests/test_extraction_agent.py`, `tests/test_retrieval_agent.py`, `tests/test_classification_agent.py`, `tests/test_report_node.py` — all against the real API (`HFClient.extract_entities`/`.generate`/`.chat_with_tools`, real state keys), each guarded with `pytest.importorskip(...)` so they skip cleanly (not fail) until B merges.
- New `tests/test_supervisor.py` (`importorskip`-guarded): asserts `dispatch_tool` called exactly once regardless of employee count (1 vs 5); asserts `route_to_employee_fanout` returns one `Send` per employee targeting `"employee_pipeline"`, each payload shaped like `EmployeeTaskState`.
- New `tests/test_fanout_concurrency.py`: a timing assertion (mock a `0.25s` sleep per `_retriever.query` call across N=4 employees; assert `elapsed < delay * N * 0.6`, proving genuine concurrency not serial execution) **plus** a correctness-parity assertion (parallel graph run vs. manual sequential loop produce identical `{(employee_id, ncci_code)}` sets, since parallel completion order isn't guaranteed).
- New `tests/test_tool_boundaries.py`: two-layered per agent — **structural** (`"dispatch_tool" not in dir(classification_agent)`) plus **behavioral** (patch the tool at its source, assert `assert_not_called()` after running the agent) — structural alone could pass by accident, behavioral alone doesn't prove the boundary is by design.
- New `tests/test_critic_agent.py`: 3–5 synthetic disagreement cases (mocked LLM returns `agrees: False` for a code contradicted by its citations) asserting `recommended_action == "flag_for_review"`, plus a blind-review test asserting the classification `rationale` string never appears in the prompt sent to `chat_with_tools`.
- Rewrite `tests/test_workflow.py` against the real `run_workflow(file_path: str)` signature and real state keys (`error`, `audit_report`, `ncci_suggestions`) — kept minimal/`importorskip`-guarded until Stage 3's full wiring.

**Merge order (orchestrator, sequential, `git merge --no-ff`): E → B → A**, running `pytest tests/ -q` after each:
```
pytest tests/ -q                    # after E: 0 failed, new B/A/C/D-dependent tests skip cleanly
pytest tests/ -k workstream_b -q    # after B: all pass
pytest tests/ -k workstream_a -q    # after A: all pass
pytest tests/ -q                    # full suite: 0 failed, only fanout/critic tests still skipped
```

**Checkpoint 1 — pause for review** after E→B→A are merged and green.

---

## Stage 2 — Workstreams C, D (parallel worktree agents, off post-A feature branch)

> **Correction found by empirical testing (post-approval, before Stage 1 spawn):** a plain `Send()` fan-out only replicates the *immediately targeted* node — any node reached afterward via a normal `add_edge` runs once, globally, after all branches converge, not once per employee (verified directly against the installed LangGraph 1.2.7; a naive `retrieval_agent → classification_agent` chain of top-level nodes loses per-employee identity and the shared `Annotated` lists are visible-but-unpartitioned mid-run). Fix, also verified empirically (4 parallel 0.2s branches completed in ~0.21s with correct per-employee isolation): wrap the per-employee stages in a small, separately-compiled **subgraph** (`EmployeeTaskState` schema, added to `agents/schemas.py`) and make *that* the `Send` target. Each stage stays its own file/function exactly as planned — only the wiring changes.

### Workstream C — owns `agents/workflow.py`, new `agents/employee_subgraph.py`, new `agents/nodes/aggregate_node.py`
- **New `agents/employee_subgraph.py`**: compiles a small `StateGraph(EmployeeTaskState)` with nodes `retrieval_agent` (from Workstream B) → `classification_agent` (from Workstream B), and exposes `employee_pipeline_node(state: EmployeeTaskState) -> dict` — a thin wrapper that calls `compiled_subgraph.invoke(state)` and reshapes the result into the parent `AuditState`'s single-item-list contributions (`retrieval_outputs`, `ncci_suggestions`, `audit_trail`). Critic is **not** in the subgraph yet in this stage (Workstream D's `critic_agent.py` doesn't exist in C's isolated worktree) — deferred to Stage 3, which adds the third node to this same file.
- **New `agents/nodes/aggregate_node.py`**: runs after all per-employee subgraph invocations reconverge in the parent graph. Validates `len(ncci_suggestions) == len(employee_records)`; for any employee present in `employee_records` but missing from `ncci_suggestions` (branch failure), marks `"NOT CLASSIFIED — retry required"` and logs a warning `AgentStep` (per the "degrade gracefully" decision — does not fail the whole run). Does not itself merge lists — the `operator.add` reducers already did that; this node is validation + shape-prep only.
- **Modify `agents/workflow.py`**: rewire to `intake → extraction_agent → supervisor →(Send fan-out to "employee_pipeline")→ aggregate_node → report_node → END`, where `"employee_pipeline"` is `employee_pipeline_node` from `agents/employee_subgraph.py`. Update `run_workflow`'s `initial_state` with the new fields (`prior_classifications: {}`, `retrieval_outputs: []`, `critic_assessments: []`, `audit_trail: []`).

### Workstream D — owns `agents/nodes/critic_agent.py` only (never touches `workflow.py` or the subgraph file)
New file: `agents/nodes/critic_agent.py`. `critic_agent(state: EmployeeTaskState) -> dict` builds `CriticInput` from `state["classification_output"]` (**strips `rationale` before building the prompt** — structural, not conventions-based, blind-review enforcement) + `state["retrieval_output"]["policy_citations"]` (raw). Forced tool call `finalize_critic_review` (`agrees`, `concern`, `recommended_action`, `critic_confidence`). Returns `{"critic_output": {...}, "audit_trail": [...]}` — plain fields, since within one isolated subgraph invocation there's only ever one writer. Unit-tested by calling the function directly against an `EmployeeTaskState` fixture, not through any compiled graph — it isn't wired into the subgraph until Stage 3.

**Merge order: C → D**, `pytest tests/ -q` after each (fanout tests flip to pass after C; critic tests flip to pass after D as standalone-function tests, but end-to-end critic-in-subgraph assertions stay skipped until Stage 3).

**Checkpoint 2 — pause for review** after C and D are merged and green.

---

## Stage 3 — Final integration (orchestrator directly, not a worktree agent)

1. **`agents/employee_subgraph.py`**: splice `critic_agent` in as the third subgraph node, after `classification_agent`; update `employee_pipeline_node`'s reshaping logic to also emit `critic_assessments`.
2. **`api/main.py`**: `/audit/run` response gains `critic_assessments` and `audit_trail` — additive only, no existing fields removed.
3. **`app.py`**: fix `_build_dataframe`'s fragile name-join the same way as `report_node.py` (switch to `employee_id`); add a minimal Critic status column/flag to the Employee Classification DataFrame in Tab 1 (per the locked UI-surfacing decision) so `flag_for_review` is actually visible to the auditor, not just in the JSON audit trail.
4. Unskip `tests/test_workflow.py`'s full end-to-end assertions (critic present in trail, `sufficiency_met` field present, aggregate validation, N-employee fan-out produces N suggestions + N critic assessments, degraded-employee case shows `"NOT CLASSIFIED — retry required"`).

**Final verification:**
```
pytest tests/ -q                         # 0 failed, 0 skipped
python3 -c "from agents.workflow import run_workflow; r = run_workflow('data/sample_payroll_register.pdf'); print(r['error'], len(r['ncci_suggestions']), len(r['critic_assessments']))"
```
Then start the app (`python app.py`) and manually run the sample payroll register through the Gradio UI to confirm the DataFrame renders the new Critic column and the report still generates correctly end-to-end — this is a UI change (app.py), so per project norms it needs an actual browser check, not just pytest green.

**Checkpoint 3 — final review.** `main` is untouched throughout; nothing is pushed or redeployed. Merging `feature/multi-agent-redesign` into `main` (and any HF Spaces redeploy) is an explicit, separate, later action the user takes — not part of this plan.

---

## Critical files

| File | Role |
|---|---|
| `agents/schemas.py` (new) | Typed handoff contracts — everything else builds against this |
| `agents/state.py` | `AuditState` + `operator.add` reducers for parallel-written fields |
| `agents/nodes/supervisor.py` (new) | One-time PolicyCenter fetch + `Send`-based per-employee fan-out |
| `agents/employee_subgraph.py` (new) | Compiled per-employee subgraph (retrieval→classification→critic) + `Send`-target wrapper; the fix for LangGraph's per-branch-identity limitation |
| `agents/nodes/extraction_agent.py` (renamed) | Document-level NER+LLM extraction, assigns `employee_id` |
| `agents/nodes/retrieval_agent.py` (new) | Iterative self-evaluating RAG loop, replaces static fallback logic |
| `agents/nodes/classification_agent.py` (new) | Forced-tool-call classification, no direct tool access |
| `agents/nodes/critic_agent.py` (new) | Blind-review verifier, structurally withholds rationale |
| `agents/nodes/aggregate_node.py` (new) | Post-fan-out validation + graceful partial-failure handling |
| `agents/workflow.py` | Graph rewiring — intake→extraction→supervisor→fan-out→...→aggregate→report |
| `mcp_servers/policycenter.py`, `rag/retriever.py`, `models/hf_client.py` | **Reused as-is** — underlying primitives (`dispatch_tool`, `NCCIRetriever`, `HFClient.chat_with_tools`) called by the new agents, no signature changes |
| `report_node.py`, `app.py` | Both need the `name`→`employee_id` join fix; `app.py` also gets the Critic status column |
| `tests/conftest.py`, `tests/pytest.ini` (new) | Shared fixtures + markers for the new staged test suite |

## Verification summary

Each stage ends with `pytest tests/ -q` run by the orchestrator after every merge (not just at the end) — failing tests block progression to the next stage. Stage-specific commands are listed inline above. Final Stage 3 verification adds a manual Gradio smoke test since `app.py` changes are a UI change that pytest can't confirm visually.
