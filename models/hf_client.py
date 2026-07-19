import os
import random
import re
import threading
import time
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from huggingface_hub import InferenceClient

load_dotenv()

NER_MODEL      = "dslim/bert-base-NER"
EMBED_MODEL    = "BAAI/bge-large-en-v1.5"
GENERATE_MODEL = "llama-3.3-70b-versatile"   # Groq model ID

_MAX_RATE_LIMIT_RETRIES = 6
_DEFAULT_BACKOFF_SECONDS = 2.0
_JITTER_FRACTION = 0.3  # +/- 30% randomization on every wait
_RETRY_HINT_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)

# Bounds how many Groq requests this process has in flight at once. The
# multi-agent fan-out sends several employees' retrieval/classification/critic
# calls concurrently at the GRAPH level by design (that's the redesign's whole
# latency win — LangGraph still runs N employees' branches concurrently
# regardless of this value), but Groq's free/on-demand tier caps at a modest
# tokens-per-minute budget shared across the whole organization. Without a
# local cap, N employees' calls can all fire at once, all get 429'd together,
# and — without jitter — all retry at close to the same instant, recreating
# the exact same burst on every retry round (a "thundering herd") instead of
# converging.
#
# Set to 1 (serialize the actual Groq HTTP calls) rather than a higher number:
# observed live on a fresh org's tight per-minute budget, 2 concurrent calls
# still produced enough contention that some employees exhausted even 6
# retries. Serializing the calls themselves (not the graph-level concurrency)
# trades a bit of wall-clock time for reliability — worth it since a failed
# branch is a visible, debuggable "NOT CLASSIFIED" row now (see
# agents/employee_subgraph.py's fault-isolation boundary), but still better
# to avoid triggering when the budget is this tight.
_MAX_CONCURRENT_GROQ_CALLS = 1
_groq_call_slots = threading.Semaphore(_MAX_CONCURRENT_GROQ_CALLS)


def _retry_after_seconds(exc: RateLimitError) -> float:
    """How long to wait before retrying, in order of preference: the response's
    Retry-After header, Groq's "try again in Xs" hint in the error message,
    then a fixed fallback. Jittered so concurrent callers that got rate-limited
    in the same instant don't all retry in lockstep and re-trigger the same
    burst on every round."""
    base = _DEFAULT_BACKOFF_SECONDS
    header = exc.response.headers.get("retry-after") if exc.response is not None else None
    if header:
        try:
            base = float(header)
        except ValueError:
            pass
    else:
        match = _RETRY_HINT_RE.search(str(exc))
        if match:
            base = float(match.group(1))

    jitter = base * _JITTER_FRACTION
    return base + random.uniform(-jitter, jitter)


def _call_with_retry(fn, *args, **kwargs):
    """Call a Groq API function, bounding concurrency and retrying on 429
    rate-limit responses.

    Concurrent per-employee LLM calls (the whole point of the multi-agent
    fan-out) can burst past Groq's per-minute token limit even when total
    usage is unremarkable, because the old sequential pipeline spread the
    same calls out over the run's full wall-clock time and this one doesn't.
    A semaphore caps how many requests are actually in flight at once (see
    _MAX_CONCURRENT_GROQ_CALLS), and jittered retries on the ones that still
    get rate-limited keep the concurrency benefit instead of serializing
    everything or capping fan-out width to dodge a free-tier-specific ceiling.
    """
    for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
        with _groq_call_slots:
            try:
                return fn(*args, **kwargs)
            except RateLimitError as exc:
                if attempt == _MAX_RATE_LIMIT_RETRIES:
                    raise
                wait = _retry_after_seconds(exc)
        # Sleep outside the semaphore so other queued callers get their turn
        # to try while this one is backing off.
        print(f"Groq rate limit hit (attempt {attempt + 1}/{_MAX_RATE_LIMIT_RETRIES}); retrying in {wait:.1f}s", flush=True)
        time.sleep(wait)


class HFClient:
    def __init__(self) -> None:
        hf_token   = os.getenv("HF_TOKEN")
        groq_token = os.getenv("GROQ_API_KEY")
        # HF Serverless for NER + embeddings; Groq for LLM generation
        self._hf  = InferenceClient(provider="hf-inference", token=hf_token)
        # max_retries=0: the Groq SDK has its own built-in retry (default 2,
        # with its own silent backoff sleep) that runs *inside* every
        # .create() call, invisible to and uncoordinated with our own
        # _call_with_retry below — the two layered on top of each other
        # meant a single rate-limited call could silently retry up to 2
        # times inside the SDK, then up to 3 more times in our wrapper, each
        # with its own sleep, with no logging for the SDK's half at all.
        # Disabling the SDK's layer makes _call_with_retry the single,
        # fully-visible source of retry/backoff behavior.
        self._gen = Groq(api_key=groq_token, max_retries=0)

    def extract_entities(self, text: str) -> list[dict]:
        """Run NER over text, return deduplicated [{entity, label, score}]."""
        raw = self._hf.token_classification(text, model=NER_MODEL)
        seen: set[str] = set()
        result: list[dict] = []
        for item in raw:
            word = item.get("word", "").strip()
            if not word or word.startswith("##") or word in seen:
                continue
            seen.add(word)
            result.append({
                "entity": word,
                "label": item.get("entity_group", item.get("entity", "")),
                "score": round(float(item.get("score", 0.0)), 4),
            })
        return result

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts (for ChromaDB ingestion)."""
        raw = self._hf.feature_extraction(texts, model=EMBED_MODEL)
        return [list(map(float, vec)) for vec in raw]

    def generate(self, prompt: str, system: str) -> str:
        """Generate a text response using Llama 3.3 70B via Groq.
        Retries on 429 rate-limit responses — see _call_with_retry."""
        response = _call_with_retry(
            self._gen.chat.completions.create,
            model=GENERATE_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def chat_with_tools(self, messages: list[dict], tools: list[dict], tool_choice: str = "auto"):
        """Run a Groq chat completion with tool definitions.
        Returns the raw response so callers can inspect finish_reason and tool_calls.
        Pass tool_choice='required' to force the model to call a tool (structured output).
        Retries on 429 rate-limit responses — see _call_with_retry.

        max_tokens=2048 (not 1024): observed live — llama-3.3-70b's "reasoning"-style
        tool arguments (e.g. evaluate_retrieval, finalize_critic_review) can run long
        enough that 1024 truncates the generation mid-JSON, before the closing brace
        and </function> tag are emitted. A truncated tool call isn't valid JSON, so
        Groq's server-side parser rejects it outright with a 400 "Failed to call a
        function" — this doesn't raise RateLimitError, so _call_with_retry's retry
        loop never engages for it, and (since it's the same malformed request every
        time) retrying wouldn't help anyway; the actual fix is giving the model
        enough budget to finish the structured output it started.
        """
        return _call_with_retry(
            self._gen.chat.completions.create,
            model=GENERATE_MODEL,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=2048,
        )


if __name__ == "__main__":
    client = HFClient()

    # --- NER smoke test ---
    sample_text = (
        "ABC Roofing LLC employs John Smith as a roofer. "
        "Q1 2024 wages: $18,500. Subcontractor: Rivera Exteriors LLC."
    )
    print("=== extract_entities ===")
    entities = client.extract_entities(sample_text)
    for e in entities:
        print(f"  {e['label']:6s}  {e['entity']:30s}  score={e['score']:.3f}")

    # --- Embeddings smoke test ---
    print("\n=== embed_text ===")
    sample_texts = ["Roofer installing shingles", "Clerical office employee", "Plumber — residential"]
    vectors = client.embed_text(sample_texts)
    for text, vec in zip(sample_texts, vectors):
        print(f"  '{text}' → dim={len(vec)}, first3={vec[:3]}")

    # --- Generation smoke test ---
    print("\n=== generate ===")
    answer = client.generate(
        system="You are a workers' compensation premium audit specialist with deep knowledge of NCCI class codes.",
        prompt=(
            "A payroll register shows an employee listed as 'roofer' at a residential roofing company. "
            "What NCCI class code applies? Give the code and a one-sentence rationale."
        ),
    )
    print(f"  {answer}")
