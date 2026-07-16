import os
import re
import time
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from huggingface_hub import InferenceClient

load_dotenv()

NER_MODEL      = "dslim/bert-base-NER"
EMBED_MODEL    = "BAAI/bge-large-en-v1.5"
GENERATE_MODEL = "llama-3.3-70b-versatile"   # Groq model ID

_MAX_RATE_LIMIT_RETRIES = 3
_DEFAULT_BACKOFF_SECONDS = 2.0
_RETRY_HINT_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


def _retry_after_seconds(exc: RateLimitError) -> float:
    """How long to wait before retrying, in order of preference: the response's
    Retry-After header, Groq's "try again in Xs" hint in the error message,
    then a fixed fallback."""
    header = exc.response.headers.get("retry-after") if exc.response is not None else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = _RETRY_HINT_RE.search(str(exc))
    if match:
        return float(match.group(1))
    return _DEFAULT_BACKOFF_SECONDS


def _call_with_retry(fn, *args, **kwargs):
    """Call a Groq API function, retrying on 429 rate-limit responses.

    Concurrent per-employee LLM calls (the whole point of the multi-agent
    fan-out) can burst past Groq's per-minute token limit even when total
    usage is unremarkable, because the old sequential pipeline spread the
    same calls out over the run's full wall-clock time and this one doesn't.
    Retrying with the wait time Groq itself reports keeps the concurrency
    benefit instead of serializing everything or capping fan-out width to
    dodge a free-tier-specific ceiling.
    """
    for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as exc:
            if attempt == _MAX_RATE_LIMIT_RETRIES:
                raise
            wait = _retry_after_seconds(exc) + 0.5  # small buffer past Groq's own estimate
            print(f"Groq rate limit hit (attempt {attempt + 1}/{_MAX_RATE_LIMIT_RETRIES}); retrying in {wait:.1f}s")
            time.sleep(wait)


class HFClient:
    def __init__(self) -> None:
        hf_token   = os.getenv("HF_TOKEN")
        groq_token = os.getenv("GROQ_API_KEY")
        # HF Serverless for NER + embeddings; Groq for LLM generation
        self._hf  = InferenceClient(provider="hf-inference", token=hf_token)
        self._gen = Groq(api_key=groq_token)

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
        Retries on 429 rate-limit responses — see _call_with_retry."""
        return _call_with_retry(
            self._gen.chat.completions.create,
            model=GENERATE_MODEL,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=1024,
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
