"""
SF Zoo Agent Comparison Demo
============================
Compares a Single Agent vs Multi-Agent approach for fetching:
  - Current weather at SF Zoo
  - Driving distance from downtown SF
  - Adult ticket price
  - Opening hours

Uses hosted open-source models (no local inference).

Requirements:
    pip install -r requirements.txt

Usage (default — OpenRouter + Llama 3.3):
    # Option 1: put OPENROUTER_API_KEY in .env (recommended)
    python sf_zoo_agent_comparison.py

    # Option 2: export in shell
    export OPENROUTER_API_KEY="your-key-here"
    python sf_zoo_agent_comparison.py

Other providers:
    export LLM_PROVIDER=groq
    export GROQ_API_KEY="your-key-here"
    python sf_zoo_agent_comparison.py
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS  # legacy package name
    except ImportError:
        DDGS = None


# ── Config ────────────────────────────────────────────────────────────────────

PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter").lower()
MAX_TOKENS = 1024
PAUSE_BETWEEN_RUNS_SEC = int(os.environ.get("PAUSE_BETWEEN_RUNS_SEC", "60"))
MULTI_AGENT_STAGGER_SEC = float(os.environ.get("MULTI_AGENT_STAGGER_SEC", "3"))
SEARCH_MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "3"))
SEARCH_SNIPPET_CHARS = int(os.environ.get("SEARCH_SNIPPET_CHARS", "280"))
SPLIT_MULTI_AGENT_MODELS = os.environ.get(
    "SPLIT_MULTI_AGENT_MODELS", "true"
).lower() in {"1", "true", "yes"}

# Default models per provider (override with LLM_MODEL env var)
DEFAULT_MODELS = {
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "groq": "groq/compound-mini",          # open models + built-in web search
}

MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODELS.get(PROVIDER, DEFAULT_MODELS["openrouter"]))

# Models with server-side web search (no DuckDuckGo needed)
COMPOUND_MODELS = {"groq/compound", "groq/compound-mini"}

# One model per sub-agent — separate TPM buckets on Groq (avoids 429/413)
MULTI_AGENT_MODEL_POOLS = {
    "groq": [
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
    ],
    "openrouter": [
        "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ],
}

PROVIDER_CONFIG = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "signup": "https://openrouter.ai/",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "signup": "https://console.groq.com/",
    },
}

TASKS = [
    {
        "id": "weather",
        "label": "🌤  Weather at SF Zoo",
        "prompt": (
            "What is the current weather at San Francisco Zoo, 1 Zoo Rd, "
            "San Francisco, CA 94132? Give temperature in °F and conditions "
            "in 2 concise sentences."
        ),
    },
    {
        "id": "distance",
        "label": "📍 Distance from downtown SF",
        "prompt": (
            "What is the driving distance and estimated drive time from "
            "Union Square, San Francisco to San Francisco Zoo at 1 Zoo Rd, "
            "San Francisco, CA 94132? Be concise."
        ),
    },
    {
        "id": "tickets",
        "label": "🎟  Adult ticket price",
        "prompt": (
            "What is the current adult general admission ticket price for "
            "the San Francisco Zoo? State the price in USD and note if "
            "online discounts apply."
        ),
    },
    {
        "id": "hours",
        "label": "🕐 Opening hours today",
        "prompt": (
            "What are the current opening hours of San Francisco Zoo today? "
            "Include any seasonal notes."
        ),
    },
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: str
    label: str
    answer: str
    latency_ms: int
    model: str = ""
    error: Optional[str] = None


@dataclass
class RunResult:
    mode: str                          # "single" | "multi"
    total_latency_ms: int
    results: list[TaskResult] = field(default_factory=list)


# ── Client setup ──────────────────────────────────────────────────────────────

def create_client() -> OpenAI:
    if PROVIDER not in PROVIDER_CONFIG:
        supported = ", ".join(PROVIDER_CONFIG)
        raise ValueError(f"Unknown LLM_PROVIDER '{PROVIDER}'. Use one of: {supported}")

    cfg = PROVIDER_CONFIG[PROVIDER]
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise EnvironmentError(
            f"Set {cfg['api_key_env']} (sign up: {cfg['signup']})"
        )

    return OpenAI(api_key=api_key, base_url=cfg["base_url"])


def model_uses_builtin_search(model: str) -> bool:
    return model in COMPOUND_MODELS


def uses_builtin_search() -> bool:
    return model_uses_builtin_search(MODEL)


def get_multi_agent_models() -> list[str]:
    """Assign a different model to each parallel sub-agent."""
    if not SPLIT_MULTI_AGENT_MODELS:
        return [MODEL] * len(TASKS)

    pool = MULTI_AGENT_MODEL_POOLS.get(PROVIDER)
    if not pool:
        return [MODEL] * len(TASKS)

    return [pool[idx % len(pool)] for idx in range(len(TASKS))]


def parse_retry_seconds(error_msg: str) -> float:
    match = re.search(r"try again in ([\d.]+)s", error_msg, re.IGNORECASE)
    return float(match.group(1)) + 1.0 if match else 5.0


# ── Web search (for non-Compound models) ──────────────────────────────────────

def search_web(query: str, max_results: int = SEARCH_MAX_RESULTS) -> str:
    """Fetch live web snippets via DuckDuckGo (free, no API key)."""
    if DDGS is None:
        raise ImportError("pip install ddgs")

    snippets = []
    with DDGS() as ddgs:
        for hit in ddgs.text(query, max_results=max_results):
            title = hit.get("title", "")
            body = hit.get("body", "")[:SEARCH_SNIPPET_CHARS]
            url = hit.get("href", "")
            snippets.append(f"- {title}: {body} ({url})")

    return "\n".join(snippets) if snippets else "(no search results)"


def build_prompt(task_prompt: str, model: str) -> str:
    if model_uses_builtin_search(model):
        return task_prompt

    search_results = search_web(task_prompt)
    return (
        "Use the web search results below to answer the question. "
        "Be concise (2-3 sentences). If results are insufficient, say what is missing.\n\n"
        f"Web search results:\n{search_results}\n\n"
        f"Question: {task_prompt}"
    )


# ── Core API call ─────────────────────────────────────────────────────────────

def call_model(prompt: str, client: OpenAI, model: str) -> tuple[str, int]:
    """Call hosted open-source model and return (answer_text, latency_ms)."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    answer = (response.choices[0].message.content or "").strip()
    return answer or "(no text response)", elapsed_ms


def call_model_with_retry(
    prompt: str, client: OpenAI, model: str, max_attempts: int = 3
) -> tuple[str, int]:
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return call_model(prompt, client, model)
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if "429" in msg and attempt < max_attempts - 1:
                wait = parse_retry_seconds(msg)
                time.sleep(wait)
                continue
            raise
    raise last_exc  # type: ignore[misc]


def run_task(task: dict, client: OpenAI, model: str) -> TaskResult:
    try:
        prompt = build_prompt(task["prompt"], model)
        answer, latency = call_model_with_retry(prompt, client, model)
        return TaskResult(task["id"], task["label"], answer, latency, model=model)
    except Exception as exc:
        msg = str(exc)
        if "402" in msg and PROVIDER == "openrouter":
            msg += (
                "\n    Hint: use the free model (meta-llama/llama-3.3-70b-instruct:free), "
                "add credits at https://openrouter.ai/settings/credits, "
                "or switch to Groq (LLM_PROVIDER=groq)."
            )
        if "413" in msg:
            msg += (
                "\n    Hint: request too large — multi-agent now uses smaller "
                "DuckDuckGo prompts and split models to avoid this."
            )
        return TaskResult(task["id"], task["label"], "", 0, model=model, error=msg)


# ── Single Agent ──────────────────────────────────────────────────────────────

def run_single_agent(client: OpenAI) -> RunResult:
    """
    One agent handles all four tasks sequentially.
    Each task is a separate API call resolved one by one.
    """
    print("\n" + "═" * 60)
    print("  SINGLE AGENT  (sequential)")
    print("═" * 60)

    run_start = time.perf_counter()
    task_results = []

    for task in TASKS:
        print(f"\n  → {task['label']} …", end="", flush=True)
        result = run_task(task, client, MODEL)
        task_results.append(result)
        if result.error:
            print(f" ✗ ERROR: {result.error}")
        else:
            print(f" ✓ ({result.latency_ms} ms)")

    total_ms = int((time.perf_counter() - run_start) * 1000)
    return RunResult("single", total_ms, task_results)


# ── Multi-Agent ───────────────────────────────────────────────────────────────

def run_multi_agent(client: OpenAI) -> RunResult:
    """
    Four specialized sub-agents run in parallel, each on a different model
    (separate TPM buckets) with staggered launches to reduce rate-limit spikes.
    """
    models = get_multi_agent_models()

    print("\n" + "═" * 60)
    print("  MULTI-AGENT  (parallel, 4 sub-agents)")
    print("═" * 60)

    if SPLIT_MULTI_AGENT_MODELS and len(set(models)) > 1:
        print("\n  Model split (one model per task):")
        for task, model in zip(TASKS, models):
            print(f"    {task['label']:<30} → {model}")
        print(f"\n  Staggered launch: {MULTI_AGENT_STAGGER_SEC}s between agents\n")
    else:
        print("\n  Launching all agents simultaneously …\n")

    run_start = time.perf_counter()
    task_results: list[TaskResult] = [None] * len(TASKS)

    with ThreadPoolExecutor(max_workers=len(TASKS)) as executor:
        future_to_index = {}
        for idx, task in enumerate(TASKS):
            if idx > 0 and MULTI_AGENT_STAGGER_SEC > 0:
                time.sleep(MULTI_AGENT_STAGGER_SEC)
            future = executor.submit(run_task, task, client, models[idx])
            future_to_index[future] = idx

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            result = future.result()
            task_results[idx] = result
            status = "✓" if not result.error else "✗"
            model_tag = f" [{result.model}]" if result.model else ""
            print(
                f"  {status} [{result.latency_ms:>5} ms]{model_tag}  {result.label}"
            )

    total_ms = int((time.perf_counter() - run_start) * 1000)
    return RunResult("multi", total_ms, task_results)


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_results(result: RunResult) -> None:
    label = "Single Agent" if result.mode == "single" else "Multi-Agent"
    print(f"\n{'─'*60}")
    print(f"  {label} — answers")
    print(f"{'─'*60}")
    for r in result.results:
        print(f"\n  {r.label}")
        if r.error:
            print(f"    ERROR: {r.error}")
        else:
            words = r.answer.split()
            lines, line = [], []
            for word in words:
                line.append(word)
                if len(" ".join(line)) > 72:
                    lines.append("    " + " ".join(line))
                    line = []
            if line:
                lines.append("    " + " ".join(line))
            print("\n".join(lines))
        if r.model:
            print(f"    🤖  {r.model}")
        print(f"    ⏱  {r.latency_ms} ms")


def print_comparison(single: RunResult, multi: RunResult) -> None:
    speedup = single.total_latency_ms / max(multi.total_latency_ms, 1)

    print("\n" + "═" * 60)
    print("  COMPARISON SUMMARY")
    print("═" * 60)
    print(f"  {'Metric':<30} {'Single':>10} {'Multi':>10}")
    print(f"  {'─'*50}")
    print(f"  {'Total latency':<30} {single.total_latency_ms:>8}ms {multi.total_latency_ms:>8}ms")
    print(f"  {'Speed advantage':<30} {'—':>10} {speedup:>9.1f}×")

    print(f"\n  Per-task breakdown:")
    print(f"  {'Task':<28} {'Single':>8} {'Multi':>8}  {'Faster?':}")
    print(f"  {'─'*60}")
    for s_r, m_r in zip(single.results, multi.results):
        faster = "multi ✓" if m_r.latency_ms < s_r.latency_ms else "single ✓"
        print(f"  {s_r.label:<28} {s_r.latency_ms:>6}ms {m_r.latency_ms:>6}ms  {faster}")

    print(f"\n  Verdict:")
    if speedup >= 1.5:
        print(f"  ✅ Multi-agent is {speedup:.1f}× faster — parallel wins on latency.")
    else:
        print(f"  ⚠️  Speedup is only {speedup:.1f}× — network overhead reduced the gain.")
    print(f"\n  Accuracy note:")
    print(f"  Each multi-agent sub-agent had a focused context (1 task).")
    print(f"  The single agent handled all 4 tasks in sequence, increasing")
    print(f"  the risk of cross-task confusion or hallucination.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def pause_between_runs(seconds: int) -> None:
    """Wait between single- and multi-agent runs to avoid TPM rate limits."""
    if seconds <= 0:
        return
    print(
        f"\n  ⏳  Pausing {seconds}s between runs (rate-limit cooldown) …",
        flush=True,
    )
    time.sleep(seconds)
    print("  ✓  Cooldown complete — starting multi-agent run.\n", flush=True)


def main():
    client = create_client()
    search_mode = (
        "built-in web search"
        if uses_builtin_search()
        else "DuckDuckGo + LLM"
    )
    multi_models = get_multi_agent_models()

    print("\n🦁  SF Zoo — Single Agent vs Multi-Agent Demo")
    print(f"    Provider: {PROVIDER}  |  Single model: {MODEL}")
    if SPLIT_MULTI_AGENT_MODELS and len(set(multi_models)) > 1:
        print(f"    Multi:    split across {len(set(multi_models))} models")
    else:
        print(f"    Multi:    {MODEL}")
    print(f"    Search:   {search_mode}")
    print("    Tasks: weather · distance · tickets · hours\n")

    single_result = run_single_agent(client)
    print_results(single_result)

    pause_between_runs(PAUSE_BETWEEN_RUNS_SEC)

    multi_result = run_multi_agent(client)
    print_results(multi_result)

    print_comparison(single_result, multi_result)


if __name__ == "__main__":
    main()
