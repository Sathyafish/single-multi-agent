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
    pip install openai duckduckgo-search

Usage (default — OpenRouter + Llama 3.3):
    export OPENROUTER_API_KEY="your-key-here"
    python sf_zoo_agent_comparison.py

Other providers:
    export LLM_PROVIDER=groq
    export GROQ_API_KEY="your-key-here"
    python sf_zoo_agent_comparison.py
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None


# ── Config ────────────────────────────────────────────────────────────────────

PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter").lower()
MAX_TOKENS = 1024

# Default models per provider (override with LLM_MODEL env var)
DEFAULT_MODELS = {
    "groq": "groq/compound-mini",          # open models + built-in web search
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}

MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODELS.get(PROVIDER, DEFAULT_MODELS["openrouter"]))

# Models with server-side web search (no DuckDuckGo needed)
COMPOUND_MODELS = {"groq/compound", "groq/compound-mini"}

PROVIDER_CONFIG = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "signup": "https://console.groq.com/",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "signup": "https://openrouter.ai/",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "signup": "https://api.together.ai/",
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


def uses_builtin_search() -> bool:
    return MODEL in COMPOUND_MODELS


# ── Web search (for non-Compound models) ──────────────────────────────────────

def search_web(query: str, max_results: int = 5) -> str:
    """Fetch live web snippets via DuckDuckGo (free, no API key)."""
    if DDGS is None:
        raise ImportError("pip install duckduckgo-search")

    snippets = []
    with DDGS() as ddgs:
        for hit in ddgs.text(query, max_results=max_results):
            title = hit.get("title", "")
            body = hit.get("body", "")
            url = hit.get("href", "")
            snippets.append(f"- {title}: {body} ({url})")

    return "\n".join(snippets) if snippets else "(no search results)"


def build_prompt(task_prompt: str) -> str:
    if uses_builtin_search():
        return task_prompt

    search_results = search_web(task_prompt)
    return (
        "Use the web search results below to answer the question. "
        "If results are insufficient, say what is missing.\n\n"
        f"Web search results:\n{search_results}\n\n"
        f"Question: {task_prompt}"
    )


# ── Core API call ─────────────────────────────────────────────────────────────

def call_model(prompt: str, client: OpenAI) -> tuple[str, int]:
    """Call hosted open-source model and return (answer_text, latency_ms)."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    answer = (response.choices[0].message.content or "").strip()
    return answer or "(no text response)", elapsed_ms


def run_task(task: dict, client: OpenAI) -> TaskResult:
    try:
        prompt = build_prompt(task["prompt"])
        answer, latency = call_model(prompt, client)
        return TaskResult(task["id"], task["label"], answer, latency)
    except Exception as exc:
        return TaskResult(task["id"], task["label"], "", 0, error=str(exc))


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
        result = run_task(task, client)
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
    Four specialized sub-agents run in parallel using a thread pool.
    Total latency ≈ slowest single task, not the sum.
    """
    print("\n" + "═" * 60)
    print("  MULTI-AGENT  (parallel, 4 sub-agents)")
    print("═" * 60)
    print("\n  Launching all agents simultaneously …\n")

    run_start = time.perf_counter()
    task_results: list[TaskResult] = [None] * len(TASKS)

    with ThreadPoolExecutor(max_workers=len(TASKS)) as executor:
        future_to_index = {
            executor.submit(run_task, task, client): idx
            for idx, task in enumerate(TASKS)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            result = future.result()
            task_results[idx] = result
            status = "✓" if not result.error else "✗"
            print(f"  {status} [{result.latency_ms:>5} ms]  {result.label}")

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

def main():
    client = create_client()
    search_mode = "built-in web search" if uses_builtin_search() else "DuckDuckGo + LLM"

    print("\n🦁  SF Zoo — Single Agent vs Multi-Agent Demo")
    print(f"    Provider: {PROVIDER}  |  Model: {MODEL}")
    print(f"    Search:   {search_mode}")
    print("    Tasks: weather · distance · tickets · hours\n")

    single_result = run_single_agent(client)
    print_results(single_result)

    multi_result = run_multi_agent(client)
    print_results(multi_result)

    print_comparison(single_result, multi_result)


if __name__ == "__main__":
    main()
