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
import threading
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
    key_label: str = ""
    error: Optional[str] = None


@dataclass
class RunResult:
    mode: str                          # "single" | "multi"
    total_latency_ms: int
    results: list[TaskResult] = field(default_factory=list)


# ── Client setup ──────────────────────────────────────────────────────────────

def load_api_keys() -> list[str]:
    """Load one or more API keys (GROQ_API_KEY, GROQ_API_KEY_2, …)."""
    cfg = PROVIDER_CONFIG[PROVIDER]
    env = cfg["api_key_env"]
    keys: list[str] = []

    bulk = os.environ.get(f"{env}S")
    if bulk:
        keys.extend(k.strip() for k in bulk.split(",") if k.strip())

    for name in [env] + [f"{env}_{i}" for i in range(2, 10)]:
        value = os.environ.get(name)
        if value and value not in keys:
            keys.append(value)

    return keys


def create_clients() -> list[OpenAI]:
    if PROVIDER not in PROVIDER_CONFIG:
        supported = ", ".join(PROVIDER_CONFIG)
        raise ValueError(f"Unknown LLM_PROVIDER '{PROVIDER}'. Use one of: {supported}")

    keys = load_api_keys()
    if not keys:
        cfg = PROVIDER_CONFIG[PROVIDER]
        raise EnvironmentError(
            f"Set {cfg['api_key_env']} (sign up: {cfg['signup']})"
        )

    cfg = PROVIDER_CONFIG[PROVIDER]
    return [OpenAI(api_key=key, base_url=cfg["base_url"]) for key in keys]


def client_for_index(clients: list[OpenAI], idx: int) -> tuple[OpenAI, str]:
    """Round-robin client selection across available API keys."""
    slot = idx % len(clients)
    return clients[slot], f"key #{slot + 1}"


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

_search_lock = threading.Lock()


def search_web(query: str, max_results: int = SEARCH_MAX_RESULTS) -> str:
    """Fetch live web snippets via DuckDuckGo (free, no API key)."""
    if DDGS is None:
        raise ImportError("pip install ddgs")

    snippets = []
    # DDGS is not thread-safe — serialize all search calls
    with _search_lock:
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=max_results):
                title = hit.get("title", "")
                body = hit.get("body", "")[:SEARCH_SNIPPET_CHARS]
                url = hit.get("href", "")
                snippets.append(f"- {title}: {body} ({url})")

    return "\n".join(snippets) if snippets else "(no search results)"


def format_search_prompt(task_prompt: str, search_results: str) -> str:
    return (
        "Use the web search results below to answer the question. "
        "Be concise (2-3 sentences). If results are insufficient, say what is missing.\n\n"
        f"Web search results:\n{search_results}\n\n"
        f"Question: {task_prompt}"
    )


def build_prompt(
    task_prompt: str,
    model: str,
    search_results: Optional[str] = None,
) -> str:
    if model_uses_builtin_search(model):
        return task_prompt

    if search_results is None:
        search_results = search_web(task_prompt)
    return format_search_prompt(task_prompt, search_results)


def prefetch_prompts(tasks: list[dict], models: list[str]) -> dict[str, Optional[str]]:
    """
    Run all DuckDuckGo searches sequentially before parallel LLM calls.
    Avoids SSL/thread errors (e.g. 'Unsupported protocol version 0x304').
    """
    cache: dict[str, Optional[str]] = {}
    needs_search = any(
        not model_uses_builtin_search(model) for model in models
    )
    if not needs_search:
        for task in tasks:
            cache[task["id"]] = task["prompt"]
        return cache

    print("\n  Prefetching web searches (sequential) …")
    for task, model in zip(tasks, models):
        if model_uses_builtin_search(model):
            cache[task["id"]] = task["prompt"]
            continue
        print(f"    → {task['label']} …", end="", flush=True)
        try:
            results = search_web(task["prompt"])
            cache[task["id"]] = format_search_prompt(task["prompt"], results)
            print(" ✓")
        except Exception as exc:
            cache[task["id"]] = None
            print(f" ✗ {exc}")
        time.sleep(0.5)

    return cache


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
    prompt: str,
    clients: list[OpenAI],
    model: str,
    client_idx: int,
    max_attempts: int = 3,
) -> tuple[str, int, int]:
    """Returns (answer, latency_ms, client_slot_used)."""
    last_exc: Optional[Exception] = None
    slot = client_idx % len(clients)

    for attempt in range(max_attempts):
        try:
            answer, latency = call_model(prompt, clients[slot], model)
            return answer, latency, slot
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if "429" in msg and attempt < max_attempts - 1:
                if len(clients) > 1:
                    slot = (slot + 1) % len(clients)
                wait = parse_retry_seconds(msg)
                time.sleep(wait)
                continue
            raise
    raise last_exc  # type: ignore[misc]


def run_task(
    task: dict,
    clients: list[OpenAI],
    model: str,
    client_idx: int,
    prebuilt_prompt: Optional[str] = None,
    search_failed: bool = False,
) -> TaskResult:
    _, key_label = client_for_index(clients, client_idx)
    try:
        if search_failed:
            return TaskResult(
                task["id"],
                task["label"],
                "",
                0,
                model=model,
                key_label=key_label,
                error="Web search failed during prefetch",
            )
        if prebuilt_prompt is None:
            prompt = build_prompt(task["prompt"], model)
        else:
            prompt = prebuilt_prompt

        answer, latency, slot = call_model_with_retry(
            prompt, clients, model, client_idx
        )
        key_label = f"key #{slot + 1}"
        return TaskResult(
            task["id"], task["label"], answer, latency, model=model, key_label=key_label
        )
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
        return TaskResult(
            task["id"], task["label"], "", 0, model=model, key_label=key_label, error=msg
        )


# ── Single Agent ──────────────────────────────────────────────────────────────

def run_single_agent(clients: list[OpenAI]) -> RunResult:
    """
    One agent handles all four tasks sequentially.
    Tasks alternate across API keys when multiple keys are configured.
    """
    print("\n" + "═" * 60)
    print("  SINGLE AGENT  (sequential)")
    print("═" * 60)
    if len(clients) > 1:
        print(f"\n  API keys: {len(clients)} (round-robin per task)")

    run_start = time.perf_counter()
    task_results = []

    for idx, task in enumerate(TASKS):
        _, key_label = client_for_index(clients, idx)
        print(f"\n  → {task['label']} [{key_label}] …", end="", flush=True)
        result = run_task(task, clients, MODEL, idx)
        task_results.append(result)
        if result.error:
            print(f" ✗ ERROR: {result.error}")
        else:
            print(f" ✓ ({result.latency_ms} ms)")

    total_ms = int((time.perf_counter() - run_start) * 1000)
    return RunResult("single", total_ms, task_results)


# ── Multi-Agent ───────────────────────────────────────────────────────────────

def run_multi_agent(clients: list[OpenAI]) -> RunResult:
    """
    Four specialized sub-agents run in parallel, each on a different model
    and API key (separate TPM buckets) with staggered launches.
    """
    models = get_multi_agent_models()

    print("\n" + "═" * 60)
    print("  MULTI-AGENT  (parallel, 4 sub-agents)")
    print("═" * 60)

    if len(clients) > 1:
        print(f"\n  API keys: {len(clients)} (round-robin per task)")

    if SPLIT_MULTI_AGENT_MODELS and len(set(models)) > 1:
        print("\n  Model + key split (one per task):")
        for idx, (task, model) in enumerate(zip(TASKS, models)):
            _, key_label = client_for_index(clients, idx)
            print(f"    {task['label']:<30} → {model}  ({key_label})")
        print(f"\n  Staggered launch: {MULTI_AGENT_STAGGER_SEC}s between agents\n")
    else:
        print("\n  Launching all agents simultaneously …\n")

    prompt_cache = prefetch_prompts(TASKS, models)

    run_start = time.perf_counter()
    task_results: list[TaskResult] = [None] * len(TASKS)

    with ThreadPoolExecutor(max_workers=len(TASKS)) as executor:
        future_to_index = {}
        for idx, task in enumerate(TASKS):
            if idx > 0 and MULTI_AGENT_STAGGER_SEC > 0:
                time.sleep(MULTI_AGENT_STAGGER_SEC)
            cached = prompt_cache.get(task["id"])
            future = executor.submit(
                run_task,
                task,
                clients,
                models[idx],
                idx,
                cached if cached is not None else None,
                search_failed=cached is None
                and not model_uses_builtin_search(models[idx]),
            )
            future_to_index[future] = idx

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            result = future.result()
            task_results[idx] = result
            status = "✓" if not result.error else "✗"
            model_tag = f" [{result.model}]" if result.model else ""
            key_tag = f" ({result.key_label})" if result.key_label else ""
            print(
                f"  {status} [{result.latency_ms:>5} ms]{model_tag}{key_tag}  {result.label}"
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
        if r.key_label:
            print(f"    🔑  {r.key_label}")
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
    clients = create_clients()
    search_mode = (
        "built-in web search"
        if uses_builtin_search()
        else "DuckDuckGo + LLM"
    )
    multi_models = get_multi_agent_models()

    print("\n🦁  SF Zoo — Single Agent vs Multi-Agent Demo")
    print(f"    Provider: {PROVIDER}  |  Single model: {MODEL}")
    if len(clients) > 1:
        print(f"    API keys:  {len(clients)} Groq keys (load-balanced)")
    if SPLIT_MULTI_AGENT_MODELS and len(set(multi_models)) > 1:
        print(f"    Multi:    split across {len(set(multi_models))} models")
    else:
        print(f"    Multi:    {MODEL}")
    print(f"    Search:   {search_mode}")
    print("    Tasks: weather · distance · tickets · hours\n")

    single_result = run_single_agent(clients)
    print_results(single_result)

    pause_between_runs(PAUSE_BETWEEN_RUNS_SEC)

    multi_result = run_multi_agent(clients)
    print_results(multi_result)

    print_comparison(single_result, multi_result)


if __name__ == "__main__":
    main()
