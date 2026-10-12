"""
scripts/load_test.py
====================
LLM Cost Autopilot — Load Test Script (Phase 5).

Usage
-----
    # Run A: smart routing (normal operation)
    python scripts/load_test.py --mode smart --requests 500

    # Run B: baseline (all requests forced to the top model)
    python scripts/load_test.py --mode baseline --requests 500

    # Run both and produce the full report automatically
    python scripts/load_test.py --mode both --requests 500

What it measures
----------------
- Total cost (actual model + verification reference model calls)
- Gross savings (baseline_cost - smart_cost)
- Net savings (gross - verification_cost)
- Avg and p95 latency for both runs
- Routing distribution by tier
- Average quality score and escalation rate from the DB

NOTE: This script makes REAL API calls and incurs REAL cost.
      Run with a small --requests count first to estimate cost.

Output
------
Writes ``load_test_report.md`` to the project root.

IMPORTANT: Numbers in the report come from actual test runs.
           Do not fabricate data. Placeholders are left where the
           script cannot yet compute a value (e.g. verification cost
           requires the DB to be populated after the run).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any

import httpx

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_API_BASE = "http://localhost:8000"
_REPORT_PATH = _PROJECT_ROOT / "load_test_report.md"

# ---------------------------------------------------------------------------
# Diverse prompt set — varied complexity to exercise all tiers
# ---------------------------------------------------------------------------

PROMPTS: list[str] = [
    # Simple — factual, one-shot
    "What is the capital of France?",
    "Convert 100 Fahrenheit to Celsius.",
    "Who wrote Pride and Prejudice?",
    "What is 17 multiplied by 23?",
    "List the planets in the solar system.",
    "What does HTTP stand for?",
    "What year did World War II end?",
    "What is the boiling point of water in Celsius?",
    "Name three primary colours.",
    "What is the speed of light in km/s?",
    # Moderate — requires reasoning
    "Explain the difference between TCP and UDP protocols.",
    "Write a Python function that checks if a string is a palindrome.",
    "What are the pros and cons of microservices architecture?",
    "Explain what a closure is in JavaScript with an example.",
    "Summarise the key ideas of the CAP theorem.",
    "How does HTTPS differ from HTTP in terms of security?",
    "Explain recursion with a simple code example.",
    "What is the difference between SQL and NoSQL databases?",
    "Write a regex to validate an email address.",
    "Describe the observer design pattern.",
    # Complex — multi-step, analytical, or creative
    "Design a scalable REST API for a social media platform. Consider authentication, rate limiting, and pagination.",
    "Compare and contrast transformer and RNN architectures for NLP tasks. What are the trade-offs?",
    "Explain how Kubernetes achieves high availability and what happens during a node failure.",
    "Write a detailed plan to migrate a monolithic Python application to microservices with minimal downtime.",
    "Analyse the time and space complexity of quicksort, merge sort, and heapsort.",
    "How would you implement a distributed rate limiter using Redis? Describe the algorithm.",
    "Explain the RAFT consensus algorithm and how it handles leader election.",
    "Design a real-time bidding system that handles 100,000 bids per second.",
    "Compare LangChain, LlamaIndex, and raw prompting for building a RAG system.",
    "What are the main challenges of building a multi-tenant SaaS application at scale?",
]


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------


async def run_single(
    client: httpx.AsyncClient,
    prompt: str,
    force_model: str | None = None,
) -> dict[str, Any]:
    """Send one request to POST /v1/completions and return the result dict."""
    body: dict[str, Any] = {"prompt": prompt, "max_tokens": 512}
    if force_model:
        body["force_model"] = force_model

    start = time.perf_counter()
    try:
        resp = await client.post(f"{_API_BASE}/v1/completions", json=body, timeout=60.0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            data["_latency_ms"] = elapsed_ms
            data["_error"] = None
            return data
        else:
            return {"_latency_ms": elapsed_ms, "_error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"_latency_ms": elapsed_ms, "_error": str(exc)}


# ---------------------------------------------------------------------------
# Run a batch
# ---------------------------------------------------------------------------


async def run_batch(
    n_requests: int,
    force_model: str | None = None,
    concurrency: int = 5,
) -> list[dict]:
    """Run n_requests against the API with limited concurrency."""
    results: list[dict] = []
    semaphore = asyncio.Semaphore(concurrency)
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n_requests)]

    async def bounded(prompt: str) -> dict:
        async with semaphore:
            return await run_single(client, prompt, force_model=force_model)

    async with httpx.AsyncClient() as client:
        tasks = [bounded(p) for p in prompts]
        results = await asyncio.gather(*tasks)

    return results


# ---------------------------------------------------------------------------
# Summarise results
# ---------------------------------------------------------------------------


def summarise(results: list[dict]) -> dict:
    """Compute summary statistics from a batch of results."""
    successes = [r for r in results if r.get("_error") is None]
    errors = [r for r in results if r.get("_error") is not None]

    latencies = [r["_latency_ms"] for r in successes]
    costs = [r.get("cost_usd", 0.0) for r in successes]
    tiers = [r.get("complexity_tier", "unknown") for r in successes]

    tier_dist: dict[str, int] = {}
    for t in tiers:
        tier_dist[t] = tier_dist.get(t, 0) + 1

    p95 = (
        sorted(latencies)[int(len(latencies) * 0.95)]
        if latencies
        else 0.0
    )

    return {
        "total": len(results),
        "success": len(successes),
        "errors": len(errors),
        "total_cost_usd": sum(costs),
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": p95,
        "tier_distribution": tier_dist,
        "error_sample": [r["_error"] for r in errors[:3]],
    }


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def _write_report(smart: dict | None, baseline: dict | None, n: int) -> None:
    """Write load_test_report.md."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# LLM Cost Autopilot — Load Test Report",
        "",
        f"Generated: {ts}  |  Requests per run: {n}",
        "",
        "> **Note:** Verification cost (reference model calls from Phase 3) is not",
        "> included in the per-request cost_usd field below. It is logged separately",
        "> in the `verification_log` table. Query `db/queries.py::get_quality_metrics()`",
        "> after the run to compute net savings.",
        "",
        "---",
        "",
    ]

    def _tier_table(dist: dict) -> str:
        rows = "\n".join(
            f"| {tier:<10} | {count:>8} | {count / sum(dist.values()) * 100:>6.1f}% |"
            for tier, count in sorted(dist.items())
        )
        return f"| Tier       | Requests | Share  |\n|------------|----------|--------|\n{rows}"

    if smart:
        lines += [
            "## Run A — Smart Routing",
            "",
            f"- **Total requests:** {smart['total']}",
            f"- **Successful:**     {smart['success']}",
            f"- **Errors:**         {smart['errors']}",
            f"- **Total cost (USD):** `${smart['total_cost_usd']:.6f}`",
            f"- **Avg latency:**    {smart['avg_latency_ms']:.0f} ms",
            f"- **p95 latency:**    {smart['p95_latency_ms']:.0f} ms",
            "",
            "### Routing distribution",
            "",
            _tier_table(smart["tier_distribution"]),
            "",
        ]

    if baseline:
        lines += [
            "## Run B — Baseline (All Top-Tier)",
            "",
            f"- **Total requests:** {baseline['total']}",
            f"- **Successful:**     {baseline['success']}",
            f"- **Errors:**         {baseline['errors']}",
            f"- **Total cost (USD):** `${baseline['total_cost_usd']:.6f}`",
            f"- **Avg latency:**    {baseline['avg_latency_ms']:.0f} ms",
            f"- **p95 latency:**    {baseline['p95_latency_ms']:.0f} ms",
            "",
        ]

    if smart and baseline:
        gross_savings = baseline["total_cost_usd"] - smart["total_cost_usd"]
        gross_pct = (
            gross_savings / baseline["total_cost_usd"] * 100
            if baseline["total_cost_usd"] > 0
            else 0.0
        )
        latency_delta = smart["avg_latency_ms"] - baseline["avg_latency_ms"]

        lines += [
            "## Savings Summary",
            "",
            f"| Metric                         | Value              |",
            f"|--------------------------------|--------------------|",
            f"| Baseline cost                  | ${baseline['total_cost_usd']:.6f}      |",
            f"| Smart routing cost             | ${smart['total_cost_usd']:.6f}      |",
            f"| **Gross savings**              | **${gross_savings:.6f}** |",
            f"| **Gross savings %**            | **{gross_pct:.1f}%**       |",
            f"| Net savings (after verif. cost)| `<!-- TODO: compute from DB after run -->` |",
            f"| Avg latency delta              | {latency_delta:+.0f} ms        |",
            f"| p95 latency delta              | {smart['p95_latency_ms'] - baseline['p95_latency_ms']:+.0f} ms |",
            "",
            "## Quality Metrics",
            "",
            "<!-- TODO: Run `python -c \"from db.queries import get_quality_metrics; print(get_quality_metrics())\"` -->",
            "<!-- after populating the DB, then fill in avg_quality_score and escalation_rate_pct here. -->",
            "",
            "| Metric                  | Value     |",
            "|-------------------------|-----------|",
            "| Avg quality score       | `[run DB query]` |",
            "| Escalation rate         | `[run DB query]` |",
            "",
        ]

    lines += [
        "---",
        "",
        "_Report generated by `scripts/load_test.py`._",
    ]

    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {_REPORT_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(mode: str, n: int) -> None:
    """Run the load test in the specified mode."""
    # Detect baseline model — use first high-tier model in registry
    baseline_model_id: str | None = None
    try:
        from models.registry import list_models_by_tier
        high = list_models_by_tier("high")
        if high:
            baseline_model_id = high[0].model_id
            print(f"Baseline model: {baseline_model_id}")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not load registry: {exc}")

    smart_summary: dict | None = None
    baseline_summary: dict | None = None

    if mode in ("smart", "both"):
        print(f"\n[Run A] Smart routing — {n} requests …")
        smart_results = await run_batch(n, force_model=None, concurrency=5)
        smart_summary = summarise(smart_results)
        print(f"  Cost: ${smart_summary['total_cost_usd']:.6f}  |  Avg: {smart_summary['avg_latency_ms']:.0f}ms  |  p95: {smart_summary['p95_latency_ms']:.0f}ms")

    if mode in ("baseline", "both"):
        if not baseline_model_id:
            print("No high-tier model found — cannot run baseline. Add a 'high' quality_tier model to registry.")
        else:
            print(f"\n[Run B] Baseline (force {baseline_model_id}) — {n} requests …")
            baseline_results = await run_batch(n, force_model=baseline_model_id, concurrency=5)
            baseline_summary = summarise(baseline_results)
            print(f"  Cost: ${baseline_summary['total_cost_usd']:.6f}  |  Avg: {baseline_summary['avg_latency_ms']:.0f}ms  |  p95: {baseline_summary['p95_latency_ms']:.0f}ms")

    _write_report(smart_summary, baseline_summary, n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Cost Autopilot load test")
    parser.add_argument("--mode", choices=["smart", "baseline", "both"], default="both")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests per run")
    args = parser.parse_args()

    asyncio.run(main(mode=args.mode, n=args.requests))
