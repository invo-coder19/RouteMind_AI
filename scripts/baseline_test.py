"""
scripts.baseline_test — Phase 1 baseline evaluation.

Sends 10 diverse prompts to every model in MODEL_REGISTRY, captures
performance metrics, and writes results to ``baseline_results.csv``.

Prompt categories
-----------------
  IDs 1–3   : Simple factual questions (expected: cheap model handles fine)
  IDs 4–5   : Summarization tasks
  IDs 6–7   : Classification tasks
  IDs 8–10  : Multi-step reasoning (expected: may need higher-tier model)

Usage
-----
  # From the project root with venv active:
  python scripts/baseline_test.py

  # With a custom output path:
  python scripts/baseline_test.py --output results/my_baseline.csv

Output
------
  baseline_results.csv  — one row per (prompt × model) combination
  Console summary       — total cost and average latency per model

Notes
-----
  - If any single provider fails for a prompt, the error is logged and the
    script continues. A full provider outage will skip all prompts for that
    model but will NOT abort the run.
  - Ollama models require a running local Ollama instance. If Ollama is
    unreachable, all three Ollama models will be skipped with a warning.
  - Results are appended if the output file already exists, allowing
    incremental runs. Delete the file first for a clean baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so the script can be run directly.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.exceptions import ProviderError
from core.router_client import send_request
from models.registry import MODEL_REGISTRY, ModelConfig
from models.response import LLMResponse

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baseline_test")

# ---------------------------------------------------------------------------
# Test prompts (10 total)
# ---------------------------------------------------------------------------

TEST_PROMPTS: list[dict[str, str]] = [
    # Simple factual (IDs 1–3)
    {
        "prompt_id": "factual_01",
        "category": "factual",
        "text": "What is the capital city of Japan?",
    },
    {
        "prompt_id": "factual_02",
        "category": "factual",
        "text": "How many days are in a leap year?",
    },
    {
        "prompt_id": "factual_03",
        "category": "factual",
        "text": "What element has the chemical symbol 'Au'?",
    },
    # Summarization (IDs 4–5)
    {
        "prompt_id": "summary_01",
        "category": "summarization",
        "text": (
            "Summarize the following passage in exactly two sentences:\n\n"
            "The Internet of Things (IoT) refers to a network of physical devices "
            "embedded with sensors, software, and connectivity that allows them to "
            "collect and exchange data with other devices and systems over the internet. "
            "Common examples include smart thermostats, wearable fitness trackers, and "
            "connected industrial machinery. IoT enables greater automation and efficiency "
            "but also introduces new security and privacy challenges because each connected "
            "device can be a potential entry point for attackers."
        ),
    },
    {
        "prompt_id": "summary_02",
        "category": "summarization",
        "text": (
            "Give a one-paragraph summary suitable for a non-technical executive:\n\n"
            "Large Language Models (LLMs) are deep learning systems trained on vast text "
            "corpora to predict and generate human-like text. They are used for tasks ranging "
            "from question answering and code generation to creative writing. However, they can "
            "hallucinate — confidently stating incorrect information — and they are expensive to "
            "train and run at scale. Organizations deploying LLMs must balance capability, cost, "
            "accuracy, and responsible AI practices."
        ),
    },
    # Classification (IDs 6–7)
    {
        "prompt_id": "classify_01",
        "category": "classification",
        "text": (
            "Classify the sentiment of the following customer review as POSITIVE, NEGATIVE, "
            "or NEUTRAL. Reply with just the label and a one-sentence justification.\n\n"
            "Review: 'The delivery arrived two days late and the packaging was damaged, "
            "but the product itself works perfectly and customer support was very helpful.'"
        ),
    },
    {
        "prompt_id": "classify_02",
        "category": "classification",
        "text": (
            "Classify the following support ticket into exactly one category from this list: "
            "[BILLING, TECHNICAL, ACCOUNT_ACCESS, FEATURE_REQUEST, OTHER]. "
            "Reply with just the category name.\n\n"
            "Ticket: 'Hi, I was charged twice for my subscription this month. "
            "I only see one active account in my profile. Can someone please look into this?'"
        ),
    },
    # Multi-step reasoning (IDs 8–10)
    {
        "prompt_id": "reasoning_01",
        "category": "reasoning",
        "text": (
            "A store sells apples for $0.75 each and oranges for $1.20 each. "
            "Maria buys 4 apples and some oranges. Her total bill is $8.40. "
            "How many oranges did Maria buy? Show your step-by-step working."
        ),
    },
    {
        "prompt_id": "reasoning_02",
        "category": "reasoning",
        "text": (
            "Consider a software team that follows this deployment policy:\n"
            "- Code must pass all unit tests before merging.\n"
            "- A feature branch can only be deployed to staging after it is merged to main.\n"
            "- A feature can only be deployed to production after it has been on staging for "
            "at least 24 hours and has been approved by a senior engineer.\n\n"
            "Alice's feature branch passes all tests at 9:00 AM on Monday. "
            "It is merged to main at 10:00 AM. A senior engineer approves it at 3:00 PM Monday. "
            "What is the earliest time and day Alice's feature can be deployed to production? "
            "Reason through each policy constraint step by step."
        ),
    },
    {
        "prompt_id": "reasoning_03",
        "category": "reasoning",
        "text": (
            "You are given three statements. Determine which conclusion is logically valid.\n\n"
            "Statements:\n"
            "1. All data scientists know Python.\n"
            "2. Some machine learning engineers are data scientists.\n"
            "3. No Python developer works exclusively with spreadsheets.\n\n"
            "Conclusions:\n"
            "A. Some machine learning engineers know Python.\n"
            "B. All machine learning engineers know Python.\n"
            "C. No data scientist works exclusively with spreadsheets.\n"
            "D. Both A and C.\n\n"
            "Identify all valid conclusions and explain your reasoning for each."
        ),
    },
]

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

CSV_FIELDNAMES: list[str] = [
    "prompt_id",
    "category",
    "model_id",
    "provider",
    "output_text",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "cost_usd",
    "status",       # "ok" or "error"
    "error_detail", # empty on success
]

# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------

async def _run_single(
    prompt_entry: dict[str, str],
    model_config: ModelConfig,
) -> dict[str, str | float | int]:
    """Run one prompt against one model and return a result dict.

    Args:
        prompt_entry: Entry from TEST_PROMPTS with ``prompt_id``,
                      ``category``, and ``text`` keys.
        model_config: The model to call.

    Returns:
        A dict matching CSV_FIELDNAMES. On error, status is ``"error"``
        and ``error_detail`` contains the exception message.
    """
    base_row: dict[str, str | float | int] = {
        "prompt_id": prompt_entry["prompt_id"],
        "category": prompt_entry["category"],
        "model_id": model_config.model_id,
        "provider": model_config.provider,
        "output_text": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0.0,
        "cost_usd": 0.0,
        "status": "ok",
        "error_detail": "",
    }

    try:
        response: LLMResponse = await send_request(
            prompt=prompt_entry["text"],
            model_config=model_config,
            system_prompt="You are a helpful, concise assistant.",
            max_tokens=512,
        )
        base_row.update(
            {
                "output_text": response.output_text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": round(response.latency_ms, 2),
                "cost_usd": response.cost_usd,
                "status": "ok",
            }
        )
        logger.info(
            "✓  %-15s  %-40s  %6.0f ms  $%.6f",
            prompt_entry["prompt_id"],
            model_config.model_id,
            response.latency_ms,
            response.cost_usd,
        )
    except ProviderError as exc:
        base_row["status"] = "error"
        base_row["error_detail"] = str(exc)
        logger.warning(
            "✗  %-15s  %-40s  %s",
            prompt_entry["prompt_id"],
            model_config.model_id,
            exc,
        )
    except Exception as exc:  # noqa: BLE001 — unexpected; log and continue
        base_row["status"] = "error"
        base_row["error_detail"] = f"Unexpected: {type(exc).__name__}: {exc}"
        logger.error(
            "✗  %-15s  %-40s  Unexpected error: %s",
            prompt_entry["prompt_id"],
            model_config.model_id,
            exc,
            exc_info=True,
        )

    return base_row


async def run_baseline(output_path: Path) -> list[dict[str, str | float | int]]:
    """Send all 10 prompts to every model in MODEL_REGISTRY.

    Calls are made sequentially (model by model) to avoid hammering rate limits.
    Within each model, prompts are sent sequentially for the same reason.

    Args:
        output_path: Path to write (or append to) the CSV results file.

    Returns:
        List of all result dicts (one per prompt × model).
    """
    all_results: list[dict[str, str | float | int]] = []

    write_header = not output_path.exists()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for model_config in MODEL_REGISTRY:
            logger.info(
                "\n%s\nTesting model: %s (%s)\n%s",
                "=" * 70,
                model_config.display_name or model_config.model_id,
                model_config.provider,
                "=" * 70,
            )
            for prompt_entry in TEST_PROMPTS:
                row = await _run_single(prompt_entry, model_config)
                writer.writerow(row)
                csv_file.flush()  # Write incrementally in case of early abort
                all_results.append(row)

    return all_results


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict[str, str | float | int]]) -> None:
    """Print a per-model cost and latency summary to stdout.

    Args:
        results: The full list of result dicts returned by ``run_baseline``.
    """
    from collections import defaultdict

    model_stats: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"total_cost": 0.0, "total_latency_ms": 0.0, "ok": 0, "error": 0}
    )

    for row in results:
        mid = str(row["model_id"])
        stats = model_stats[mid]
        if row["status"] == "ok":
            stats["total_cost"] = float(stats["total_cost"]) + float(row["cost_usd"])
            stats["total_latency_ms"] = float(stats["total_latency_ms"]) + float(row["latency_ms"])
            stats["ok"] = int(stats["ok"]) + 1
        else:
            stats["error"] = int(stats["error"]) + 1

    print("\n" + "=" * 78)
    print(f"{'MODEL':<44} {'TOTAL COST':>12} {'AVG LATENCY':>13} {'OK/ERR':>8}")
    print("=" * 78)

    for model_id, stats in sorted(model_stats.items()):
        ok_count = int(stats["ok"])
        err_count = int(stats["error"])
        total_cost = float(stats["total_cost"])
        avg_latency = (
            float(stats["total_latency_ms"]) / ok_count if ok_count else 0.0
        )
        print(
            f"{model_id:<44}  "
            f"${total_cost:>10.6f}  "
            f"{avg_latency:>10.0f} ms  "
            f"{ok_count}/{err_count}"
        )

    print("=" * 78)

    # Grand total
    grand_cost = sum(
        float(stats["total_cost"]) for stats in model_stats.values()
    )
    total_ok = sum(int(s["ok"]) for s in model_stats.values())
    total_err = sum(int(s["error"]) for s in model_stats.values())
    print(
        f"\nGrand total cost for this run : ${grand_cost:.6f}"
    )
    print(f"Successful calls              : {total_ok}")
    print(f"Failed calls                  : {total_err}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 baseline: send 10 prompts to all registered models."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("baseline_results.csv"),
        help="Path to the output CSV file (default: baseline_results.csv)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the baseline test and print the summary."""
    args = _parse_args()
    output_path: Path = args.output

    logger.info(
        "Starting baseline test: %d prompts × %d models = %d calls",
        len(TEST_PROMPTS),
        len(MODEL_REGISTRY),
        len(TEST_PROMPTS) * len(MODEL_REGISTRY),
    )
    logger.info("Results will be written to: %s", output_path.resolve())

    results = asyncio.run(run_baseline(output_path))
    _print_summary(results)

    logger.info("\nDone. Results saved to %s", output_path.resolve())


if __name__ == "__main__":
    main()
