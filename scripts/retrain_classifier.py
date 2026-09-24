"""
scripts/retrain_classifier.py
=============================
Weekly retraining job for the LLM Cost Autopilot complexity classifier.

INTENDED USAGE
--------------
Run manually:
    python scripts/retrain_classifier.py

Or schedule weekly (cron example — do NOT build the scheduler here):
    0 3 * * 0  /path/to/.venv/Scripts/python.exe /path/to/scripts/retrain_classifier.py

What it does
------------
1. Pulls all unprocessed routing failures from the feedback store since the
   last retraining run (tracked in ``scripts/last_retrain_timestamp.txt``).
2. Relabels each failed prompt with the *correct* tier — based on the rule:
      "If the cheap model (tier X) failed, the prompt actually needed tier X+1."
   Specifically:
      - cheap was 'simple' → correct tier = 'moderate'
      - cheap was 'moderate' → correct tier = 'complex'
      - cheap was 'complex' → already the top tier; skip (can't escalate further)
3. Appends the relabeled prompts to ``data/labeled_prompts.csv``.
4. Re-runs the dataset split (``data/split_dataset.py``) and training
   (``classifier/train.py``) on the updated dataset.
5. Prints before-accuracy / after-accuracy so improvement is visible.
6. Marks processed failures in the feedback store.
7. Writes the current timestamp to ``scripts/last_retrain_timestamp.txt``
   so the next run only processes new failures.

Relabeling rationale
--------------------
When a routing failure occurs it means the classifier underestimated the
prompt's complexity.  The correct tier is one step *above* what was assigned.
This is conservative: we don't skip tiers (e.g. simple → complex) based on
a single failure.  Over time, if a prompt repeatedly fails at 'moderate', the
classifier will accumulate enough signal to route it to 'complex'.
"""
from __future__ import annotations

import csv
import io
import logging
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Configure logging before any project imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("retrain_classifier")

from verification.feedback_store import (  # noqa: E402
    get_failed_verifications,
    mark_retraining_processed,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
_DATA_DIR = _PROJECT_ROOT / "data"
_LABELED_CSV = _DATA_DIR / "labeled_prompts.csv"
_SPLIT_SCRIPT = _DATA_DIR / "split_dataset.py"
_TRAIN_SCRIPT = _PROJECT_ROOT / "classifier" / "train.py"
_TIMESTAMP_FILE = _SCRIPTS_DIR / "last_retrain_timestamp.txt"
_PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Tier escalation map
# ---------------------------------------------------------------------------

#: Maps the *assigned* (wrong) tier to the *correct* tier.
#: If cheap was 'complex' there's nowhere to escalate — skip that row.
_TIER_ESCALATION: dict[str, str | None] = {
    "simple": "moderate",
    "moderate": "complex",
    "complex": None,  # already top tier — nothing to do
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_last_retrain_time() -> datetime:
    """Read the timestamp of the last retraining run.

    Returns the Unix epoch (1970-01-01) if the file doesn't exist yet,
    so the first run processes *all* failures.
    """
    if _TIMESTAMP_FILE.exists():
        ts_str = _TIMESTAMP_FILE.read_text(encoding="utf-8").strip()
        try:
            return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("Could not parse timestamp file — using epoch.")
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _write_current_retrain_time() -> None:
    """Persist the current UTC time so the next run knows where to start."""
    _TIMESTAMP_FILE.write_text(
        datetime.now(timezone.utc).isoformat(), encoding="utf-8"
    )


def _read_existing_prompts() -> set[str]:
    """Return the set of prompt_text values already in labeled_prompts.csv.

    Used to deduplicate — we don't add a prompt that's already labeled.
    """
    if not _LABELED_CSV.exists():
        return set()
    with _LABELED_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["prompt_text"] for row in reader}


def _append_new_labels(new_rows: list[tuple[str, str]]) -> int:
    """Append (prompt_text, complexity_label) tuples to labeled_prompts.csv.

    Parameters
    ----------
    new_rows:
        List of (prompt_text, complexity_label) pairs to append.

    Returns
    -------
    int
        Number of rows actually appended (after deduplication).
    """
    if not new_rows:
        return 0

    existing = _read_existing_prompts()
    to_add = [(p, l) for p, l in new_rows if p not in existing]

    if not to_add:
        logger.info("All %d new labels already exist in dataset — skipping append.", len(new_rows))
        return 0

    _LABELED_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Append mode — create header only if file is new.
    write_header = not _LABELED_CSV.exists()
    with _LABELED_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writerow(["prompt_text", "complexity_label"])
        writer.writerows(to_add)

    logger.info("Appended %d new rows to %s", len(to_add), _LABELED_CSV)
    return len(to_add)


def _run_script(script_path: pathlib.Path, description: str) -> str:
    """Run a Python script in the current venv and return its stdout.

    Raises
    ------
    RuntimeError
        If the script exits with a non-zero return code.
    """
    logger.info("Running %s ...", description)
    result = subprocess.run(
        [_PYTHON, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {result.returncode}):\n{result.stderr}"
        )
    return result.stdout


def _extract_accuracy(script_output: str) -> float | None:
    """Parse the 'Test accuracy: X.XX%' line from train.py output."""
    for line in script_output.splitlines():
        if "Test accuracy:" in line:
            # e.g. "Test accuracy: 87.10%"
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    return float(parts[1].strip().rstrip("%")) / 100.0
                except ValueError:
                    pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point — pull failures, relabel, retrain, report."""
    logger.info("=" * 60)
    logger.info("LLM Cost Autopilot — Weekly Classifier Retraining Job")
    logger.info("=" * 60)

    # -------------------------------------------------------- 1. Fetch failures
    since = _read_last_retrain_time()
    logger.info("Fetching failures since %s", since.isoformat())
    failures = get_failed_verifications(since=since)

    if not failures:
        logger.info("No new routing failures found. Nothing to retrain.")
        _write_current_retrain_time()
        return

    logger.info("Found %d routing failure(s) to process.", len(failures))

    # ------------------------------------------------- 2. Relabel failed prompts
    new_labels: list[tuple[str, str]] = []
    skipped_top_tier: list[str] = []

    for row in failures:
        tier_assigned: str = row.get("tier_assigned", "simple")
        correct_tier = _TIER_ESCALATION.get(tier_assigned)

        if correct_tier is None:
            # Already at 'complex' — can't escalate further
            logger.warning(
                "request_id=%s | tier='complex' already — skipping relabeling",
                row["request_id"],
            )
            skipped_top_tier.append(row["request_id"])
            continue

        prompt: str = row["prompt"]
        new_labels.append((prompt, correct_tier))
        logger.info(
            "Relabeling: %r | %s -> %s",
            prompt[:60] + ("..." if len(prompt) > 60 else ""),
            tier_assigned,
            correct_tier,
        )

    # -------------------------------------------------------- 3. Measure before accuracy
    before_accuracy: float | None = None
    try:
        train_output_before = _run_script(_TRAIN_SCRIPT, "classifier/train.py (before)")
        before_accuracy = _extract_accuracy(train_output_before)
        logger.info("Before-retraining accuracy: %.2f%%", (before_accuracy or 0) * 100)
    except RuntimeError as exc:
        logger.warning("Could not get before-accuracy: %s", exc)

    # ----------------------------------------- 4. Append new labels + re-split + retrain
    appended = _append_new_labels(new_labels)

    if appended == 0 and not skipped_top_tier:
        logger.info("No new data to add after deduplication. Skipping retrain.")
        _write_current_retrain_time()
        return

    if appended > 0:
        try:
            _run_script(_SPLIT_SCRIPT, "data/split_dataset.py")
        except RuntimeError as exc:
            logger.error("Dataset split failed: %s", exc)
            sys.exit(1)

        try:
            train_output_after = _run_script(_TRAIN_SCRIPT, "classifier/train.py (after)")
        except RuntimeError as exc:
            logger.error("Retraining failed: %s", exc)
            sys.exit(1)

        after_accuracy = _extract_accuracy(train_output_after)
        logger.info("After-retraining accuracy:  %.2f%%", (after_accuracy or 0) * 100)

        # -------------------------------------------- 5. Print before/after delta
        print("\n" + "=" * 60)
        print("  Retraining Summary")
        print("=" * 60)
        print(f"  Failures processed   : {len(failures)}")
        print(f"  New rows appended    : {appended}")
        print(f"  Skipped (top tier)   : {len(skipped_top_tier)}")
        if before_accuracy is not None and after_accuracy is not None:
            delta = after_accuracy - before_accuracy
            arrow = "^" if delta > 0 else ("v" if delta < 0 else "=")
            print(f"  Accuracy before      : {before_accuracy:.2%}")
            print(f"  Accuracy after       : {after_accuracy:.2%}")
            print(f"  Delta                : {arrow} {abs(delta):.2%}")
        print("=" * 60 + "\n")

    # ----------------------------------------- 6. Mark processed + update timestamp
    processed_ids = [row["request_id"] for row in failures]
    mark_retraining_processed(processed_ids)
    _write_current_retrain_time()
    logger.info("Done. Timestamp updated to %s", _TIMESTAMP_FILE)


if __name__ == "__main__":
    main()
