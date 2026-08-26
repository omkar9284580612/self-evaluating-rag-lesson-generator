"""
memory_store.py
----------------
Persistent memory for the self-evaluating lesson generator.

There are two types of memory:

1. Per-run audit logs
   memory/runs/<topic>_<timestamp>.json

   These store:
   - topic
   - final status
   - every generation/evaluation attempt
   - the final lesson

2. Cross-run lessons learned
   memory/lessons_learned.json

   Every failed rubric checkpoint is counted.

   When a checkpoint fails repeatedly across runs, it becomes a
   standing rule. That rule is injected into future generation prompts.

This provides the "self-evolving" behavior required by the project.
"""

import json
import os
from datetime import datetime, timezone


# ---------------------------------------------------------------------
# Storage locations
# ---------------------------------------------------------------------

MEMORY_DIR = "memory"
RUNS_DIR = os.path.join(MEMORY_DIR, "runs")
LESSONS_FILE = os.path.join(MEMORY_DIR, "lessons_learned.json")

# A failure pattern becomes a permanent standing rule after
# appearing this many times across evaluations.
GRADUATION_THRESHOLD = 2


# ---------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------

def _ensure_dirs():
    """Create the memory directories if they do not already exist."""
    os.makedirs(RUNS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Lessons learned
# ---------------------------------------------------------------------

def load_lessons() -> dict:
    """
    Load the cross-run lessons learned from disk.

    If the file does not exist, return a clean empty structure.
    """

    _ensure_dirs()

    if not os.path.exists(LESSONS_FILE):
        return {
            "failure_counts": {},
            "standing_rules": [],
        }

    try:
        with open(LESSONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # If the memory file is damaged or unreadable, start safely
        # instead of crashing the whole lesson pipeline.
        return {
            "failure_counts": {},
            "standing_rules": [],
        }

    # Make sure expected keys always exist.
    data.setdefault("failure_counts", {})
    data.setdefault("standing_rules", [])

    return data


def save_lessons(data: dict):
    """Save cross-run lessons learned to disk."""

    _ensure_dirs()

    with open(LESSONS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def record_failures_and_get_standing_rules(failures: list) -> str:
    """
    Record failed rubric checkpoints and promote repeated failures
    into standing rules.

    Example:

        no_unexplained_jargon fails
        -> count becomes 1

        no_unexplained_jargon fails again
        -> count becomes 2
        -> standing rule is created

    The resulting standing rules are returned as text so they can be
    injected into the next generation prompt.
    """

    data = load_lessons()

    failure_counts = data["failure_counts"]
    standing_rules = data["standing_rules"]

    # IDs of rules that have already graduated.
    existing_rule_ids = {
        rule.get("id")
        for rule in standing_rules
        if isinstance(rule, dict)
    }

    for failure in failures:
        failure_id = failure.get("id")
        reason = failure.get(
            "reason",
            "Repeated failure detected for this checkpoint."
        )

        # Ignore malformed evaluator entries.
        if not failure_id:
            continue

        # Increment the failure count.
        failure_counts[failure_id] = (
            failure_counts.get(failure_id, 0) + 1
        )

        # Promote repeated failures into standing rules.
        if (
            failure_counts[failure_id] >= GRADUATION_THRESHOLD
            and failure_id not in existing_rule_ids
        ):
            standing_rules.append(
                {
                    "id": failure_id,
                    "rule": (
                        f"Past runs repeatedly failed on "
                        f"'{failure_id}': {reason}"
                    ),
                }
            )

            existing_rule_ids.add(failure_id)

    data["failure_counts"] = failure_counts
    data["standing_rules"] = standing_rules

    save_lessons(data)

    return get_standing_rules_text(data)


def get_standing_rules_text(data: dict = None) -> str:
    """
    Convert standing rules into text suitable for a generation prompt.

    Returns "(none yet)" when no standing rules exist.
    """

    if data is None:
        data = load_lessons()

    standing_rules = data.get("standing_rules", [])

    if not standing_rules:
        return "(none yet)"

    lines = []

    for rule in standing_rules:
        rule_text = rule.get("rule")

        if rule_text:
            lines.append(f"- {rule_text}")

    if not lines:
        return "(none yet)"

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Per-run audit log
# ---------------------------------------------------------------------

def save_run_log(
    topic: str,
    attempts: list,
    final_status: str,
    final_lesson: str,
):
    """
    Save a complete audit record for one pipeline run.

    The log contains:
    - topic
    - timestamp
    - final status
    - every attempt
    - final lesson
    """

    _ensure_dirs()

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    # Convert the topic into a safe filename.
    safe_topic = "".join(
        c if c.isalnum() else "_"
        for c in topic
    ).strip("_")[:40]

    if not safe_topic:
        safe_topic = "lesson"

    filename = f"{safe_topic}_{timestamp}.json"
    path = os.path.join(RUNS_DIR, filename)

    log = {
        "topic": topic,
        "timestamp": timestamp,
        "final_status": final_status,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "final_lesson": final_lesson,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            log,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return path