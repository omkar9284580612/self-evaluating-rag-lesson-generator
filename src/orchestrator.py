"""
orchestrator.py
---------------
LangGraph implementation of the generate -> evaluate -> regenerate workflow.

Flow:

START
  |
  v
GENERATE
  |
  v
EVALUATE
  |
  +---- PASS -----------------> SHIP
  |
  +---- FAIL -> PREPARE RETRY -> GENERATE
  |
  +---- MAX RETRIES ----------> MAX_RETRIES
"""

import os
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END

from src.generator import Generator
from src.evaluator import Evaluator
from src.memory_store import (
    record_failures_and_get_standing_rules,
    get_standing_rules_text,
    save_run_log,
)


# Number of retries AFTER the initial generation.
#
# MAX_RETRIES = 2 means:
#   Attempt 1 = initial generation
#   Attempt 2 = retry 1
#   Attempt 3 = retry 2
#
# Therefore maximum total attempts = 3.
MAX_RETRIES = 2


class LessonState(TypedDict, total=False):
    topic: str
    lesson: str
    evaluation: dict
    attempts_log: list
    attempt_num: int
    failure_feedback: Optional[str]
    persistent_lessons: str
    inject_error: bool
    final_status: str
    guard_failure: Optional[str]


def basic_lesson_guard(lesson: str):
    """
    Deterministic safety check for obviously incomplete output.

    This guard does NOT replace the LLM evaluator.
    It only catches very obvious generation problems such as:
      - empty output
      - output that is far too short
      - missing basic lesson structure
      - output ending in an obviously incomplete sentence

    The actual quality decision is still made by Evaluator.
    """

    if not lesson or not lesson.strip():
        return "Generated lesson is empty."

    text = lesson.strip()

    # Word-count safety check.
    # The generator is instructed to produce 300-400 words.
    word_count = len(text.split())

    if word_count < 200:
        return (
            f"Generated lesson is too short "
            f"({word_count} words). It may be incomplete."
        )

    # The generator is required to use five teaching sections.
    # These checks are intentionally generic so the pipeline works
    # for topics other than RAG.
    required_patterns = [
        "## 1.",
        "## 2.",
        "## 3.",
        "## 4.",
        "## 5.",
    ]

    missing = [
        pattern
        for pattern in required_patterns
        if pattern not in text
    ]

    if missing:
        return (
            "Lesson is missing required numbered teaching sections: "
            + ", ".join(missing)
        )

    # Very simple truncation detection.
    # A generated lesson should normally finish with punctuation.
    if text[-1] not in ".!?`*)#":
        return (
            "Lesson appears to end abruptly without completing "
            "the final sentence."
        )

    return None


def build_graph(generator: Generator, evaluator: Evaluator):

    def generate_node(state: LessonState):
        """
        Generate a lesson draft.
        """

        attempt = state.get("attempt_num", 1)

        if attempt == 1 and state.get("inject_error", False):
            lesson = generator.generate_with_deliberate_error(
                state["topic"]
            )
        else:
            lesson = generator.generate(
                state["topic"],
                failure_feedback=state.get("failure_feedback"),
                persistent_lessons=state.get(
                    "persistent_lessons",
                    "(none yet)"
                ),
            )

        guard_failure = basic_lesson_guard(lesson)

        print(
            f"Generated draft for attempt {attempt}."
        )

        if guard_failure:
            print(
                f"  - [GUARD] {guard_failure}"
            )

        return {
            "lesson": lesson,
            "guard_failure": guard_failure,
        }


    def evaluate_node(state: LessonState):
        """
        Evaluate the generated lesson.

        The LLM evaluator remains the primary evaluator.
        The deterministic guard adds an extra failure when
        obvious incompleteness is detected.
        """

        lesson = state["lesson"]

        # Always run the actual LLM evaluator.
        evaluation = evaluator.evaluate(
            lesson,
            topic=state["topic"],
        )

        # Run deterministic guard as an additional safety check.
        guard_failure = basic_lesson_guard(lesson)

        if guard_failure:
            guard_result = {
                "id": "coherent_flow",
                "status": "FAIL",
                "reason": guard_failure,
            }

            # Add guard failure only if coherent_flow wasn't already
            # reported as a failure by the LLM evaluator.
            existing_ids = {
                failure["id"]
                for failure in evaluation["failures"]
            }

            if "coherent_flow" not in existing_ids:
                evaluation["results"].append(guard_result)
                evaluation["failures"].append(guard_result)

            evaluation["all_pass"] = False

        entry = {
            "attempt": state["attempt_num"],
            "evaluation": evaluation,
            "lesson_snapshot": lesson,
        }

        print(
            f"Evaluated attempt {state['attempt_num']}: "
            f"{'PASS' if evaluation['all_pass'] else 'FAIL'}"
        )

        return {
            "evaluation": evaluation,
            "attempts_log": (
                state.get("attempts_log", [])
                + [entry]
            ),
        }


    def prepare_retry_node(state: LessonState):
        """
        Prepare feedback for the next generation attempt.
        """

        evaluation = state["evaluation"]
        failures = evaluation["failures"]

        for failure in failures:
            print(
                f"  - [{failure['id']}] "
                f"{failure['reason']}"
            )

        # Update persistent cross-run memory.
        standing_rules = (
            record_failures_and_get_standing_rules(
                failures
            )
        )

        # Convert current failures into targeted retry feedback.
        feedback = Evaluator.format_failures_for_feedback(
            failures
        )

        return {
            "failure_feedback": feedback,
            "persistent_lessons": standing_rules,
            "attempt_num": state["attempt_num"] + 1,
        }


    def route_after_evaluation(state: LessonState):
        """
        Decide whether to ship, retry, or terminate.
        """

        if state["evaluation"]["all_pass"]:
            return "ship"

        # Example with MAX_RETRIES = 2:
        #
        # attempt 1 -> retry
        # attempt 2 -> retry
        # attempt 3 -> max_retries
        #
        if state["attempt_num"] >= MAX_RETRIES + 1:
            return "max_retries"

        return "retry"


    def ship_node(state: LessonState):
        """
        Save a successful run.
        """

        save_run_log(
            state["topic"],
            state.get("attempts_log", []),
            final_status="PASSED",
            final_lesson=state["lesson"],
        )

        return {
            "final_status": "PASSED"
        }


    def max_retries_node(state: LessonState):
        """
        Save a run that exhausted its retry budget.
        """

        save_run_log(
            state["topic"],
            state.get("attempts_log", []),
            final_status="MAX_RETRIES_REACHED",
            final_lesson=state["lesson"],
        )

        return {
            "final_status": "MAX_RETRIES_REACHED"
        }


    # ---------------------------------------------------------
    # Build LangGraph
    # ---------------------------------------------------------

    graph = StateGraph(LessonState)

    graph.add_node(
        "generate",
        generate_node
    )

    graph.add_node(
        "evaluate",
        evaluate_node
    )

    graph.add_node(
        "prepare_retry",
        prepare_retry_node
    )

    graph.add_node(
        "ship",
        ship_node
    )

    graph.add_node(
        "max_retries",
        max_retries_node
    )

    graph.add_edge(
        START,
        "generate"
    )

    graph.add_edge(
        "generate",
        "evaluate"
    )

    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "ship": "ship",
            "retry": "prepare_retry",
            "max_retries": "max_retries",
        },
    )

    graph.add_edge(
        "prepare_retry",
        "generate"
    )

    graph.add_edge(
        "ship",
        END
    )

    graph.add_edge(
        "max_retries",
        END
    )

    return graph.compile()


def run_pipeline(
    topic: str,
    generator: Generator,
    evaluator: Evaluator,
    inject_error: bool = False,
):
    """
    Run the complete lesson generation pipeline.
    """

    print("\n=== LangGraph lesson pipeline ===")

    initial_state: LessonState = {
        "topic": topic,
        "attempt_num": 1,
        "attempts_log": [],
        "failure_feedback": None,
        "persistent_lessons": get_standing_rules_text(),
        "inject_error": inject_error,
    }

    app = build_graph(
        generator,
        evaluator,
    )

    final_state = app.invoke(
        initial_state
    )

    return {
        "status": final_state["final_status"],
        "lesson": final_state["lesson"],
        "attempts": final_state.get(
            "attempts_log",
            []
        ),
    }


def write_outputs(
    topic: str,
    result: dict,
    outputs_dir: str = "outputs",
):
    """
    Write the final lesson and human-readable rejection log.
    """

    os.makedirs(
        outputs_dir,
        exist_ok=True
    )

    lesson_path = os.path.join(
        outputs_dir,
        "lesson_final.md"
    )

    with open(
        lesson_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            result["lesson"]
        )

    log_lines = [
        f"# Rejection Log: {topic}\n",
        f"Final status: **{result['status']}**\n",
    ]

    for attempt in result["attempts"]:

        evaluation = attempt["evaluation"]

        status = (
            "PASSED"
            if evaluation["all_pass"]
            else "FAILED"
        )

        log_lines.append(
            f"\n## Attempt {attempt['attempt']} -- {status}"
        )

        for result_item in evaluation["results"]:

            mark = (
                "PASS"
                if result_item["status"].upper() == "PASS"
                else "FAIL"
            )

            log_lines.append(
                f"- **[{mark}] "
                f"{result_item['id']}**: "
                f"{result_item['reason']}"
            )

        if not evaluation["all_pass"]:

            feedback = "; ".join(
                f"{failure['id']}: "
                f"{failure['reason']}"
                for failure in evaluation["failures"]
            )

            log_lines.append(
                "\n**Feedback passed to the next generation:** "
                + feedback
            )

    log_path = os.path.join(
        outputs_dir,
        "rejection_log.md"
    )

    with open(
        log_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "\n".join(log_lines)
        )

    return lesson_path, log_path