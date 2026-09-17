"""
evaluator.py
------------
Evaluates a generated lesson against six binary quality checkpoints.
"""

from src.llm_client import LLMClient, extract_json
from src.rubric import RUBRIC, rubric_as_prompt_block


EVAL_PROMPT = """
You are a strict curriculum quality reviewer.

Evaluate the lesson against EVERY checkpoint.

For every checkpoint:
- Return PASS or FAIL only.
- There is no partial credit.
- Give a short reason.
- Do not invent missing content.
- Judge only the lesson provided.

CHECKPOINTS:

{rubric_block}

LESSON:

---
{lesson}
---

Return ONLY valid JSON.

Use exactly this structure:

{{
  "results": [
    {{
      "id": "checkpoint_id",
      "status": "PASS",
      "reason": "short reason"
    }}
  ]
}}

Return exactly one result for every checkpoint.
Use the checkpoint IDs exactly as provided.
"""


class Evaluator:

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def evaluate(
        self,
        lesson_markdown: str,
        topic: str = "the requested topic",
    ) -> dict:

        prompt = EVAL_PROMPT.format(
            rubric_block=rubric_as_prompt_block(topic),
            lesson=lesson_markdown,
        )

        raw = self.llm.complete(
            prompt,
            temperature=0.0,
            max_tokens=2500,
            json_mode=True,
        )

        parsed = extract_json(raw)

        results = parsed.get("results")

        if not isinstance(results, list):
            raise ValueError(
                "Evaluator response must contain a results list."
            )

        expected_ids = [
            item["id"]
            for item in RUBRIC
        ]

        expected_set = set(expected_ids)

        returned_ids = [
            item.get("id")
            for item in results
        ]

        if len(returned_ids) != len(set(returned_ids)):
            raise ValueError(
                "Evaluator returned duplicate checkpoint IDs."
            )

        missing = expected_set - set(returned_ids)

        unexpected = set(returned_ids) - expected_set

        if missing:
            raise ValueError(
                f"Evaluator response missing checkpoints: "
                f"{sorted(missing)}"
            )

        if unexpected:
            raise ValueError(
                f"Evaluator returned unexpected checkpoints: "
                f"{sorted(unexpected)}"
            )

        normalized = []

        for result in results:

            checkpoint_id = result.get("id")

            status = str(
                result.get("status", "")
            ).upper()

            reason = str(
                result.get("reason", "")
            ).strip()

            if status not in {"PASS", "FAIL"}:
                raise ValueError(
                    f"Invalid status for {checkpoint_id}: "
                    f"{status}"
                )

            if not reason:
                raise ValueError(
                    f"Empty evaluator reason for "
                    f"{checkpoint_id}"
                )

            normalized.append(
                {
                    "id": checkpoint_id,
                    "status": status,
                    "reason": reason,
                }
            )

        by_id = {
            result["id"]: result
            for result in normalized
        }

        ordered = [
            by_id[checkpoint_id]
            for checkpoint_id in expected_ids
        ]

        failures = [
            result
            for result in ordered
            if result["status"] == "FAIL"
        ]

        return {
            "all_pass": len(failures) == 0,
            "results": ordered,
            "failures": failures,
        }

    @staticmethod
    def format_failures_for_feedback(
        failures: list,
    ) -> str:

        return "\n".join(
            f"- [{failure['id']}] "
            f"{failure['reason']}"
            for failure in failures
        )