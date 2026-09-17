"""
generator.py
-------------
Generates beginner-friendly lessons and regenerates rejected lessons
using evaluator feedback.
"""

from src.llm_client import LLMClient


AUDIENCE_BRIEF = (
    "The learner is a 12th-grade graduate from India with limited "
    "English vocabulary and zero assumed computer science or machine "
    "learning background. The learner wants to start a career in AI "
    "and technology."
)


BASE_INSTRUCTIONS = """
You are writing a standalone beginner lesson about:

"{topic}"

AUDIENCE:
{audience_brief}

Create a COMPLETE lesson of approximately 250-350 words.

Use EXACTLY these five sections:

# Introduction to {topic}

## 1. What is {topic}?

Define the topic in simple English. Explain what it is and what problem
it helps solve. Use the topic's correct terminology.

## 2. Why does {topic} matter?

Explain the practical value of the topic and the problem it helps solve.

## 3. How does {topic} work?

Explain the main parts or steps in simple language. Define each technical
term when it first appears.

## 4. Simple real-world example

Give ONE complete, step-by-step example related to the topic.

## 5. Key takeaway

Give exactly three short bullet points summarizing the lesson.

STRICT RULES:

- Use simple English.
- Use short sentences.
- Define technical terms when first introduced.
- Avoid unnecessary jargon.
- Do not use unexplained terms.
- Do not make absolute claims.
- Complete every section.
- Never stop in the middle of a sentence.
- Do not repeat sections.
- Do not add a preamble.
- Output ONLY Markdown lesson content.
- The lesson MUST end with section 5.
- The final character must be part of a complete sentence or bullet.
"""


RETRY_ADDENDUM = """
The previous lesson was rejected by the evaluator.

You MUST fix every failure listed below.

FAILURE FEEDBACK:
{failure_feedback}

PERSISTENT LESSONS:
{persistent_lessons}

Do not remove sections that already passed.

Before producing the final answer, internally verify:

1. All five sections exist.
2. The explanation of RAG is technically correct.
3. The mechanism explains retrieval → context → generation.
4. A complete real-world example exists.
5. The lesson ends with three takeaway bullets.
6. No sentence is unfinished.
7. No section is missing.
"""


class Generator:

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(
        self,
        topic: str,
        failure_feedback: str = None,
        persistent_lessons: str = "",
    ) -> str:

        prompt = BASE_INSTRUCTIONS.format(
            topic=topic,
            audience_brief=AUDIENCE_BRIEF,
        )

        if failure_feedback:
            prompt += "\n\n" + RETRY_ADDENDUM.format(
                failure_feedback=failure_feedback,
                persistent_lessons=persistent_lessons or "(none yet)",
            )

        return self.llm.complete(
            prompt,
            temperature=0.3,
            max_tokens=1800,
        )

    def generate_with_deliberate_error(
        self,
        topic: str,
    ) -> str:
        """
        Generate a normal lesson and then deterministically inject
        a factual error for the evaluator demonstration.
        """

        lesson = self.generate(topic)

        return self._inject_error(lesson)

    @staticmethod
    def _inject_error(lesson: str) -> str:

        old = (
            "The retrieved information is added to the question "
            "as context."
        )

        bad = (
            "The retrieved information is used to fine-tune "
            "the LLM's internal weights."
        )

        if old in lesson:
            return lesson.replace(old, bad, 1)

        marker = "## 3. How does RAG work?"

        if marker in lesson:
            return lesson.replace(
                marker,
                marker
                + "\n\n"
                + "RAG works by fine-tuning the LLM's internal "
                  "weights using retrieved documents.",
                1,
            )

        return lesson