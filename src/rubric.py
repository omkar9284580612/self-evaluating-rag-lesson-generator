"""
rubric.py
---------
The rubric is deliberately a list of hard, binary (pass/fail) checkpoints --
no 1-10 scores, no partial credit. This mirrors what the JD asks for:
"eval sets, rubrics, and guardrails that decide what's good enough to ship."

A score like "7/10 beginner-friendly" is not actionable for an automated
regenerate step. "FAIL: uses the term 'embedding vector' without ever
defining it" is directly actionable -- it tells the generator exactly
what to fix. That's the whole design reason for binary + reasons.
"""

RUBRIC = [
    {
        "id": "accurate_grounded",
        "label": "Accurate & grounded",
        "check": (
            "Every factual claim about RAG is technically correct and not "
            "invented/hallucinated. No made-up statistics, no wrong "
            "definitions, no confusing RAG with fine-tuning or plain "
            "prompting."
        ),
    },
    {
        "id": "beginner_friendly",
        "label": "Beginner-friendly language",
        "check": (
            "Written for a 12th-grade graduate with limited English "
            "vocabulary and a non-English-medium background. Short "
            "sentences. No idioms. No assumed prior CS/ML vocabulary."
        ),
    },
    {
        "id": "teaches_by_example",
        "label": "Teaches by example",
        "check": (
            "Includes at least one concrete, worked example or analogy "
            "(e.g. 'open-book exam' style analogy, or a real query walked "
            "through step by step) -- not just abstract definitions."
        ),
    },
    {
        "id": "no_unexplained_jargon",
        "label": "Clear, no unexplained jargon",
        "check": (
            "Every technical term used (retrieval, embedding, vector "
            "database, LLM, context window, hallucination, etc.) is "
            "defined in plain words the first time it appears."
        ),
    },
    {
        "id": "covers_key_points",
        "label": "Covers the key points",
        "check": (
            "Covers, at minimum: (1) what RAG is, (2) the problem it "
            "solves / why it matters (LLMs don't know your private or "
            "recent data, and can hallucinate), (3) how it works at a "
            "high level (retrieve relevant chunks, then feed them to the "
            "LLM as context before it answers)."
        ),
    },
    {
        "id": "coherent_flow",
        "label": "Coherent teaching flow",
        "check": (
            "Ideas build in a logical order (motivation -> concept -> "
            "mechanism -> example -> recap). No topic introduced before "
            "the reader has what they need to understand it. Ends with a "
            "short recap or takeaway."
        ),
    },
]


def rubric_as_prompt_block(topic: str = "the requested topic") -> str:
    lines = []
    for i, item in enumerate(RUBRIC, start=1):
        check = item["check"].replace("RAG", topic)
        lines.append(f"{i}. [{item['id']}] {item['label']}: {check}")
    return "\n".join(lines)
