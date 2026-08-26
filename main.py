"""
main.py
-------
CLI entry point for the self-evaluating lesson generator.
"""

import argparse
from dotenv import load_dotenv

from src.llm_client import LLMClient
from src.generator import Generator
from src.evaluator import Evaluator
from src.orchestrator import run_pipeline, write_outputs


load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Self-evaluating lesson content generator"
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Topic to teach, for example: 'Introduction to RAG'",
    )

    parser.add_argument(
        "--inject-error",
        action="store_true",
        help=(
            "Inject a deliberate factual error into the first draft "
            "for demonstrating evaluator rejection."
        ),
    )

    args = parser.parse_args()

    llm = LLMClient()

    generator = Generator(llm)
    evaluator = Evaluator(llm)

    result = run_pipeline(
        args.topic,
        generator,
        evaluator,
        inject_error=args.inject_error,
    )

    lesson_path, log_path = write_outputs(
        args.topic,
        result,
    )

    print(f"\nFinal lesson written to: {lesson_path}")
    print(f"Rejection log written to: {log_path}")


if __name__ == "__main__":
    main()