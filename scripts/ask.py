"""Ask the forecast a question from the command line, using the configured model.

Usage:
    python scripts/ask.py "Which evening this week is best for a run?"
    python scripts/ask.py --fixture tests/fixtures/valid.json "Do I need a coat on Thursday?"

Without --fixture it fetches the live forecast (needs VISUAL_CROSSING_API_KEY).
"""

import argparse
import json
from pathlib import Path

from weather_rag.answer import answer_question
from weather_rag.chunks import forecast_chunks, guide_chunks
from weather_rag.fetch import fetch_forecast
from weather_rag.llm import get_provider
from weather_rag.parse import parse_forecast
from weather_rag.retrieve import Retriever

GUIDE = Path(__file__).resolve().parent.parent / "corpus" / "what_to_wear.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--fixture", type=Path, help="use a saved API response instead")
    args = parser.parse_args()

    raw = json.loads(args.fixture.read_text()) if args.fixture else fetch_forecast()
    guides = guide_chunks(GUIDE.read_text(), "what-to-wear") if GUIDE.exists() else []
    retriever = Retriever(forecast_chunks(parse_forecast(raw)), guides)
    provider = get_provider()

    result = answer_question(args.question, retriever, provider)

    print(f"[{result.status}] via {provider.name}")
    if result.answer:
        print(result.answer)
    for line in (result.reason, result.note):
        if line:
            print(f"- {line}")
    print("Based on:", ", ".join(chunk.label for chunk in result.sources) or "nothing")
    print("\nHow this answer was produced:")
    for step in result.trace:
        print(f"  {'✓' if step.ok else '✗'} {step.text}")


if __name__ == "__main__":
    main()
