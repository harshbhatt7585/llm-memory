from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from google import genai
except ImportError as exc:  # pragma: no cover - import error is user-actionable
    raise SystemExit(
        "google-genai package is required. Install with 'pip install google-genai'."
    ) from exc

try:  # Optional dependency so we can read .env files automatically if available
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


DEFAULT_PROMPT = "Explain how AI works in a few words."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick Gemini LLM smoke test")
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model to query (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text to send to the Gemini API",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override GEMINI_API_KEY environment variable",
    )
    return parser.parse_args()


def resolve_api_key(explicit_key: str | None) -> str:
    api_key = (
        explicit_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        raise SystemExit(
            "Missing Gemini API key. Set GEMINI_API_KEY env var or pass --api-key."
        )
    return api_key


def run_test(model_name: str, prompt: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini API returned an empty response")
    return text.strip()


def main() -> None:
    if load_dotenv is not None:
        dotenv_path = Path(".env")
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
    args = parse_args()
    api_key = resolve_api_key(args.api_key)
    try:
        output = run_test(args.model, args.prompt, api_key)
    except Exception as exc:  # pragma: no cover - surfaces API errors to user
        print(f"Gemini API test failed: {exc}", file=sys.stderr)
        raise
    else:
        print("Gemini LLM response:\n")
        print(output)


if __name__ == "__main__":
    main()
