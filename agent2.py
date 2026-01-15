import json
import os
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment")
    return genai.Client(api_key=api_key)


def generate_response(
    prompt: str,
    system_instruction: str | None = None,
    model_name: str | None = None,
) -> str:
    """Simple single-turn generation."""
    client = _get_client()
    config = None
    if system_instruction:
        config = types.GenerateContentConfig(system_instruction=system_instruction)
    
    response = client.models.generate_content(
        model=model_name or MODEL_NAME,
        contents=prompt,
        config=config,
    )
    return response.text.strip()


def generate_chat_completion(
    contents: List[types.Content],
    system_instruction: str | None = None,
    model_name: str | None = None,
) -> str:
    """Multi-turn chat completion using typed Content objects."""
    client = _get_client()
    config = None
    if system_instruction:
        config = types.GenerateContentConfig(system_instruction=system_instruction)
    
    response = client.models.generate_content(
        model=model_name or MODEL_NAME,
        contents=contents,
        config=config,
    )
    return response.text.strip()


def parse_agent_response(response: str) -> dict:
    try:
        return json.loads(response)
    except ValueError:
        return {"query": response.strip(), "context": ""}


def query_generate_agent(question: str) -> str:
    system_prompt = (
        "You are a memory search agent. All conversation history lives in a vector DB, "
        "so you must iteratively craft retrieval queries until you can answer the user. "
        "Given a question like 'What did I order last night at the <restaurant_name> restaurant?', "
        "you should probe for 'restaurant', <restaurant_name>. Always respond with JSON in the "
        'form {"query": <query>, "context": <context>} describing the retrieval you need.'
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=f"Given Question: {question}")],
        ),
    ]

    response = generate_chat_completion(
        contents=contents,
        system_instruction=system_prompt,
    )
    parsed_response = parse_agent_response(response)
    return parsed_response.get("query", "")


if __name__ == "__main__":
    prompt = "What did I eat on Friday?"
    response = query_generate_agent(prompt)
    print(response)
