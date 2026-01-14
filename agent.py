from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_dataset(path: str | Path) -> List[Dict]:
    with open(path, "r") as fh:
        return json.load(fh)


dataset = load_dataset("dataset.json")


model_id = "Qwen/Qwen2.5-0.5B-Instruct"
device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)


preferred_dtype = (
    torch.bfloat16 if device.type == "cuda" else
    torch.float16 if device.type == "mps" else
    torch.float32
)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=preferred_dtype,
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)


SYSTEM_PROMPT = (
    "You are memory agent, who goes through all the conversation to find the relevant answer. "
    "If the conversation contains the answer, respond with `Answer: <the answer>`. "
    "If you cannot find it, respond with `Answer: I don't know`."
)


def format_session(session: List[Dict[str, str]]) -> str:
    """Turn the structured chat session into plain text for prompting."""
    formatted_turns = [
        f"{turn['role'].capitalize()}: {turn['content']}" for turn in session
    ]
    return "\n".join(formatted_turns)


def query_model(context_text: str) -> str:
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question and conversation:\n\n{context_text}",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[:, input_length:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return decoded[0]


def extract_answer(model_response: str) -> str:
    marker = "Answer:"
    if marker not in model_response:
        return model_response.strip()
    return model_response.split(marker, maxsplit=1)[1].strip()


results = []

for item in dataset:
    question = item["question"]
    sessions = item["session"]
    current_idx = 0
    answer = "I don't know"

    while current_idx < len(sessions):
        current_context = sessions[current_idx]
        context = f"Question: {question}\n\nConversation:\n{format_session(current_context)}"
        response = query_model(context)
        print(response)
        parsed_answer = extract_answer(response)
        current_idx += 1

        if parsed_answer.lower().startswith("i don't know"):
            continue

        answer = parsed_answer
        break

    results.append(
        {
            "question": question,
            "answer": answer,
            "chunks_examined": current_idx,
            "chunks_total": len(sessions),
        }
    )


output_path = Path("results.json")
with output_path.open("w") as fh:
    json.dump(results, fh, indent=2)

print(f"Saved {len(results)} answers to {output_path}")
