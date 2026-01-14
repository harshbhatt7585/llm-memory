from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


DATASET_PATH = Path("dataset.json")
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MAX_NEW_TOKENS = 64  # JSON response is short, no need for 128


def load_dataset(path: Path) -> List[Dict]:
    with path.open("r") as fh:
        return json.load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple memory agent for conversation deep search")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Dataset JSON file")
    parser.add_argument("--output", type=Path, default=Path("results.json"), help="Path to save agent answers")
    parser.add_argument("--start", type=int, default=0, help="Question index to start from")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="How many questions to process (<=0 means process all remaining)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit how many conversation chunks are inspected per question",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print model responses for each chunk while processing",
    )
    return parser.parse_args()


def format_session(session: List[Dict[str, str]]) -> str:
    formatted_turns = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in session]
    return "\n".join(formatted_turns)


@torch.inference_mode()
def query_model(context_text: str, system_prompt: str) -> str:
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_text},
    ]

    if tokenizer.chat_template is not None:
        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model_device)
    else:
        prompt = f"SYSTEM: {system_prompt}\nUSER: {context_text}\nASSISTANT:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model_device)

    # Greedy decoding is faster than sampling for deterministic JSON output
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        temperature=0.1,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.2,
        early_stopping=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[:, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


def parse_agent_response(raw: str) -> Dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"found": False, "answer": raw.strip()}


def resolve_model_device() -> torch.device:
    requested = os.environ.get("LLM_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Default to CPU unless the user explicitly opts in to MPS via env var.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


model_device = resolve_model_device()
preferred_dtype = (
    torch.bfloat16 if model_device.type == "cuda" else
    torch.float16 if model_device.type == "mps" else
    torch.float32
)

print(f"Loading model on {model_device} with {preferred_dtype}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=preferred_dtype,
).to(model_device).eval()

# Compile model for faster inference (PyTorch 2.0+)
if hasattr(torch, "compile") and model_device.type in ("cuda", "cpu"):
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled with torch.compile()")
    except Exception as e:
        print(f"torch.compile() not available: {e}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SYSTEM_PROMPT = """You are a memory agent.
You must answer ONLY using the provided conversation chunk.
If the chunk contains the answer, output JSON:
{"found": true, "answer": "..."}
If not, output:
{"found": false, "answer": ""}
Output VALID JSON only (double quotes).

EXAMPLE:
"X went to the gym on Monday at 7PM, he came back and slept. He woke up at 9AM next day and went to the office."
Question: At what time did Harsh went to the gym on Monday?
Answer: {"found": true, "answer": "He went to the gym at 7PM"}

Question: Why X was upset in the office?
Answer: {"found": false, "answer": ""}
"""


def process_question(
    question_idx: int,
    item: Dict,
    *,
    max_chunks: int | None = None,
    verbose: bool = False,
) -> Dict:
    """Process a question by scanning the conversation chunks sequentially."""
    question = item.get("question", "")
    sessions = item.get("session", [])

    if not sessions:
        return {
            "question": question,
            "answer": "",
            "chunks_examined": 0,
            "chunks_total": 0,
            "retrieved_chunks": [],
            "chunk_logs": [],
        }

    limit = None if max_chunks is None or max_chunks <= 0 else max_chunks
    chunk_logs = []
    final_answer = ""

    for chunk_idx, chunk in enumerate(sessions):
        if limit is not None and chunk_idx >= limit:
            break

        context = f"Question: {question}\n\nConversation:\n{format_session(chunk)}"
        response = query_model(context, SYSTEM_PROMPT)

        print(f"--------------------------------")
        print(f"Question: {question}")
        print(f"Conversation: {chunk}")
        print(f"[chunk {chunk_idx}] {response}")
        print(f"--------------------------------")
        parsed = parse_agent_response(response)

        chunk_logs.append(
            {
                "chunk_index": chunk_idx,
                "response": response,
                "parsed": parsed,
            }
        )

        if parsed.get("found"):
            final_answer = parsed.get("answer", "")
            break
    else:
        if chunk_logs:
            final_answer = chunk_logs[-1]["parsed"].get("answer", "")

    retrieved_chunks = [entry["chunk_index"] for entry in chunk_logs]
    return {
        "question": question,
        "answer": final_answer,
        "chunks_examined": len(chunk_logs),
        "chunks_total": len(sessions),
        "retrieved_chunks": retrieved_chunks,
        "chunk_logs": chunk_logs,
    }


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    if not dataset:
        raise RuntimeError("Dataset is empty. Nothing to evaluate.")

    if args.start < 0 or args.start >= len(dataset):
        raise ValueError(f"Start index {args.start} is outside the dataset range (size={len(dataset)})")

    remaining = len(dataset) - args.start
    count = remaining if args.count <= 0 else min(args.count, remaining)
    subset = dataset[args.start : args.start + count]

    print(f"Processing {len(subset)} questions starting at index {args.start}...")
    iterator = enumerate(subset, start=args.start)
    results = []
    for question_idx, item in tqdm(iterator, total=len(subset), desc="Questions"):
        result = process_question(
            question_idx,
            item,
            max_chunks=args.max_chunks,
            verbose=args.verbose,
        )
        results.append(result)

    with args.output.open("w") as fh:
        json.dump(results, fh, indent=2)

    total_chunks = sum(r["chunks_examined"] for r in results)
    found_count = sum(1 for r in results if r["answer"])
    print(f"\nSaved {len(results)} answers to {args.output}")
    print(f"Found answers: {found_count}/{len(results)}, Total chunks examined: {total_chunks}")




def search_agent(question: str, model, trials=20) -> str:
    """This agent will search the conversation in the vector db, it will create query for vector db"""


    messages = [
        {"role": "system", 
        "content": """You are a search agent who will make query for vector db to search the conversation, iterate over different queries to find the answer"""
        },
        {
            "role": "user", 
            "content": f"Generate a query for vector db to search the conversation for the question: {question}"
        }
    ]
    for i in range(trials):


        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model_device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            temperature=0.1,
            top_p=0.95,
            top_k=40,
        )

        response = tokenizer.decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        results = search_vector_db(response)

        messages.append({
            "role": "assistant",
            "content": response
        })


        messages.append({
            "role": "user",
            "content": f"the results are: {results}, Can you find the answer to the question: {question}? If you found the answer, output JSON: {"found": true, "answer": "..."}, If not, output: {"found": false, "answer": ""}"
        })

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model_device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            temperature=0.1,
            top_p=0.95,
            top_k=40,
        )
        response = tokenizer.decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        parsed = parse_agent_response(response)
        if parsed.get("found"):
            return parsed.get("answer", "")
        

    return ""






        

     


if __name__ == "__main__":
    main()
