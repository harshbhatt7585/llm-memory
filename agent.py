from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
from tqdm import tqdm

# Fix FAISS segfault on macOS by disabling OpenMP parallelism
faiss.omp_set_num_threads(1)


DATASET_PATH = Path("dataset.json")
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_CACHE_PATH = Path("chunk_embeddings.pt")
FAISS_INDEX_PATH = Path("conversation.index")
FAISS_METADATA_PATH = Path("conversation_metadata.json")
RETRIEVAL_TOP_K = 5
MAX_NEW_TOKENS = 64  # JSON response is short, no need for 128


def load_dataset(path: Path) -> List[Dict]:
    with path.open("r") as fh:
        return json.load(fh)


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


class SemanticRetriever:
    def __init__(
        self,
        dataset: List[Dict],
        dataset_hash: str,
        model_name: str = EMBED_MODEL_ID,
        cache_path: Path = EMBED_CACHE_PATH,
    ) -> None:
        self.dataset = dataset
        self.dataset_hash = dataset_hash
        self.model_name = model_name
        self.cache_path = cache_path
        self.embed_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_torch_device = torch.device(self.embed_device)
        self.embedder = SentenceTransformer(self.model_name, device=self.embed_device)
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.metadata = self._load_metadata()
        self.chunk_lookup = {entry["chunk_id"]: entry for entry in self.metadata}

    def _load_metadata(self) -> List[Dict]:
        with FAISS_METADATA_PATH.open("r") as fh:
            payload = json.load(fh)
        stored_hash = payload.get("dataset_hash")
        if stored_hash != self.dataset_hash:
            raise RuntimeError(
                "Metadata hash mismatch. Regenerate the FAISS index via prepare_dataset.py"
            )
        chunks = payload.get("chunks", [])
        chunks.sort(key=lambda item: item["chunk_id"])  # keep aligned with FAISS ids
        return chunks

    def retrieve(self, question_idx: int, question: str, top_k: int) -> List[Tuple[int, float]]:
        query_embedding = self.embedder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        search_k = min(max(top_k * 5, top_k), self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)

        filtered: List[Tuple[int, float]] = []
        for chunk_id, score in zip(indices[0], scores[0]):
            if chunk_id == -1:
                continue
            meta = self.chunk_lookup.get(int(chunk_id))
            if not meta or meta["question_idx"] != question_idx:
                continue
            filtered.append((meta["chunk_idx"], float(score)))
            if len(filtered) == top_k:
                break
        return filtered


dataset = load_dataset(DATASET_PATH)
dataset_hash = compute_file_hash(DATASET_PATH)

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
    retriever: SemanticRetriever,
) -> Dict:
    """Process a single question and return the result dict."""
    question = item["question"]
    sessions = item["session"]

    if not sessions:
        return {
            "question": question,
            "answer": "",
            "chunks_examined": 0,
            "chunks_total": 0,
            "retrieved_chunks": [],
            "chunk_logs": [],
        }

    retrieved = retriever.retrieve(question_idx, question, RETRIEVAL_TOP_K)
    score_lookup = dict(retrieved)
    chunk_order = [idx for idx, _ in retrieved]

    if not chunk_order:
        chunk_order = list(range(min(RETRIEVAL_TOP_K, len(sessions))))

    chunk_logs = []
    final_answer = ""

    for chunk_idx in chunk_order:
        context = f"Question: {question}\n\nConversation:\n{format_session(sessions[chunk_idx])}"
        response = query_model(context, SYSTEM_PROMPT)
        print(response)
        parsed = parse_agent_response(response)
        
        chunk_logs.append({
            "chunk_index": chunk_idx,
            "similarity": score_lookup.get(chunk_idx),
            "response": response,
            "parsed": parsed,
        })
        
        if parsed.get("found"):
            final_answer = parsed.get("answer", "")
            break
    else:
        # No answer found in any chunk
        if chunk_logs:
            final_answer = chunk_logs[-1]["parsed"].get("answer", "")

    return {
        "question": question,
        "answer": final_answer,
        "chunks_examined": len(chunk_logs),
        "chunks_total": len(sessions),
        "retrieved_chunks": chunk_order,
        "chunk_logs": chunk_logs,
    }


def main():
    print(f"Processing {len(dataset)} questions...")
    retriever = SemanticRetriever(dataset, dataset_hash)
    
    results = []
    for question_idx, item in enumerate(tqdm(dataset, desc="Questions")):
        if question_idx != 0:
            break
        result = process_question(question_idx, item, retriever)
        results.append(result)
    
    output_path = Path("results.json")
    with output_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    
    # Summary stats
    total_chunks = sum(r["chunks_examined"] for r in results)
    found_count = sum(1 for r in results if r["answer"])
    print(f"\nSaved {len(results)} answers to {output_path}")
    print(f"Found answers: {found_count}/{len(results)}, Total chunks examined: {total_chunks}")


if __name__ == "__main__":
    main()
