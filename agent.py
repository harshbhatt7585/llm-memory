from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET_PATH = Path("dataset.json")
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_CACHE_PATH = Path("chunk_embeddings.pt")
RETRIEVAL_TOP_K = 5


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
    formatted_turns = [
        f"{turn['role'].capitalize()}: {turn['content']}" for turn in session
    ]
    return "\n".join(formatted_turns)


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
        prompt = []
        for turn in conversation:
            role = turn["role"].upper()
            prompt.append(f"{role}: {turn['content']}")
        prompt.append("ASSISTANT:")
        prompt_text = "\n".join(prompt)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model_device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.18,
            num_beams=1,
            no_repeat_ngram_size=3,
            early_stopping=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[:, input_length:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return decoded[0]


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
        self.metadata, self.embeddings = self._load_or_build_embeddings()
        self.question_to_positions = self._build_question_map()

    def _build_question_map(self) -> Dict[int, List[int]]:
        mapping: Dict[int, List[int]] = {}
        for idx, meta in enumerate(self.metadata):
            mapping.setdefault(meta["question_idx"], []).append(idx)
        return mapping

    def _load_or_build_embeddings(self):
        if self.cache_path.exists():
            stored = torch.load(self.cache_path, map_location="cpu")
            if stored.get("dataset_hash") == self.dataset_hash:
                metadata = stored["metadata"]
                embeddings = stored["embeddings"]
            else:
                metadata, embeddings = self._compute_and_store_embeddings()
        else:
            metadata, embeddings = self._compute_and_store_embeddings()

        embeddings = embeddings.to(self.embed_torch_device)
        return metadata, embeddings

    def _compute_and_store_embeddings(self):
        chunk_texts = []
        metadata = []
        for q_idx, item in enumerate(self.dataset):
            for chunk_idx, chunk in enumerate(item["session"]):
                metadata.append({"question_idx": q_idx, "chunk_idx": chunk_idx})
                chunk_texts.append(format_session(chunk))

        embeddings = self.embedder.encode(
            chunk_texts,
            convert_to_tensor=True,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        embeddings_cpu = embeddings.cpu()
        torch.save(
            {
                "metadata": metadata,
                "embeddings": embeddings_cpu,
                "dataset_hash": self.dataset_hash,
            },
            self.cache_path,
        )
        return metadata, embeddings_cpu

    def retrieve(self, question_idx: int, question: str, top_k: int) -> List[Tuple[int, float]]:
        candidate_positions = self.question_to_positions.get(question_idx, [])
        if not candidate_positions:
            return []

        chunk_embeddings = self.embeddings[candidate_positions]
        query_embedding = self.embedder.encode(
            question,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        if query_embedding.device != chunk_embeddings.device:
            query_embedding = query_embedding.to(chunk_embeddings.device)

        scores = torch.matmul(chunk_embeddings, query_embedding.unsqueeze(-1)).squeeze(-1)
        k = min(top_k, scores.shape[0])
        if k == 0:
            return []

        top_scores, top_indices = torch.topk(scores, k=k)
        results: List[Tuple[int, float]] = []
        for score, local_idx in zip(top_scores.tolist(), top_indices.tolist()):
            meta_idx = candidate_positions[local_idx]
            chunk_idx = self.metadata[meta_idx]["chunk_idx"]
            results.append((chunk_idx, float(score)))
        return results


dataset = load_dataset(DATASET_PATH)
dataset_hash = compute_file_hash(DATASET_PATH)

model_device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
preferred_dtype = (
    torch.bfloat16 if model_device.type == "cuda" else
    torch.float16 if model_device.type == "mps" else
    torch.float32
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=preferred_dtype,
).to(model_device).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

SYSTEM_PROMPT = """You are a memory agent.
You must answer ONLY using the provided conversation chunk.
If the chunk contains the answer, output JSON:
{"found": true, "answer": "..."}
If not, output:
{"found": false, "answer": ""}
Output VALID JSON only (double quotes).

EXAMPLE:
"Harsh went to the gym on Monday at 7PM, he came back and slept. He woke up at 9AM next day and went to the office."
Question: At what time did Harsh went to the gym on Monday?
Answer: {"found": true, "answer": "He went to the gym at 7PM"}

Question: Why Harsh was upset in the office?
Answer: {"found": false, "answer": ""}
"""


retriever = SemanticRetriever(dataset, dataset_hash)
results = []

for question_idx, item in enumerate(dataset):
    question = item["question"]
    sessions = item["session"]

    if not sessions:
        results.append(
            {
                "question": question,
                "answer": "",
                "chunks_examined": 0,
                "chunks_total": 0,
                "retrieved_chunks": [],
                "chunk_logs": [],
            }
        )
        continue

    retrieved = retriever.retrieve(question_idx, question, RETRIEVAL_TOP_K)
    score_lookup = {chunk_idx: score for chunk_idx, score in retrieved}
    chunk_order = [idx for idx, _ in retrieved]

    if not chunk_order:
        chunk_order = list(range(min(RETRIEVAL_TOP_K, len(sessions))))

    chunk_logs = []
    final_answer = ""
    found_answer = False

    for chunk_idx in chunk_order:
        current_context = sessions[chunk_idx]
        context = (
            f"Question: {question}\n\nConversation:\n{format_session(current_context)}"
        )
        response = query_model(context, SYSTEM_PROMPT)
        print(response)
        print("--------------------------------")
        parsed = parse_agent_response(response)
        chunk_logs.append(
            {
                "chunk_index": chunk_idx,
                "similarity": score_lookup.get(chunk_idx),
                "response": response,
                "parsed": parsed,
            }
        )
        if parsed.get("found"):
            final_answer = parsed.get("answer", "")
            found_answer = True
            break

    if not found_answer and chunk_logs:
        final_answer = chunk_logs[-1]["parsed"].get("answer", "")

    results.append(
        {
            "question": question,
            "answer": final_answer,
            "chunks_examined": len(chunk_logs),
            "chunks_total": len(sessions),
            "retrieved_chunks": chunk_order,
            "chunk_logs": chunk_logs,
        }
    )

output_path = Path("results.json")
with output_path.open("w") as fh:
    json.dump(results, fh, indent=2)

print(f"Saved {len(results)} answers to {output_path}")
