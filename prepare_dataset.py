from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss


DATASET_PATH = Path("dataset.json")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = Path("conversation.index")
FAISS_METADATA_PATH = Path("conversation_metadata.json")


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


def collect_chunks(dataset: List[Dict]) -> Tuple[List[str], List[Dict]]:
    chunk_texts: List[str] = []
    metadata: List[Dict] = []

    for question_idx, item in enumerate(dataset):
        question = item.get("question", "")
        for chunk_idx, chunk in enumerate(item.get("session", [])):
            text = format_session(chunk)
            chunk_id = len(chunk_texts)
            chunk_texts.append(text)
            metadata.append(
                {
                    "chunk_id": chunk_id,
                    "question_idx": question_idx,
                    "chunk_idx": chunk_idx,
                    "question": question,
                    "text": text,
                }
            )
    return chunk_texts, metadata


def build_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(model_name, device=device)
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings


def build_index(embeddings: np.ndarray) -> faiss.IndexIDMap2:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")
    dim = embeddings.shape[1]
    base_index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base_index)
    ids = np.arange(len(embeddings), dtype="int64")
    index.add_with_ids(embeddings, ids)
    return index


def save_metadata(metadata: List[Dict], dataset_hash: str) -> None:
    payload = {
        "dataset_hash": dataset_hash,
        "embedding_model": EMBED_MODEL_ID,
        "chunks": metadata,
    }
    with FAISS_METADATA_PATH.open("w") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    dataset = load_dataset(DATASET_PATH)
    if not dataset:
        raise RuntimeError("Dataset is empty. Nothing to index.")

    dataset_hash = compute_file_hash(DATASET_PATH)
    chunk_texts, metadata = collect_chunks(dataset)
    print(f"Collected {len(chunk_texts)} chunks from {len(dataset)} questions")

    embeddings = build_embeddings(chunk_texts, EMBED_MODEL_ID)
    index = build_index(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    save_metadata(metadata, dataset_hash)

    print(
        f"Saved FAISS index with {index.ntotal} entries to {FAISS_INDEX_PATH} and metadata to {FAISS_METADATA_PATH}"
    )


if __name__ == "__main__":
    main()
