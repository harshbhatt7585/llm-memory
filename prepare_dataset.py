from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm


DATASET_PATH = Path("dataset.json")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = Path("conversation.index")
FAISS_METADATA_PATH = Path("conversation_metadata.json")
MAX_CHUNK_WORDS = 500  # Maximum words per chunk before splitting
SIMILARITY_THRESHOLD = 0.5  # Threshold for semantic breakpoints (lower = more splits)

# Global embedder for semantic chunking (lazy loaded)
_semantic_embedder: Optional[SentenceTransformer] = None


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


def get_semantic_embedder() -> SentenceTransformer:
    """Lazy load the embedder for semantic chunking."""
    global _semantic_embedder
    if _semantic_embedder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _semantic_embedder = SentenceTransformer(EMBED_MODEL_ID, device=device)
    return _semantic_embedder


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex-based tokenization."""
    # Split on sentence-ending punctuation followed by space or newline
    # Handles: periods, exclamation marks, question marks
    # Preserves the punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings and strip whitespace
    return [s.strip() for s in sentences if s.strip()]


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def semantic_chunk_text(
    text: str, 
    max_words: int, 
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> List[str]:
    """
    Split text into chunks using semantic similarity.
    
    This approach:
    1. Splits text into sentences
    2. Computes embeddings for each sentence
    3. Finds breakpoints where consecutive sentences have low similarity (topic shift)
    4. Groups sentences into chunks respecting both semantic boundaries and max size
    """
    sentences = split_into_sentences(text)
    
    # Compute embeddings for all sentences
    embedder = get_semantic_embedder()
    sentence_embeddings = embedder.encode(sentences, convert_to_numpy=True)
    
    # Find semantic breakpoints (where similarity drops below threshold)
    breakpoints = []
    for i in range(len(sentences) - 1):
        similarity = compute_cosine_similarity(
            sentence_embeddings[i], 
            sentence_embeddings[i + 1]
        )
        if similarity < similarity_threshold:
            breakpoints.append(i + 1)  # Break after sentence i
    
    # Group sentences into chunks based on breakpoints and max_words
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0
    breakpoint_set = set(breakpoints)
    
    for i, sentence in enumerate(sentences):
        sentence_words = len(sentence.split())
        
        # Check if adding this sentence would exceed max_words
        if current_word_count + sentence_words > max_words and current_chunk_sentences:
            # Save current chunk and start new one
            chunks.append(' '.join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_word_count = sentence_words
        else:
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_words
            
            # Check for semantic breakpoint
            if i + 1 in breakpoint_set and current_chunk_sentences:
                chunks.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = []
                current_word_count = 0
    
    # Add remaining sentences
    if current_chunk_sentences:
        chunks.append(' '.join(current_chunk_sentences))
    
    return chunks


def collect_chunks(dataset: List[Dict]) -> Tuple[List[str], List[Dict]]:
    chunk_texts: List[str] = []
    metadata: List[Dict] = []

    for question_idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Questions"):
        for chunk_idx, chunk in tqdm(enumerate(item.get("session", [])), total=len(item.get("session", [])), desc="Chunks"):
            text = format_session(chunk)
            word_count = len(text.split())
            
            # Split if text exceeds max word threshold using semantic chunking
            if word_count > MAX_CHUNK_WORDS:
                split_pieces = semantic_chunk_text(text, MAX_CHUNK_WORDS)
                for split_idx, piece in enumerate(split_pieces):
                    chunk_id = len(chunk_texts)
                    chunk_texts.append(piece)
                    metadata.append(
                        {
                            "chunk_id": chunk_id,
                            "chunk_idx": chunk_idx,
                            "split_idx": split_idx,
                            "total_splits": len(split_pieces),
                            "text": piece,
                        }
                    )
            else:
                chunk_id = len(chunk_texts)
                chunk_texts.append(text)
                metadata.append(
                    {
                        "chunk_id": chunk_id,
                        "chunk_idx": chunk_idx,
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
