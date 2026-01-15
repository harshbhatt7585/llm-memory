import json
import os
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")


class Tools:
    """Abstraction for tools that can be used in the agent."""
    
    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    def __call__(self, **kwargs) -> dict:
        """Execute the tool with the given arguments."""
        raise NotImplementedError("Subclasses must implement this method.")


class RAGTool(Tools):
    """Tool for retrieving information from the RAG index."""
    def __call__(self, query: str, k: int = 5) -> dict:
        """Retrieve information from the RAG index."""
        return retrieve_top_k_chunks(query, embedder, index, metadata, k)


def load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def generate_embedding(embedder: SentenceTransformer, text: str, faiss: faiss.Index) -> np.ndarray:
    """Return embedding with batch dimension for Faiss search."""
    encoding = embedder.encode(text)
    return normalize_embedding(np.asarray(encoding, dtype=np.float32).reshape(1, -1))



def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    return embedding / np.linalg.norm(embedding)


def load_faiss_index(index_path: str) -> faiss.Index:
    return faiss.read_index(index_path)


def search_faiss_index(index: faiss.Index, encoding: np.ndarray, k: int = 10) -> tuple:
    """Return distances and indices from FAISS search."""
    return index.search(encoding, k=k)


def load_dataset(dataset_path: str):
    with open(dataset_path, "r") as f:
        return json.load(f)


def load_metadata(metadata_path: str) -> dict:
    """Load conversation metadata that maps chunk IDs to chunk data."""
    with open(metadata_path, "r") as f:
        return json.load(f)


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


def retrieve_top_k_chunks(
    query: str,
    embedder: SentenceTransformer,
    index: faiss.Index,
    metadata: dict,
    k: int = 5
) -> List[dict]:
    encoding = generate_embedding(embedder, query, index)
    distances, indices = search_faiss_index(index, encoding, k=k)
    chunks = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(metadata["chunks"]):
            chunk_data = metadata["chunks"][idx].copy()
            chunk_data["similarity_score"] = float(dist)
            chunk_data["rank"] = i + 1
            chunks.append(chunk_data)

    return chunks


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



def search_agent(question: str, chunks: List[dict]):

    rag_tool = RAGTool(name="RAGTool", description="Retrieve information from the RAG index.", parameters={"query": str, "k": int})

    system_prompt = (
        "You are a search agent who searches the answer to the question from the conversation.",
        "Given a question like 'What did I order last night at the <restaurant_name> restaurant?', "
        "You have the following tools to use: RAGTool",
        "RAGTool is a tool that can be used to retrieve information from the RAG index.",
        "RAGTool takes a query and a number of chunks to retrieve.",
        "RAGTool returns a list of chunks that are similar to the query.",
        "You have to crwal the conversation to find the answer to the question.",
        "Think step by step and reason about the question and the context to find the answer using the RAGTool.",
        "To call the RAGTool, you need to use the following JSON format: {'tool': 'RAGTool', 'args': {'query': <query>, 'k': <k>}}",
        "You have to find the answer in the conversation",
        """return the answer in the form of JSON: {"found": true, "answer": <answer>}""",
        "If you think there is no valid answer to question"
        """return the answer in the form of JSON: {"found": false, "answer": ""}""",
    )
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=f"Context: {chunks if chunks else ''} Question: {question}")],
        ),
    ]
    print("calling gemini request ")

    response = generate_chat_completion(
        contents=contents,
        system_instruction=system_prompt,
    )

    print("response: ", response)


    parsed_response = parse_agent_response(response)

    if parsed_response.get("tool") == "RAGTool":
        
        args = parsed_response.get("args")
        if args:
            print("calling RAGTool")
            chunk = rag_tool(args["query"], args["k"])
            return search_agent(question, chunk) 
        else:
            print("no args found")

    return parsed_response

    





if __name__ == "__main__":
    # Example: Retrieve top 3 chunks for a query
    dataset = load_dataset("dataset.json")
    question = dataset[0]["question"]

    global embedder, index, metadata
    
    # Load required components
    embedder = load_embedder("sentence-transformers/all-MiniLM-L6-v2")
    index = load_faiss_index("conversation.index")
    metadata = load_metadata("conversation_metadata.json")
    
    # Perform similarity search
    # chunks = retrieve_top_k_chunks(query, embedder, index, metadata, k=3)
 
    answer = search_agent(question, None)
    print(answer)