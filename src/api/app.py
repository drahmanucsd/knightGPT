import os
import sys
# Ensure 'src' directory is on Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from typing import List, Optional
from dataclasses import dataclass
from ollama import chat
from retrieval.retriever import Retriever
from utils.logging import configure_logging
import json

# Configure centralized logging
configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Paths and initialization
chunks_path = os.getenv("CHUNKS_PATH", "data/chunks.json")
graph_path = os.getenv("GRAPH_PATH", "data/graph.graphml")
graph_format = os.getenv("GRAPH_FORMAT", "graphml")
model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
metadata_path = os.getenv("METADATA_PATH", "data/processed_pages.json")

# Load PDF metadata for title and DOI
try:
    with open(metadata_path, 'r', encoding='utf-8') as f:
        md = json.load(f).get('metadata', {})
        PDF_TITLE = md.get('title', 'Unknown Title')
        PDF_DOI = md.get('doi', 'Unknown DOI')
except Exception as e:
    logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
    PDF_TITLE = 'Unknown Title'
    PDF_DOI = 'Unknown DOI'

# Initialize Retriever for CLI mode
retriever = Retriever(
    chunks_path=chunks_path,
    graph_path=graph_path,
    graph_format=graph_format,
    model_name=model_name
)
logger.info("Retriever initialized for CLI mode")

@dataclass
class Citation:
    text: str
    node_id: str
    page: Optional[int]
    paragraph_index: Optional[int]
    chunk_index: Optional[int]


def process_query(query: str, top_k: int = 5, hops: int = 1):
    """
    Retrieve relevant chunks and query llama3 interactively in the terminal.
    """
    # Retrieve chunks
    chunks = retriever.retrieve(query=query, top_k=top_k, hops=hops)
    if not chunks:
        logger.error("No relevant documents found.")
        return

    # Build context and citations list
    context_lines: List[str] = []
    citations: List[Citation] = []
    for idx, c in enumerate(chunks, start=1):
        tag = f"[^{idx}]"
        context_lines.append(f"{tag} {c['text']}")
        citations.append(Citation(
            text=c['text'],
            node_id=c['node_id'],
            page=c.get('page'),
            paragraph_index=c.get('paragraph_index'),
            chunk_index=c.get('chunk_index')
        ))
    context = "\n\n".join(context_lines)

    # Prepare prompts
    system_prompt = (
        "You are an expert microbiome researcher. "
        "Answer the question using the provided context paragraphs. "
        "Include inline citations like [^1], [^2], etc."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Query llama3 via Ollama
    try:
        resp = chat(
            model="llama3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
    except Exception as e:
        logger.error(f"LLM service error: {e}")
        return

    # Extract and print answer
    content = getattr(resp, 'content', None) or getattr(getattr(resp, 'message', {}), 'content', '')
    answer = content.strip()
    print(f"\n=== Answer (based on '{PDF_TITLE}' DOI: {PDF_DOI}) ===")
    print(answer)
    print("\n=== Citations ===")
    for idx, cit in enumerate(citations, start=1):
        print(f"[^ {idx}] Title: {PDF_TITLE}, DOI: {PDF_DOI}, Page {cit.page}, Para {cit.paragraph_index}, Chunk {cit.chunk_index}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLI for Microbiome RAG Chatbot")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve initially")
    parser.add_argument("--hops", type=int, default=1, help="Number of graph hops for context expansion")
    args = parser.parse_args()

    print("Microbiome RAG Chatbot CLI (type 'exit' to quit)")
    while True:
        query = input("\nEnter your question: ")
        if query.lower().strip() in ("exit", "quit"):
            print("Goodbye!")
            break
        process_query(query, top_k=args.top_k, hops=args.hops)
