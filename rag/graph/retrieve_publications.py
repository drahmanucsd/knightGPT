#!/usr/bin/env python3
import argparse
import numpy as np
from ollama import embeddings
from rag.graph.store import GraphStore


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve(query: str, graph_path: str, model_name: str = "bge-base-en-v1.5", top_k: int = 5):
    """
    Retrieve the top_k most similar Publication nodes to the free-text query.
    
    Args:
        query (str): The user's natural-language query.
        graph_path (str): Path to the GEXF graph file.
        model_name (str): Ollama embedding model name.
        top_k (int): Number of publications to return.
    
    Returns:
        List of tuples: [(pub_id, score, metadata_dict), ...]
    """
    # 1. Load graph
    gs = GraphStore()
    gs.load(graph_path)
    
    # 2. Embed the query
    q_vec = np.array(embeddings(model=model_name, prompt=query))
    
    # 3. Score each Publication node
    scores = []
    for node_id, attrs in gs.G.nodes(data=True):
        if attrs.get("type") != "Publication":
            continue
        emb = np.array(attrs["embedding"])
        score = cosine_sim(q_vec, emb)
        scores.append((node_id, score, attrs))
    
    # 4. Return top_k sorted by descending similarity
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve top Publications for a given query from the graph."
    )
    parser.add_argument(
        "query", type=str, help="Natural-language query string"
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="rag/graph/graph_publications.gexf",
        help="Path to your saved publications graph (GEXF)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Ollama embedding model name"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="How many top publications to return"
    )
    args = parser.parse_args()
    
    results = retrieve(
        query=args.query,
        graph_path=args.graph,
        model_name=args.model,
        top_k=args.top_k
    )
    
    print(f"\nTop {len(results)} publications matching “{args.query}”:\n")
    for rank, (pub_id, score, meta) in enumerate(results, start=1):
        title = meta.get("title", "<no title>")
        year  = meta.get("year", "<no year>")
        print(f"{rank}. [{pub_id}] \"{title}\" ({year}) — score: {score:.4f}")

if __name__ == "__main__":
    main()
