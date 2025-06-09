#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from ollama import embeddings
from rag.graph.store import GraphStore

MANIFEST_PATH = Path(__file__).parent / "publications_manifest.json"

def load_manifest_embeddings(manifest_path: Path):
    """Returns a dict mapping pub_id → embedding (list of floats)."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return { pub["id"]: pub["embedding"] for pub in data }

def unpack_embedding(resp) -> list[float]:
    """
    Given an Ollama EmbeddingsResponse (or dict/list), extract the raw vector.
    """
    # 1) Pydantic-like model with .dict()
    if hasattr(resp, "dict"):
        d = resp.dict()
        for v in d.values():
            if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                return v
        raise ValueError("No numeric list found in EmbeddingsResponse.dict()")
    # 2) dict with "embeddings" key
    if isinstance(resp, dict) and "embeddings" in resp:
        return resp["embeddings"]
    # 3) plain list
    if isinstance(resp, list):
        return resp
    # 4) fallback to list(resp)
    return list(resp)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve(
    query: str,
    graph_path: str,
    model_name: str = "bge-base-en-v1.5",
    top_k: int = 5
):
    # 1) Load the graph
    gs = GraphStore()
    gs.load(graph_path)

    # 2) Load node embeddings from manifest
    id2emb = load_manifest_embeddings(MANIFEST_PATH)

    # 3) Embed the query and unpack
    resp_q = embeddings(model=model_name, prompt=query)
    q_vec = np.array(unpack_embedding(resp_q))

    # 4) Score each Publication node
    results = []
    for node_id, attrs in gs.G.nodes(data=True):
        if attrs.get("type") != "Publication":
            continue
        emb_list = id2emb.get(node_id)
        if not emb_list:
            continue
        score = cosine_sim(q_vec, np.array(emb_list))
        results.append((node_id, score, attrs))

    # 5) Sort & return top_k
    results.sort(key=lambda x: -x[1])
    return results[:top_k]

def main():
    p = argparse.ArgumentParser(
        description="Retrieve top Publications for a given query from the graph."
    )
    p.add_argument("query", help="Natural-language query string")
    p.add_argument(
        "--graph",
        default="rag/graph/graph_publications.gexf",
        help="Path to your saved publications graph (GEXF)"
    )
    p.add_argument(
        "--model",
        default="bge-base-en-v1.5",
        help="Ollama embedding model name"
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="How many top publications to return"
    )
    args = p.parse_args()

    hits = retrieve(
        query=args.query,
        graph_path=args.graph,
        model_name=args.model,
        top_k=args.top_k
    )

    if not hits:
        print("No matching publications found.")
        return

    print(f"\nTop {len(hits)} publications for “{args.query}”:\n")
    for i, (pub_id, score, meta) in enumerate(hits, start=1):
        title = meta.get("title", "<no title>")
        year  = meta.get("year", "<no year>")
        print(f"{i}. [{pub_id}] “{title}” ({year}) — score: {score:.4f}")

if __name__ == "__main__":
    main()
