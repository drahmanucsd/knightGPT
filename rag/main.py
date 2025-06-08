#!/usr/bin/env python3
import argparse
from pathlib import Path

from ollama import chat, embeddings
from rag.graph.retrieve_publications import retrieve

# —————————————————————————————————————————————
SYSTEM_PROMPT = """
You are a microbiome literature assistant.  
Answer ONLY using the abstracts provided below.  
Cite Publication IDs in your answer.
"""

def assemble_context(pub_results, max_tokens=2000):
    pieces = []
    for pub_id, score, attrs in pub_results:
        header = f"[{pub_id}] {attrs.get('title')}\n"
        pieces.append(header + attrs.get("abstract", "") + "\n\n")
    return "".join(pieces)

def main():
    p = argparse.ArgumentParser(description="Graph-RAG interactive loop")
    p.add_argument(
        "--graph",
        type=Path,
        default=Path(__file__).parent / "graph" / "graph_publications.gexf",
        help="Path to the publications graph"
    )
    p.add_argument(
        "--embed_model",
        type=str,
        default="llama3",
        help="Ollama embedding model for retrieval"
    )
    p.add_argument(
        "--chat_model",
        type=str,
        default="llama3",
        help="Ollama chat model for answering"
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="How many publications to retrieve per query"
    )
    args = p.parse_args()

    print("🤖 Microbiome assistant ready. Type your question (or 'exit' to quit):")
    while True:
        query = input("> ").strip()
        if not query or query.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        # 1) Retrieve top-K Publication nodes
        pubs = retrieve(
            query=query,
            graph_path=str(args.graph),
            model_name=args.embed_model,
            top_k=args.top_k
        )

        print(pubs)
        # 2) Assemble context
        context = assemble_context(pubs)

        # 3) Call Ollama
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context},
            {"role": "user",   "content": query}
        ]
        response = chat(model=args.chat_model, messages=messages)

        # 4) Extract & print
        try:
            answer = response.message.content
        except Exception:
            print("⚠️  Couldn't parse response; raw:", response)
            continue

        print("\n" + answer + "\n")

if __name__ == "__main__":
    main()
