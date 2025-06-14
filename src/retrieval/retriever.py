import json
import logging
from typing import List, Dict, Optional

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Retriever:
    """
    Retriever loads preprocessed chunks with embeddings and a semantic graph,
    then retrieves relevant chunks for a query based on embedding similarity
    and optional graph-based neighborhood expansion.
    """
    def __init__(
        self,
        chunks_path: str,
        graph_path: str,
        graph_format: str = 'graphml',
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        :param chunks_path: Path to JSON file with chunk dicts (must include 'node_id' and 'embedding')
        :param graph_path: Path to graph file (GraphML or GEXF)
        :param graph_format: 'graphml' or 'gexf'
        :param model_name: Sentence-Transformers model name for query embedding
        """
        logger.info(f"Loading chunks from {chunks_path}")
        with open(chunks_path, 'r') as f:
            self.chunks: List[Dict] = json.load(f)

        self.ids = [chunk['node_id'] for chunk in self.chunks]
        self.embeddings = np.array([chunk['embedding'] for chunk in self.chunks])
        # normalize embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.normed = self.embeddings / np.clip(norms, a_min=1e-8, a_max=None)

        logger.info(f"Loading graph from {graph_path} ({graph_format})")
        if graph_format == 'graphml':
            self.graph = nx.read_graphml(graph_path)
        elif graph_format == 'gexf':
            self.graph = nx.read_gexf(graph_path)
        else:
            raise ValueError(f"Unsupported graph format: {graph_format}")
        logger.info(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed the input query into the same vector space as chunks.
        """
        vec = self.model.encode([query], convert_to_numpy=True)
        # normalize
        norm = np.linalg.norm(vec)
        return (vec / max(norm, 1e-8)).squeeze()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 1
    ) -> List[Dict]:
        """
        Retrieve top_k chunks semantically closest to query and expand via graph.

        :param query: User query string
        :param top_k: Number of initial chunks to retrieve
        :param hops: Number of graph hops to expand context
        :return: List of retrieved chunk dicts with metadata
        """
        q_vec = self.embed_query(query)
        sims = np.dot(self.normed, q_vec)
        # get top_k indices
        best_idx = np.argsort(sims)[::-1][:top_k]
        selected_ids = [self.ids[i] for i in best_idx]
        logger.info(f"Top {top_k} nodes: {selected_ids}")

        # Expand via graph neighbors
        all_ids = set(selected_ids)
        if hops > 0:
            for node_id in selected_ids:
                lengths = nx.single_source_shortest_path_length(self.graph, node_id, cutoff=hops)
                all_ids.update(lengths.keys())
            logger.info(f"Expanded to {len(all_ids)} nodes with hops={hops}")

        # Collect chunk metadata
        id_to_chunk = {c['node_id']: c for c in self.chunks}
        results = [id_to_chunk[nid] for nid in all_ids if nid in id_to_chunk]
        # Sort results by semantic similarity descending
        results.sort(key=lambda c: sims[self.ids.index(c['node_id'])], reverse=True)
        return results

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Retrieve relevant chunks for a query')
    parser.add_argument('--chunks', type=str, required=True,
                        help='Path to JSON chunks file')
    parser.add_argument('--graph', type=str, required=True,
                        help='Path to graph file')
    parser.add_argument('--format', type=str, choices=['graphml','gexf'], default='graphml',
                        help='Graph file format')
    parser.add_argument('--query', type=str, required=True,
                        help='Query text')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of initial chunks to retrieve')
    parser.add_argument('--hops', type=int, default=1,
                        help='Number of graph hops for expansion')
    args = parser.parse_args()

    retriever = Retriever(
        chunks_path=args.chunks,
        graph_path=args.graph,
        graph_format=args.format
    )
    results = retriever.retrieve(args.query, top_k=args.top_k, hops=args.hops)
    # Output simplified JSON
    output = [
        {
            'node_id': c['node_id'],
            'page': c.get('page'),
            'paragraph_index': c.get('paragraph_index'),
            'chunk_index': c.get('chunk_index'),
            'text': c.get('text')
        } for c in results
    ]
    print(json.dumps(output, indent=2))
