#!/usr/bin/env python3
"""Build a paragraph-level graph for retrieval-augmented generation."""

import json
import re
from pathlib import Path
from typing import List

import numpy as np
from ollama import chat, embeddings

from .store import GraphStore

# ---------------------------------------------------------------
# Paths
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR.parent / "publications_raw"
EMB_DIR = BASE_DIR.parent / "embeddings" / "paragraphs"
GRAPH_OUT = BASE_DIR / "graph_paragraphs.gexf"
MANIFEST = BASE_DIR / "paragraphs_manifest.json"

# ---------------------------------------------------------------
# Helper functions

def extract_text(fp: Path) -> str:
    """Return raw text from a .txt or .pdf file."""
    if fp.suffix.lower() == ".pdf":
        # Prefer pdfplumber, fallback to PyMuPDF if available
        try:
            import pdfplumber
        except Exception:
            try:
                import fitz
            except Exception as exc:  # pragma: no cover - environment-dependent
                raise RuntimeError("No PDF parser available") from exc
            doc = fitz.open(str(fp))
            return "\n\n".join(page.get_text("text") for page in doc)
        with pdfplumber.open(str(fp)) as pdf:
            return "\n\n".join((p.extract_text() or "") for p in pdf.pages)
    return fp.read_text(encoding="utf-8")


def infer_metadata_via_llm(text: str) -> dict:
    """Infer title, authors, and doi using an LLM."""
    SYSTEM = """
You are a metadata extraction assistant.
Given the beginning of an academic paper, return ONLY a JSON object with:
  title (string),
  authors (array of strings),
  doi (string or null)
"""
    snippet = text[:4000]
    resp = chat(
        model="llama3",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": snippet},
        ],
    )
    if isinstance(resp, dict):
        content = resp.get("text") or resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        content = str(resp)
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"title": None, "authors": [], "doi": None}


def chunk_paragraphs(text: str, max_tokens: int = 200) -> List[str]:
    """Split text into roughly token-limited paragraphs."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    for p in paras:
        words = p.split()
        while len(words) > max_tokens:
            chunks.append(" ".join(words[:max_tokens]))
            words = words[max_tokens:]
        if words:
            chunks.append(" ".join(words))
    return chunks


def ensure_embeddings(texts: List[str], emb_dir: Path, model_name: str = "bge-base-en-v1.5") -> List[List[float]]:
    emb_dir.mkdir(parents=True, exist_ok=True)
    vectors = []
    for i, t in enumerate(texts):
        fp = emb_dir / f"{i}.json"
        if fp.exists():
            vectors.append(json.loads(fp.read_text()))
            continue
        resp = embeddings(model=model_name, prompt=t)
        if hasattr(resp, "model_dump"):
            vec = next(v for v in resp.model_dump().values() if isinstance(v, list))
        elif hasattr(resp, "dict"):
            vec = next(v for v in resp.dict().values() if isinstance(v, list))
        elif isinstance(resp, dict) and "embeddings" in resp:
            vec = resp["embeddings"]
        else:
            vec = list(resp)
        fp.write_text(json.dumps(vec), encoding="utf-8")
        vectors.append(vec)
    return vectors


# ---------------------------------------------------------------
if __name__ == "__main__":
    paragraphs = []
    meta_lookup = {}
    for fp in sorted(RAW_DIR.iterdir()):
        if fp.suffix.lower() not in (".txt", ".pdf"):
            continue
        raw = extract_text(fp)
        if not raw.strip():
            continue
        meta = infer_metadata_via_llm(raw)
        meta_lookup[fp.stem] = meta
        paras = chunk_paragraphs(raw)
        for idx, text in enumerate(paras):
            paragraphs.append({
                "id": f"{fp.stem}_p{idx}",
                "text": text,
                "file_id": fp.stem,
            })

    texts = [p["text"] for p in paragraphs]
    vectors = ensure_embeddings(texts, EMB_DIR)
    for vec, p in zip(vectors, paragraphs):
        p["embedding"] = vec

    MANIFEST.write_text(json.dumps({"paragraphs": paragraphs, "meta": meta_lookup}, indent=2))

    gs = GraphStore()
    for p in paragraphs:
        meta = meta_lookup.get(p["file_id"], {})
        gs.G.add_node(
            p["id"],
            type="Paragraph",
            text=p["text"],
            title=meta.get("title") or p["file_id"],
            authors=json.dumps(meta.get("authors", [])),
            doi=meta.get("doi") or "",
        )

    # add similarity edges (top 3 per node)
    vectors_np = [np.array(v) for v in vectors]
    for i, (pid, vec_i) in enumerate(zip([p["id"] for p in paragraphs], vectors_np)):
        sims = []
        for j, vec_j in enumerate(vectors_np):
            if i == j:
                continue
            score = float(np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j)))
            sims.append((score, paragraphs[j]["id"]))
        sims.sort(key=lambda x: -x[0])
        for score, nid in sims[:3]:
            gs.add_edge(pid, nid, weight=score)

    gs.save(str(GRAPH_OUT))
    print(f"[save] graph -> {GRAPH_OUT}")
