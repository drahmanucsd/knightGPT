#!/usr/bin/env python3
import json
from pathlib import Path

from ollama import chat, embeddings
from rag.graph.store import GraphStore

# —————————————————————————————————————————————
# Paths (relative to this file)
BASE_DIR  = Path(__file__).parent
RAW_DIR   = BASE_DIR.parent / "publications_raw"
EMB_DIR   = BASE_DIR.parent / "embeddings" / "publications"
MANIFEST  = BASE_DIR / "publications_manifest.json"
GRAPH_OUT = BASE_DIR / "graph_publications.gexf"

import warnings

def extract_text(fp: Path) -> str:
    if fp.suffix.lower() == ".pdf":
        # Default to pdfplumber, fallback to PyMuPDF if available
        try:
            import pdfplumber
        except Exception:
            try:
                import fitz
            except Exception as exc:  # pragma: no cover - env dependent
                raise RuntimeError("No PDF parser available") from exc
            doc = fitz.open(str(fp))
            return "\n\n".join(page.get_text("text") for page in doc)
        with pdfplumber.open(str(fp)) as pdf:
            return "\n\n".join((p.extract_text() or "") for p in pdf.pages)
    return fp.read_text(encoding="utf-8")


import re
import json
from ollama import chat

def infer_metadata_via_llm(text: str) -> dict:
    """
    Ask the LLM (llama3) to extract title, authors, year, abstract.
    Then pull out the first JSON object by slicing from the first '{'
    to the last '}' in the response.
    """
    SYSTEM = """
You are a metadata-extraction assistant.
Given the start of a scientific paper or document, output *only* a JSON object with:
  title (string),
  authors (array of strings),
  year (integer),
  abstract (string)
If a field is missing, set it to null or an empty list.
"""
    snippet = text[:5000]
    resp = chat(
        model="llama3",
        messages=[
            {"role":"system","content":SYSTEM},
            {"role":"user",  "content":snippet}
        ]
    )

    # Pull out the assistant’s reply text
    if isinstance(resp, dict):
        if "choices" in resp:
            content = resp["choices"][0]["message"]["content"]
        elif "message" in resp and isinstance(resp["message"], dict):
            content = resp["message"].get("content","")
        else:
            content = resp.get("text","")
    else:
        content = str(resp)

    # Debug print
    print(f"[LLM meta] {content[:200].replace('\\n',' ')}…")

    # Extract JSON by slicing from first '{' to last '}'
    start = content.find("{")
    end   = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = content[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("[warn] JSON parse failed, got:", json_str[:200])
    else:
        print("[warn] No JSON object found in LLM response")

    # Fallback
    return {"title": None, "authors": [], "year": None, "abstract": None}



def parse_publications(raw_dir: Path) -> list[dict]:
    """
    Load each .txt/.pdf, infer metadata via LLM, fallback abstract=1st paragraph.
    """
    pubs = []
    for fp in sorted(raw_dir.iterdir()):
        if fp.suffix.lower() not in (".txt", ".pdf"):
            continue

        raw = extract_text(fp)
        if not raw.strip():
            print(f"[warn] {fp.name} empty, skipping")
            continue

        # 1) LLM inference
        meta = infer_metadata_via_llm(raw)
        title    = meta.get("title")    or fp.stem
        authors  = meta.get("authors")  or []
        year     = meta.get("year")
        abstract = meta.get("abstract") or ""

        # 2) fallback on paragraph
        if not abstract.strip():
            paras = [
                p.strip() for p in raw.replace("\r\n","\n")\
                                       .split("\n\n") if p.strip()
            ]
            if len(paras) > 1:
                abstract = paras[1]
            else:
                abstract = paras[0]

        print(f"[parsed] {fp.name}  title={title!r}  authors={authors}  "
              f"year={year}  abstract_len={len(abstract)}")
        pubs.append({
            "id":       fp.stem,
            "title":    title,
            "authors":  authors,
            "year":     year,
            "abstract": abstract
        })

    return pubs

def ensure_embeddings(pubs: list[dict], emb_dir: Path, model_name: str="llama3"):
    """
    Compute or load cached embeddings for each publication’s abstract.
    EmbeddingsResponse → list[float].
    """
    emb_dir.mkdir(parents=True, exist_ok=True)
    for pub in pubs:
        ef = emb_dir / f"{pub['id']}.json"
        if not ef.exists():
            print(f"[embed] {pub['id']} → {model_name}")
            resp = embeddings(model=model_name, prompt=pub["abstract"])
            # unpack vector
            if hasattr(resp, "model_dump"):
                d = resp.model_dump()
                vec = next(
                    v
                    for v in d.values()
                    if isinstance(v, list)
                    and all(isinstance(x, (int, float)) for x in v)
                )
            elif hasattr(resp, "dict"):
                d = resp.dict()
                vec = next(v for v in d.values()
                           if isinstance(v, list) and all(isinstance(x,(int,float)) for x in v))
            elif isinstance(resp, dict) and "embeddings" in resp:
                vec = resp["embeddings"]
            elif isinstance(resp, list):
                vec = resp
            else:
                vec = list(resp)
            ef.write_text(json.dumps(vec), encoding="utf-8")
        pub["embedding"] = json.loads(ef.read_text(encoding="utf-8"))

# —————————————————————————————————————————————
if __name__ == "__main__":
    print("[parse] raw files → metadata via LLM")
    publications = parse_publications(RAW_DIR)

    print("[embed] generate or load embeddings")
    ensure_embeddings(publications, EMB_DIR)

    print("[save] manifest")
    MANIFEST.write_text(json.dumps(publications, indent=2), encoding="utf-8")

    print("[build] graph nodes")
    gs = GraphStore()
    for pub in publications:
        gs.add_publication(
            pub_id   = pub["id"],
            title    = pub["title"],
            authors  = pub["authors"],
            year     = pub["year"],
            abstract = pub["abstract"]
        )
    gs.save(str(GRAPH_OUT))
    print(f"[save] graph → {GRAPH_OUT}")
