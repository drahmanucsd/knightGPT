#!/usr/bin/env python3
import json
from pathlib import Path

import pdfplumber
from ollama import embeddings, chat
from rag.graph.store import GraphStore

# —————————————————————————————————————————————
# Paths (all relative to this file)
BASE_DIR  = Path(__file__).parent
RAW_DIR   = BASE_DIR.parent / "publications_raw"
EMB_DIR   = BASE_DIR.parent / "embeddings" / "publications"
MANIFEST  = BASE_DIR / "publications_manifest.json"
GRAPH_OUT = BASE_DIR / "graph_publications.gexf"

# —————————————————————————————————————————————
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                pages.append(txt)
    return "\n".join(pages)

def infer_metadata_via_llm(text_snippet: str) -> dict:
    """
    Use the LLM to extract title, authors, year, and abstract from the snippet.
    Returns a dict with keys: title, authors, year, abstract.
    """
    SYSTEM = (
        "You are a metadata-extraction assistant.\n"
        "Given the start of a scientific paper (front matter + abstract),\n"
        "output a JSON object with exactly these fields:\n"
        "  title (string),\n"
        "  authors (array of strings),\n"
        "  year (integer),\n"
        "  abstract (string)\n"
        "If any field is missing, set it to null or an empty list."
    )
    # truncate so we stay within the model window
    snippet = text_snippet[:5000]
    msgs = [
        {"role":"system","content":SYSTEM},
        {"role":"user",  "content":snippet}
    ]
    resp = chat(model="llama3", messages=msgs)
    try:
        return json.loads(resp["choices"][0]["message"]["content"])
    except Exception:
        return {"title": None, "authors": [], "year": None, "abstract": None}

def parse_publications(raw_dir: Path) -> list[dict]:
    """
    Scan RAW_DIR for .txt and .pdf files.
    Try front-matter parsing; if absent, fall back to LLM-inferred metadata.
    """
    pubs = []
    for fp in sorted(raw_dir.iterdir()):
        if fp.suffix.lower() == ".txt":
            raw = fp.read_text(encoding="utf-8")
        elif fp.suffix.lower() == ".pdf":
            raw = extract_text_from_pdf(fp)
        else:
            continue

        lines = raw.splitlines()
        title, authors, year = None, None, None
        abstract_lines = []
        meta_done = False

        # 1) manual front-matter
        for line in lines:
            if not meta_done and line.startswith("Title:"):
                title = line.split(":",1)[1].strip()
            elif not meta_done and line.startswith("Authors:"):
                authors = [a.strip() for a in line.split(":",1)[1].split(",") if a.strip()]
            elif not meta_done and line.startswith("Year:"):
                try:
                    year = int(line.split(":",1)[1].strip())
                except ValueError:
                    year = None
            elif not meta_done and line.strip()=="" and (title or authors or year):
                meta_done = True
            elif meta_done:
                abstract_lines.append(line)

        # 2) fallback if no manual abstract
        if not abstract_lines:
            inferred = infer_metadata_via_llm(raw)
            title    = inferred.get("title")   or fp.stem
            authors  = inferred.get("authors") or []
            year     = inferred.get("year")
            abstract = inferred.get("abstract") or raw.strip()
        else:
            abstract = "\n".join(abstract_lines).strip()
            title    = title    or fp.stem
            authors  = authors  or []
            # year stays as parsed (or None)

        # 3) skip if we end up with no abstract at all
        if not abstract:
            print(f"[warn] Skipping {fp.name}: empty text")
            continue

        pubs.append({
            "id":       fp.stem,
            "title":    title,
            "authors":  authors,
            "year":     year,
            "abstract": abstract
        })

    return pubs

def ensure_embeddings(pubs: list[dict], emb_dir: Path, model_name: str="llama3"):
    """Compute or load cached embeddings for each publication abstract."""
    emb_dir.mkdir(parents=True, exist_ok=True)

    for pub in pubs:
        ef = emb_dir / f"{pub['id']}.json"
        if not ef.exists():
            print(f"[embed] {pub['id']} → {model_name}")
            resp = embeddings(model=model_name, prompt=pub["abstract"])

            # --- unpack the EmbeddingsResponse into a simple list ---
            # 1) if it's a pydantic model with .dict()
            if hasattr(resp, "dict"):
                d = resp.dict()
                # look for the first list-of-floats field
                for v in d.values():
                    if isinstance(v, list) and all(isinstance(x, (float,int)) for x in v):
                        vec = v
                        break
                else:
                    raise ValueError("No vector field found in EmbeddingsResponse.dict()")
            # 2) if it’s a dict already
            elif isinstance(resp, dict) and "embeddings" in resp:
                vec = resp["embeddings"]
            # 3) if it’s already a list
            elif isinstance(resp, list):
                vec = resp
            else:
                # last resort: try to treat it as an iterable
                vec = list(resp)

            # now vec is JSON-serializable
            ef.write_text(json.dumps(vec), encoding="utf-8")

        # load back in
        pub["embedding"] = json.loads(ef.read_text(encoding="utf-8"))


# —————————————————————————————————————————————
if __name__ == "__main__":
    # 1. load or parse
    if MANIFEST.exists():
        print("[load] existing manifest")
        publications = json.loads(MANIFEST.read_text(encoding="utf-8"))
    else:
        print("[parse] raw files (.txt + .pdf)")
        publications = parse_publications(RAW_DIR)

    # 2. embeddings
    ensure_embeddings(publications, EMB_DIR)

    # 3. save manifest
    MANIFEST.write_text(json.dumps(publications, indent=2), encoding="utf-8")
    print(f"[save] manifest → {MANIFEST}")

    # 4. build graph
    print("[build] GraphStore with Publication nodes")
    gs = GraphStore()
    for pub in publications:
        # 1) Replace None with GEXF‐friendly defaults
        title     = pub["title"]    or ""
        authors   = pub["authors"]  or []     # lists are allowed but you can json.dumps(authors) if you want a string
        year      = pub["year"]      if pub["year"] is not None else 0
        abstract  = pub["abstract"]  or ""
        embedding = pub["embedding"] or []     # vector of floats is OK, but you could also json.dumps(embedding)

        gs.add_publication(
            pub_id    = pub["id"],
            title     = title,
            authors   = authors,
            year      = year,
            abstract  = abstract,
        )
    gs.save(str(GRAPH_OUT))

    print(f"[save] graph → {GRAPH_OUT}")
