#KnightGPT
## Overview

This repository provides a **Graph-RAG** (Retrieval-Augmented Generation) pipeline tailored for **microbiome literature**. It ingests raw `.txt` or `.pdf` papers, uses an LLM to extract metadata (title, authors, year, abstract), builds a publication graph, and lets you query it via embeddings + graph‐based retrieval before answering with a local LLM (Ollama/LLaMA3).

---

## Repository Structure

```
.
├── README.md
├── rag
│   ├── main.py                   # CLI entrypoint for the Microbiome assistant
│   └── publications_raw          # Drop your .txt/.pdf papers here
└── rag/graph
    ├── __init__.py
    ├── build_publications.py     # Ingest + metadata‐LLM + embedding + graph build
    ├── build_paragraph_graph.py  # Build paragraph-level graph with similarity edges
    ├── retrieve_publications.py  # Seed retrieval over Publication nodes
    ├── store.py                  # Simple NetworkX wrapper for Publication graph
    ├── embeddings/               # Caches per‐publication embedding JSON files
    ├── publications_manifest.json# Metadata + embeddings index
    ├── paragraphs_manifest.json  # Paragraph nodes + metadata
    ├── graph_publications.gexf   # Publication-level graph
    └── graph_paragraphs.gexf     # Paragraph-level similarity graph
```

---

## Prerequisites

* **Python 3.10+**
* Ollama CLI & Python bindings configured with a local LLaMA3 model
* `pip install -r requirements.txt` dependencies:

  ```txt
  networkx
  pdfplumber     # preferred PDF text extractor
  ollama
  numpy
  pymupdf        # optional fallback if pdfplumber isn't available
  ```

---

## Installation

1. **Clone** the repo and enter it:

   ```bash
   git clone https://github.com/drahmanucsd/knightGPT.git
   cd knightGPT
   ```

2. **Install** Python dependencies:

   ```bash
   pip install networkx pdfplumber ollama numpy
   # optionally: pip install pymupdf  # used if pdfplumber missing
   ```

3. **Verify** your Ollama setup:

   ```bash
   ollama list
   # ensure llama3 (or llama2) is available locally
   ```

---

## Usage

### 1. Add Publications

Place your PDF or TXT files in:

```
rag/publications_raw/
```

Each file’s basename (e.g. `OMNICellTOSG.pdf`) becomes its publication ID.

### 2. Build the Publications Graph

Runs LLM metadata extraction, caches embeddings, and writes a GEXF graph.

```bash
python -m rag.graph.build_publications
```

* Outputs:

  * `rag/graph/publications_manifest.json`
  * `rag/graph/embeddings/<ID>.json`
  * `rag/graph/graph_publications.gexf`

### 2b. Build the Paragraph Graph

Generate paragraph-level nodes with similarity edges:

```bash
python -m rag.graph.build_paragraph_graph
```

* Outputs:

  * `rag/graph/paragraphs_manifest.json`
  * `rag/embeddings/paragraphs/`
  * `rag/graph/graph_paragraphs.gexf`

### 3. Test Retrieval

Query the publication graph for top-k relevant papers:

```bash
python -m rag.graph.retrieve_publications "gut microbiome butyrate" --top_k 3
```

You’ll see a ranked list of `[ID] Title (Year) — score: …`

### 4. Run the Microbiome Assistant

Launch the interactive CLI:

```bash
python -m rag.main
```

* **Prompt**: enter your question (e.g. “Which microbes produce butyrate in the human gut?”)
* **Output**: the model cites only publication IDs and returns a concise, context-grounded answer.

---

## Configuration

* **Embedding model**: change `--model bge-base-en-v1.5` to any supported Ollama embedder
* **LLM model**: edit `chat(model="llama3",…)` in both `build_publications.py` and `main.py`
* **Top-k**: adjust `--top_k` on both retrieval and `main.py` invocation

---

## Troubleshooting

* **“CropBox missing” warnings**: harmless; suppress with:

  ```python
  import warnings
  warnings.filterwarnings("ignore", message="CropBox missing")
  ```

* **PDF text jumbles**: consider switching to PyMuPDF:

  ```python
  import fitz
  def extract_text(fp):
      doc = fitz.open(str(fp))
      return "\n\n".join(page.get_text("text") for page in doc)
  ```

* **Metadata parse failures**: check `[LLM meta]` debug prints for malformed JSON. Tighten the prompt or increase snippet length.

* **Empty retrievals**: ensure embeddings ran successfully (`rag/graph/embeddings/…`) and that your query shares vocabulary with abstracts.

---

## Extending

* **Citation edges**: add CrossRef lookups to create `cites` edges in `build_publications.py`.
* **Entity nodes**: integrate NER (species, genes, pathways) and `mentions` edges for multi-hop queries.
* **Full-text chunking**: split PDFs into passages and attach passage nodes for finer-grained retrieval.
* **Persistent backend**: swap `networkx` + GEXF for Neo4j, RedisGraph + RedisVector, or Weaviate for scale.

---

## License

This project is released under the MIT License. Feel free to copy, modify, and redistribute!
