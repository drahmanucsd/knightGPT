# Microbiome RAG Chatbot

This repository implements a Retrieval-Augmented Generation (RAG) chatbot specialized in microbiome research. It extracts text and metadata from scientific PDFs, builds a semantic paragraph-level knowledge graph, and enables interactive querying via Ollama's llama3.

---

## Repository Structure

```
microbiome-rag-bot/
├── data/
│   ├── raw_pdfs/             # Source PDF files
│   ├── processed_pages.json   # Output from pdf_reader.py (metadata + pages)
│   ├── chunks.json            # Paragraph chunks (output of chunker.py)
│   ├── chunks_with_emb.json   # Chunks enriched with embeddings
│   └── graph.graphml          # Semantic graph in GraphML format
│
├── src/
│   ├── ingestion/
│   │   └── pdf_reader.py      # PDF → pages + metadata
│   │
│   ├── chunking/
│   │   └── chunker.py         # Pages → paragraph chunks
│   │
│   ├── embedding/
│   │   └── embedder.py        # Chunks → vector embeddings
│   │
│   ├── graph/
│   │   └── builder.py         # Build semantic graph via cosine similarity
│   │
│   ├── storage/
│   │   └── storage.py         # (Optional) Persist graph to Neo4j
│   │
│   ├── retrieval/
│   │   └── retriever.py       # Queryable RAG retriever
│   │
│   ├── utils/
│   │   └── logging.py         # Centralized logging configuration
│   │
│   └── cli.py                 # Terminal-based chatbot CLI
│
├── architecture.md            # System architecture overview
├── requirements.txt           # Python dependencies
└── README.md                  # (This file)
```

---

## Quick Start

### 1. Setup

1. **Clone and navigate**

   ```bash
   git clone <repo-url>
   cd microbiome-rag-bot
   ```

2. **Python environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Ollama & llama3**

   ```bash
   # macOS (or see ollama.com for other platforms)
   brew install ollama
   ollama pull llama3
   ```

---

### 2. Build the Knowledge Graph

1. **Extract pages + metadata**

   ```bash
   python src/ingestion/pdf_reader.py \
     data/raw_pdfs/your_paper.pdf \
     -o data/processed_pages.json
   ```

   Output: `data/processed_pages.json` with `metadata` (title, authors, DOI, date) and `pages`.

2. **Chunk paragraphs**

   ```bash
   python src/chunking/chunker.py \
     --pages data/processed_pages.json \
     --output data/chunks.json \
     --max_tokens 500
   ```

   Output: `data/chunks.json` with paragraph-level chunks.

3. **Embed chunks**

   ```bash
   python src/embedding/embedder.py \
     --input data/chunks.json \
     --output data/chunks_with_emb.json \
     --model all-MiniLM-L6-v2
   ```

4. **Build semantic graph**

   ```bash
   python src/graph/builder.py \
     --input data/chunks_with_emb.json \
     --output data/graph.graphml \
     --threshold 0.7
   ```

5. *(Optional)* **Load into Neo4j**

   ```bash
   python src/storage/storage.py \
     --action write \
     --input data/chunks_with_emb.json \
     --uri bolt://localhost:7687 \
     --user neo4j --password <pw>
   ```

---

### 3. Run the Chatbot (CLI)

Launch the terminal-based interface:

```bash
python src/cli.py --top_k 5 --hops 1
```

* **`top_k`**: initial number of chunks to retrieve.
* **`hops`**: number of graph hops for context expansion.

Type your question at the prompt. Citations will display with Title & DOI.

---

## Troubleshooting

* **ModuleNotFoundError**: Ensure you run Python from the repo root (or adjust `PYTHONPATH`), e.g.:

  ```bash
  python -m src.cli
  ```

* **Spacing issues**: The pipeline uses `pdfminer.six` exclusively for text extraction, splitting on form-feed (`\f`). Ensure no remnants of two-column code remain.

* **Ollama errors**: Verify `ollama` daemon is running and `llama3` is pulled:

  ```bash
  ollama list
  ```

---

## Future Enhancements

* FastAPI / HTTP endpoint
* FAISS or Pinecone integration
* Figure / table parsing
* Streaming answers
* API key protection

---

*Version 1.0*
