## Microbiome RAG Chatbot Architecture

This document outlines the end-to-end system architecture for a Retrieval-Augmented Generation (RAG) chatbot specialized in microbiome research. It describes each component, its responsibilities, and how they interact.

---

### 1. High-Level Components

| Layer                  | Component            | Responsibility                                                                                                          |
| ---------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Data Ingestion**     | `PDFReader`          | Parse raw PDF publications (two-column support) and emit plain-text pages.                                              |
| **Preprocessing**      | `Chunker`            | Split page text into paragraph-sized chunks, enforce token limits, assign metadata (IDs, page, paragraph, chunk index). |
|                        | `MetadataExtractor`  | Detect and extract DOI, then fetch title, authors, and publication date via Crossref.                                   |
| **Embedding**          | `Embedder`           | Encode each text chunk into a fixed-size vector using SentenceTransformers (or OpenAI).                                 |
| **Graph Construction** | `GraphBuilder`       | Build a semantic graph where nodes are chunks and edges link similar nodes (cosine similarity).                         |
| **Persistence**        | `GraphStorage`       | Persist/retrieve the semantic graph to/from Neo4j; export/import via JSON or GraphML/GEXF.                              |
| **Retrieval**          | `Retriever`          | Embed user queries, retrieve top‑K matching chunks, and optionally expand via graph hops.                               |
| **API / RAG**          | `FastAPI` (`app.py`) | Expose `/query` endpoint, orchestrate retrieval, format prompts with citation tags, call LLM.                           |
| **LLM**                | OpenAI GPT‑4         | Generate answers using retrieved context and inline citations.                                                          |
| **Logging & Config**   | `logging.py`         | Centralized logging configuration across all modules.                                                                   |

---

### 2. Data Flow & Sequence

1. **PDF Ingestion**

   * User deposits PDF files into `data/raw_pdfs/`.
   * `PDFReader.read_pdf()` processes each file to extract a list of page strings.

2. **Text Chunking & Metadata Extraction**

   * `Chunker.chunk_pages()` splits page strings into paragraph chunks, assigns UUIDs and metadata.
   * `MetadataExtractor.extract_and_fetch()` scans first-page text for DOI, queries Crossref, and attaches metadata to chunks.

3. **Embedding & Graph Building**

   * `Embedder.embed_chunks()` encodes each chunk’s text into an embedding vector.
   * `GraphBuilder.build()` creates nodes with metadata and edges where cosine similarity ≥ threshold.

4. **Persistence**

   * `GraphStorage.write_graph()` writes the in-memory NetworkX graph to Neo4j (bolt://), enabling fast graph queries.
   * Optionally export to GraphML/GEXF for offline analysis.

5. **Query Handling**

   * On startup, `app.py` initializes `Retriever` by loading chunks JSON and graph from file/DB.
   * Client calls `POST /query` with a microbiome question.
   * `Retriever.retrieve()` embeds the query, selects top‑K chunk nodes, and expands via N‑hop neighbor traversal.

6. **RAG Generation**

   * `app.py` formats retrieved chunks into a context block with citation markers \[^1], \[^2], …
   * Calls OpenAI GPT‑4 with system and user prompts.
   * Returns generated answer and structured citation metadata via JSON.

---

### 3. Deployment & Operations

* **Containerization**: All components run in a Docker container defined by `Dockerfile`.
* **Environment Variables**:

  * `OPENAI_API_KEY` for LLM access.
  * `CHUNKS_PATH`, `GRAPH_PATH`, `GRAPH_FORMAT`, `EMBEDDING_MODEL` for retrieval configuration.
* **CI/CD**: Use GitHub Actions to lint, test (pytest), build Docker image, and deploy to staging/production.
* **Monitoring & Logging**:

  * Centralized logs via `logging.py` to console and optional file.
  * Neo4j metrics for graph size and query performance.

---

### 4. Scaling Considerations

* **PDF Parsing**: Parallelize ingestion over multiple worker processes.
* **Embedding**: Batch-encode with GPU support; consider caching embeddings for unchanged chunks.
* **Graph DB**: Use Neo4j clustering for high availability and horizontal scaling.
* **API**: Leverage autoscaling for FastAPI/Uvicorn workers behind a load balancer.

---

### 5. Future Enhancements

* **Vector Store**: Integrate FAISS or Pinecone for approximate nearest-neighbor retrieval.
* **Multi-modal**: Add image-processing for figures and diagrams in publications.
* **Feedback Loop**: Collect user feedback to refine retrieval thresholds and LLM prompts.
* **Authentication**: Secure API with OAuth2 or API keys.
* **Streaming**: Implement streaming responses for long answers.

---

*Document version 1.0*
