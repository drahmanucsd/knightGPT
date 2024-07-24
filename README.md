# knightGPT

knightGPT is a Retrieval Augmented Generation (RAG) system that integrates with Ollama for model support. This project enhances traditional Language Models (LLMs) by incorporating text-based retrieval for more contextual and relevant responses.

## Features

- **Retrieval Augmented Generation (RAG)**: Utilizes embeddings to find relevant context for generating responses.
- **Ollama Integration**: Seamlessly integrates with Ollama for efficient local model deployment and inference.
- **Flexible Embedding Models**: Currently supports 'nomic-embed-text' and 'bge-base-en-v1.5' for generating embeddings.
- **Customizable System Prompt**: Easily adaptable for various domains and use cases.
- **Efficient Embedding Management**: Saves and loads embeddings to/from JSON files for faster subsequent runs.

## Prerequisites

- Python 3.8+
- Git
- [Ollama](https://ollama.ai/) installed and configured

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/knightGPT.git
   cd knightGPT
   ```

2. Create and activate a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install ollama numpy
   ```

## Usage

1. Ensure Ollama is running and the desired models are available (llama3, bge-base-en-v1.5, and optionally nomic-embed-text and mistral).

2. Place your text file (e.g., "peter-pan.txt") in the project directory.

3. Run the main application:

   ```
   python main.py
   ```

4. Enter your question when prompted.

## Configuration

You can customize knightGPT's behavior by modifying the following variables in `main.py`:

- `SYSTEM_PROMPT`: Adjust the system prompt to change the behavior of the assistant.
- Embedding model: Change `bge-base-en-v1.5` to `nomic-embed-text` or another compatible model.
- LLM model: Change `llama3` to `mistral` or another Ollama-supported model.
- Number of similar chunks: Modify the `[:5]` in `find_most_similar` to retrieve more or fewer chunks.

## How It Works

1. The script parses a text file into paragraphs.
2. It generates embeddings for each paragraph using the specified embedding model.
3. When a question is asked, it finds the most similar paragraphs using cosine similarity.
4. The relevant paragraphs are used as context for the LLM to generate an answer.

## Future Improvements

- Implement graph-based RAG for more sophisticated retrieval.
- Add support for multiple document types.
- Improve embedding caching mechanism.
- Implement a more user-friendly interface.

## Acknowledgments

- The Ollama team for their excellent model serving framework
- Creators of the embedding models and LLMs used in this project

---

Happy querying with knightGPT! 🚀📚
