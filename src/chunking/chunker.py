import uuid
import logging
from typing import List, Dict

import spacy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load spaCy model with sentencizer for sentence boundary detection
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
    nlp.add_pipe("sentencizer")
except OSError:
    # If model not found, fall back to blank English with sentencizer
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

class Chunker:
    """
    Chunker splits page texts into paragraph-sized chunks suitable for RAG nodes.
    - Splits on blank lines for paragraphs
    - Enforces a max token count per chunk
    - Splits oversized paragraphs on sentence boundaries
    """

    def __init__(self, max_tokens: int = 500):
        """
        :param max_tokens: Approximate maximum number of tokens (words) per chunk
        """
        self.max_tokens = max_tokens

    def _token_count(self, text: str) -> int:
        """Approximate token count by splitting on whitespace"""
        return len(text.split())

    def chunk_pages(self, pages: List[str]) -> List[Dict]:
        """
        Splits each page's text into chunks with metadata.

        :param pages: List of page text strings
        :return: List of chunk dicts: {
            "node_id": str,
            "page": int,
            "paragraph_index": int,
            "chunk_index": int,
            "text": str
        }
        """
        all_chunks: List[Dict] = []
        for page_num, page_text in enumerate(pages, start=1):
            # Split into paragraphs by two or more newlines
            paras = [p.strip() for p in page_text.split("\n\n") if p.strip()]
            logger.info(f"Page {page_num}: {len(paras)} paragraphs detected")
            for para_idx, para in enumerate(paras, start=1):
                tokens = self._token_count(para)
                if tokens <= self.max_tokens:
                    # fits in one chunk
                    chunk = {
                        "node_id": str(uuid.uuid4()),
                        "page": page_num,
                        "paragraph_index": para_idx,
                        "chunk_index": 1,
                        "text": para
                    }
                    all_chunks.append(chunk)
                else:
                    # split paragraph into sentences and form sub-chunks
                    doc = nlp(para)
                    sentences = [sent.text.strip() for sent in doc.sents]
                    logger.debug(f"Paragraph {para_idx} on page {page_num} has {len(sentences)} sentences")
                    chunk_text = ""
                    chunk_i = 1
                    for sent in sentences:
                        if self._token_count(chunk_text + " " + sent) <= self.max_tokens:
                            chunk_text = (chunk_text + " " + sent).strip()
                        else:
                            # emit previous chunk
                            all_chunks.append({
                                "node_id": str(uuid.uuid4()),
                                "page": page_num,
                                "paragraph_index": para_idx,
                                "chunk_index": chunk_i,
                                "text": chunk_text
                            })
                            chunk_i += 1
                            chunk_text = sent
                    # emit final chunk_text
                    if chunk_text:
                        all_chunks.append({
                            "node_id": str(uuid.uuid4()),
                            "page": page_num,
                            "paragraph_index": para_idx,
                            "chunk_index": chunk_i,
                            "text": chunk_text
                        })
        logger.info(f"Generated {len(all_chunks)} total chunks")
        return all_chunks

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Chunk PDF page texts into paragraph-sized nodes")
    parser.add_argument("--pages", type=str, required=True,
                        help="Path to JSON file containing list of page texts")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output JSON chunks file")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum tokens per chunk")
    args = parser.parse_args()

    # Load pages
    with open(args.pages, 'r') as f:
        pages = json.load(f)

    chunker = Chunker(max_tokens=args.max_tokens)
    chunks = chunker.chunk_pages(pages)

    # Save to output
    with open(args.output, 'w') as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {args.output}")
