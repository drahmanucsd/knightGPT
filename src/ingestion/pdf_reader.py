import logging
import json
import os
import re
from typing import List, Dict, Any

from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from ollama import chat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PDFReader:
    """
    PDFReader uses pdfminer.six for metadata and text extraction,
    including DOI detection via regex from the first page of text.
    Falls back to llama3 for missing metadata fields, using simplified response parsing.
    """

    DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)

    def __init__(self):
        pass

    def _decode_pdf_string(self, value: Any) -> Any:
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8', errors='ignore')
            except Exception:
                return value.decode('latin-1', errors='ignore')
        return value

    def extract_metadata(self, pdf_path: str, first_page_text: str) -> Dict[str, Any]:
        """
        Extracts metadata using pdfminer; falls back to llama3 for any missing fields.
        """
        metadata: Dict[str, Any] = {"title": None, "authors": [], "publication_date": None, "doi": None}
        # 1) PDF document info
        with open(pdf_path, 'rb') as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser)
            info = doc.info[0] if doc.info else {}

            # Title
            raw_title = info.get('Title')
            metadata['title'] = self._decode_pdf_string(raw_title)

            # Author(s)
            raw_author = info.get('Author')
            author_str = self._decode_pdf_string(raw_author)
            if author_str:
                metadata['authors'] = [a.strip() for a in re.split(r'[;,]', author_str) if a.strip()]

            # Publication Date
            raw_date = info.get('CreationDate') or info.get('ModDate')
            date_str = self._decode_pdf_string(raw_date)
            if date_str and date_str.startswith('D:'):
                parts = date_str[2:10]
                if len(parts) == 8 and parts.isdigit():
                    metadata['publication_date'] = f"{parts[0:4]}-{parts[4:6]}-{parts[6:8]}"

        # 2) DOI detection via regex
        doi_match = self.DOI_REGEX.search(first_page_text)
        if doi_match:
            metadata['doi'] = doi_match.group(0).rstrip('.')
            logger.info(f"Detected DOI via regex: {metadata['doi']}")

        # 3) Fallback: llama3 for missing fields
        missing = [key for key, val in metadata.items() if not val]
        if missing:
            logger.info(f"Falling back to llama3 for metadata fields: {missing}")
            prompt = (
                f"Extract only the following metadata fields: {', '.join(missing)}. "
                f"Return a JSON object. Text: {first_page_text[:2000]}"
            )
            try:
                resp = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
                content = getattr(resp, 'content', None) or getattr(resp, 'message', {}).get('content', '')
                # Extract JSON substring
                start = content.find('{')
                end = content.rfind('}')
                json_str = content[start:end+1] if start != -1 and end != -1 else content
                llm_meta = json.loads(json_str)
                for key in missing:
                    if key in llm_meta and llm_meta[key]:
                        metadata[key] = llm_meta[key]
            except Exception as e:
                logger.error(f"LLM metadata extraction failed: {e}")

        return metadata

    def read_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Reads a PDF file and returns metadata and page texts using pdfminer.six.
        Splits on form-feed characters to separate pages.
        """
        logger.info(f"Extracting text and metadata from PDF: {pdf_path}")
        full_text = extract_text(pdf_path)
        raw_pages = full_text.split("\f")
        pages = [p.strip() for p in raw_pages if p.strip()]
        logger.info(f"Extracted {len(pages)} pages from PDF")

        first_page_text = pages[0] if pages else ''
        metadata = self.extract_metadata(pdf_path, first_page_text)

        return {"metadata": metadata, "pages": pages}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract metadata (including DOI) and text from PDF into JSON'
    )
    parser.add_argument('pdf_path', type=str, help='Path to PDF file to read')
    parser.add_argument('-o', '--output_json', type=str, help='Path to write output JSON file')
    args = parser.parse_args()

    reader = PDFReader()
    data = reader.read_pdf(args.pdf_path)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote output JSON: {args.output_json}")
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))
