import argparse
import os
import pdfplumber
import chromadb
from models.hf_client import HFClient

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_BATCH_SIZE = 32
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "ncci_codes"


def extract_pages(pdf_path: str) -> list[tuple[str, str]]:
    """Extract text page by page. Returns list of (page_id, page_text)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((f"{os.path.basename(pdf_path)}::p{i + 1}", text))
    return pages


def chunk_text(page_id: str, text: str) -> list[tuple[str, str]]:
    """Slide a 500-token window with 50-token overlap over text.
    Returns list of (chunk_id, chunk_text)."""
    words = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append((f"{page_id}::c{idx}", " ".join(words[start:end])))
        if end >= len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1
    return chunks


def ingest(pdf_path: str, chroma_path: str = CHROMA_PATH) -> None:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Source   : {os.path.abspath(pdf_path)}")

    pages = extract_pages(pdf_path)
    if not pages:
        print("No extractable text found in PDF.")
        return
    print(f"Pages    : {len(pages)}")

    all_chunks: list[tuple[str, str, str]] = []
    for page_id, page_text in pages:
        for chunk_id, chunk in chunk_text(page_id, page_text):
            all_chunks.append((chunk_id, chunk, page_id))
    print(f"Chunks   : {len(all_chunks)}  (size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP})")

    ids = [c[0] for c in all_chunks]
    texts = [c[1] for c in all_chunks]
    metadatas = [{"source": c[2]} for c in all_chunks]

    print(f"Embedding: 0/{len(texts)}", end="", flush=True)
    hf = HFClient()
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        embeddings.extend(hf.embed_text(batch))
        print(f"\rEmbedding: {min(i + EMBED_BATCH_SIZE, len(texts))}/{len(texts)}", end="", flush=True)
    print()

    db = chromadb.PersistentClient(path=chroma_path)
    collection = db.get_or_create_collection(COLLECTION_NAME)
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    print(f"\nDone.")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Vectors    : {len(embeddings)}")
    print(f"Stored at  : {os.path.abspath(chroma_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into the NCCI ChromaDB knowledge base."
    )
    parser.add_argument("--source", required=True, help="Path to the PDF to ingest")
    parser.add_argument("--chroma-path", default=CHROMA_PATH, help="ChromaDB directory (default: ./chroma_db)")
    args = parser.parse_args()
    ingest(args.source, args.chroma_path)
