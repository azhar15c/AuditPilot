import os
import chromadb
from models.hf_client import HFClient

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "ncci_codes"


class NCCIRetriever:
    def __init__(self, chroma_path: str = CHROMA_PATH) -> None:
        self._hf = HFClient()
        db = chromadb.PersistentClient(path=chroma_path)
        self._collection = db.get_or_create_collection(COLLECTION_NAME)

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Embed query_text and return top_k nearest chunks.
        Each result: {chunk_text, distance, metadata}."""
        query_embedding = self._hf.embed_text([query_text])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        return [
            {
                "chunk_text": doc,
                "distance": round(float(dist), 4),
                "metadata": meta or {},
            }
            for doc, dist, meta in zip(documents, distances, metadatas)
        ]

    def format_context(self, results: list[dict]) -> str:
        """Format retrieved chunks into a string ready to inject into an LLM prompt."""
        if not results:
            return "No relevant NCCI reference material found."
        sections = []
        for i, r in enumerate(results, start=1):
            source = r["metadata"].get("source", "unknown")
            sections.append(
                f"[Ref {i} | {source} | distance: {r['distance']}]\n{r['chunk_text']}"
            )
        return "\n\n---\n\n".join(sections)
