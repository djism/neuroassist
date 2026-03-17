from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import EMBEDDING_MODEL


class LocalEmbeddings(Embeddings):
    """
    Free, local embedding model using sentence-transformers.
    BAAI/bge-small-en-v1.5 — production quality, runs on CPU, no API cost.
    LangChain compatible — works directly with ChromaDB.
    """

    def __init__(self):
        print(f"⏳ Loading embedding model: {EMBEDDING_MODEL}")
        print("   (First run downloads ~130MB — subsequent runs are instant)")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Embedding model loaded: {EMBEDDING_MODEL}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents — called during ingestion."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,   # normalizing improves retrieval quality
            show_progress_bar=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query — called at runtime for every user question."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )
        return embedding.tolist()


# Singleton — load model once, reuse everywhere
_embeddings_instance = None

def get_embeddings() -> LocalEmbeddings:
    """
    Returns a singleton embedding model instance.
    Model loads once on first call, reused on all subsequent calls.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = LocalEmbeddings()
    return _embeddings_instance


if __name__ == "__main__":
    # Quick test
    print("Testing embedding model...")
    embeddings = get_embeddings()

    test_texts = [
        "What does the Shrestha lab study?",
        "Fiber photometry measures calcium signals in neurons",
        "The SAA pipeline processes behavioral data"
    ]

    print("\n📐 Testing embed_documents...")
    doc_embeddings = embeddings.embed_documents(test_texts)
    print(f"   Embedded {len(doc_embeddings)} documents")
    print(f"   Embedding dimension: {len(doc_embeddings[0])}")

    print("\n🔍 Testing embed_query...")
    query_embedding = embeddings.embed_query("What is fiber photometry?")
    print(f"   Query embedding dimension: {len(query_embedding)}")

    print("\n✅ Embeddings working correctly!")