import chromadb
from langchain_community.vectorstores import Chroma
from typing import List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (
    CHROMA_PERSIST_DIR,
    PAPERS_COLLECTION,
    CODE_COLLECTION,
    TOP_K_RESULTS
)
from backend.rag.embeddings import get_embeddings


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Returns a persistent ChromaDB client.
    Data is saved to disk so it survives restarts.
    """
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR
    )


def get_vectorstore(collection_name: str) -> Chroma:
    """
    Returns a LangChain Chroma vectorstore for a given collection.
    Used for both papers and code collections.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    return vectorstore


def get_papers_vectorstore() -> Chroma:
    """Returns the lab papers vectorstore."""
    return get_vectorstore(PAPERS_COLLECTION)


def get_code_vectorstore() -> Chroma:
    """Returns the pipeline code vectorstore."""
    return get_vectorstore(CODE_COLLECTION)


def get_collection_stats() -> dict:
    """
    Returns stats about both collections.
    Useful for checking ingestion worked correctly.
    """
    client = get_chroma_client()
    stats = {}

    for name in [PAPERS_COLLECTION, CODE_COLLECTION]:
        try:
            collection = client.get_collection(name)
            stats[name] = {
                "count": collection.count(),
                "status": "ready" if collection.count() > 0 else "empty"
            }
        except Exception:
            stats[name] = {
                "count": 0,
                "status": "not created yet"
            }

    return stats


def similarity_search(
    query: str,
    collection_name: str,
    k: int = TOP_K_RESULTS
) -> List[dict]:
    """
    Runs similarity search on a given collection.
    Returns list of dicts with content and metadata.
    """
    vectorstore = get_vectorstore(collection_name)
    results = vectorstore.similarity_search_with_score(query, k=k)

    formatted = []
    for doc, score in results:
        formatted.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": round(float(score), 4)
        })

    return formatted


def delete_collection(collection_name: str) -> None:
    """
    Deletes a collection — useful for re-ingestion.
    """
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
        print(f"🗑️  Deleted collection: {collection_name}")
    except Exception as e:
        print(f"⚠️  Could not delete {collection_name}: {e}")


if __name__ == "__main__":
    print("Checking ChromaDB collections...\n")
    stats = get_collection_stats()

    for collection, info in stats.items():
        status_icon = "✅" if info["status"] == "ready" else "⚠️"
        print(f"{status_icon} {collection}")
        print(f"   Status : {info['status']}")
        print(f"   Chunks : {info['count']}")
        print()

    if all(info["count"] == 0 for info in stats.values()):
        print("💡 Collections are empty — run scripts/run_ingestion.py first")
    else:
        print("✅ ChromaDB is ready!")