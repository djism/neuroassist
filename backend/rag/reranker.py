import sys
from pathlib import Path
from typing import List
from sentence_transformers import CrossEncoder

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TOP_K_RESULTS

# CrossEncoder model — scores query+chunk pairs for true relevance
# Much more accurate than embedding similarity alone
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Two-stage retrieval reranker using CrossEncoder.

    Stage 1 (vectorstore): fetch top 20 chunks by embedding similarity
    Stage 2 (reranker):    score each chunk against the query directly
                           and keep only the truly relevant ones

    CrossEncoder looks at query + chunk TOGETHER — unlike embeddings
    which encode them separately. This catches relevance that
    semantic similarity misses.
    """

    def __init__(self):
        print(f"⏳ Loading reranker model: {RERANKER_MODEL}")
        self.model = CrossEncoder(RERANKER_MODEL)
        print(f"✅ Reranker loaded: {RERANKER_MODEL}")

    def rerank(
        self,
        query: str,
        chunks: List[dict],
        top_k: int = TOP_K_RESULTS
    ) -> List[dict]:
        """
        Reranks retrieved chunks by true relevance to the query.

        Args:
            query: The user's question
            chunks: List of chunk dicts from vectorstore
            top_k: How many to keep after reranking

        Returns:
            Top-k chunks sorted by reranker score (most relevant first)
        """
        if not chunks:
            return []

        # Build (query, chunk_text) pairs for the CrossEncoder
        pairs = [(query, chunk["content"]) for chunk in chunks]

        # Score all pairs — CrossEncoder reads them together
        scores = self.model.predict(pairs)

        # Attach reranker scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk["reranker_score"] = float(score)

        # Sort by reranker score descending — best first
        reranked = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)

        # Keep only top_k
        top_chunks = reranked[:top_k]

        return top_chunks

    def is_relevant(self, query: str, chunk: str, threshold: float = -2.0) -> bool:
        """
        Returns True if a single chunk is relevant enough to answer the query.
        Used by the fallback mechanism to decide if PubMed search is needed.

        Threshold: CrossEncoder scores range roughly -10 to +10.
        -2.0 is a conservative threshold — anything below means
        the chunk probably doesn't answer the question well.
        """
        score = self.model.predict([(query, chunk)])[0]
        return float(score) > threshold

    def chunks_are_sufficient(
        self,
        query: str,
        chunks: List[dict],
        threshold: float = -2.0,
        min_good_chunks: int = 1
    ) -> bool:
        """
        Checks if retrieved chunks are sufficient to answer the query.
        Used to decide whether to fall back to PubMed.

        Returns True if at least min_good_chunks pass the threshold.
        Returns False if the local knowledge base doesn't have the answer.
        """
        if not chunks:
            return False

        good_chunks = sum(
            1 for chunk in chunks
            if chunk.get("reranker_score", -999) > threshold
        )

        return good_chunks >= min_good_chunks


# Singleton — load model once
_reranker_instance = None

def get_reranker() -> Reranker:
    """Returns singleton Reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance


if __name__ == "__main__":
    print("Testing Reranker...\n")
    reranker = get_reranker()

    query = "How does z-score normalization work in the pipeline?"

    # Simulate chunks of varying relevance
    test_chunks = [
        {
            "content": "The z-score normalization is applied after motion correction. "
                       "It standardizes the signal using the mean and std of the baseline period.",
            "metadata": {"source": "preprocessing.py"},
            "relevance_score": 0.45
        },
        {
            "content": "The SAA paradigm involves a shuttlebox where mice learn to avoid "
                       "a shock by crossing to the other side.",
            "metadata": {"source": "Lab_Paper1.pdf"},
            "relevance_score": 0.41
        },
        {
            "content": "self.zscore = (self.dff - baseline_mean) / baseline_std "
                       "where baseline is defined as the pre-CS period.",
            "metadata": {"source": "base_mouse.py"},
            "relevance_score": 0.39
        },
        {
            "content": "Dopamine signals were measured using fiber photometry with "
                       "dLight1.3b sensor in the nucleus accumbens.",
            "metadata": {"source": "Lab_Paper3.pdf"},
            "relevance_score": 0.38
        }
    ]

    print(f"Query: {query}")
    print(f"Chunks before reranking: {len(test_chunks)}")
    print("\nBefore reranking (by embedding similarity):")
    for c in test_chunks:
        print(f"  {c['metadata']['source']:30} score: {c['relevance_score']:.3f}")

    reranked = reranker.rerank(query, test_chunks, top_k=2)

    print(f"\nAfter reranking (by CrossEncoder) — top 2:")
    for c in reranked:
        print(f"  {c['metadata']['source']:30} reranker score: {c['reranker_score']:.3f}")

    sufficient = reranker.chunks_are_sufficient(query, reranked)
    print(f"\nChunks sufficient to answer? {sufficient}")

    print("\n✅ Reranker working correctly!")