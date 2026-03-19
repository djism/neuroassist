import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TOP_K_RESULTS, PAPERS_COLLECTION, CODE_COLLECTION
from backend.rag.vectorstore import similarity_search
from backend.rag.reranker import get_reranker
from backend.rag.knowledge_graph.graph_retriever import get_graph_retriever


class UnifiedRetriever:
    """
    Single interface combining:
    - ChromaDB vector search (semantic similarity)
    - CrossEncoder reranking (true relevance scoring)
    - NetworkX knowledge graph (logic-based traversal)

    The agent calls this — it never talks to ChromaDB,
    the reranker, or NetworkX directly.
    """

    def __init__(self):
        self.graph = get_graph_retriever()
        self.reranker = get_reranker()

    # ── Vector search with reranking ──────────────────────────────────────────

    def search_papers(
        self,
        query: str,
        k: int = TOP_K_RESULTS
    ) -> list[dict]:
        """
        Two-stage retrieval over lab papers:
        1. ChromaDB fetches top 20 by embedding similarity
        2. CrossEncoder reranks and returns top k by true relevance
        """
        # Stage 1: broad vector search
        raw_results = similarity_search(query, PAPERS_COLLECTION, k=k)

        # Stage 2: rerank for true relevance
        reranked = self.reranker.rerank(query, raw_results, top_k=k)

        return reranked

    def search_code(
        self,
        query: str,
        k: int = TOP_K_RESULTS
    ) -> list[dict]:
        """
        Two-stage retrieval over pipeline code:
        1. ChromaDB fetches top 20 by embedding similarity
        2. CrossEncoder reranks and returns top k by true relevance
        """
        # Stage 1: broad vector search
        raw_results = similarity_search(query, CODE_COLLECTION, k=k)

        # Stage 2: rerank for true relevance
        reranked = self.reranker.rerank(query, raw_results, top_k=k)

        return reranked

    def papers_are_sufficient(self, query: str, chunks: list[dict]) -> bool:
        """
        Checks if retrieved paper chunks actually answer the query.
        Used to decide whether PubMed fallback is needed.
        Returns False if local knowledge base doesn't have the answer.
        """
        return self.reranker.chunks_are_sufficient(
            query,
            chunks,
            threshold=-2.0,
            min_good_chunks=1
        )

    # ── Graph queries ─────────────────────────────────────────────────────────

    def get_lab_overview(self) -> str:
        """Returns structured lab overview from knowledge graph."""
        overview = self.graph.get_lab_overview()
        lines = []

        lab = overview.get("lab", {})
        pi = overview.get("pi", {})
        lines.append(f"Lab: {lab.get('label', 'Shrestha Lab')}")
        lines.append(f"PI : {pi.get('label', 'Prof. Prerana Shrestha')}")
        lines.append(f"Department: {lab.get('description', '')}")

        techniques = overview.get("techniques", [])
        if techniques:
            lines.append(f"\nTechniques used:")
            for t in techniques:
                lines.append(f"  - {t['label']}: {t.get('description', '')}")

        paradigms = overview.get("paradigms", [])
        if paradigms:
            lines.append(f"\nBehavioral paradigms studied:")
            for p in paradigms:
                lines.append(f"  - {p['label']}: {p.get('description', '')}")

        brain_regions = overview.get("brain_regions", [])
        if brain_regions:
            lines.append(f"\nBrain regions studied:")
            for b in brain_regions:
                lines.append(f"  - {b['label']}: {b.get('description', '')}")

        lines.append(f"\nTotal papers published: {overview.get('paper_count', 0)}")
        return "\n".join(lines)

    def get_file_dependencies(self, filename: str) -> str:
        """Returns dependency tree for a code file."""
        deps = self.graph.get_file_dependencies(filename)
        if not deps:
            return f"No dependency information found for '{filename}'"

        lines = [f"Dependencies of {filename}:"]
        for d in deps:
            lines.append(f"  → {d['label']}: {d.get('description', '')}")
        return "\n".join(lines)

    def get_file_description(self, filename: str) -> str:
        """Returns description of a specific code file."""
        info = self.graph.get_file_description(filename)
        if not info:
            return f"No information found for '{filename}'"
        return f"{info['label']}: {info.get('description', 'No description available')}"

    def get_all_code_files(self) -> str:
        """Returns list of all pipeline files with descriptions."""
        files = self.graph.get_all_code_files()
        if not files:
            return "No code files found in knowledge graph."

        lines = ["Pipeline code files:"]
        for f in files:
            lines.append(f"  - {f['label']}: {f.get('description', '')}")
        return "\n".join(lines)

    def search_graph(self, keyword: str) -> str:
        """Keyword search across all graph nodes."""
        results = self.graph.search_by_keyword(keyword)
        if not results:
            return f"Nothing found in knowledge graph for '{keyword}'"
        return self.graph.format_for_llm(results)

    # ── Combined retrieval ────────────────────────────────────────────────────

    def retrieve_for_question(
        self,
        question: str,
        search_papers: bool = True,
        search_code: bool = False,
        k: int = TOP_K_RESULTS
    ) -> str:
        """
        Combined retrieval with reranking — fetches from whichever
        sources are relevant and merges into one context string.
        """
        context_parts = []

        if search_papers:
            paper_results = self.search_papers(question, k=k)
            if paper_results:
                context_parts.append("RELEVANT LAB PAPER CONTENT:")
                context_parts.append("=" * 40)
                for i, r in enumerate(paper_results, 1):
                    source = r["metadata"].get("source", "unknown")
                    score = r.get("reranker_score", r.get("relevance_score", 0))
                    context_parts.append(f"[{i}] Source: {source} (score: {score:.3f})")
                    context_parts.append(r["content"])
                    context_parts.append("-" * 40)

        if search_code:
            code_results = self.search_code(question, k=k)
            if code_results:
                context_parts.append("\nRELEVANT PIPELINE CODE:")
                context_parts.append("=" * 40)
                for i, r in enumerate(code_results, 1):
                    source = r["metadata"].get("source", "unknown")
                    context_parts.append(f"[{i}] File: {source}")
                    context_parts.append(r["content"])
                    context_parts.append("-" * 40)

        graph_results = self.graph.search_by_keyword(
            question.split()[0]
        )
        if graph_results:
            context_parts.append("\nRELATED CONCEPTS FROM KNOWLEDGE GRAPH:")
            context_parts.append("=" * 40)
            context_parts.append(self.graph.format_for_llm(graph_results))

        if not context_parts:
            return "No relevant context found in knowledge base."

        return "\n".join(context_parts)

    def format_vector_results(self, results: list[dict]) -> str:
        """Formats raw vector search results into clean text for LLM."""
        if not results:
            return "No results found."

        lines = []
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "unknown")
            score = r.get("reranker_score", r.get("relevance_score", 0))
            lines.append(f"[{i}] Source: {source} | Score: {score:.3f}")
            lines.append(r["content"])
            lines.append("")

        return "\n".join(lines)


# Singleton
_retriever_instance = None

def get_retriever() -> UnifiedRetriever:
    """Returns singleton UnifiedRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = UnifiedRetriever()
    return _retriever_instance


if __name__ == "__main__":
    print("Testing UnifiedRetriever with reranking...\n")
    retriever = get_retriever()

    print("=" * 55)
    print("TEST 1: Paper search with reranking")
    print("=" * 55)
    results = retriever.search_papers(
        "dopamine signaling during fear conditioning", k=3
    )
    for r in results:
        print(f"Source        : {r['metadata']['source']}")
        print(f"Reranker score: {r.get('reranker_score', 'N/A'):.3f}")
        print(f"Preview       : {r['content'][:100]}")
        print()

    print("=" * 55)
    print("TEST 2: Sufficiency check")
    print("=" * 55)
    sufficient = retriever.papers_are_sufficient(
        "dopamine signaling during fear conditioning", results
    )
    print(f"Papers sufficient? {sufficient}")

    print("\n=" * 55)
    print("TEST 3: Code search with reranking")
    print("=" * 55)
    results = retriever.search_code("z-score normalization baseline", k=3)
    for r in results:
        print(f"File          : {r['metadata']['source']}")
        print(f"Reranker score: {r.get('reranker_score', 'N/A'):.3f}")
        print(f"Preview       : {r['content'][:100]}")
        print()

    print("✅ UnifiedRetriever with reranking working correctly!")