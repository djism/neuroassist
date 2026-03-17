import sys
from pathlib import Path
from typing import Optional
import networkx as nx

sys.path.append(str(Path(__file__).resolve().parents[3]))
from backend.rag.knowledge_graph.graph_builder import load_graph, build_full_graph, GRAPH_PATH


class GraphRetriever:
    """
    Queries the NetworkX knowledge graph using logic-based traversal.
    Complements vector search by answering structural/relational questions
    that semantic similarity alone cannot answer well.

    Examples of what this handles:
    - "What techniques does the lab use?"
    - "What files does saa_mouse.py depend on?"
    - "Which papers study dopamine?"
    - "What paradigms has the lab published about?"
    """

    def __init__(self):
        self.G = self._load_or_build_graph()

    def _load_or_build_graph(self) -> nx.DiGraph:
        """Loads graph from disk, builds it if not found."""
        G = load_graph()
        if G is None:
            print("⚠️  Knowledge graph not found — building now...")
            G = build_full_graph()
        return G

    # ── Lab-level queries ─────────────────────────────────────────────────────

    def get_lab_techniques(self) -> list[dict]:
        """Returns all techniques the lab uses."""
        return self._get_neighbors(
            "Shrestha_Lab",
            relation="uses_technique",
            node_type="technique"
        )

    def get_lab_paradigms(self) -> list[dict]:
        """Returns all behavioral paradigms the lab studies."""
        return self._get_neighbors(
            "Shrestha_Lab",
            relation="studies_paradigm",
            node_type="paradigm"
        )

    def get_lab_brain_regions(self) -> list[dict]:
        """Returns all brain regions the lab studies."""
        return self._get_neighbors(
            "Shrestha_Lab",
            relation="studies_brain_region",
            node_type="brain_region"
        )

    def get_all_papers(self) -> list[dict]:
        """Returns all lab papers."""
        return self._get_neighbors(
            "Shrestha_Lab",
            relation="published",
            node_type="paper"
        )

    def get_lab_overview(self) -> dict:
        """
        Returns a full structured overview of the lab.
        Used when someone asks 'What does this lab do?'
        """
        return {
            "lab": self._get_node_info("Shrestha_Lab"),
            "pi": self._get_node_info("Prof_Shrestha"),
            "techniques": self.get_lab_techniques(),
            "paradigms": self.get_lab_paradigms(),
            "brain_regions": self.get_lab_brain_regions(),
            "paper_count": len(self.get_all_papers())
        }

    # ── Paper queries ─────────────────────────────────────────────────────────

    def get_papers_by_topic(self, topic: str) -> list[dict]:
        """
        Returns papers related to a given topic node.
        e.g. topic='dopamine' returns papers that study dopamine.
        """
        results = []
        topic_node = self._find_node_by_label(topic)
        if not topic_node:
            return []

        # Find all papers that have an edge pointing to this topic
        for source, target, data in self.G.edges(data=True):
            if target == topic_node and self.G.nodes[source].get("type") == "paper":
                results.append(self._get_node_info(source))

        return results

    def get_paper_topics(self, paper_id: str) -> list[dict]:
        """Returns all topics a specific paper studies."""
        return self._get_neighbors(paper_id, relation="studies")

    # ── Code queries ──────────────────────────────────────────────────────────

    def get_file_dependencies(self, filename: str) -> list[dict]:
        """
        Returns all files that a given file depends on (uses/extends).
        e.g. 'saa_mouse.py' → all files it imports/uses
        """
        deps = []
        if not self.G.has_node(filename):
            # Try partial match
            filename = self._find_code_file(filename)
            if not filename:
                return []

        for _, target, data in self.G.out_edges(filename, data=True):
            if data.get("relation") in ["uses", "extends", "runs"]:
                deps.append(self._get_node_info(target))

        return deps

    def get_file_description(self, filename: str) -> Optional[dict]:
        """Returns description of a specific code file."""
        node = self._find_code_file(filename)
        if node:
            return self._get_node_info(node)
        return None

    def get_all_code_files(self) -> list[dict]:
        """Returns all pipeline code files."""
        return [
            self._get_node_info(n)
            for n, d in self.G.nodes(data=True)
            if d.get("type") == "code_file"
        ]

    def get_pipeline_entry_point(self) -> Optional[dict]:
        """Returns the main entry point of the pipeline."""
        return self._get_node_info("run_analysis.py")

    def get_files_for_paradigm(self, paradigm: str) -> list[dict]:
        """
        Returns code files related to a paradigm.
        e.g. 'SAA' → files that implement SAA analysis
        """
        results = []
        paradigm_node = self._find_node_by_label(paradigm)
        if not paradigm_node:
            paradigm_node = paradigm  # try direct node id

        for source, target, data in self.G.edges(data=True):
            if (target == paradigm_node and
                    self.G.nodes[source].get("type") == "code_file"):
                results.append(self._get_node_info(source))

        return results

    # ── Generic graph queries ─────────────────────────────────────────────────

    def search_by_keyword(self, keyword: str) -> list[dict]:
        """
        Searches all nodes whose label or description contains the keyword.
        Case-insensitive.
        """
        keyword_lower = keyword.lower()
        results = []

        for node_id, data in self.G.nodes(data=True):
            label = data.get("label", "").lower()
            description = data.get("description", "").lower()

            if keyword_lower in label or keyword_lower in description:
                results.append(self._get_node_info(node_id))

        return results

    def get_related_nodes(self, node_id: str, depth: int = 1) -> list[dict]:
        """
        Returns all nodes within a given traversal depth from a node.
        Useful for exploring the graph around a concept.
        """
        if not self.G.has_node(node_id):
            return []

        related = set()
        current_level = {node_id}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                neighbors = (
                    list(self.G.successors(node)) +
                    list(self.G.predecessors(node))
                )
                next_level.update(neighbors)
            related.update(next_level)
            current_level = next_level

        related.discard(node_id)  # exclude the starting node
        return [self._get_node_info(n) for n in related]

    # ── Format for LLM context ────────────────────────────────────────────────

    def format_for_llm(self, results: list[dict] | dict) -> str:
        """
        Formats graph query results as clean text for the LLM.
        The LLM receives this as context alongside vector search results.
        """
        if not results:
            return "No relevant information found in knowledge graph."

        if isinstance(results, dict):
            results = [results]

        lines = []
        for item in results:
            if not item:
                continue
            label = item.get("label", item.get("id", "Unknown"))
            node_type = item.get("type", "")
            description = item.get("description", "")

            line = f"[{node_type.upper()}] {label}"
            if description:
                line += f": {description}"
            lines.append(line)

        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_node_info(self, node_id: str) -> Optional[dict]:
        """Returns node data dict, or None if node doesn't exist."""
        if not self.G.has_node(node_id):
            return None
        data = dict(self.G.nodes[node_id])
        data["id"] = node_id
        return data

    def _get_neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
        node_type: Optional[str] = None
    ) -> list[dict]:
        """
        Returns neighbors of a node, optionally filtered by
        edge relation type and/or neighbor node type.
        """
        if not self.G.has_node(node_id):
            return []

        results = []
        for _, target, edge_data in self.G.out_edges(node_id, data=True):
            # Filter by relation if specified
            if relation and edge_data.get("relation") != relation:
                continue
            # Filter by node type if specified
            target_type = self.G.nodes[target].get("type", "")
            if node_type and target_type != node_type:
                continue
            info = self._get_node_info(target)
            if info:
                results.append(info)

        return results

    def _find_node_by_label(self, label: str) -> Optional[str]:
        """
        Finds a node ID by matching its label.
        Case-insensitive partial match.
        """
        label_lower = label.lower()
        for node_id, data in self.G.nodes(data=True):
            if label_lower in data.get("label", "").lower():
                return node_id
        return None

    def _find_code_file(self, filename: str) -> Optional[str]:
        """
        Finds a code file node by partial filename match.
        e.g. 'saa' matches 'saa_mouse.py'
        """
        filename_lower = filename.lower()
        for node_id, data in self.G.nodes(data=True):
            if (data.get("type") == "code_file" and
                    filename_lower in node_id.lower()):
                return node_id
        return None


# Singleton
_retriever_instance = None

def get_graph_retriever() -> GraphRetriever:
    """Returns singleton GraphRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = GraphRetriever()
    return _retriever_instance


if __name__ == "__main__":
    print("Testing GraphRetriever...\n")
    retriever = get_graph_retriever()

    print("=" * 55)
    print("TEST 1: Lab Overview")
    print("=" * 55)
    overview = retriever.get_lab_overview()
    print(f"Lab      : {overview['lab']['label']}")
    print(f"PI       : {overview['pi']['label']}")
    print(f"Papers   : {overview['paper_count']}")
    print(f"Techniques: {[t['label'] for t in overview['techniques']]}")
    print(f"Paradigms : {[p['label'] for p in overview['paradigms']]}")

    print("\n" + "=" * 55)
    print("TEST 2: File dependencies for saa_mouse.py")
    print("=" * 55)
    deps = retriever.get_file_dependencies("saa_mouse.py")
    for d in deps:
        print(f"  → {d['label']}: {d.get('description', '')[:60]}")

    print("\n" + "=" * 55)
    print("TEST 3: Papers about dopamine")
    print("=" * 55)
    papers = retriever.get_papers_by_topic("dopamine")
    for p in papers:
        print(f"  → {p['label']}")

    print("\n" + "=" * 55)
    print("TEST 4: Keyword search — 'preprocessing'")
    print("=" * 55)
    results = retriever.search_by_keyword("preprocessing")
    for r in results:
        print(f"  → [{r['type']}] {r['label']}")

    print("\n" + "=" * 55)
    print("TEST 5: Format for LLM")
    print("=" * 55)
    formatted = retriever.format_for_llm(overview['techniques'])
    print(formatted)

    print("\n✅ GraphRetriever working correctly!")