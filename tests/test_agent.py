import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


# ── Config tests ──────────────────────────────────────────────────────────────

def test_config_imports():
    from config import (
        GROQ_API_KEY, NCBI_API_KEY, LLM_MODEL,
        EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS
    )
    assert LLM_MODEL == "llama-3.3-70b-versatile"
    assert EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
    assert CHUNK_SIZE == 512
    assert CHUNK_OVERLAP == 64
    assert TOP_K_RESULTS == 5


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_question_request_valid():
    from backend.api.schemas import QuestionRequest
    req = QuestionRequest(question="What does the Shrestha lab study?")
    assert req.question == "What does the Shrestha lab study?"


def test_question_request_too_short():
    from backend.api.schemas import QuestionRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        QuestionRequest(question="Hi")


def test_answer_response_valid():
    from backend.api.schemas import AnswerResponse
    resp = AnswerResponse(
        answer="The lab studies...",
        sources=["Lab_Paper1.pdf"],
        route="graph",
        faithfulness_score=0.92,
        eval_error=None
    )
    assert resp.answer == "The lab studies..."
    assert resp.route == "graph"
    assert resp.faithfulness_score == 0.92


# ── Agent state tests ─────────────────────────────────────────────────────────

def test_initial_state():
    from backend.agent.state import get_initial_state
    state = get_initial_state("What does the lab study?")
    assert state["question"] == "What does the lab study?"
    assert state["route"] is None
    assert state["answer"] is None
    assert state["messages"] == []
    assert state["sources"] is None


def test_state_update():
    from backend.agent.state import get_initial_state
    state = get_initial_state("test question")
    state["route"] = "papers"
    state["answer"] = "Test answer"
    assert state["route"] == "papers"
    assert state["answer"] == "Test answer"


# ── Router tests ──────────────────────────────────────────────────────────────

def test_router_graph_route():
    from backend.agent.state import get_initial_state
    from backend.agent.nodes import router_node
    state = get_initial_state("What does the Shrestha lab study?")
    result = router_node(state)
    assert result["route"] == "graph"


def test_router_code_route():
    from backend.agent.state import get_initial_state
    from backend.agent.nodes import router_node
    state = get_initial_state("How does saa_mouse.py work?")
    result = router_node(state)
    assert result["route"] == "code"


def test_router_papers_route():
    from backend.agent.state import get_initial_state
    from backend.agent.nodes import router_node
    state = get_initial_state("What did the lab find about dopamine?")
    result = router_node(state)
    assert result["route"] == "papers"


def test_router_pubmed_route():
    from backend.agent.state import get_initial_state
    from backend.agent.nodes import router_node
    state = get_initial_state("Find recent papers on active avoidance")
    result = router_node(state)
    assert result["route"] == "pubmed"


# ── Knowledge graph tests ─────────────────────────────────────────────────────

def test_graph_builds():
    from backend.rag.knowledge_graph.graph_builder import build_full_graph
    G = build_full_graph()
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0
    assert G.has_node("Shrestha_Lab")
    assert G.has_node("saa_mouse.py")


def test_graph_lab_techniques():
    from backend.rag.knowledge_graph.graph_retriever import GraphRetriever
    retriever = GraphRetriever()
    techniques = retriever.get_lab_techniques()
    assert len(techniques) > 0
    labels = [t["label"] for t in techniques]
    assert "Fiber Photometry" in labels


def test_graph_file_dependencies():
    from backend.rag.knowledge_graph.graph_retriever import GraphRetriever
    retriever = GraphRetriever()
    deps = retriever.get_file_dependencies("saa_mouse.py")
    assert len(deps) > 0
    dep_labels = [d["label"] for d in deps]
    assert "base_mouse.py" in dep_labels


def test_graph_keyword_search():
    from backend.rag.knowledge_graph.graph_retriever import GraphRetriever
    retriever = GraphRetriever()
    results = retriever.search_by_keyword("dopamine")
    assert len(results) > 0


# ── Vectorstore tests ─────────────────────────────────────────────────────────

def test_collection_stats():
    from backend.rag.vectorstore import get_collection_stats
    stats = get_collection_stats()
    assert "lab_papers" in stats
    assert "pipeline_code" in stats


def test_papers_collection_not_empty():
    from backend.rag.vectorstore import get_collection_stats
    stats = get_collection_stats()
    assert stats["lab_papers"]["count"] > 0, \
        "Papers collection is empty — run scripts/run_ingestion.py first"


def test_code_collection_not_empty():
    from backend.rag.vectorstore import get_collection_stats
    stats = get_collection_stats()
    assert stats["pipeline_code"]["count"] > 0, \
        "Code collection is empty — run scripts/run_ingestion.py first"