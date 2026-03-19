import sys
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, END

sys.path.append(str(Path(__file__).resolve().parents[2]))
from backend.agent.state import AgentState, get_initial_state
from backend.agent.nodes import (
    router_node,
    papers_retriever_node,
    code_retriever_node,
    graph_retriever_node,
    papers_and_code_retriever_node,
    pubmed_retriever_node,
    pubmed_fallback_node,
    generator_node,
    evaluator_node
)
from config import RAGAS_ENABLED


# ── Conditional edge: router → retriever ─────────────────────────────────────

def route_to_retriever(
    state: AgentState
) -> Literal["papers", "code", "graph", "pubmed", "papers_and_code"]:
    """Routes to correct retriever based on router decision."""
    route = state.get("route", "papers")
    print(f"   📍 Routing to: {route}")
    return route


# ── Conditional edge: after retrieval → fallback or generator ────────────────

def check_fallback(
    state: AgentState
) -> Literal["pubmed_fallback", "generator"]:
    """
    After papers retrieval, checks if we need PubMed fallback.
    If the reranker flagged results as insufficient → go to PubMed.
    Otherwise → go straight to generator.

    This is the key to autonomous behavior:
    the agent knows when it doesn't know something.
    """
    needs_fallback = state.get("needs_pubmed_fallback", False)
    if needs_fallback:
        print("   🔄 Insufficient local context — triggering PubMed fallback")
        return "pubmed_fallback"
    return "generator"


# ── Conditional edge: after generation → evaluator or end ────────────────────

def route_to_evaluator(state: AgentState) -> Literal["evaluate", "end"]:
    """Decides whether to run RAGAS evaluation."""
    if RAGAS_ENABLED:
        return "evaluate"
    return "end"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """
    Builds and compiles the full LangGraph agent.

    Graph flow:

    START
      ↓
    router_node
      ↓ (conditional — 5 routes)
    ┌──────────────────────────────────────────────────────┐
    │ papers          → papers_retriever                   │
    │ code            → code_retriever                     │
    │ graph           → graph_retriever                    │
    │ pubmed          → pubmed_retriever                   │
    │ papers_and_code → papers_and_code_retriever          │
    └──────────────────────────────────────────────────────┘
      ↓
    papers_retriever → check_fallback (conditional)
      ├── insufficient → pubmed_fallback → generator
      └── sufficient  → generator
      ↓
    generator_node
      ↓ (conditional)
    evaluator_node → END
    """
    graph = StateGraph(AgentState)

    # ── Add all nodes ──────────────────────────────────────────────────────
    graph.add_node("router", router_node)
    graph.add_node("papers", papers_retriever_node)
    graph.add_node("code", code_retriever_node)
    graph.add_node("graph", graph_retriever_node)
    graph.add_node("pubmed", pubmed_retriever_node)
    graph.add_node("papers_and_code", papers_and_code_retriever_node)
    graph.add_node("pubmed_fallback", pubmed_fallback_node)
    graph.add_node("generator", generator_node)
    graph.add_node("evaluator", evaluator_node)

    # ── Entry point ────────────────────────────────────────────────────────
    graph.set_entry_point("router")

    # ── Router → retrievers (conditional) ─────────────────────────────────
    graph.add_conditional_edges(
        "router",
        route_to_retriever,
        {
            "papers": "papers",
            "code": "code",
            "graph": "graph",
            "pubmed": "pubmed",
            "papers_and_code": "papers_and_code"
        }
    )

    # ── Papers → check if fallback needed (conditional) ───────────────────
    # Only papers route checks for fallback — other routes are authoritative
    graph.add_conditional_edges(
        "papers",
        check_fallback,
        {
            "pubmed_fallback": "pubmed_fallback",
            "generator": "generator"
        }
    )

    # ── All other retrievers → generator directly ─────────────────────────
    for retriever in ["code", "graph", "pubmed", "papers_and_code"]:
        graph.add_edge(retriever, "generator")

    # ── PubMed fallback → generator ────────────────────────────────────────
    graph.add_edge("pubmed_fallback", "generator")

    # ── Generator → evaluator or END (conditional) ────────────────────────
    graph.add_conditional_edges(
        "generator",
        route_to_evaluator,
        {
            "evaluate": "evaluator",
            "end": END
        }
    )

    # ── Evaluator → END ────────────────────────────────────────────────────
    graph.add_edge("evaluator", END)

    compiled = graph.compile()
    print("✅ Agent graph compiled successfully")
    return compiled


# ── Main agent class ──────────────────────────────────────────────────────────

class NeuroAssistAgent:
    """
    Main agent class. Wraps the compiled LangGraph
    and provides a clean ask() interface.
    """

    def __init__(self):
        print("🚀 Initializing NeuroAssist Agent...")
        self.graph = build_agent_graph()
        print("✅ Agent ready!\n")

    def ask(self, question: str) -> dict:
        """
        Ask the agent a question.

        Returns:
            dict with answer, sources, route, faithfulness_score, eval_error
        """
        print(f"\n{'='*55}")
        print(f"QUESTION: {question}")
        print(f"{'='*55}")

        initial_state = get_initial_state(question)
        final_state = self.graph.invoke(initial_state)

        return {
            "answer": final_state.get("answer", "No answer generated"),
            "sources": final_state.get("sources", []),
            "route": final_state.get("route", "unknown"),
            "faithfulness_score": final_state.get("faithfulness_score"),
            "eval_error": final_state.get("eval_error"),
            "used_pubmed_fallback": "PubMed (fallback)" in final_state.get("sources", [])
        }

    def print_response(self, response: dict) -> None:
        """Pretty prints a response dict."""
        print(f"\n{'─'*55}")
        print("ANSWER:")
        print(f"{'─'*55}")
        print(response["answer"])
        print(f"\n{'─'*55}")
        print(f"Route              : {response['route']}")
        print(f"Sources            : {response['sources']}")
        print(f"PubMed fallback    : {response.get('used_pubmed_fallback', False)}")
        if response["faithfulness_score"] is not None:
            print(f"Faithfulness score : {response['faithfulness_score']:.2f}")
        print(f"{'─'*55}\n")


# ── Singleton ─────────────────────────────────────────────────────────────────

_agent_instance = None

def get_agent() -> NeuroAssistAgent:
    """Returns singleton agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = NeuroAssistAgent()
    return _agent_instance


if __name__ == "__main__":
    print("Testing full NeuroAssist Agent with all fixes...\n")

    agent = NeuroAssistAgent()

    test_questions = [
        "What does the Shrestha lab study?",
        "How does the snippet extractor work?",
        "What did the lab find about dopamine signaling?",
        "What is CRISPR?",  # Not in lab papers — should trigger PubMed fallback
    ]

    for question in test_questions:
        response = agent.ask(question)
        agent.print_response(response)

    print("✅ Full agent test complete!")