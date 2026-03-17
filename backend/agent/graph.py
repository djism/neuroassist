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
    generator_node,
    evaluator_node
)
from config import RAGAS_ENABLED


# ── Conditional edge function ─────────────────────────────────────────────────

def route_to_retriever(
    state: AgentState
) -> Literal["papers", "code", "graph", "pubmed", "papers_and_code"]:
    """
    Reads the route from state and directs the graph
    to the correct retriever node.
    This is the conditional edge after the router node.
    """
    route = state.get("route", "papers")
    print(f"   📍 Routing to: {route}")
    return route


def route_to_evaluator(state: AgentState) -> Literal["evaluate", "end"]:
    """
    After generation, decides whether to run RAGAS evaluation.
    Skips evaluation if RAGAS_ENABLED is False in config.
    """
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
    router_node          ← decides which path to take
      ↓ (conditional)
    ┌─────────────────────────────────────────────┐
    │  papers           → papers_retriever_node   │
    │  code             → code_retriever_node     │
    │  graph            → graph_retriever_node    │
    │  pubmed           → pubmed_retriever_node   │
    │  papers_and_code  → papers_and_code_node    │
    └─────────────────────────────────────────────┘
      ↓
    generator_node       ← builds context + calls LLM
      ↓ (conditional)
    evaluator_node       ← RAGAS faithfulness scoring
      ↓
    END
    """
    # Create graph with our state schema
    graph = StateGraph(AgentState)

    # ── Add all nodes ─────────────────────────────────────────────────────────
    graph.add_node("router", router_node)
    graph.add_node("papers", papers_retriever_node)
    graph.add_node("code", code_retriever_node)
    graph.add_node("graph", graph_retriever_node)
    graph.add_node("pubmed", pubmed_retriever_node)
    graph.add_node("papers_and_code", papers_and_code_retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("evaluator", evaluator_node)

    # ── Set entry point ───────────────────────────────────────────────────────
    graph.set_entry_point("router")

    # ── Add conditional edge from router ──────────────────────────────────────
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

    # ── All retrievers feed into generator ───────────────────────────────────
    for retriever in ["papers", "code", "graph", "pubmed", "papers_and_code"]:
        graph.add_edge(retriever, "generator")

    # ── Generator conditionally feeds into evaluator or END ──────────────────
    graph.add_conditional_edges(
        "generator",
        route_to_evaluator,
        {
            "evaluate": "evaluator",
            "end": END
        }
    )

    # ── Evaluator always ends ─────────────────────────────────────────────────
    graph.add_edge("evaluator", END)

    # ── Compile ───────────────────────────────────────────────────────────────
    compiled = graph.compile()
    print("✅ Agent graph compiled successfully")
    return compiled


# ── Main agent runner ─────────────────────────────────────────────────────────

class NeuroAssistAgent:
    """
    Main agent class. Wraps the compiled LangGraph
    and provides a clean `ask()` interface.
    """

    def __init__(self):
        print("🚀 Initializing NeuroAssist Agent...")
        self.graph = build_agent_graph()
        print("✅ Agent ready!\n")

    def ask(self, question: str) -> dict:
        """
        Ask the agent a question and get a structured response.

        Args:
            question: Natural language question

        Returns:
            dict with keys:
                - answer: str
                - sources: list[str]
                - route: str (which retriever was used)
                - faithfulness_score: float or None
                - eval_error: str or None
        """
        print(f"\n{'='*55}")
        print(f"QUESTION: {question}")
        print(f"{'='*55}")

        # Build initial state
        initial_state = get_initial_state(question)

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Package the response
        response = {
            "answer": final_state.get("answer", "No answer generated"),
            "sources": final_state.get("sources", []),
            "route": final_state.get("route", "unknown"),
            "faithfulness_score": final_state.get("faithfulness_score"),
            "eval_error": final_state.get("eval_error")
        }

        return response

    def print_response(self, response: dict) -> None:
        """Pretty prints a response dict."""
        print(f"\n{'─'*55}")
        print("ANSWER:")
        print(f"{'─'*55}")
        print(response["answer"])
        print(f"\n{'─'*55}")
        print(f"Route   : {response['route']}")
        print(f"Sources : {response['sources']}")
        if response["faithfulness_score"] is not None:
            print(f"Faithfulness: {response['faithfulness_score']:.2f}")
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
    print("Testing full NeuroAssist Agent...\n")

    agent = NeuroAssistAgent()

    # Test questions covering all routes
    test_questions = [
        "What does the Shrestha lab study?",
        "How does the snippet extractor work?",
        "What did the lab find about dopamine signaling?",
    ]

    for question in test_questions:
        response = agent.ask(question)
        agent.print_response(response)

    print("✅ Full agent test complete!")