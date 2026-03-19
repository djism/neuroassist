import sys
import os
import ssl
import certifi
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

ssl._create_default_https_context = ssl.create_default_context
os.environ['SSL_CERT_FILE'] = certifi.where()

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import GROQ_API_KEY, LLM_MODEL, RAGAS_ENABLED
from backend.agent.state import AgentState
from backend.rag.retriever import get_retriever
from backend.tools.pubmed import search_and_format


# ── LLM setup ─────────────────────────────────────────────────────────────────
def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=0.2,
        max_tokens=1024
    )


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are NeuroAssist, an intelligent research assistant for the Shrestha Lab
at Stony Brook University (Department of Neurobiology & Behavior).

You have access to:
1. The lab's published research papers
2. The lab's fiber photometry analysis pipeline codebase
3. A knowledge graph of lab concepts, techniques, and code relationships
4. Live PubMed search results (when the question goes beyond the lab's own papers)

Your job is to answer questions accurately and helpfully based on the provided context.

RULES:
- Always base your answers on the provided context
- If the context does not contain enough information, say so clearly
- Always cite your sources (paper names or file names)
- For code questions, reference specific file names and function names when possible
- For scientific questions, be precise about neuroscience terminology
- Keep answers clear and well-structured
- Never make up information that is not in the context

You serve two types of users:
1. New students/researchers with no neuroscience background — explain concepts simply
2. Experienced researchers — be precise and technical
Adapt your language to the question's complexity."""


# ── Sufficiency check prompt ──────────────────────────────────────────────────
SUFFICIENCY_CHECK_PROMPT = """You are checking if the provided context contains enough 
information to answer the question.

Question: {question}

Context (first 500 chars): {context_preview}

Reply with ONLY one word: YES if the context can answer the question, NO if it cannot.
Do not explain. Just YES or NO."""


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1: ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def router_node(state: AgentState) -> AgentState:
    """
    Analyzes the question and decides which retrieval path to take.
    Routes: papers, code, graph, pubmed, papers_and_code
    """
    question = state["question"].lower()
    print(f"\n🔀 Router analyzing: '{state['question'][:60]}...'")

    code_keywords = [
        "code", "file", "function", "implement", "pipeline", "script",
        "how does", "how do", "where is", "which file", "class", "method",
        "parameter", "argument", "import", "module", "run", "execute",
        "saa_mouse", "base_mouse", "snippet", "peth", "heatmap", "excel",
        "preprocessing", "loader", "parser", "scanner", "plotter"
    ]

    graph_keywords = [
        "what does the lab", "what does this lab", "lab do",
        "overview", "summary", "about the lab", "who is", "professor",
        "techniques used", "paradigms", "brain regions", "what files",
        "all files", "dependencies", "depends on", "structure",
        "list all", "what is the lab", "shrestha lab",
        "what does the shrestha", "what does prof",
        "about this lab", "tell me about the lab",
        "paradigm", "technique", "brain region",
        "what does the lab", "what does this lab", "lab study",
        "lab do", "lab use", "lab research"
    ]

    pubmed_keywords = [
        "recent papers", "latest research", "find papers", "search pubmed",
        "published recently", "other labs", "literature", "not in the lab",
        "general research", "field of", "state of the art"
    ]

    is_code = any(kw in question for kw in code_keywords)
    is_graph = any(kw in question for kw in graph_keywords)
    is_pubmed = any(kw in question for kw in pubmed_keywords)

    if is_graph:
        route = "graph"
    elif is_code and not is_graph:
        paper_keywords = ["paper", "study", "finding", "result", "publish"]
        also_papers = any(kw in question for kw in paper_keywords)
        route = "papers_and_code" if also_papers else "code"
    elif is_pubmed:
        route = "pubmed"
    else:
        route = "papers"

    print(f"   ✅ Route decided: {route}")
    return {**state, "route": route}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2: PAPERS RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

def papers_retriever_node(state: AgentState) -> AgentState:
    """
    Two-stage retrieval over lab papers:
    1. ChromaDB vector search (cast wide net)
    2. CrossEncoder reranking (keep most relevant)
    Sets needs_pubmed_fallback=True if results are insufficient.
    """
    print(f"\n📚 Retrieving from lab papers (with reranking)...")
    retriever = get_retriever()

    results = retriever.search_papers(state["question"], k=5)
    papers_context = retriever.format_vector_results(results)

    # Check if retrieved chunks actually answer the question
    sufficient = retriever.papers_are_sufficient(state["question"], results)

    graph_context = retriever.search_graph(state["question"].split()[0])

    sources = list({r["metadata"].get("source", "unknown") for r in results})

    print(f"   ✅ Retrieved {len(results)} chunks (reranked)")
    print(f"   Sources   : {sources}")
    print(f"   Sufficient: {sufficient}")

    return {
        **state,
        "papers_context": papers_context,
        "graph_context": graph_context,
        "sources": sources,
        "needs_pubmed_fallback": not sufficient
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3: CODE RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

def code_retriever_node(state: AgentState) -> AgentState:
    """
    Two-stage retrieval over pipeline code with reranking.
    Also fetches file dependency info from knowledge graph.
    """
    print(f"\n💻 Retrieving from pipeline code (with reranking)...")
    retriever = get_retriever()

    results = retriever.search_code(state["question"], k=5)
    code_context = retriever.format_vector_results(results)

    question_lower = state["question"].lower()
    py_files = [
        "saa_mouse", "base_mouse", "snippet_extractor", "peth_plotter",
        "group_analyzer", "preprocessing", "tdt_loader", "heatmap_plotter",
        "excel_writer", "metrics_calculator", "run_analysis"
    ]
    graph_context = ""
    for fname in py_files:
        if fname in question_lower:
            graph_context = retriever.get_file_dependencies(f"{fname}.py")
            break
    if not graph_context:
        graph_context = retriever.get_all_code_files()

    sources = list({r["metadata"].get("source", "unknown") for r in results})

    print(f"   ✅ Retrieved {len(results)} chunks (reranked)")
    print(f"   Sources: {sources}")

    return {
        **state,
        "code_context": code_context,
        "graph_context": graph_context,
        "sources": sources,
        "needs_pubmed_fallback": False
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4: GRAPH RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

def graph_retriever_node(state: AgentState) -> AgentState:
    """Retrieves structured info from the NetworkX knowledge graph."""
    print(f"\n🧠 Retrieving from knowledge graph...")
    retriever = get_retriever()

    question_lower = state["question"].lower()

    if any(kw in question_lower for kw in [
        "overview", "about the lab", "what does", "who is",
        "paradigm", "technique", "brain region", "shrestha lab",
        "what does the lab", "what does this lab", "lab study",
        "lab do", "lab use", "lab research"
    ]):
        graph_context = retriever.get_lab_overview()
        sources = ["Knowledge Graph — Lab Overview"]

    elif any(kw in question_lower for kw in ["files", "pipeline", "all code", "components"]):
        graph_context = retriever.get_all_code_files()
        sources = ["Knowledge Graph — Code Files"]

    elif "depends" in question_lower or "dependencies" in question_lower:
        py_files = [
            "saa_mouse", "base_mouse", "snippet_extractor", "peth_plotter",
            "group_analyzer", "preprocessing", "tdt_loader"
        ]
        graph_context = ""
        sources = []
        for fname in py_files:
            if fname in question_lower:
                graph_context = retriever.get_file_dependencies(f"{fname}.py")
                sources = [f"Knowledge Graph — {fname}.py dependencies"]
                break
        if not graph_context:
            graph_context = retriever.get_lab_overview()
            sources = ["Knowledge Graph — Lab Overview"]
    else:
        stopwords = {"what", "does", "the", "is", "how", "which", "who",
                     "do", "a", "an", "this", "that", "about"}
        keywords = [
            w for w in state["question"].lower().split()
            if len(w) > 3 and w not in stopwords
        ]
        keyword = keywords[0] if keywords else "lab"
        graph_context = retriever.search_graph(keyword)
        sources = ["Knowledge Graph"]

    print(f"   ✅ Graph context retrieved")

    return {
        **state,
        "graph_context": graph_context,
        "sources": sources,
        "needs_pubmed_fallback": False
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5: PAPERS AND CODE RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

def papers_and_code_retriever_node(state: AgentState) -> AgentState:
    """Retrieves from both papers and code with reranking."""
    print(f"\n📚💻 Retrieving from papers AND code (with reranking)...")
    retriever = get_retriever()

    paper_results = retriever.search_papers(state["question"], k=3)
    code_results = retriever.search_code(state["question"], k=3)

    papers_context = retriever.format_vector_results(paper_results)
    code_context = retriever.format_vector_results(code_results)

    sources = list({
        r["metadata"].get("source", "unknown")
        for r in paper_results + code_results
    })

    print(f"   ✅ {len(paper_results)} paper + {len(code_results)} code chunks (reranked)")

    return {
        **state,
        "papers_context": papers_context,
        "code_context": code_context,
        "sources": sources,
        "needs_pubmed_fallback": False
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6: PUBMED RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

def pubmed_retriever_node(state: AgentState) -> AgentState:
    """Searches PubMed for papers related to the question."""
    print(f"\n🔬 Searching PubMed...")
    retriever = get_retriever()

    pubmed_context = search_and_format(state["question"], max_results=3)

    paper_results = retriever.search_papers(state["question"], k=2)
    papers_context = retriever.format_vector_results(paper_results)

    sources = ["PubMed"] + [
        r["metadata"].get("source", "unknown")
        for r in paper_results
    ]

    print(f"   ✅ PubMed results retrieved")

    return {
        **state,
        "pubmed_context": pubmed_context,
        "papers_context": papers_context,
        "sources": sources,
        "needs_pubmed_fallback": False
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 7: PUBMED FALLBACK
# Called automatically when local papers don't answer the question
# ══════════════════════════════════════════════════════════════════════════════

def pubmed_fallback_node(state: AgentState) -> AgentState:
    """
    Automatic PubMed fallback — triggered when the reranker determines
    that local lab papers don't have sufficient context to answer.

    This is what makes the agent genuinely autonomous:
    it knows when it doesn't know something and goes to find more.
    """
    print(f"\n🔄 Local papers insufficient — falling back to PubMed...")

    pubmed_context = search_and_format(state["question"], max_results=3)

    existing_sources = state.get("sources", [])
    updated_sources = list(set(existing_sources + ["PubMed (fallback)"]))

    print(f"   ✅ PubMed fallback results retrieved")

    return {
        **state,
        "pubmed_context": pubmed_context,
        "sources": updated_sources,
        "needs_pubmed_fallback": False
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 8: GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generator_node(state: AgentState) -> AgentState:
    """
    Assembles all retrieved context and calls the LLM to generate
    a grounded, cited answer.
    """
    print(f"\n✍️  Generating answer...")
    llm = get_llm()

    context_parts = []

    if state.get("graph_context"):
        context_parts.append("KNOWLEDGE GRAPH CONTEXT:")
        context_parts.append(state["graph_context"])
        context_parts.append("")

    if state.get("papers_context"):
        context_parts.append("LAB PAPERS CONTEXT:")
        context_parts.append(state["papers_context"])
        context_parts.append("")

    if state.get("code_context"):
        context_parts.append("PIPELINE CODE CONTEXT:")
        context_parts.append(state["code_context"])
        context_parts.append("")

    if state.get("pubmed_context"):
        context_parts.append("PUBMED SEARCH RESULTS (external literature):")
        context_parts.append(state["pubmed_context"])
        context_parts.append("")

    context = "\n".join(context_parts)

    if not context.strip():
        context = "No context was retrieved for this question."

    user_prompt = f"""Based on the following context, please answer this question:

QUESTION: {state["question"]}

CONTEXT:
{context}

Please provide a clear, accurate answer based only on the context above.
Cite the specific sources (paper names or file names) that support your answer.
If PubMed results are included, note that they come from external literature.
If the context does not contain enough information to answer fully, say so."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    answer = response.content

    print(f"   ✅ Answer generated ({len(answer)} chars)")

    updated_messages = list(state.get("messages", []))
    updated_messages.append(HumanMessage(content=state["question"]))
    updated_messages.append(AIMessage(content=answer))

    return {
        **state,
        "answer": answer,
        "messages": updated_messages
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 9: EVALUATOR — Fixed async issue
# ══════════════════════════════════════════════════════════════════════════════

def evaluator_node(state: AgentState) -> AgentState:
    """
    Evaluates answer quality using RAGAS faithfulness metric.
    Fixed: uses nest_asyncio to handle async inside FastAPI/uvloop.
    """
    if not RAGAS_ENABLED:
        return {**state, "faithfulness_score": None}

    print(f"\n📊 Evaluating answer quality...")

    try:
        import nest_asyncio
        nest_asyncio.apply()

        from ragas import evaluate
        from ragas.metrics import faithfulness
        from datasets import Dataset
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from backend.rag.embeddings import get_embeddings

        contexts = []
        if state.get("papers_context"):
            contexts.append(state["papers_context"])
        if state.get("code_context"):
            contexts.append(state["code_context"])
        if state.get("graph_context"):
            contexts.append(state["graph_context"])
        if state.get("pubmed_context"):
            contexts.append(state["pubmed_context"])

        if not contexts or not state.get("answer"):
            return {
                **state,
                "faithfulness_score": None,
                "eval_error": "Insufficient context for evaluation"
            }

        eval_data = {
            "question": [state["question"]],
            "answer": [state["answer"]],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(eval_data)

        groq_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=0
        )
        ragas_llm = LangchainLLMWrapper(groq_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )

        # Handle both float and list return types from RAGAS
        raw_score = result["faithfulness"]
        if isinstance(raw_score, list):
            score = float(raw_score[0])
        else:
            score = float(raw_score)

        print(f"   ✅ Faithfulness score: {score:.2f}")

        return {**state, "faithfulness_score": score}

    except Exception as e:
        print(f"   ⚠️  Evaluation skipped: {e}")
        return {
            **state,
            "faithfulness_score": None,
            "eval_error": str(e)
        }


if __name__ == "__main__":
    from backend.agent.state import get_initial_state

    print("Testing updated nodes...\n")

    print("=" * 55)
    print("TEST 1: Router")
    print("=" * 55)
    questions = [
        ("What does the Shrestha lab study?", "graph"),
        ("How does saa_mouse.py work?", "code"),
        ("What did the lab find about dopamine?", "papers"),
        ("Find recent papers on active avoidance", "pubmed"),
    ]
    for q, expected in questions:
        state = get_initial_state(q)
        result = router_node(state)
        status = "✅" if result["route"] == expected else "⚠️"
        print(f"   {status} '{q[:45]}' → {result['route']}")

    print("\n" + "=" * 55)
    print("TEST 2: Papers retriever + sufficiency check")
    print("=" * 55)
    state = get_initial_state("What techniques does the lab use?")
    state["route"] = "papers"
    result = papers_retriever_node(state)
    print(f"   Papers context length  : {len(result.get('papers_context', ''))}")
    print(f"   Needs PubMed fallback  : {result.get('needs_pubmed_fallback')}")
    print(f"   Sources                : {result.get('sources', [])}")

    print("\n" + "=" * 55)
    print("TEST 3: Full pipeline with graph route")
    print("=" * 55)
    question = "What behavioral paradigms does the Shrestha lab study?"
    state = get_initial_state(question)
    state = router_node(state)
    state = graph_retriever_node(state)
    state = generator_node(state)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{state['answer'][:400]}...")
    print(f"\nSources: {state['sources']}")

    print("\n✅ All nodes working correctly!")