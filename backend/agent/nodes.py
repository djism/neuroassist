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
from config import GROQ_API_KEY, LLM_MODEL
from backend.agent.state import AgentState
from backend.rag.retriever import get_retriever
from backend.tools.pubmed import search_and_format


# ── LLM setup ─────────────────────────────────────────────────────────────────
def get_llm() -> ChatGroq:
    """Returns Groq LLM instance."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=0.2,       # low temperature = more factual, less creative
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


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1: ROUTER
# Decides which tools to use based on the question
# ══════════════════════════════════════════════════════════════════════════════

def router_node(state: AgentState) -> AgentState:
    """
    Analyzes the question and decides which retrieval path to take.

    Routes:
    - "papers"          → question about lab research, findings, methods
    - "code"            → question about pipeline code, implementation
    - "graph"           → question about lab overview, relationships, structure
    - "pubmed"          → question needs external literature beyond lab papers
    - "papers_and_code" → question touches both research and implementation
    """
    question = state["question"].lower()

    print(f"\n🔀 Router analyzing: '{state['question'][:60]}...'")

    # Code-related keywords
    code_keywords = [
        "code", "file", "function", "implement", "pipeline", "script",
        "how does", "how do", "where is", "which file", "class", "method",
        "parameter", "argument", "import", "module", "run", "execute",
        "saa_mouse", "base_mouse", "snippet", "peth", "heatmap", "excel",
        "preprocessing", "loader", "parser", "scanner", "plotter"
    ]

    # Graph/overview keywords
    graph_keywords = [
    "what does the lab", "what does this lab", "lab do",
    "overview", "summary", "about the lab", "who is", "professor",
    "techniques used", "paradigms", "brain regions", "what files",
    "all files", "dependencies", "depends on", "structure",
    "list all", "what is the lab", "shrestha lab",
    "what does the shrestha", "what does prof",
    "about this lab", "tell me about the lab"
    ]

    # PubMed keywords — needs external literature
    pubmed_keywords = [
        "recent papers", "latest research", "find papers", "search pubmed",
        "published recently", "other labs", "literature", "not in the lab",
        "general research", "field of", "state of the art"
    ]

    # Determine route
    is_code = any(kw in question for kw in code_keywords)
    is_graph = any(kw in question for kw in graph_keywords)
    is_pubmed = any(kw in question for kw in pubmed_keywords)

    if is_graph:
        route = "graph"
    elif is_code and not is_graph:
        # Check if it also needs paper context
        paper_keywords = ["paper", "study", "finding", "result", "publish"]
        also_papers = any(kw in question for kw in paper_keywords)
        route = "papers_and_code" if also_papers else "code"
    elif is_pubmed:
        route = "pubmed"
    else:
        route = "papers"  # default — most questions are about research

    print(f"   ✅ Route decided: {route}")
    return {**state, "route": route}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2: PAPERS RETRIEVER
# Fetches context from lab papers vector store
# ══════════════════════════════════════════════════════════════════════════════

def papers_retriever_node(state: AgentState) -> AgentState:
    """
    Retrieves relevant chunks from the lab papers ChromaDB collection.
    Also adds graph context for richer answers.
    """
    print(f"\n📚 Retrieving from lab papers...")
    retriever = get_retriever()

    # Vector search over papers
    results = retriever.search_papers(state["question"], k=5)
    papers_context = retriever.format_vector_results(results)

    # Also get graph context for related concepts
    graph_context = retriever.search_graph(state["question"].split()[0])

    # Extract source names
    sources = list({
        r["metadata"].get("source", "unknown")
        for r in results
    })

    print(f"   ✅ Retrieved {len(results)} paper chunks")
    print(f"   Sources: {sources}")

    return {
        **state,
        "papers_context": papers_context,
        "graph_context": graph_context,
        "sources": sources
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3: CODE RETRIEVER
# Fetches context from pipeline code vector store
# ══════════════════════════════════════════════════════════════════════════════

def code_retriever_node(state: AgentState) -> AgentState:
    """
    Retrieves relevant chunks from the pipeline code ChromaDB collection.
    Also fetches file dependency information from knowledge graph.
    """
    print(f"\n💻 Retrieving from pipeline code...")
    retriever = get_retriever()

    # Vector search over code
    results = retriever.search_code(state["question"], k=5)
    code_context = retriever.format_vector_results(results)

    # Get file dependency graph context
    # Try to find a filename mentioned in the question
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

    # If no specific file mentioned, get all files list
    if not graph_context:
        graph_context = retriever.get_all_code_files()

    # Extract source names
    sources = list({
        r["metadata"].get("source", "unknown")
        for r in results
    })

    print(f"   ✅ Retrieved {len(results)} code chunks")
    print(f"   Sources: {sources}")

    return {
        **state,
        "code_context": code_context,
        "graph_context": graph_context,
        "sources": sources
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4: GRAPH RETRIEVER
# Fetches structured info from knowledge graph
# ══════════════════════════════════════════════════════════════════════════════

def graph_retriever_node(state: AgentState) -> AgentState:
    """
    Retrieves structured information from the NetworkX knowledge graph.
    Best for overview, relationship, and structural questions.
    """
    print(f"\n🧠 Retrieving from knowledge graph...")
    retriever = get_retriever()

    question_lower = state["question"].lower()

    # Decide what to fetch from graph
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
        # Try to find filename in question
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
        # Generic keyword search — skip short/common words
        stopwords = {"what", "does", "the", "is", "how", "which", "who", "does", "do", "a", "an"}
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
        "sources": sources
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5: PAPERS AND CODE RETRIEVER
# Fetches from both paper and code collections
# ══════════════════════════════════════════════════════════════════════════════

def papers_and_code_retriever_node(state: AgentState) -> AgentState:
    """
    Retrieves from both papers and code collections.
    Used when question spans both research and implementation.
    """
    print(f"\n📚💻 Retrieving from papers AND code...")
    retriever = get_retriever()

    # Search both
    paper_results = retriever.search_papers(state["question"], k=3)
    code_results = retriever.search_code(state["question"], k=3)

    papers_context = retriever.format_vector_results(paper_results)
    code_context = retriever.format_vector_results(code_results)

    sources = list({
        r["metadata"].get("source", "unknown")
        for r in paper_results + code_results
    })

    print(f"   ✅ {len(paper_results)} paper chunks + {len(code_results)} code chunks")

    return {
        **state,
        "papers_context": papers_context,
        "code_context": code_context,
        "sources": sources
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6: PUBMED RETRIEVER
# Fetches live papers from PubMed API
# ══════════════════════════════════════════════════════════════════════════════

def pubmed_retriever_node(state: AgentState) -> AgentState:
    """
    Searches PubMed for papers related to the question.
    Also searches lab papers as supplementary context.
    """
    print(f"\n🔬 Searching PubMed...")
    retriever = get_retriever()

    # Search PubMed
    pubmed_context = search_and_format(state["question"], max_results=3)

    # Also get some lab paper context for comparison
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
        "sources": sources
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 7: GENERATOR
# Builds context and calls the LLM to generate the answer
# ══════════════════════════════════════════════════════════════════════════════

def generator_node(state: AgentState) -> AgentState:
    """
    Assembles all retrieved context and calls the LLM to generate
    a grounded, cited answer.
    """
    print(f"\n✍️  Generating answer...")
    llm = get_llm()

    # Build context block from whatever was retrieved
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
        context_parts.append("PUBMED SEARCH RESULTS:")
        context_parts.append(state["pubmed_context"])
        context_parts.append("")

    context = "\n".join(context_parts)

    if not context.strip():
        context = "No context was retrieved for this question."

    # Build the prompt
    user_prompt = f"""Based on the following context, please answer this question:

QUESTION: {state["question"]}

CONTEXT:
{context}

Please provide a clear, accurate answer based only on the context above.
Cite the specific sources (paper names or file names) that support your answer.
If the context does not contain enough information to answer fully, say so."""

    # Call LLM
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    answer = response.content

    print(f"   ✅ Answer generated ({len(answer)} chars)")

    # Update message history
    updated_messages = list(state.get("messages", []))
    updated_messages.append(HumanMessage(content=state["question"]))
    updated_messages.append(AIMessage(content=answer))

    return {
        **state,
        "answer": answer,
        "messages": updated_messages
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 8: EVALUATOR
# Scores the answer using RAGAS metrics
# ══════════════════════════════════════════════════════════════════════════════

def evaluator_node(state: AgentState) -> AgentState:
    """
    Evaluates the generated answer using RAGAS faithfulness metric.
    Faithfulness measures whether the answer is grounded in the context.
    Score of 1.0 = fully faithful, 0.0 = hallucinated.
    """
    print(f"\n📊 Evaluating answer quality...")

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness
        from datasets import Dataset
        from langchain_groq import ChatGroq
        from langchain_core.embeddings import Embeddings

        # Build context list
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

        # Prepare dataset for RAGAS
        eval_data = {
            "question": [state["question"]],
            "answer": [state["answer"]],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(eval_data)

        # Use Groq LLM for evaluation
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from backend.rag.embeddings import get_embeddings

        groq_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=0
        )
        ragas_llm = LangchainLLMWrapper(groq_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )

        score = float(result["faithfulness"])
        print(f"   ✅ Faithfulness score: {score:.2f}")

        return {
            **state,
            "faithfulness_score": score
        }

    except Exception as e:
        # Evaluation failure should never break the answer
        print(f"   ⚠️  Evaluation skipped: {e}")
        return {
            **state,
            "faithfulness_score": None,
            "eval_error": str(e)
        }


if __name__ == "__main__":
    from backend.agent.state import get_initial_state

    print("Testing individual nodes...\n")

    # Test router
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
        print(f"   {status} '{q[:45]}' → {result['route']} (expected: {expected})")

    # Test papers retriever
    print("\n" + "=" * 55)
    print("TEST 2: Papers retriever")
    print("=" * 55)
    state = get_initial_state("What techniques does the lab use?")
    state["route"] = "papers"
    result = papers_retriever_node(state)
    print(f"   Papers context length : {len(result.get('papers_context', ''))}")
    print(f"   Sources               : {result.get('sources', [])}")

    # Test generator
    print("\n" + "=" * 55)
    print("TEST 3: Full pipeline — router → retriever → generator")
    print("=" * 55)
    question = "What behavioral paradigms does the Shrestha lab study?"
    state = get_initial_state(question)
    state = router_node(state)
    state = graph_retriever_node(state)
    state = generator_node(state)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{state['answer']}")
    print(f"\nSources: {state['sources']}")

    print("\n✅ Nodes working correctly!")