# 🧠 NeuroAssist
### Hybrid GraphRAG Research Agent for Neuroscience Labs

<p align="center">
  <img src="https://img.shields.io/badge/LangGraph-Agent-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/NetworkX-Knowledge_Graph-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Groq-Llama_3.3_70B-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/CrossEncoder-Reranking-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-teal?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub_Actions-black?style=for-the-badge&logo=githubactions" />
</p>

<p align="center">
  <a href="https://github.com/djism/neuroassist/actions/workflows/ci.yml">
    <img src="https://github.com/djism/neuroassist/actions/workflows/ci.yml/badge.svg" alt="NeuroAssist CI" />
  </a>
</p>

---

## What Is This?

NeuroAssist is a **production-grade agentic RAG system** built for the [Shrestha Lab](https://www.shresthalab.org/research) at SUNY Stony Brook (Dept. of Neurobiology & Behavior).

The lab uses a complex fiber photometry analysis pipeline to study dopamine signaling in fear learning and active avoidance behavior. New students and collaborators waste days trying to understand the codebase and research context — NeuroAssist solves this with a single intelligent interface.

**Ask it anything about the lab:**
> *"What does this lab study?"* → answers from knowledge graph
> *"How does the snippet extractor work?"* → answers from codebase
> *"What did the lab find about dopamine in SAA?"* → answers from published papers
> *"Find recent papers on active avoidance"* → live PubMed search

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Question                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Agent Router                       │
│         Classifies intent → selects retrieval strategy         │
└──────┬──────────┬───────────────┬──────────────┬───────────────┘
       │          │               │              │
       ▼          ▼               ▼              ▼
  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────────┐
  │ Papers  │ │  Code   │ │  Graph   │ │   PubMed     │
  │   RAG   │ │   RAG   │ │ Retrieval│ │  Live Search │
  │ChromaDB │ │ChromaDB │ │NetworkX  │ │  NCBI API    │
  └────┬────┘ └────┬────┘ └────┬─────┘ └──────┬───────┘
       │           │                          │
       ▼           ▼                          │
  ┌─────────────────────┐                     │
  │  CrossEncoder       │                     │
  │  Reranker           │                     │
  │  (sufficiency check)│                     │
  └────────┬────────────┘                     │
           │                                  │
    sufficient?                               │
     NO ──────────────────────────────────────┘
           │                    (automatic PubMed fallback)
          YES
           │
           └───────────────────────┐
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Generator Node                               │
│         Groq Llama 3.3 70B — grounded answer synthesis         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAGAS Evaluator                              │
│            Faithfulness scoring — hallucination detection       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                    Answer + Sources + Score
```

---

## What Makes This a Hybrid GraphRAG System

Most RAG systems use **only vector search** — finding chunks by semantic similarity. NeuroAssist combines three fundamentally different retrieval strategies:

| Strategy | Technology | Best For |
|---|---|---|
| **Semantic search** | ChromaDB + BAAI embeddings | "What did the lab find about dopamine?" |
| **Logic-based traversal** | NetworkX Knowledge Graph | "What files does saa_mouse.py depend on?" |
| **Two-stage reranking** | CrossEncoder ms-marco-MiniLM | Scoring true relevance, not just similarity |
| **Live search** | PubMed NCBI API | "Find recent papers on active avoidance" |
| **Automatic fallback** | Reranker sufficiency check | When local knowledge is insufficient |

The LangGraph agent **intelligently routes** each question to the right strategy — and automatically falls back to PubMed when the reranker determines local knowledge is insufficient.

---

## Two-Stage Retrieval With Reranking

Standard RAG retrieves by embedding similarity — which finds semantically *close* chunks, not necessarily *relevant* ones. NeuroAssist uses a two-stage approach:

```
Stage 1 — ChromaDB vector search:
  Embed question → find top 20 similar chunks (cast wide net)

Stage 2 — CrossEncoder reranking:
  Score each (question, chunk) pair together → keep top 5 by true relevance

Sufficiency check:
  If no chunk scores above threshold → trigger automatic PubMed fallback
```

The CrossEncoder reads the question and chunk *together* — unlike embeddings which encode them separately. This catches relevance that pure similarity search misses.

---

## Knowledge Base

| Source | Content | Chunks |
|---|---|---|
| 10 published lab papers | Neuroscience findings, methods, results | ~2,000 |
| Prof. CV + lab intro + research summary | PI background, lab overview | ~50 |
| 24 pipeline Python files | Full fiber photometry codebase | ~750 |
| Knowledge Graph | 45+ nodes, 60+ relationships | Structured |

---

## Key Features

- **Multi-tool LangGraph Agent** — routes questions across 5 specialized tools with reasoning between each step
- **Hybrid GraphRAG** — combines ChromaDB semantic retrieval with NetworkX logic-based knowledge graph traversal (inspired by Microsoft GraphRAG, 2024)
- **Two-Stage Reranking** — CrossEncoder `ms-marco-MiniLM-L-6-v2` reranks retrieved chunks for true relevance over surface similarity
- **Automatic PubMed Fallback** — agent detects when local knowledge is insufficient and autonomously fetches live papers from NCBI
- **RAGAS Evaluation** — every answer scored for faithfulness using RAGAS framework (consistently 0.9–1.0 on code and paper questions)
- **Source Citations** — every answer cites exact paper names or file names
- **Production REST API** — FastAPI with Swagger docs, health checks, and versioned endpoints
- **Fully Containerized** — Docker + docker-compose, one command to run everything
- **CI/CD Pipeline** — GitHub Actions: lint checks + 17 unit tests + Docker build verification on every push

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Agent Orchestration** | LangGraph |
| **LLM** | Llama 3.3 70B via Groq API (free) |
| **Embeddings** | BAAI/bge-small-en-v1.5 (local, free) |
| **Vector Store** | ChromaDB (2 collections) |
| **Reranker** | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| **Knowledge Graph** | NetworkX |
| **RAG Framework** | LangChain |
| **Evaluation** | RAGAS (faithfulness scoring) |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Containerization** | Docker + docker-compose |
| **CI/CD** | GitHub Actions |
| **External API** | NCBI PubMed via Biopython |

**Total cost to run: $0** — Groq free tier, local embeddings, open source everything.

---

## Project Structure

```
neuroassist/
├── backend/
│   ├── agent/
│   │   ├── graph.py               # LangGraph agent — nodes, edges, fallback routing
│   │   ├── nodes.py               # Router, retrievers, fallback, generator, evaluator
│   │   └── state.py               # AgentState schema
│   ├── rag/
│   │   ├── embeddings.py          # Local BAAI embedding model
│   │   ├── vectorstore.py         # ChromaDB — 2 collections
│   │   ├── reranker.py            # CrossEncoder two-stage reranking + sufficiency check
│   │   ├── retriever.py           # Unified retriever interface
│   │   ├── ingestion/
│   │   │   ├── ingest_papers.py   # PDF → chunks → ChromaDB
│   │   │   └── ingest_code.py     # .py files → chunks → ChromaDB
│   │   └── knowledge_graph/
│   │       ├── graph_builder.py   # NetworkX graph construction
│   │       └── graph_retriever.py # Logic-based graph queries
│   ├── tools/
│   │   └── pubmed.py              # NCBI PubMed live search
│   └── api/
│       ├── main.py                # FastAPI app + lifespan
│       ├── routes.py              # /ask /health /stats /examples
│       └── schemas.py             # Pydantic request/response models
├── frontend/
│   └── app.py                     # Streamlit UI
├── scripts/
│   └── run_ingestion.py           # One command to build knowledge base
├── tests/
│   └── test_agent.py              # 17 unit tests
├── config.py                      # Centralized configuration
├── docker-compose.yml             # Run everything with one command
├── Dockerfile.backend
└── Dockerfile.frontend
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- Docker Desktop
- [Groq API key](https://console.groq.com) (free)
- [NCBI API key](https://ncbi.nlm.nih.gov/account) (free)

### 1. Clone and setup

```bash
git clone https://github.com/djism/neuroassist.git
cd neuroassist

python -m venv neuroassist-env
source neuroassist-env/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Add your data

```
data/papers/      ← Add lab PDFs here
data/code_files/  ← Add pipeline .py files here
```

### 4. Build the knowledge base

```bash
python scripts/run_ingestion.py
python backend/rag/knowledge_graph/graph_builder.py
```

### 5. Run with Docker

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |

---

## API Reference

### `POST /api/v1/ask`

```json
// Request
{
  "question": "How does the snippet extractor work?"
}

// Response
{
  "answer": "The snippet extractor (snippet_extractor.py) extracts peri-event time snippets...",
  "sources": ["snippet_extractor.py", "saa_mouse.py"],
  "route": "code",
  "faithfulness_score": 1.00,
  "eval_error": null
}
```

### `GET /api/v1/health`

```json
{
  "status": "healthy",
  "collections": {
    "lab_papers":    {"count": 2024, "status": "ready"},
    "pipeline_code": {"count": 749,  "status": "ready"}
  },
  "message": "NeuroAssist is ready!"
}
```

---

## Example Questions

**Lab Overview**
- *"What does the Shrestha lab study?"*
- *"What techniques does the lab use?"*
- *"Who leads this lab and what is their background?"*
- *"What brain regions does the lab focus on?"*

**Published Findings**
- *"What did the lab find about dopamine signaling?"*
- *"What is the SAA paradigm?"*
- *"What is fiber photometry used for in this lab?"*
- *"What is dLight1.3b?"*

**Pipeline Code**
- *"How does the snippet extractor work?"*
- *"What does saa_mouse.py depend on?"*
- *"Where is z-score normalization implemented?"*
- *"What does base_mouse.py do?"*

**Live PubMed Search**
- *"Find recent papers on dopamine and active avoidance"*
- *"Search for fiber photometry in fear conditioning"*
- *"Find latest research on VTA dopamine neurons"*

---

## Why I Built This

I joined the Shrestha Lab as a Senior Research Aide with a data science background but no neuroscience experience. Understanding both the research context (10 published papers) and the codebase (24 Python files, 6 pipeline components) simultaneously took weeks.

NeuroAssist compresses that onboarding from weeks to minutes — for me, for future students, and for any collaborating lab that wants to use or understand the pipeline.

---

## Author

**Dhananjay Sharma**
M.S. Data Science, SUNY Stony Brook (May 2026)

<p>
  <a href="https://www.linkedin.com/in/dsharma2496/">LinkedIn</a> ·
  <a href="https://djism.github.io/">Portfolio</a> ·
  <a href="https://github.com/djism">GitHub</a>
</p>

---

<p align="center">Built with ❤️ for the <a href="https://www.shresthalab.org/research">Shrestha Lab</a> · SUNY Stony Brook · Dept. of Neurobiology & Behavior</p>