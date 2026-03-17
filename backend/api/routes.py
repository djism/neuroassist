import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

sys.path.append(str(Path(__file__).resolve().parents[2]))
from backend.api.schemas import QuestionRequest, AnswerResponse, HealthResponse
from backend.agent.graph import get_agent
from backend.rag.vectorstore import get_collection_stats

router = APIRouter()


# ── Health check ──────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns status of ChromaDB collections and agent readiness.
    Call this first to verify everything is running.
    """
    try:
        stats = get_collection_stats()
        all_ready = all(info["status"] == "ready" for info in stats.values())

        return HealthResponse(
            status="healthy" if all_ready else "degraded",
            collections=stats,
            message=(
                "NeuroAssist is ready!" if all_ready
                else "Some collections are empty. Run scripts/run_ingestion.py first."
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# ── Ask endpoint ──────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint — ask NeuroAssist a question.

    The agent will:
    1. Route the question to the right retriever
    2. Retrieve relevant context
    3. Generate a grounded answer with citations
    4. Score the answer with RAGAS

    Example questions:
    - "What does the Shrestha lab study?"
    - "How does saa_mouse.py work?"
    - "What did the lab find about dopamine?"
    - "Find recent papers on active avoidance"
    """
    try:
        agent = get_agent()
        response = agent.ask(request.question)

        return AnswerResponse(
            answer=response["answer"],
            sources=response["sources"],
            route=response["route"],
            faithfulness_score=response.get("faithfulness_score"),
            eval_error=response.get("eval_error")
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )


# ── Collections stats endpoint ────────────────────────────────────────────────

@router.get("/stats")
async def get_stats():
    """
    Returns detailed stats about the knowledge base.
    Shows chunk counts for papers and code collections.
    """
    try:
        stats = get_collection_stats()
        return JSONResponse(content={
            "collections": stats,
            "total_chunks": sum(info["count"] for info in stats.values())
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Example questions endpoint ────────────────────────────────────────────────

@router.get("/examples")
async def get_example_questions():
    """
    Returns example questions to help users get started.
    Displayed in the Streamlit UI on first load.
    """
    return JSONResponse(content={
        "examples": {
            "lab_research": [
                "What does the Shrestha lab study?",
                "What techniques does the lab use?",
                "What brain regions does the lab focus on?",
                "Who leads the Shrestha lab?",
                "What behavioral paradigms does the lab study?"
            ],
            "papers": [
                "What did the lab find about dopamine signaling?",
                "What is fiber photometry used for in this lab?",
                "What is Pavlovian Threat Conditioning?",
                "What is the SAA paradigm?",
                "What is dLight1.3b?"
            ],
            "pipeline_code": [
                "How does the snippet extractor work?",
                "What does saa_mouse.py do?",
                "What files does the pipeline consist of?",
                "What does base_mouse.py do?",
                "How is z-score normalization implemented?"
            ],
            "pubmed": [
                "Find recent papers on active avoidance and dopamine",
                "Search for papers on fiber photometry in fear conditioning",
                "Find latest research on VTA dopamine neurons"
            ]
        }
    })