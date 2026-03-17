import sys
import os
import ssl
import certifi
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ssl._create_default_https_context = ssl.create_default_context
os.environ['SSL_CERT_FILE'] = certifi.where()

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import validate_config, API_HOST, API_PORT
from backend.api.routes import router
from backend.agent.graph import get_agent


# ── Lifespan — runs on startup and shutdown ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: validate config and pre-load the agent so the
    first request doesn't have a cold-start delay.
    Shutdown: clean up resources.
    """
    # Startup
    print("\n" + "=" * 55)
    print("  NeuroAssist API — Starting Up")
    print("=" * 55)

    try:
        validate_config()
    except EnvironmentError as e:
        print(f"\n❌ Config error:\n{e}")
        print("Please check your .env file and try again.")
        raise

    # Pre-load agent — loads embedding model + graph into memory
    print("\n⏳ Pre-loading agent (this takes ~10s on first run)...")
    get_agent()
    print("✅ Agent pre-loaded and ready\n")

    yield

    # Shutdown
    print("\n👋 NeuroAssist API shutting down")


# ── Create FastAPI app ────────────────────────────────────────────────────────

app = FastAPI(
    title="NeuroAssist API",
    description="""
    Intelligent research assistant for the Shrestha Lab at Stony Brook University.
    
    Combines:
    - **RAG over lab papers** — semantic search over published research
    - **RAG over pipeline code** — semantic search over the fiber photometry codebase  
    - **Knowledge Graph** — structured lab entity and relationship queries
    - **PubMed Search** — live external literature search
    
    All powered by a LangGraph agent with Llama 3.3 70B via Groq.
    """,
    version="1.0.0",
    lifespan=lifespan
)


# ── CORS middleware ───────────────────────────────────────────────────────────
# Allows the Streamlit frontend to call this API

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Register routes ───────────────────────────────────────────────────────────

app.include_router(router, prefix="/api/v1")


# ── Root endpoint ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "NeuroAssist API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
        "ask": "/api/v1/ask"
    }


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print(f"\n🚀 Starting NeuroAssist API on http://{API_HOST}:{API_PORT}")
    print(f"📖 API docs at http://localhost:{API_PORT}/docs\n")
    uvicorn.run(
        "backend.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )