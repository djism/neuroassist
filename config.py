import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
CODE_DIR = DATA_DIR / "code_files"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db"))
PAPERS_COLLECTION = os.getenv("PAPERS_COLLECTION", "lab_papers")
CODE_COLLECTION = os.getenv("CODE_COLLECTION", "pipeline_code")

# ── Embedding Model (runs locally, free) ─────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# ── LLM (Groq, free) ─────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# ── PubMed ────────────────────────────────────────────────────────────────────
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
PUBMED_MAX_RESULTS = 5  # max papers to fetch per PubMed search

# ── RAG Chunking ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 512        # characters per chunk
CHUNK_OVERLAP = 64      # overlap between chunks to preserve context

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS = 5       # how many chunks to retrieve per query

# ── RAGAS Evaluation ─────────────────────────────────────────────────────────
RAGAS_ENABLED = True    # set to False to skip evaluation (faster responses)

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Validation ───────────────────────────────────────────────────────────────
def validate_config():
    """Call this on startup to catch missing env vars early."""
    errors = []
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is missing from .env")
    if not NCBI_API_KEY:
        errors.append("NCBI_API_KEY is missing from .env")
    if not PAPERS_DIR.exists():
        errors.append(f"Papers directory not found: {PAPERS_DIR}")
    if not CODE_DIR.exists():
        errors.append(f"Code directory not found: {CODE_DIR}")
    if errors:
        raise EnvironmentError("\n".join(errors))
    print("✅ Config validated successfully")
    print(f"   LLM        : {LLM_MODEL}")
    print(f"   Embeddings : {EMBEDDING_MODEL}")
    print(f"   Papers dir : {PAPERS_DIR}")
    print(f"   Code dir   : {CODE_DIR}")
    print(f"   ChromaDB   : {CHROMA_PERSIST_DIR}")

if __name__ == "__main__":
    validate_config()