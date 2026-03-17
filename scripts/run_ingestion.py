import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import validate_config
from backend.rag.ingestion.ingest_papers import ingest_papers
from backend.rag.ingestion.ingest_code import ingest_code
from backend.rag.vectorstore import get_collection_stats


def run_full_ingestion(reset: bool = False):
    """
    Master ingestion script — runs both papers and code ingestion.
    Run this once before starting the app.
    Run again with --reset flag whenever you add new files.
    """

    print("\n" + "=" * 55)
    print("  NeuroAssist — Full Knowledge Base Ingestion")
    print("=" * 55)

    if reset:
        print("\n⚠️  RESET MODE — existing collections will be wiped")
    else:
        print("\n📌 TIP: Run with --reset to wipe and re-ingest everything")

    # ── Step 0: Validate config ───────────────────────────────────────────────
    print("\n🔧 Validating config...")
    try:
        validate_config()
    except EnvironmentError as e:
        print(f"\n❌ Config error:\n{e}")
        sys.exit(1)

    total_start = time.time()

    # ── Step 1: Ingest lab papers ─────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("  PHASE 1: Lab Papers")
    print("─" * 55)
    papers_start = time.time()

    try:
        papers_chunks = ingest_papers(reset=reset)
        papers_time = round(time.time() - papers_start, 1)
        print(f"\n⏱️  Papers ingestion took {papers_time}s")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Papers ingestion failed: {e}")
        sys.exit(1)

    # ── Step 2: Ingest pipeline code ──────────────────────────────────────────
    print("\n" + "─" * 55)
    print("  PHASE 2: Pipeline Code")
    print("─" * 55)
    code_start = time.time()

    try:
        code_chunks = ingest_code(reset=reset)
        code_time = round(time.time() - code_start, 1)
        print(f"\n⏱️  Code ingestion took {code_time}s")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Code ingestion failed: {e}")
        sys.exit(1)

    # ── Step 3: Final summary ─────────────────────────────────────────────────
    total_time = round(time.time() - total_start, 1)

    print("\n" + "=" * 55)
    print("  INGESTION COMPLETE")
    print("=" * 55)

    print("\n📊 Final Collection Stats:")
    stats = get_collection_stats()
    for collection, info in stats.items():
        status_icon = "✅" if info["status"] == "ready" else "⚠️"
        print(f"   {status_icon} {collection}")
        print(f"      Chunks : {info['count']}")
        print(f"      Status : {info['status']}")

    print(f"\n📦 Total chunks ingested:")
    print(f"   Papers : {papers_chunks}")
    print(f"   Code   : {code_chunks}")
    print(f"   Total  : {papers_chunks + code_chunks}")
    print(f"\n⏱️  Total time : {total_time}s")

    print("\n🚀 Knowledge base is ready!")
    print("   Next step: run the knowledge graph builder")
    print("   python backend/rag/knowledge_graph/graph_builder.py")
    print()


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    run_full_ingestion(reset=reset)