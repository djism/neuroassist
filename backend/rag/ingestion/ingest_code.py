import os
import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(str(Path(__file__).resolve().parents[3]))
from config import (
    CODE_DIR,
    CODE_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from backend.rag.vectorstore import get_code_vectorstore, delete_collection


# File types we want to index
SUPPORTED_EXTENSIONS = {".py", ".md", ".txt", ".rst"}


def load_code_files(code_dir: Path) -> List[Document]:
    """
    Loads all supported code and documentation files
    from the code_files directory.
    Each file becomes one or more LangChain Documents with metadata.
    """
    all_documents = []

    # Find all supported files
    code_files = [
        f for f in code_dir.iterdir()
        if f.is_file() and f.suffix in SUPPORTED_EXTENSIONS
    ]

    if not code_files:
        raise FileNotFoundError(
            f"No supported files found in {code_dir}\n"
            f"Supported types: {SUPPORTED_EXTENSIONS}\n"
            "Please add pipeline .py files to data/code_files/"
        )

    print(f"📁 Found {len(code_files)} files\n")

    for file_path in sorted(code_files):
        print(f"   Loading: {file_path.name}")
        try:
            # Read raw text
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            if not content.strip():
                print(f"   ⚠️  Skipping {file_path.name} — file is empty")
                continue

            # Determine file type for metadata
            file_type = "python_code" if file_path.suffix == ".py" else "documentation"

            # Create document with rich metadata
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "source_type": file_type,
                    "file_path": str(file_path),
                    "extension": file_path.suffix,
                    "file_size_chars": len(content)
                }
            )

            all_documents.append(doc)
            print(f"   ✅ {len(content):,} characters loaded from {file_path.name}")

        except Exception as e:
            print(f"   ⚠️  Skipping {file_path.name}: {e}")

    print(f"\n💻 Total files loaded: {len(all_documents)}")
    return all_documents


def chunk_code_documents(documents: List[Document]) -> List[Document]:
    """
    Splits code and documentation files into chunks.
    Uses different separators optimized for code structure —
    splits at class/function boundaries before splitting mid-block.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\nclass ",       # split at class definitions first
            "\ndef ",         # then function definitions
            "\n\n",           # then blank lines
            "\n",             # then single newlines
            " ",              # then spaces
            ""                # last resort: split anywhere
        ],
        length_function=len
    )

    chunks = splitter.split_documents(documents)

    # Preserve source filename in every chunk metadata
    for chunk in chunks:
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "unknown"

    print(f"✂️  Split into {len(chunks)} chunks")
    print(f"   Chunk size    : {CHUNK_SIZE} characters")
    print(f"   Chunk overlap : {CHUNK_OVERLAP} characters")
    return chunks


def ingest_code(reset: bool = False) -> int:
    """
    Full ingestion pipeline for pipeline code files.

    Args:
        reset: If True, deletes existing collection before ingesting.
               Use this when you add new code files.

    Returns:
        Number of chunks ingested.
    """
    print("=" * 55)
    print("  NeuroAssist — Pipeline Code Ingestion")
    print("=" * 55)

    # Optionally reset collection
    if reset:
        print("\n🔄 Resetting code collection...")
        delete_collection(CODE_COLLECTION)

    # Step 1: Load code files
    print("\n📂 Step 1: Loading code files...")
    documents = load_code_files(CODE_DIR)

    if not documents:
        print("❌ No documents loaded. Exiting.")
        return 0

    # Step 2: Chunk documents
    print("\n✂️  Step 2: Chunking code files...")
    chunks = chunk_code_documents(documents)

    # Step 3: Embed and store in ChromaDB
    print("\n🧠 Step 3: Embedding and storing in ChromaDB...")
    print("   This may take a few minutes on first run...")

    vectorstore = get_code_vectorstore()

    # Add in batches to avoid memory issues
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"   Batch {batch_num}/{total_batches} — {len(batch)} chunks")
        vectorstore.add_documents(batch)

    print(f"\n✅ Ingestion complete!")
    print(f"   {len(chunks)} chunks stored in collection: {CODE_COLLECTION}")

    # Quick verification
    print("\n🔍 Verification — running test query...")
    results = vectorstore.similarity_search(
        "How does the SAA pipeline process data?", k=2
    )
    print(f"   Test query returned {len(results)} results")
    if results:
        preview = results[0].page_content[:150].replace("\n", " ")
        source = results[0].metadata.get("source", "unknown")
        print(f"   Top result from : {source}")
        print(f"   Preview         : {preview}...")

    return len(chunks)


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    ingest_code(reset=reset)