import os
import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

sys.path.append(str(Path(__file__).resolve().parents[3]))
from config import (
    PAPERS_DIR,
    PAPERS_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from backend.rag.vectorstore import get_papers_vectorstore, delete_collection


def load_pdfs(papers_dir: Path) -> List[Document]:
    """
    Loads all PDF files from the papers directory.
    Each page becomes a LangChain Document with metadata.
    """
    all_documents = []
    pdf_files = list(papers_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {papers_dir}\n"
            "Please add lab papers to data/papers/"
        )

    print(f"📄 Found {len(pdf_files)} PDF files\n")

    for pdf_path in pdf_files:
        print(f"   Loading: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # Add clean metadata to every page
            for page in pages:
                page.metadata.update({
                    "source": pdf_path.name,
                    "source_type": "lab_paper",
                    "file_path": str(pdf_path)
                })

            all_documents.extend(pages)
            print(f"   ✅ {len(pages)} pages loaded from {pdf_path.name}")

        except Exception as e:
            print(f"   ⚠️  Skipping {pdf_path.name}: {e}")

    print(f"\n📚 Total pages loaded: {len(all_documents)}")
    return all_documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into smaller chunks for better retrieval.
    RecursiveCharacterTextSplitter tries to split at natural boundaries
    like paragraphs and sentences before splitting mid-sentence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunks = splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks")
    print(f"   Chunk size    : {CHUNK_SIZE} characters")
    print(f"   Chunk overlap : {CHUNK_OVERLAP} characters")
    return chunks


def ingest_papers(reset: bool = False) -> int:
    """
    Full ingestion pipeline for lab papers.
    
    Args:
        reset: If True, deletes existing collection before ingesting.
               Use this when you add new papers.
    
    Returns:
        Number of chunks ingested.
    """
    print("=" * 55)
    print("  NeuroAssist — Lab Papers Ingestion")
    print("=" * 55)

    # Optionally reset collection
    if reset:
        print("\n🔄 Resetting papers collection...")
        delete_collection(PAPERS_COLLECTION)

    # Step 1: Load PDFs
    print("\n📂 Step 1: Loading PDFs...")
    documents = load_pdfs(PAPERS_DIR)

    if not documents:
        print("❌ No documents loaded. Exiting.")
        return 0

    # Step 2: Chunk documents
    print("\n✂️  Step 2: Chunking documents...")
    chunks = chunk_documents(documents)

    # Step 3: Embed and store in ChromaDB
    print("\n🧠 Step 3: Embedding and storing in ChromaDB...")
    print("   This may take a few minutes on first run...")

    vectorstore = get_papers_vectorstore()

    # Add in batches to avoid memory issues
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"   Batch {batch_num}/{total_batches} — {len(batch)} chunks")
        vectorstore.add_documents(batch)

    print(f"\n✅ Ingestion complete!")
    print(f"   {len(chunks)} chunks stored in collection: {PAPERS_COLLECTION}")

    # Quick verification
    print("\n🔍 Verification — running test query...")
    results = vectorstore.similarity_search(
        "What does the Shrestha lab study?", k=2
    )
    print(f"   Test query returned {len(results)} results")
    if results:
        preview = results[0].page_content[:150].replace("\n", " ")
        print(f"   Top result preview: {preview}...")

    return len(chunks)


if __name__ == "__main__":
    # Pass --reset flag to wipe and re-ingest
    reset = "--reset" in sys.argv
    ingest_papers(reset=reset)