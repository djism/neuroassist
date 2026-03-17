import sys
import os
import ssl
import certifi
import time
from pathlib import Path
from typing import Optional
from Bio import Entrez, Medline

# Fix for Mac SSL certificate verification
ssl._create_default_https_context = ssl.create_default_context
os.environ['SSL_CERT_FILE'] = certifi.where()

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import NCBI_API_KEY, PUBMED_MAX_RESULTS

# Required by NCBI — identify your application
Entrez.email = "neuroassist@research.com"
Entrez.api_key = NCBI_API_KEY


def search_pubmed(query: str, max_results: int = PUBMED_MAX_RESULTS) -> list[dict]:
    """
    Searches PubMed for papers matching the query.
    Returns a list of paper dicts with title, abstract, authors, year.

    Args:
        query: Natural language or keyword search query
        max_results: Maximum number of papers to return

    Returns:
        List of dicts with paper metadata and abstract
    """
    try:
        # Step 1: Search for matching paper IDs
        print(f"   🔍 Searching PubMed: '{query}'")
        search_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance"
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results.get("IdList", [])

        if not id_list:
            print(f"   ℹ️  No PubMed results found for: '{query}'")
            return []

        print(f"   📄 Found {len(id_list)} papers — fetching details...")

        # Small delay to respect NCBI rate limits
        time.sleep(0.5)

        # Step 2: Fetch full details for those IDs
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=id_list,
            rettype="medline",
            retmode="text"
        )
        records = list(Medline.parse(fetch_handle))
        fetch_handle.close()

        # Step 3: Parse into clean dicts
        papers = []
        for record in records:
            paper = _parse_medline_record(record)
            if paper:
                papers.append(paper)

        print(f"   ✅ Retrieved {len(papers)} papers from PubMed")
        return papers

    except Exception as e:
        print(f"   ⚠️  PubMed search error: {e}")
        return []


def _parse_medline_record(record: dict) -> Optional[dict]:
    """
    Parses a raw Medline record into a clean dict.
    Handles missing fields gracefully.
    """
    try:
        # Extract authors
        authors = record.get("AU", [])
        authors_str = ", ".join(authors[:3])  # first 3 authors
        if len(authors) > 3:
            authors_str += " et al."

        # Extract year from date
        pub_date = record.get("DP", "")
        year = pub_date[:4] if pub_date else "Unknown"

        # Extract abstract
        abstract = record.get("AB", "")
        if not abstract:
            abstract = "Abstract not available."

        # Build clean paper dict
        paper = {
            "pmid": record.get("PMID", ""),
            "title": record.get("TI", "Title not available"),
            "authors": authors_str,
            "year": year,
            "journal": record.get("JT", record.get("TA", "Unknown journal")),
            "abstract": abstract,
            "doi": record.get("LID", ""),
            "source": "PubMed",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{record.get('PMID', '')}/"
        }

        return paper

    except Exception:
        return None


def format_papers_for_llm(papers: list[dict]) -> str:
    """
    Formats PubMed results as clean text context for the LLM.
    Each paper gets a numbered block with title, authors, year, abstract.
    """
    if not papers:
        return "No relevant papers found on PubMed."

    lines = ["PUBMED SEARCH RESULTS:", "=" * 40]

    for i, paper in enumerate(papers, 1):
        lines.append(f"\n[Paper {i}]")
        lines.append(f"Title   : {paper['title']}")
        lines.append(f"Authors : {paper['authors']}")
        lines.append(f"Year    : {paper['year']}")
        lines.append(f"Journal : {paper['journal']}")
        lines.append(f"URL     : {paper['url']}")
        lines.append(f"Abstract: {paper['abstract'][:500]}...")
        lines.append("-" * 40)

    return "\n".join(lines)


def search_and_format(query: str, max_results: int = PUBMED_MAX_RESULTS) -> str:
    """
    Convenience function — searches PubMed and returns
    formatted text ready to pass to the LLM as context.
    This is what the LangGraph agent tool calls directly.
    """
    papers = search_pubmed(query, max_results)
    return format_papers_for_llm(papers)


if __name__ == "__main__":
    print("Testing PubMed tool...\n")

    # Test 1: Basic search
    print("=" * 55)
    print("TEST 1: Search for fiber photometry papers")
    print("=" * 55)
    papers = search_pubmed("fiber photometry dopamine fear conditioning", max_results=3)

    if papers:
        for i, p in enumerate(papers, 1):
            print(f"\nPaper {i}:")
            print(f"  Title  : {p['title'][:80]}...")
            print(f"  Authors: {p['authors']}")
            print(f"  Year   : {p['year']}")
            print(f"  URL    : {p['url']}")
    else:
        print("No results returned")

    # Test 2: Format for LLM
    print("\n" + "=" * 55)
    print("TEST 2: Formatted output for LLM")
    print("=" * 55)
    formatted = search_and_format(
        "signaled active avoidance dopamine", max_results=2
    )
    print(formatted[:800])

    print("\n✅ PubMed tool working correctly!")