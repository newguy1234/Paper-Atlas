"""MCP tools for Paper Atlas. Run with: python -m paper_atlas.mcp_server"""

from __future__ import annotations

import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from paper_atlas.llm import DEFAULT_MODEL, DEFAULT_OLLAMA_URL
from paper_atlas.service import (ask_collection, discover, download_selected_open_pdfs,
                                 expand_collection, extract_selected_many, get_graph, import_bibtex)
from paper_atlas.store import PaperStore

STORE = PaperStore()
server = MCPServer(
    name="paper-atlas",
    log_level="WARNING",
    instructions=("Search scholarly indexes, save local paper collections, download provider-listed "
                  "open-access PDFs, extract methods and claims with exact-quote validation, ask "
                  "saved collections using local Ollama, and inspect evidence-linked paper graphs. "
                  "Inferred relations are marked separately from bibliographic citation records."),
)


def search_literature(query: str, limit: int = 20, from_year: int | None = None,
                      providers: list[str] | None = None,
                      scope: str = "cs_methods", semantic: bool = False) -> dict[str, Any]:
    """Search OpenAlex, arXiv, and Semantic Scholar; returns a saved collection ID."""
    return discover(query, limit=limit, from_year=from_year,
                    providers=tuple(providers or ("openalex", "arxiv", "semanticscholar")),
                    scope=scope, semantic=semantic, store=STORE)


def expand_related_papers(collection_id: str, paper_id: str, limit: int = 12) -> dict[str, Any]:
    """Add Semantic Scholar recommendations, OpenAlex citers, and semantic neighbors of a seed paper."""
    return expand_collection(paper_id, collection_id=collection_id, limit=limit, store=STORE)


def import_bibtex_collection(text: str, collection_name: str = "Imported BibTeX") -> dict[str, Any]:
    """Import common BibTeX entries into a local collection."""
    return import_bibtex(text, store=STORE, collection_name=collection_name)


def inspect_paper_graph(collection_id: str, limit: int = 40) -> dict[str, Any]:
    """Return a paper/topic/method/claim graph with evidence and provenance."""
    return get_graph(collection_id, limit=limit, store=STORE)


def download_open_access_pdfs(paper_ids: list[str], output_dir: str = "") -> dict[str, Any]:
    """Download only provider-listed open PDFs and public arXiv PDFs to the local library."""
    return download_selected_open_pdfs(paper_ids, output_dir=output_dir or None, store=STORE)


def extract_methods_and_claims(paper_ids: list[str], model: str = DEFAULT_MODEL,
                               ollama_url: str = DEFAULT_OLLAMA_URL,
                               pdf_dir: str = "") -> dict[str, Any]:
    """Extract facts from selected papers using local Ollama; reject quotes absent from source."""
    return extract_selected_many(paper_ids, model=model, base_url=ollama_url,
                                 pdf_dir=pdf_dir or None, store=STORE)


def ask_saved_collection(collection_id: str, question: str, model: str = DEFAULT_MODEL,
                         ollama_url: str = DEFAULT_OLLAMA_URL) -> dict[str, Any]:
    """Answer from a saved collection with paper IDs mapped to source records."""
    return ask_collection(question, collection_id, model=model, base_url=ollama_url, store=STORE)


server.tool()(search_literature)
server.tool()(expand_related_papers)
server.tool()(import_bibtex_collection)
server.tool()(inspect_paper_graph)
server.tool()(download_open_access_pdfs)
server.tool()(extract_methods_and_claims)
server.tool()(ask_saved_collection)


if __name__ == "__main__":
    server.run("streamable-http" if "--http" in sys.argv else "stdio")
