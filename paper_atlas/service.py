from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from paper_atlas.bibtex import parse_bibtex
from paper_atlas.graph import build_graph, graph_payload
from paper_atlas.llm import (DEFAULT_MODEL, DEFAULT_OLLAMA_URL, analyze_paper,
                             answer_from_collection, fetch_open_pdf, infer_relation)
from paper_atlas.models import Paper, merge_papers
from paper_atlas.providers import expand_from_seed, search_literature
from paper_atlas.store import PaperStore
from paper_atlas.paths import output_folder


def paper_summary(paper: Paper) -> dict[str, Any]:
    return {"id": paper.id, "title": paper.title, "year": paper.year, "authors": paper.authors[:8],
            "venue": paper.venue, "doi": paper.doi, "arxiv_id": paper.arxiv_id,
            "citation_count": paper.citation_count, "url": paper.url,
            "open_access_url": paper.open_access_url, "topics": paper.topics,
            "providers": paper.providers, "abstract": paper.abstract}


def discover(query: str, *, limit: int = 20, from_year: int | None = None,
             providers: tuple[str, ...] = ("openalex", "arxiv", "semanticscholar"),
             scope: str = "cs_methods", semantic: bool = False,
             store: PaperStore | None = None) -> dict[str, Any]:
    papers, warnings = search_literature(query, limit=limit, from_year=from_year,
                                         providers=providers, scope=scope, semantic=semantic)
    store = store or PaperStore()
    scope_label = "Computer science methods" if scope == "cs_methods" else "CS methods applied across fields"
    mode = "semantic+" if semantic else ""
    collection_id = store.create_collection(f"{scope_label} {mode}: {query}".strip(), papers, warnings)
    return {"collection_id": collection_id, "query": query, "count": len(papers),
            "scope": scope, "semantic": semantic, "warnings": warnings,
            "papers": [paper_summary(paper) for paper in papers]}


def expand_collection(paper_id: str, *, collection_id: str, limit: int = 12,
                      store: PaperStore | None = None) -> dict[str, Any]:
    store = store or PaperStore()
    paper = store.get_paper(paper_id)
    if not paper:
        raise ValueError(f"Paper not found: {paper_id}")
    related, warnings = expand_from_seed(paper, limit=limit)
    added = store.add_papers_to_collection(collection_id, related) if related else 0
    papers = store.get_papers(collection_id, limit=200)
    return {"collection_id": collection_id, "seed_paper_id": paper_id, "added": added,
            "count": len(papers), "warnings": warnings,
            "papers": [paper_summary(item) for item in papers]}


def discover_research_pack(queries: list[str], *, limit_per_query: int = 12,
                           from_year: int | None = 2023,
                           providers: tuple[str, ...] = ("openalex", "arxiv", "semanticscholar"),
                           scope: str = "cs_methods", name: str = "Research pack",
                           store: PaperStore | None = None) -> dict[str, Any]:
    """Search a set of related queries, merge duplicates, and save one ranked collection."""
    clean_queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
    if not clean_queries:
        raise ValueError("Provide at least one non-empty research query")
    from paper_atlas.providers import search_literature

    merged: dict[str, Paper] = {}
    rank_scores: Counter[str] = Counter()
    warnings: list[str] = []
    query_counts: dict[str, int] = {}
    for query in clean_queries:
        try:
            papers, query_warnings = search_literature(
                query, limit=limit_per_query, from_year=from_year,
                providers=providers, scope=scope,
            )
        except Exception as exc:
            query_counts[query] = 0
            warnings.append(f"{query}: search failed: {exc}")
            continue
        query_counts[query] = len(papers)
        warnings.extend(f"{query}: {warning}" for warning in query_warnings)
        for rank, paper in enumerate(papers, 1):
            rank_scores[paper.id] += 1.0 / (60 + rank)
            merged[paper.id] = merge_papers(merged[paper.id], paper) if paper.id in merged else paper

    ranked = sorted(merged.values(), key=lambda paper: (
        rank_scores[paper.id], paper.year or 0, paper.citation_count
    ), reverse=True)
    store = store or PaperStore()
    collection_id = store.create_collection(name, ranked, warnings)
    return {
        "collection_id": collection_id,
        "name": name,
        "scope": scope,
        "from_year": from_year,
        "query_counts": query_counts,
        "count": len(ranked),
        "successful_queries": sum(1 for query in clean_queries if not any(
            warning.startswith(f"{query}: search failed:") for warning in warnings
        )),
        "warnings": warnings,
        "papers": [paper_summary(paper) for paper in ranked],
    }


def import_bibtex(text: str, *, store: PaperStore | None = None,
                  collection_name: str = "Imported BibTeX") -> dict[str, Any]:
    store = store or PaperStore()
    papers = parse_bibtex(text)
    collection_id = store.create_collection(collection_name, papers, [])
    return {"collection_id": collection_id, "query": collection_name, "count": len(papers),
            "papers": [paper_summary(paper) for paper in papers], "warnings": []}


def get_graph(collection_id: str, *, limit: int = 40, selected_paper_id: str | None = None,
              store: PaperStore | None = None) -> dict[str, Any]:
    store = store or PaperStore()
    graph = build_graph(store, collection_id, limit=limit, selected_paper_id=selected_paper_id)
    return graph_payload(graph)


def extract_selected(paper_id: str, *, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_OLLAMA_URL,
                    pdf_dir: Path | str | None = None,
                    store: PaperStore | None = None) -> dict[str, Any]:
    store = store or PaperStore()
    paper = store.get_paper(paper_id)
    if not paper:
        raise ValueError(f"Paper not found: {paper_id}")
    return analyze_paper(paper, store, model=model, base_url=base_url, pdf_dir=output_folder(pdf_dir, store))


def download_selected_open_pdfs(paper_ids: list[str], *, output_dir: Path | str | None = None,
                                store: PaperStore | None = None) -> dict[str, Any]:
    """Download only provider-listed open PDFs or public arXiv PDFs, one by one."""
    store = store or PaperStore()
    results = []
    batch_limit = 250 * 1024 * 1024
    bytes_used = 0
    for paper_id in dict.fromkeys(paper_ids):
        paper = store.get_paper(paper_id)
        if not paper:
            results.append({"paper_id": paper_id, "status": "error", "message": "Paper not found"})
            continue
        if not (paper.open_access_url or paper.arxiv_id):
            results.append({"paper_id": paper_id, "title": paper.title,
                            "status": "unavailable", "message": "No provider-listed open PDF"})
            continue
        remaining = batch_limit - bytes_used
        if remaining <= 0:
            results.append({"paper_id": paper_id, "title": paper.title, "status": "skipped",
                            "message": "This batch reached its 250 MB download limit; run the remaining papers separately"})
            continue
        try:
            path = fetch_open_pdf(paper, max_bytes=min(35 * 1024 * 1024, remaining),
                                  destination_dir=output_folder(output_dir, store))
            if path:
                bytes_used += path.stat().st_size
            results.append({"paper_id": paper_id, "title": paper.title,
                            "status": "downloaded" if path else "unavailable",
                            "path": str(path) if path else "",
                            "message": "" if path else "No direct PDF URL was available"})
        except Exception as exc:
            results.append({"paper_id": paper_id, "title": paper.title, "status": "error", "message": str(exc)})
    return {"count": len(results), "results": results}


def extract_selected_many(paper_ids: list[str], *, model: str = DEFAULT_MODEL,
                          base_url: str = DEFAULT_OLLAMA_URL, pdf_dir: Path | str | None = None,
                          store: PaperStore | None = None) -> dict[str, Any]:
    """Extract quote-checked methods and claims for selected papers, preserving per-paper errors."""
    store = store or PaperStore()
    results = []
    for paper_id in dict.fromkeys(paper_ids):
        try:
            results.append(extract_selected(paper_id, model=model, base_url=base_url,
                                            pdf_dir=pdf_dir, store=store))
        except Exception as exc:
            results.append({"paper_id": paper_id, "facts": [], "source_kind": "failed", "error": str(exc)})
    return {"count": len(results), "results": results,
            "fact_count": sum(len(item.get("facts", [])) for item in results)}


def compare_papers(source_id: str, target_id: str, *, model: str = DEFAULT_MODEL,
                   base_url: str = DEFAULT_OLLAMA_URL, store: PaperStore | None = None) -> dict[str, Any]:
    store = store or PaperStore()
    source = store.get_paper(source_id)
    target = store.get_paper(target_id)
    if not source or not target:
        raise ValueError("Both papers must be present in the local library")
    return infer_relation(source, target, store, model=model, base_url=base_url)


def ask_collection(question: str, collection_id: str, *, model: str = DEFAULT_MODEL,
                   base_url: str = DEFAULT_OLLAMA_URL, limit: int = 8,
                    store: PaperStore | None = None) -> dict[str, Any]:
    store = store or PaperStore()
    # Make disagreement visible instead of letting a single relevance ranking
    # silently select only papers that echo the same positive result.
    views = [
        ("core findings", question, max(4, limit // 2)),
        ("benefits and gains", f"{question} benefits advantages gains improvement performance", 4),
        ("limitations and costs", f"{question} limitations failures costs tradeoffs drawbacks", 4),
        ("mixed or conflicting evidence", f"{question} inconsistent mixed conflicting results no improvement", 4),
    ]
    papers_by_id: dict[str, Paper] = {}
    paper_views: dict[str, list[str]] = {}
    view_results: list[tuple[str, list[Paper]]] = []
    for label, view_query, view_limit in views:
        matches = store.search_local(view_query, collection_id, limit=view_limit)
        view_results.append((label, matches))
        for paper in matches:
            papers_by_id.setdefault(paper.id, paper)
            paper_views.setdefault(paper.id, []).append(label)

    # Reserve space for each evidence view first, then fill with the strongest
    # remaining topical matches. Duplicates are retained once and their views noted.
    requested = max(1, limit)
    core_cap = max(1, int(requested * 0.4))
    benefit_cap = int(requested * 0.2)
    limitation_cap = int(requested * 0.2)
    conflict_cap = requested - core_cap - benefit_cap - limitation_cap
    caps = {"core findings": core_cap, "benefits and gains": benefit_cap,
            "limitations and costs": limitation_cap, "mixed or conflicting evidence": conflict_cap}
    papers: list[Paper] = []
    selected: set[str] = set()
    for label, matches in view_results:
        taken = 0
        for paper in matches:
            if paper.id in selected:
                continue
            if taken >= caps[label] or len(papers) >= max(1, limit):
                break
            papers.append(paper)
            selected.add(paper.id)
            taken += 1
    if len(papers) < max(1, limit):
        for _, matches in view_results:
            for paper in matches:
                if paper.id not in selected:
                    papers.append(paper)
                    selected.add(paper.id)
                    if len(papers) >= max(1, limit):
                        break
            if len(papers) >= max(1, limit):
                break
    facts = store.get_facts([paper.id for paper in papers])
    result = answer_from_collection(question, papers, facts, model=model, base_url=base_url)
    result["retrieved_papers"] = [paper_summary(paper) for paper in papers]
    result["evidence_views"] = {label: [paper.id for paper in matches]
                                for label, matches in view_results}
    result["paper_views"] = {paper_id: labels for paper_id, labels in paper_views.items()
                             if paper_id in selected}
    return result

