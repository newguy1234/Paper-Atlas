"""Portable exports for a saved Paper Atlas collection."""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any

from paper_atlas.graph import build_graph, graph_payload
from paper_atlas.store import PaperStore


def export_collection(collection_id: str, *, store: PaperStore | None = None) -> dict[str, Any]:
    store = store or PaperStore()
    collection = store.get_collection(collection_id)
    if not collection:
        raise ValueError(f"Collection not found: {collection_id}")
    papers = store.get_papers(collection_id, limit=1000)
    ids = [paper.id for paper in papers]
    return {
        "collection": collection,
        "papers": [paper.to_dict() for paper in papers],
        "facts": store.get_facts(ids),
        "relations": store.get_relations(ids),
        "graph": graph_payload(build_graph(store, collection_id, limit=150)),
    }


def collection_json(collection_id: str, *, store: PaperStore | None = None) -> str:
    return json.dumps(export_collection(collection_id, store=store), ensure_ascii=False, indent=2)


def collection_csv(collection_id: str, *, store: PaperStore | None = None) -> str:
    store = store or PaperStore()
    papers = store.get_papers(collection_id, limit=1000)
    output = io.StringIO(newline="")
    fields = ("id", "title", "year", "authors", "venue", "doi", "arxiv_id",
              "citation_count", "url", "open_access_url", "topics", "abstract")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for paper in papers:
        row = paper.to_dict()
        row["authors"] = "; ".join(paper.authors)
        row["topics"] = "; ".join(paper.topics)
        writer.writerow({key: row.get(key, "") for key in fields})
    return output.getvalue()


def _bibtex_escape(value: str) -> str:
    return (value.replace("\\", r"\textbackslash{}")
            .replace("{", r"\{").replace("}", r"\}")
            .replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def collection_bibtex(collection_id: str, *, store: PaperStore | None = None) -> str:
    store = store or PaperStore()
    papers = store.get_papers(collection_id, limit=1000)
    blocks = []
    for index, paper in enumerate(papers, 1):
        key_seed = paper.doi or paper.arxiv_id or paper.id
        key = re.sub(r"[^A-Za-z0-9]+", "", key_seed)[:32] or f"paper{index}"
        fields = [f"  title = {{{_bibtex_escape(paper.title)}}}"]
        if paper.authors:
            fields.append(f"  author = {{{_bibtex_escape(' and '.join(paper.authors))}}}")
        if paper.year:
            fields.append(f"  year = {{{paper.year}}}")
        if paper.venue:
            fields.append(f"  journal = {{{_bibtex_escape(paper.venue)}}}")
        if paper.doi:
            fields.append(f"  doi = {{{_bibtex_escape(paper.doi)}}}")
        if paper.url:
            fields.append(f"  url = {{{_bibtex_escape(paper.url)}}}")
        blocks.append(f"@article{{{key},\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")
