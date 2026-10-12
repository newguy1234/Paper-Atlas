from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


def clean_doi(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"^https?://(dx\.)?doi\.org/|^doi:\s*", "", value.strip(), flags=re.I).lower()


def clean_arxiv_id(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip()
    match = re.search(r"(?:arxiv\.org/(?:abs|pdf)/)?([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", value, re.I)
    if match:
        return match.group(1)
    match = re.search(r"(?:arxiv:)?([a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", value, re.I)
    return match.group(1) if match else ""


def canonical_paper_id(*, doi: str = "", arxiv_id: str = "", openalex_id: str = "",
                       semantic_scholar_id: str = "") -> str:
    doi = clean_doi(doi)
    arxiv_id = clean_arxiv_id(arxiv_id)
    if doi:
        return f"doi:{doi}"
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if openalex_id:
        return f"openalex:{openalex_id.rstrip('/').split('/')[-1]}"
    if semantic_scholar_id:
        return f"s2:{semantic_scholar_id}"
    raise ValueError("A paper needs at least one stable provider identifier")


@dataclass(slots=True)
class Paper:
    id: str
    title: str
    abstract: str = ""
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    openalex_id: str = ""
    semantic_scholar_id: str = ""
    url: str = ""
    open_access_url: str = ""
    citation_count: int = 0
    topics: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Paper":
        values = dict(data)
        for key in ("authors", "topics", "references", "providers"):
            if not isinstance(values.get(key), list):
                values[key] = []
        if values.get("year") in ("", 0):
            values["year"] = None
        if values.get("year") is not None:
            try:
                values["year"] = int(values["year"])
            except (TypeError, ValueError):
                values["year"] = None
        values["doi"] = clean_doi(values.get("doi"))
        values["arxiv_id"] = clean_arxiv_id(values.get("arxiv_id"))
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: values[key] for key in allowed if key in values})


def merge_papers(existing: Paper, incoming: Paper) -> Paper:
    """Merge provider responses without replacing richer, already-known metadata with blanks."""
    for name in ("abstract", "venue", "doi", "arxiv_id", "openalex_id", "semantic_scholar_id",
                 "url", "open_access_url"):
        if not getattr(existing, name) and getattr(incoming, name):
            setattr(existing, name, getattr(incoming, name))
    if (not existing.title or len(existing.title) < 8) and incoming.title:
        existing.title = incoming.title
    if existing.year is None:
        existing.year = incoming.year
    existing.authors = list(dict.fromkeys(existing.authors + incoming.authors))
    existing.topics = list(dict.fromkeys(existing.topics + incoming.topics))
    existing.references = list(dict.fromkeys(existing.references + incoming.references))
    existing.providers = list(dict.fromkeys(existing.providers + incoming.providers))
    existing.citation_count = max(existing.citation_count, incoming.citation_count)
    return existing
