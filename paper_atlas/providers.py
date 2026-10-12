"""Small, polite adapters for public scholarly metadata APIs.

The tool queries metadata and citation APIs; it only fetches full text from an
open-access URL returned by a provider or from arXiv's public PDF endpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from paper_atlas.models import Paper, canonical_paper_id, clean_arxiv_id, clean_doi, merge_papers

OPENALEX_API = "https://api.openalex.org/works"
ARXIV_API = "https://export.arxiv.org/api/query"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_RECS_API = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
UNPAYWALL_API = "https://api.unpaywall.org/v2"
USER_AGENT = "PaperAtlas/0.1 (local literature research tool)"
_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST_CALL = 0.0
_CACHE_LOCK = threading.Lock()
_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "paper_atlas" / "http_cache.sqlite3"
_CACHE_TTL_S = 7 * 24 * 3600


def _contact_email() -> str:
    return (os.getenv("OPENALEX_MAILTO") or os.getenv("UNPAYWALL_EMAIL")
            or os.getenv("PAPER_ATLAS_EMAIL") or "").strip()


def _user_agent() -> str:
    email = _contact_email()
    return f"PaperAtlas/0.1 (mailto:{email})" if email else USER_AGENT


def _cache_conn() -> sqlite3.Connection:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_CACHE_PATH, timeout=30)
    conn.execute("CREATE TABLE IF NOT EXISTS http_cache (cache_key TEXT PRIMARY KEY, fetched_unix REAL, body TEXT)")
    return conn


def _cache_get(cache_key: str) -> str | None:
    with _CACHE_LOCK:
        with _cache_conn() as db:
            row = db.execute("SELECT fetched_unix, body FROM http_cache WHERE cache_key=?", (cache_key,)).fetchone()
    if not row:
        return None
    if time.time() - float(row[0]) > _CACHE_TTL_S:
        return None
    return str(row[1])


def _cache_put(cache_key: str, body: str) -> None:
    with _CACHE_LOCK:
        with _cache_conn() as db:
            db.execute("INSERT OR REPLACE INTO http_cache(cache_key, fetched_unix, body) VALUES(?,?,?)",
                       (cache_key, time.time(), body))


def _cache_key(url: str, params: dict[str, Any] | None) -> str:
    payload = json.dumps({"url": url, "params": params or {}}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _get_json(url: str, *, params: dict[str, Any] | None = None,
              headers: dict[str, str] | None = None, timeout: int = 25) -> dict[str, Any]:
    params = dict(params or {})
    email = _contact_email()
    if "openalex.org" in url and email and "mailto" not in params:
        params["mailto"] = email
    cache_key = _cache_key(url, params)
    cached = _cache_get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except ValueError:
            pass
    request_headers = {"User-Agent": _user_agent(), "Accept": "application/json"}
    request_headers.update(headers or {})
    last_error: Exception | None = None
    last_status: int | None = None
    last_body = ""
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, headers=request_headers, timeout=timeout)
            if response.status_code == 429 or response.status_code in (500, 502, 503, 504):
                last_status = response.status_code
                last_body = response.text[:300]
                delay = min(30.0, float(response.headers.get("Retry-After", 2 ** attempt)))
                time.sleep(delay)
                continue
            response.raise_for_status()
            payload = response.json()
            _cache_put(cache_key, json.dumps(payload))
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(min(15, 1.5 * (2 ** attempt)))
    detail = str(last_error) if last_error else (f"HTTP {last_status}: {last_body}" if last_status else "unknown error")
    raise RuntimeError(f"Scholarly provider request failed: {detail}")


def _abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    words = sorted((int(position), word) for word, positions in index.items() for position in positions)
    return " ".join(word for _, word in words)


def _year(value: Any) -> int | None:
    try:
        year = int(str(value)[:4])
        return year if 1400 <= year <= 2200 else None
    except (TypeError, ValueError):
        return None


def _openalex_id(value: str | None) -> str:
    return (value or "").rstrip("/").split("/")[-1]


def _scope_query(query: str, scope: str) -> str:
    return f"{query} computer science computational method application algorithm" if scope == "cs_applied" else query


_CS_ARXIV_CATEGORIES = (
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY",
    "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.FL", "cs.GL", "cs.GR", "cs.HC",
    "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE",
    "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE",
    "cs.SI", "cs.SY",
)


def search_openalex(query: str, limit: int, from_year: int | None = None,
                    scope: str = "cs_methods") -> list[Paper]:
    fields = ",".join((
        "id", "doi", "title", "publication_year", "authorships", "abstract_inverted_index",
        "best_oa_location", "locations", "primary_location", "referenced_works", "cited_by_count",
        "primary_topic", "topics", "keywords",
    ))
    params: dict[str, Any] = {
        "search": _scope_query(query, scope),
        "per-page": min(100, max(1, limit)),
        # Search relevance first; date is only a tie-breaker. Sorting by date alone
        # made broad full-text matches crowd out papers that actually fit the query.
        "sort": "relevance_score:desc,publication_date:desc",
        "select": fields,
    }
    filters = []
    if from_year:
        filters.append(f"from_publication_date:{int(from_year)}-01-01")
    if scope == "cs_methods":
        # Primary topic is the paper's main field. `topics.field.id` admits any
        # secondary CS topic and leaked cross-disciplinary applications into the strict scope.
        filters.append("primary_topic.field.id:17")
    if filters:
        params["filter"] = ",".join(filters)
    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key
    payload = _get_json(OPENALEX_API, params=params)
    return [paper for work in payload.get("results", []) if (paper := _paper_from_openalex(work))]


def _paper_from_openalex(work: dict[str, Any]) -> Paper | None:
    title = " ".join((work.get("title") or "").split())
    if not title:
        return None
    oid = _openalex_id(work.get("id"))
    doi = clean_doi(work.get("doi"))
    oa = work.get("best_oa_location") or {}
    if not oa.get("pdf_url"):
        oa = next((loc for loc in work.get("locations", []) if loc and loc.get("pdf_url")), oa)
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    primary_topic = work.get("primary_topic") or {}
    primary_field = (primary_topic.get("field") or {}).get("display_name", "").strip()
    topics = [primary_field] if primary_field else []
    topics += [t.get("display_name", "").strip() for t in work.get("topics", []) if t.get("display_name")]
    topics += [t.get("display_name", "").strip() for t in work.get("keywords", []) if t.get("display_name")]
    try:
        paper_id = canonical_paper_id(doi=doi, openalex_id=oid)
    except ValueError:
        return None
    return Paper(
        id=paper_id,
        title=title,
        abstract=_abstract_from_inverted_index(work.get("abstract_inverted_index")),
        year=_year(work.get("publication_year")),
        authors=[a.get("author", {}).get("display_name", "").strip()
                 for a in work.get("authorships", []) if a.get("author", {}).get("display_name")],
        venue=(source.get("display_name") or "").strip(),
        doi=doi,
        openalex_id=oid,
        url=(work.get("doi") or f"https://openalex.org/{oid}"),
        open_access_url=(oa.get("pdf_url") or ""),
        citation_count=int(work.get("cited_by_count") or 0),
        topics=list(dict.fromkeys(topics)),
        references=[_openalex_id(x) for x in work.get("referenced_works", []) if x],
        providers=["OpenAlex"],
    )


def search_openalex_semantic(query: str, limit: int, from_year: int | None = None,
                             scope: str = "cs_methods") -> list[Paper]:
    """Nearest works by OpenAlex title/abstract embeddings. Keyword filters still apply except cited_by_count."""
    fields = ",".join((
        "id", "doi", "title", "publication_year", "authorships", "abstract_inverted_index",
        "best_oa_location", "locations", "primary_location", "referenced_works", "cited_by_count",
        "primary_topic", "topics", "keywords",
    ))
    params: dict[str, Any] = {
        "search.semantic": _scope_query(query, scope)[:2000],
        "per-page": min(50, max(1, limit)),
        "select": fields,
    }
    filters = []
    if from_year:
        filters.append(f"from_publication_date:{int(from_year)}-01-01")
    if scope == "cs_methods":
        filters.append("primary_topic.field.id:17")
    if filters:
        params["filter"] = ",".join(filters)
    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key
    payload = _get_json(OPENALEX_API, params=params)
    papers = [paper for work in payload.get("results", []) if (paper := _paper_from_openalex(work))]
    for paper in papers:
        if "OpenAlex-semantic" not in paper.providers:
            paper.providers.append("OpenAlex-semantic")
    return papers


def citing_openalex(openalex_id: str, limit: int = 12) -> list[Paper]:
    oid = _openalex_id(openalex_id)
    if not oid:
        return []
    fields = ",".join((
        "id", "doi", "title", "publication_year", "authorships", "abstract_inverted_index",
        "best_oa_location", "locations", "primary_location", "referenced_works", "cited_by_count",
        "primary_topic", "topics", "keywords",
    ))
    params = {
        "filter": f"cites:{oid}",
        "per-page": min(50, max(1, limit)),
        "sort": "cited_by_count:desc",
        "select": fields,
    }
    payload = _get_json(OPENALEX_API, params=params)
    return [paper for work in payload.get("results", []) if (paper := _paper_from_openalex(work))]


def recommend_semantic_scholar(paper_id: str, limit: int = 12) -> list[Paper]:
    if not paper_id:
        return []
    fields = "paperId,externalIds,title,abstract,year,authors,venue,url,citationCount,openAccessPdf,tldr"
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    payload = _get_json(f"{S2_RECS_API}/{paper_id}", params={"limit": min(50, max(1, limit)), "fields": fields},
                        headers=headers)
    papers: list[Paper] = []
    for item in payload.get("recommendedPapers") or payload.get("data") or []:
        paper = _paper_from_semantic_scholar(item)
        if paper:
            paper.providers = list(dict.fromkeys(paper.providers + ["Semantic Scholar recs"]))
            papers.append(paper)
    return papers


def unpaywall_pdf_url(doi: str) -> str:
    """Legal OA PDF URL from Unpaywall. Requires OPENALEX_MAILTO, UNPAYWALL_EMAIL, or PAPER_ATLAS_EMAIL."""
    doi = clean_doi(doi)
    email = _contact_email()
    if not doi or not email:
        return ""
    try:
        payload = _get_json(f"{UNPAYWALL_API}/{doi}", params={"email": email})
    except RuntimeError:
        return ""
    location = payload.get("best_oa_location") or {}
    url = (location.get("url_for_pdf") or "").strip()
    return url if url.startswith("https://") else ""


def _arxiv_request(params: dict[str, Any]) -> str:
    global _ARXIV_LAST_CALL
    with _ARXIV_LOCK:
        wait = 3.1 - (time.monotonic() - _ARXIV_LAST_CALL)
        if wait > 0:
            time.sleep(wait)
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = requests.get(ARXIV_API, params=params,
                                       headers={"User-Agent": _user_agent()}, timeout=30)
                _ARXIV_LAST_CALL = time.monotonic()
                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(20, 2 ** attempt * 2))
                    continue
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(min(20, 2 ** attempt * 2))
        raise RuntimeError(f"arXiv API request failed: {last_error}")


def search_arxiv(query: str, limit: int, from_year: int | None = None,
                 scope: str = "cs_methods") -> list[Paper]:
    quoted = query.strip().replace('"', "")
    search_query = f'all:"{quoted}"' if quoted else "all:*"
    if scope == "cs_methods":
        categories = " OR ".join(f"cat:{category}" for category in _CS_ARXIV_CATEGORIES)
        search_query = f"({search_query}) AND ({categories})"
    elif scope == "cs_applied":
        search_query = f"({search_query}) AND (all:algorithm OR all:computational OR all:method)"
    xml = _arxiv_request({
        "search_query": search_query,
        "start": 0,
        "max_results": min(100, max(1, limit)),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    root = ET.fromstring(xml)
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    papers: list[Paper] = []
    for entry in root.findall("a:entry", ns):
        abs_url = (entry.findtext("a:id", "", ns) or "").strip()
        arxiv_id = clean_arxiv_id(abs_url)
        title = " ".join((entry.findtext("a:title", "", ns) or "").split())
        if not arxiv_id or not title:
            continue
        published = entry.findtext("a:published", "", ns)
        year = _year(published)
        if from_year and year and year < from_year:
            continue
        category = entry.find("arxiv:primary_category", ns)
        topics = [category.attrib.get("term", "")] if category is not None else []
        topics.extend(x.attrib.get("term", "") for x in entry.findall("a:category", ns))
        doi = clean_doi(entry.findtext("arxiv:doi", "", ns))
        papers.append(Paper(
            id=canonical_paper_id(doi=doi, arxiv_id=arxiv_id),
            title=title,
            abstract=" ".join((entry.findtext("a:summary", "", ns) or "").split()),
            year=year,
            authors=[(a.findtext("a:name", "", ns) or "").strip()
                     for a in entry.findall("a:author", ns) if a.findtext("a:name", "", ns)],
            doi=doi,
            arxiv_id=arxiv_id,
            url=f"https://arxiv.org/abs/{arxiv_id}",
            open_access_url=f"https://arxiv.org/pdf/{arxiv_id}",
            topics=list(dict.fromkeys(t for t in topics if t)),
            providers=["arXiv"],
        ))
    return papers


def search_semantic_scholar(query: str, limit: int, from_year: int | None = None,
                            scope: str = "cs_methods") -> list[Paper]:
    fields = "paperId,externalIds,title,abstract,year,authors,venue,url,citationCount,openAccessPdf,tldr"
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    params: dict[str, Any] = {"query": _scope_query(query, scope), "limit": min(100, limit), "fields": fields}
    if scope == "cs_methods":
        params["fieldsOfStudy"] = "Computer Science"
    payload = _get_json(S2_API, params=params, headers=headers)
    papers: list[Paper] = []
    for item in payload.get("data", []):
        paper = _paper_from_semantic_scholar(item)
        if not paper:
            continue
        if from_year and paper.year and paper.year < from_year:
            continue
        papers.append(paper)
    return papers


def _paper_from_semantic_scholar(item: dict[str, Any]) -> Paper | None:
    title = " ".join((item.get("title") or "").split())
    if not title:
        return None
    ext = item.get("externalIds") or {}
    doi = clean_doi(ext.get("DOI"))
    arxiv_id = clean_arxiv_id(ext.get("ArXiv"))
    s2id = item.get("paperId") or ""
    oa = item.get("openAccessPdf") or {}
    abstract = (item.get("abstract") or "").strip()
    if not abstract and item.get("tldr"):
        abstract = (item["tldr"].get("text") or "").strip()
    try:
        paper_id = canonical_paper_id(doi=doi, arxiv_id=arxiv_id, semantic_scholar_id=s2id)
    except ValueError:
        return None
    return Paper(
        id=paper_id,
        title=title,
        abstract=abstract,
        year=_year(item.get("year")),
        authors=[a.get("name", "").strip() for a in item.get("authors", []) if a.get("name")],
        venue=(item.get("venue") or "").strip(),
        doi=doi,
        arxiv_id=arxiv_id,
        semantic_scholar_id=s2id,
        url=(item.get("url") or (f"https://www.semanticscholar.org/paper/{s2id}" if s2id else "")),
        open_access_url=(oa.get("url") or ""),
        citation_count=int(item.get("citationCount") or 0),
        providers=["Semantic Scholar"],
    )


_PROVIDERS: dict[str, Callable[[str, int, int | None, str], list[Paper]]] = {
    "openalex": search_openalex,
    "arxiv": search_arxiv,
    "semanticscholar": search_semantic_scholar,
}


def _paper_keys(paper: Paper) -> list[str]:
    out = []
    if paper.doi:
        out.append("doi:" + clean_doi(paper.doi))
    if paper.arxiv_id:
        out.append("arxiv:" + clean_arxiv_id(paper.arxiv_id))
    if paper.openalex_id:
        out.append("openalex:" + paper.openalex_id)
    if paper.semantic_scholar_id:
        out.append("s2:" + paper.semantic_scholar_id)
    title_key = re.sub(r"[^a-z0-9]+", "", paper.title.lower())
    if title_key:
        out.append(f"title:{title_key}:{paper.year or ''}")
    return out


def _fuse_lists(groups: list[tuple[str, list[Paper]]], *, limit: int) -> tuple[list[Paper], dict[str, float]]:
    found: dict[str, Paper] = {}
    aliases: dict[str, str] = {}
    rrf: dict[str, float] = {}
    for _name, items in groups:
        for rank, incoming in enumerate(items, 1):
            aliases_here = _paper_keys(incoming)
            target_id = next((aliases[k] for k in aliases_here if k in aliases), incoming.id)
            if target_id in found:
                paper = merge_papers(found[target_id], incoming)
                paper.id = target_id
            else:
                paper = incoming
                target_id = paper.id
                found[target_id] = paper
            for alias in aliases_here + _paper_keys(paper):
                aliases[alias] = target_id
            rrf[target_id] = rrf.get(target_id, 0.0) + 1.0 / (60 + rank)
    ranked = sorted(found.values(), key=lambda p: (rrf.get(p.id, 0.0), p.year or 0, p.citation_count), reverse=True)
    return ranked[:limit], rrf


def search_literature(query: str, *, limit: int = 20, from_year: int | None = None,
                      providers: tuple[str, ...] = ("openalex", "arxiv", "semanticscholar"),
                      scope: str = "cs_methods", semantic: bool = False) -> tuple[list[Paper], list[str]]:
    """Search sources in parallel, merge duplicates, and fuse provider ranks with RRF."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    query = query.strip()
    if scope not in {"cs_methods", "cs_applied"}:
        raise ValueError("Scope must be cs_methods or cs_applied")
    if not query:
        raise ValueError("Enter a topic or research question")
    limit = max(1, min(int(limit), 50))
    chosen = [name.lower() for name in providers if name.lower() in _PROVIDERS]
    if not chosen:
        raise ValueError("Choose at least one supported provider")
    jobs: list[tuple[str, Callable[[], list[Paper]]]] = [
        (name, lambda n=name: _PROVIDERS[n](query, limit, from_year, scope)) for name in chosen
    ]
    if semantic:
        jobs.append(("openalex-semantic", lambda: search_openalex_semantic(query, limit, from_year, scope)))
    errors: list[str] = []
    groups: list[tuple[str, list[Paper]]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = {pool.submit(fn): name for name, fn in jobs}
        for future in as_completed(futures):
            provider = futures[future]
            try:
                groups.append((provider, future.result()))
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
    if not groups and errors:
        raise RuntimeError("No scholarly providers returned results. " + " | ".join(errors))
    ranked, _rrf = _fuse_lists(groups, limit=limit)
    return ranked, errors


def expand_from_seed(paper: Paper, *, limit: int = 12) -> tuple[list[Paper], list[str]]:
    """Related papers from S2 recommendations, OpenAlex citers, and semantic neighbors of the abstract."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    limit = max(1, min(int(limit), 30))
    jobs: list[tuple[str, Callable[[], list[Paper]]]] = []
    if paper.semantic_scholar_id:
        jobs.append(("s2-recs", lambda: recommend_semantic_scholar(paper.semantic_scholar_id, limit)))
    if paper.openalex_id:
        jobs.append(("openalex-citing", lambda: citing_openalex(paper.openalex_id, limit)))
    seed_text = " ".join(part for part in (paper.title, paper.abstract) if part).strip()
    if seed_text:
        jobs.append(("openalex-semantic", lambda: search_openalex_semantic(seed_text, limit, None, "cs_applied")))
    if not jobs:
        return [], ["Seed paper has no Semantic Scholar or OpenAlex id and no abstract to expand from"]
    errors: list[str] = []
    groups: list[tuple[str, list[Paper]]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = {pool.submit(fn): name for name, fn in jobs}
        for future in as_completed(futures):
            name = futures[future]
            try:
                groups.append((name, future.result()))
            except Exception as exc:
                errors.append(f"{name}: {exc}")
    ranked, _rrf = _fuse_lists(groups, limit=limit)
    ranked = [item for item in ranked if item.id != paper.id]
    return ranked, errors

