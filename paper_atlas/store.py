from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper_atlas.models import Paper, merge_papers

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path(os.getenv("PAPER_ATLAS_DB", ROOT / "data" / "paper_atlas" / "library.sqlite3"))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PaperStore:
    """Local SQLite library. Provider records, extracts, and their provenance stay on disk locally."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or DEFAULT_DB).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    abstract TEXT NOT NULL DEFAULT '',
                    year INTEGER,
                    authors_json TEXT NOT NULL DEFAULT '[]',
                    venue TEXT NOT NULL DEFAULT '',
                    doi TEXT NOT NULL DEFAULT '',
                    arxiv_id TEXT NOT NULL DEFAULT '',
                    openalex_id TEXT NOT NULL DEFAULT '',
                    semantic_scholar_id TEXT NOT NULL DEFAULT '',
                    url TEXT NOT NULL DEFAULT '',
                    open_access_url TEXT NOT NULL DEFAULT '',
                    citation_count INTEGER NOT NULL DEFAULT 0,
                    topics_json TEXT NOT NULL DEFAULT '[]',
                    references_json TEXT NOT NULL DEFAULT '[]',
                    providers_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year DESC);
                CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
                CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id);
                CREATE TABLE IF NOT EXISTS collections (
                    collection_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    provider_warnings_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS collection_papers (
                    collection_id TEXT NOT NULL REFERENCES collections(collection_id) ON DELETE CASCADE,
                    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
                    rank INTEGER NOT NULL,
                    PRIMARY KEY(collection_id, paper_id)
                );
                CREATE INDEX IF NOT EXISTS idx_collection_papers_rank ON collection_papers(collection_id, rank);
                CREATE TABLE IF NOT EXISTS facts (
                    fact_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    label TEXT NOT NULL,
                    statement TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL,
                    page INTEGER,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    model TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_facts_paper ON facts(paper_id);
                CREATE TABLE IF NOT EXISTS relations (
                    relation_id TEXT PRIMARY KEY,
                    source_paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
                    target_paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
                    relation TEXT NOT NULL,
                    evidence_source TEXT NOT NULL DEFAULT '',
                    evidence_target TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.0,
                    provenance TEXT NOT NULL,
                    model TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    UNIQUE(source_paper_id, target_paper_id, relation, provenance)
                );
                CREATE TABLE IF NOT EXISTS settings (
                    setting_key TEXT PRIMARY KEY,
                    setting_value TEXT NOT NULL
                );
            """)

    def get_setting(self, key: str, default: str = "") -> str:
        with self._connect() as db:
            row = db.execute("SELECT setting_value FROM settings WHERE setting_key=?", (key,)).fetchone()
        return str(row["setting_value"]) if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._connect() as db:
            db.execute("""
                INSERT INTO settings(setting_key,setting_value) VALUES(?,?)
                ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value
            """, (key, value))

    @staticmethod
    def _paper_from_row(row: sqlite3.Row | None) -> Paper | None:
        if row is None:
            return None
        return Paper(
            id=row["paper_id"], title=row["title"], abstract=row["abstract"], year=row["year"],
            authors=json.loads(row["authors_json"]), venue=row["venue"], doi=row["doi"],
            arxiv_id=row["arxiv_id"], openalex_id=row["openalex_id"],
            semantic_scholar_id=row["semantic_scholar_id"], url=row["url"],
            open_access_url=row["open_access_url"], citation_count=row["citation_count"],
            topics=json.loads(row["topics_json"]), references=json.loads(row["references_json"]),
            providers=json.loads(row["providers_json"]),
        )

    def upsert_papers(self, papers: list[Paper]) -> None:
        with self._connect() as db:
            for incoming in papers:
                old = self._paper_from_row(db.execute(
                    "SELECT * FROM papers WHERE paper_id=?", (incoming.id,)
                ).fetchone())
                paper = merge_papers(old, incoming) if old else incoming
                db.execute("""
                    INSERT INTO papers (
                        paper_id,title,abstract,year,authors_json,venue,doi,arxiv_id,openalex_id,
                        semantic_scholar_id,url,open_access_url,citation_count,topics_json,references_json,
                        providers_json,updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(paper_id) DO UPDATE SET
                        title=excluded.title, abstract=excluded.abstract, year=excluded.year,
                        authors_json=excluded.authors_json, venue=excluded.venue, doi=excluded.doi,
                        arxiv_id=excluded.arxiv_id, openalex_id=excluded.openalex_id,
                        semantic_scholar_id=excluded.semantic_scholar_id, url=excluded.url,
                        open_access_url=excluded.open_access_url, citation_count=excluded.citation_count,
                        topics_json=excluded.topics_json, references_json=excluded.references_json,
                        providers_json=excluded.providers_json, updated_at=excluded.updated_at
                """, (
                    paper.id, paper.title, paper.abstract, paper.year, _json(paper.authors), paper.venue,
                    paper.doi, paper.arxiv_id, paper.openalex_id, paper.semantic_scholar_id, paper.url,
                    paper.open_access_url, paper.citation_count, _json(paper.topics), _json(paper.references),
                    _json(paper.providers), _now(),
                ))

    def create_collection(self, query: str, papers: list[Paper], warnings: list[str] | None = None) -> str:
        collection_id = uuid.uuid4().hex[:12]
        self.upsert_papers(papers)
        with self._connect() as db:
            db.execute("INSERT INTO collections(collection_id,query,created_at,provider_warnings_json) VALUES(?,?,?,?)",
                       (collection_id, query, _now(), _json(warnings or [])))
            db.executemany("INSERT INTO collection_papers(collection_id,paper_id,rank) VALUES(?,?,?)",
                           [(collection_id, paper.id, rank) for rank, paper in enumerate(papers, 1)])
        return collection_id

    def add_papers_to_collection(self, collection_id: str, papers: list[Paper]) -> int:
        """Append papers to an existing collection, skipping ids already present."""
        if self.get_collection(collection_id) is None:
            raise ValueError(f"Unknown collection: {collection_id}")
        self.upsert_papers(papers)
        added = 0
        with self._connect() as db:
            row = db.execute("SELECT COALESCE(MAX(rank), 0) FROM collection_papers WHERE collection_id=?",
                             (collection_id,)).fetchone()
            rank = int(row[0] if row else 0)
            for paper in papers:
                exists = db.execute(
                    "SELECT 1 FROM collection_papers WHERE collection_id=? AND paper_id=?",
                    (collection_id, paper.id),
                ).fetchone()
                if exists:
                    continue
                rank += 1
                db.execute("INSERT INTO collection_papers(collection_id,paper_id,rank) VALUES(?,?,?)",
                           (collection_id, paper.id, rank))
                added += 1
        return added

    def get_collection(self, collection_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM collections WHERE collection_id=?", (collection_id,)).fetchone()
        if row is None:
            return None
        return {"id": row["collection_id"], "query": row["query"], "created_at": row["created_at"],
                "warnings": json.loads(row["provider_warnings_json"])}

    def list_collections(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("""
                SELECT c.collection_id,c.query,c.created_at,COUNT(cp.paper_id) AS paper_count
                FROM collections c LEFT JOIN collection_papers cp USING(collection_id)
                GROUP BY c.collection_id ORDER BY c.created_at DESC LIMIT ?
            """, (max(1, min(limit, 200)),)).fetchall()
        return [{"id": r["collection_id"], "query": r["query"], "created_at": r["created_at"],
                 "paper_count": r["paper_count"]} for r in rows]

    def get_paper(self, paper_id: str) -> Paper | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        return self._paper_from_row(row)

    def get_papers(self, collection_id: str | None = None, limit: int = 200) -> list[Paper]:
        with self._connect() as db:
            if collection_id:
                rows = db.execute("""
                    SELECT p.* FROM papers p JOIN collection_papers cp USING(paper_id)
                    WHERE cp.collection_id=? ORDER BY cp.rank LIMIT ?
                """, (collection_id, max(1, min(limit, 1000)))).fetchall()
            else:
                rows = db.execute("SELECT * FROM papers ORDER BY year DESC,citation_count DESC LIMIT ?",
                                  (max(1, min(limit, 1000)),)).fetchall()
        return [paper for row in rows if (paper := self._paper_from_row(row)) is not None]

    def search_local(self, query: str, collection_id: str | None = None, limit: int = 12) -> list[Paper]:
        papers = self.get_papers(collection_id, limit=1000)
        terms = [x for x in re.findall(r"[a-z0-9][a-z0-9+-]{1,}", query.casefold())]
        if not terms:
            return papers[:limit]
        scored = []
        for paper in papers:
            title = paper.title.casefold()
            abstract = paper.abstract.casefold()
            topics = " ".join(paper.topics).casefold()
            score = 4.0 * sum(term in title for term in terms)
            score += 1.0 * sum(term in abstract for term in terms)
            score += 1.5 * sum(term in topics for term in terms)
            phrase = query.casefold().strip()
            if phrase and phrase in title:
                score += 5.0
            if score:
                scored.append((score, paper.year or 0, paper))
        scored.sort(key=lambda item: (item[0], item[1], item[2].citation_count), reverse=True)
        return [item[2] for item in scored[:max(1, limit)]]

    def save_facts(self, paper_id: str, facts: list[dict[str, Any]], model: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM facts WHERE paper_id=?", (paper_id,))
            db.executemany("""
                INSERT INTO facts(fact_id,paper_id,kind,label,statement,evidence,page,confidence,model,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
            """, [(
                uuid.uuid4().hex, paper_id, fact.get("kind", "method"), fact["label"],
                fact.get("statement", ""), fact["evidence"], fact.get("page"),
                max(0.0, min(float(fact.get("confidence", 0.7)), 1.0)), model, _now(),
            ) for fact in facts if fact.get("label") and fact.get("evidence")])

    def get_facts(self, paper_ids: list[str] | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            if paper_ids is not None:
                if not paper_ids:
                    return []
                marks = ",".join("?" for _ in paper_ids)
                rows = db.execute(f"SELECT * FROM facts WHERE paper_id IN ({marks}) ORDER BY kind,label", paper_ids).fetchall()
            else:
                rows = db.execute("SELECT * FROM facts ORDER BY paper_id,kind,label").fetchall()
        return [{"id": r["fact_id"], "paper_id": r["paper_id"], "kind": r["kind"], "label": r["label"],
                 "statement": r["statement"], "evidence": r["evidence"], "page": r["page"],
                 "confidence": r["confidence"], "model": r["model"]} for r in rows]

    def save_relation(self, relation: dict[str, Any], model: str) -> None:
        with self._connect() as db:
            db.execute("""
                INSERT OR REPLACE INTO relations(
                    relation_id,source_paper_id,target_paper_id,relation,evidence_source,evidence_target,
                    confidence,provenance,model,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """, (
                uuid.uuid4().hex, relation["source_paper_id"], relation["target_paper_id"],
                relation["relation"], relation.get("evidence_source", ""), relation.get("evidence_target", ""),
                max(0.0, min(float(relation.get("confidence", 0.6)), 1.0)), "llm_inferred", model, _now(),
            ))

    def get_relations(self, paper_ids: list[str] | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            if paper_ids is not None:
                if not paper_ids:
                    return []
                marks = ",".join("?" for _ in paper_ids)
                rows = db.execute(
                    f"SELECT * FROM relations WHERE source_paper_id IN ({marks}) AND target_paper_id IN ({marks})",
                    paper_ids + paper_ids,
                ).fetchall()
            else:
                rows = db.execute("SELECT * FROM relations ORDER BY created_at DESC").fetchall()
        return [{"id": r["relation_id"], "source_paper_id": r["source_paper_id"],
                 "target_paper_id": r["target_paper_id"], "relation": r["relation"],
                 "evidence_source": r["evidence_source"], "evidence_target": r["evidence_target"],
                 "confidence": r["confidence"], "provenance": r["provenance"], "model": r["model"]}
                for r in rows]
