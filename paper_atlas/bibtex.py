"""Small dependency-free parser for common BibTeX bibliography entries."""

from __future__ import annotations

import re
from paper_atlas.models import Paper, canonical_paper_id, clean_arxiv_id, clean_doi, merge_papers


def _entry_end(text: str, start: int, opening: str) -> int:
    closing = "}" if opening == "{" else ")"
    depth = 1
    brace_depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
            continue
        if quoted:
            continue
        if opening == "{":
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index
        else:
            if char == "{":
                brace_depth += 1
            elif char == "}" and brace_depth:
                brace_depth -= 1
            elif brace_depth == 0 and char == "(":
                depth += 1
            elif brace_depth == 0 and char == ")":
                depth -= 1
                if depth == 0:
                    return index
    raise ValueError("A BibTeX entry is missing its closing delimiter")


def _split_top_level(value: str, delimiter: str) -> list[str]:
    parts = []
    start = 0
    braces = 0
    parens = 0
    quoted = False
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
            continue
        if quoted:
            continue
        if char == "{":
            braces += 1
        elif char == "}" and braces:
            braces -= 1
        elif braces == 0 and char == "(":
            parens += 1
        elif braces == 0 and char == ")" and parens:
            parens -= 1
        elif char == delimiter and braces == 0 and parens == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return parts


def _decode_value(value: str) -> str:
    chunks = _split_top_level(value, "#")
    decoded = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) >= 2 and chunk[0] == "{" and chunk[-1] == "}":
            chunk = chunk[1:-1]
        elif len(chunk) >= 2 and chunk[0] == '"' and chunk[-1] == '"':
            chunk = chunk[1:-1]
        decoded.append(re.sub(r"\\([{}%&#_$])", r"\1", chunk).strip())
    return "".join(decoded)


def parse_bibtex(text: str) -> list[Paper]:
    """Parse common article, inproceedings, book, thesis, and misc entries."""
    if not text.strip():
        raise ValueError("Paste BibTeX entries first")
    papers: dict[str, Paper] = {}
    entry_start = re.compile(r"@\s*([A-Za-z]+)\s*([({])")
    for match in entry_start.finditer(text):
        entry_type = match.group(1).casefold()
        if entry_type in {"comment", "preamble", "string"}:
            continue
        opening = match.group(2)
        start = match.end()
        end = _entry_end(text, start, opening)
        pieces = _split_top_level(text[start:end], ",")
        if not pieces:
            continue
        cite_key = pieces[0].strip()
        if not cite_key:
            continue
        fields = {}
        for piece in pieces[1:]:
            assignment = piece.split("=", 1)
            if len(assignment) == 2:
                name = assignment[0].strip().casefold()
                if name:
                    fields[name] = _decode_value(assignment[1])
        title = " ".join(fields.get("title", "").split())
        if not title:
            continue
        doi = clean_doi(fields.get("doi"))
        arxiv_id = clean_arxiv_id(fields.get("eprint") or fields.get("arxiv"))
        year_text = fields.get("year", "")
        try:
            year = int(re.search(r"\d{4}", year_text).group(0)) if re.search(r"\d{4}", year_text) else None
        except (TypeError, ValueError):
            year = None
        authors = [" ".join(author.split()) for author in re.split(r"\s+and\s+", fields.get("author", ""), flags=re.I)
                   if author.strip()]
        venue = fields.get("journal") or fields.get("booktitle") or fields.get("publisher") or ""
        open_pdf = fields.get("pdf") or fields.get("pdfurl") or ""
        if open_pdf and not open_pdf.startswith("https://"):
            open_pdf = ""
        paper_id = canonical_paper_id(doi=doi, arxiv_id=arxiv_id) if doi or arxiv_id else f"bibtex:{cite_key.casefold()}"
        paper = Paper(
            id=paper_id, title=title, abstract=" ".join(fields.get("abstract", "").split()), year=year,
            authors=authors, venue=" ".join(venue.split()), doi=doi, arxiv_id=arxiv_id,
            url=fields.get("url", "").strip(), open_access_url=open_pdf,
            topics=[x.strip() for x in fields.get("keywords", "").split(",") if x.strip()],
            providers=["BibTeX import"],
        )
        if paper_id in papers:
            papers[paper_id] = merge_papers(papers[paper_id], paper)
        else:
            papers[paper_id] = paper
    if not papers:
        raise ValueError("No supported BibTeX entries with a title were found")
    return list(papers.values())

