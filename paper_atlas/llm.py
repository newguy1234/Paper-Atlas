"""Optional local-Ollama extraction and synthesis; every extracted fact keeps a text quote."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any

import requests

from paper_atlas.models import Paper
from paper_atlas.providers import unpaywall_pdf_url
from paper_atlas.store import PaperStore
from paper_atlas.paths import downloads_folder, output_folder

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "olmo-3:7b-instruct"
PDF_DIR = downloads_folder()
MAX_PDF_BYTES = 35 * 1024 * 1024


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _safe_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)[:90] or "paper"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{slug}_{digest}"


def fetch_open_pdf(paper: Paper, *, max_bytes: int = MAX_PDF_BYTES,
                   destination_dir: Path | str | None = None) -> Path | None:
    """Download only a provider-listed OA PDF or the paper's public arXiv PDF."""
    if paper.arxiv_id:
        url = f"https://arxiv.org/pdf/{paper.arxiv_id}"
    else:
        url = paper.open_access_url or unpaywall_pdf_url(paper.doi)
    if not url or not url.startswith("https://"):
        return None
    output_dir = output_folder(destination_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{_safe_name(paper.id)}.pdf"
    if destination.is_file() and destination.stat().st_size:
        return destination
    try:
        with requests.get(url, headers={"User-Agent": "PaperAtlas/0.1"}, stream=True,
                          timeout=(10, 45), allow_redirects=True) as response:
            response.raise_for_status()
            size = int(response.headers.get("Content-Length", "0") or 0)
            if size and size > max_bytes:
                raise ValueError(f"Open PDF is larger than the {max_bytes // (1024 * 1024)} MB download limit")
            temp = destination.with_suffix(".part")
            total = 0
            with temp.open("wb") as out:
                first = b""
                for block in response.iter_content(1024 * 128):
                    if not block:
                        continue
                    if not first:
                        first = block[:8]
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError(f"Open PDF exceeded the {max_bytes // (1024 * 1024)} MB download limit")
                    out.write(block)
            if not first.startswith(b"%PDF-"):
                temp.unlink(missing_ok=True)
                raise ValueError("Provider link did not return a PDF")
            temp.replace(destination)
        return destination
    except Exception:
        destination.with_suffix(".part").unlink(missing_ok=True)
        raise


def read_pdf_pages(path: Path, max_pages: int = 80) -> list[str]:
    import fitz

    document = fitz.open(path)
    try:
        pages = []
        for page_no, page in enumerate(document, 1):
            text = " ".join(page.get_text().split())
            if text:
                pages.append(f"[PAGE {page_no}] {text}")
            if page_no >= max_pages:
                break
        return pages
    finally:
        document.close()


def _ollama_chat(prompt: str, *, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_OLLAMA_URL,
                 temperature: float = 0.0, timeout: int = 600, json_mode: bool = True) -> str:
    payload = {"model": model, "stream": False, "messages": [
        {"role": "system", "content": ("Follow the task exactly. Return valid JSON only." if json_mode
                                          else "Follow the task exactly and use the requested response format. Do not wrap the answer in JSON.")},
        {"role": "user", "content": prompt},
    ], "options": {"temperature": temperature}}
    if json_mode:
        payload["format"] = "json"
    response = requests.post(base_url.rstrip("/") + "/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("The local model response was not a JSON object")
    return data


def analyze_paper(paper: Paper, store: PaperStore, *, model: str = DEFAULT_MODEL,
                  base_url: str = DEFAULT_OLLAMA_URL, fetch_pdf: bool = True,
                  pdf_dir: Path | str | None = None) -> dict[str, Any]:
    """Extract methods and paper-specific claims, accepting only quotes found in source text."""
    page_texts: list[str] = []
    pdf_path: Path | None = None
    download_error = ""
    if fetch_pdf and (paper.open_access_url or paper.arxiv_id):
        try:
            pdf_path = fetch_open_pdf(paper, destination_dir=pdf_dir)
        except Exception as exc:
            # Abstract extraction is still useful if the open full-text endpoint
            # is unavailable, but make that downgrade visible in the result.
            download_error = str(exc)
    if pdf_path:
        page_texts = read_pdf_pages(pdf_path)
    source_text = "\n".join(page_texts) if page_texts else f"[ABSTRACT] {paper.abstract}"
    source_text = source_text[:30000]
    if not source_text.strip():
        raise ValueError("This record has no abstract or downloadable open full text")
    prompt = f"""Extract only paper-specific information from this source.
Return JSON with this shape:
{{"methods":[{{"label":"short method name","statement":"what the paper does","evidence":"verbatim source quote","page":1,"confidence":0.8}}],
 "claims":[{{"label":"short finding or claim","statement":"what the paper reports","evidence":"verbatim source quote","page":1,"confidence":0.8}}]}}

Rules:
- Use at most 6 methods and 6 claims; prefer concrete, named methods and stated findings.
- Every evidence value must be an exact, contiguous quotation from the supplied text, under 45 words.
- Do not infer results from general background. Do not repair an unclear acronym.
- If the source is an abstract, use only its stated content and set page to null.
- Confidence is a rough extraction confidence, not a probability of truth.

Paper: {paper.title} ({paper.year or 'year unknown'})
Source text:
{source_text}
"""
    data = _parse_json(_ollama_chat(prompt, model=model, base_url=base_url))
    valid: list[dict[str, Any]] = []
    all_text = _norm(source_text)
    normalized_pages = []
    for page_text in page_texts:
        match = re.match(r"\[PAGE\s+(\d+)\]\s*(.*)", page_text)
        if match:
            normalized_pages.append((_norm(match.group(2)), int(match.group(1))))
    for kind in ("methods", "claims"):
        for item in data.get(kind, [])[:6]:
            if not isinstance(item, dict):
                continue
            label = " ".join(str(item.get("label", "")).split())[:180]
            evidence = " ".join(str(item.get("evidence", "")).split())
            normalized_evidence = _norm(evidence)
            if not label or len(evidence.split()) > 45 or normalized_evidence not in all_text:
                continue
            # Never trust the model's page number. Find the exact source page
            # containing the quote; abstract-only facts do not have page numbers.
            page = next((page_no for page_text, page_no in normalized_pages
                         if normalized_evidence in page_text), None)
            try:
                confidence = float(item.get("confidence", 0.65))
            except (TypeError, ValueError):
                confidence = 0.65
            valid.append({"kind": "method" if kind == "methods" else "claim", "label": label,
                          "statement": " ".join(str(item.get("statement", "")).split())[:600],
                          "evidence": evidence, "page": page, "confidence": max(0.0, min(confidence, 1.0))})
    store.save_facts(paper.id, valid, model)
    source_kind = "open full text" if page_texts else ("abstract (PDF unavailable)" if download_error else "abstract")
    return {"paper_id": paper.id, "facts": valid, "source_kind": source_kind,
            "pdf_path": str(pdf_path) if pdf_path else None, "download_error": download_error, "model": model}


def infer_relation(left: Paper, right: Paper, store: PaperStore, *, model: str = DEFAULT_MODEL,
                   base_url: str = DEFAULT_OLLAMA_URL) -> dict[str, Any]:
    prompt = f"""Compare these two papers using only their abstracts.
Return JSON: {{"relation":"one of supports, contradicts, qualifies, trade-off, extends, compares, applies, evaluates, complements, related",
"explanation":"briefly state whether findings conflict, agree, or differ because of task/data/metric/conditions",
"evidence_source":"verbatim quote from paper A","evidence_target":"verbatim quote from paper B","confidence":0.0}}.
Use contradicts only when the papers test materially comparable claims under comparable conditions and report incompatible results.
Use qualifies when one paper narrows or limits the other; use trade-off when benefits and costs differ. Otherwise use related.
Both evidence values must be exact quotes from the provided abstract. Do not infer a citation relationship.

Paper A [{left.id}] {left.title}: {left.abstract[:5000]}
Paper B [{right.id}] {right.title}: {right.abstract[:5000]}
"""
    data = _parse_json(_ollama_chat(prompt, model=model, base_url=base_url))
    relation = str(data.get("relation", "related")).casefold().strip()
    allowed = {"supports", "contradicts", "qualifies", "trade-off", "extends", "compares",
               "contrasts", "applies", "evaluates", "complements", "related"}
    if relation not in allowed:
        relation = "related"
    evidence_a = " ".join(str(data.get("evidence_source", "")).split())
    evidence_b = " ".join(str(data.get("evidence_target", "")).split())
    if not evidence_a or not evidence_b or _norm(evidence_a) not in _norm(left.abstract) or _norm(evidence_b) not in _norm(right.abstract):
        raise ValueError("The model did not provide verifiable quotes from both abstracts")
    try:
        confidence = float(data.get("confidence", 0.6))
    except (TypeError, ValueError):
        confidence = 0.6
    result = {"source_paper_id": left.id, "target_paper_id": right.id, "relation": relation,
              "explanation": " ".join(str(data.get("explanation", "")).split()),
              "evidence_source": evidence_a, "evidence_target": evidence_b,
              "confidence": max(0.0, min(confidence, 1.0))}
    store.save_relation(result, model)
    return result


def answer_from_collection(question: str, papers: list[Paper], facts: list[dict[str, Any]], *,
                           model: str = DEFAULT_MODEL, base_url: str = DEFAULT_OLLAMA_URL) -> dict[str, Any]:
    """Generate a quote-grounded comparison; unverified model claims never reach the UI."""
    if not papers:
        raise ValueError("No relevant papers are saved in this collection yet")
    blocks = []
    citations = {}
    facts_by_paper: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        facts_by_paper.setdefault(fact["paper_id"], []).append(fact)
    for index, paper in enumerate(papers[:8], 1):
        ref = f"P{index}"
        paper_facts = facts_by_paper.get(paper.id, [])
        page_facts = sorted({str(fact["page"]) for fact in paper_facts if fact.get("page")})
        basis = "abstract only"
        if paper_facts:
            basis = ("abstract and quote-checked full-text facts (pages " + ", ".join(page_facts) + ")"
                     if page_facts else "abstract and extracted quote-checked facts (page not recorded)")
        citations[ref] = {"paper_id": paper.id, "title": paper.title, "year": paper.year,
                          "url": paper.url or paper.open_access_url, "evidence_basis": basis}
        abstract = paper.abstract[:1200]
        details = [f"Title: {paper.title}", f"Year: {paper.year or 'unknown'}", f"Abstract: {abstract}"]
        for fact in paper_facts[:1]:
            basis = f"full-text page {fact['page']}" if fact.get("page") else "page not recorded; treat as abstract-level unless quote confirms otherwise"
            details.append(f"{fact['kind'].title()}: {fact['label']} — {fact['statement'][:300]} Evidence quote: {fact['evidence'][:350]} Evidence basis: {basis}")
        blocks.append(f"[{ref}] " + "\n".join(details))
    prompt = f"""Extract evidence for a critical literature comparison. Return one JSON object only, schema:
{{"claims":[{{"ref":"P1","kind":"benefit|downside|finding|limitation|mixed","claim":"short conservative paraphrase","quote":"exact contiguous source quote","approach":"named approach if explicit, else empty","strength":"measured_result|author_claim|unclear"}}],"comparisons":[{{"left":"P1","right":"P2","relation":"direct_conflict|qualifies|different_scope|agreement","explanation":"brief reason","left_quote":"exact quote from left record","right_quote":"exact quote from right record","same_task_metric_conditions":false}}]}}
Rules: every claim must have an exact verbatim quote from that paper's abstract or supplied quote-checked fact. Never infer a numerical result, metric, dataset, cost, or benefit absent from its quote. If the abstract does not report a downside/cost, do not invent one. Prefer fewer supported claims to broad guesses. A direct_conflict requires opposing reported outcomes on the same task, metric, and conditions; otherwise use qualifies or different_scope. Only emit comparisons when both records explicitly support the comparison. Do not use outside knowledge.

Research question: {question}
Paper records:\n{chr(10).join(blocks)}
"""
    raw = _ollama_chat(prompt, model=model, base_url=base_url, temperature=0.0, json_mode=True)
    data = _parse_json(raw)

    def quote_is_supported(ref: str, quote: str) -> bool:
        source = citations.get(ref)
        paper = next((p for p in papers[:8] if source and p.id == source["paper_id"]), None)
        if not paper or not quote.strip():
            return False
        allowed = [paper.abstract]
        allowed.extend(str(fact.get("evidence", "")) for fact in facts_by_paper.get(paper.id, []))
        needle = _norm(quote)
        return len(needle) >= 24 and any(needle in _norm(text) for text in allowed)

    claims = []
    for item in data.get("claims", []):
        if not isinstance(item, dict):
            continue
        ref, quote = str(item.get("ref", "")), str(item.get("quote", ""))
        if ref in citations and quote_is_supported(ref, quote):
            claims.append({**item, "ref": ref, "quote": quote.strip()})
    comparisons = []
    for item in data.get("comparisons", []):
        if not isinstance(item, dict):
            continue
        left, right = str(item.get("left", "")), str(item.get("right", ""))
        if (left in citations and right in citations and left != right
                and quote_is_supported(left, str(item.get("left_quote", "")))
                and quote_is_supported(right, str(item.get("right_quote", "")))):
            relation = str(item.get("relation", "different_scope"))
            # A model cannot establish comparability from abstracts alone. Keep direct conflicts
            # provisional unless it explicitly asserts matched conditions and both quotes survive.
            if relation == "direct_conflict" and not item.get("same_task_metric_conditions"):
                relation = "qualifies"
            comparisons.append({**item, "left": left, "right": right, "relation": relation})

    grouped = {kind: [claim for claim in claims if claim.get("kind") == kind]
               for kind in ("finding", "benefit", "downside", "limitation", "mixed")}
    lines = ["## Evidence synthesis", "Claims below are included only when the supplied abstract or a saved quote contains the cited text."]
    if grouped["finding"]:
        lines.append("### Findings and areas of agreement")
        lines.extend(f"- {c.get('claim','')} [{c['ref']}] ({c.get('strength','unclear')}). Evidence: “{c['quote']}”" for c in grouped["finding"])
    if grouped["benefit"] or grouped["downside"] or grouped["limitation"]:
        lines.append("### Benefits and downsides")
        lines.append("| Type | Approach | Evidence-based claim | Strength |")
        lines.append("|---|---|---|---|")
        for kind in ("benefit", "downside", "limitation"):
            for c in grouped[kind]:
                claim = str(c.get("claim", "")).replace("|", "\\|")
                approach = str(c.get("approach", "Not specified")).replace("|", "\\|")
                lines.append(f"| {kind.title()} | {approach} | {claim} [{c['ref']}] — “{c['quote']}” | {c.get('strength','unclear')} |")
        if not grouped["downside"] and not grouped["limitation"]:
            lines.append("\nThe supplied records did not provide a quote-supported downside or cost; this is missing evidence, not evidence of no downside.")
    lines.append("### Disagreements and contradictions")
    if comparisons:
        for c in comparisons:
            lines.append(f"- **{c['relation'].replace('_',' ').title()}** [{c['left']}] vs [{c['right']}]: {c.get('explanation','')} Left evidence: “{c['left_quote']}” Right evidence: “{c['right_quote']}”")
    else:
        lines.append("No quote-supported paper-to-paper contradiction was established from these records. That does not prove consensus; the searched abstracts may omit conflicting results or comparable task/metric details.")
    if grouped["mixed"]:
        lines.append("### Mixed evidence")
        lines.extend(f"- {c.get('claim','')} [{c['ref']}] — “{c['quote']}”" for c in grouped["mixed"])
    lines.append("### Evidence limits\nThe comparison is limited to the retrieved abstracts and saved quote-checked facts. An abstract-only search cannot establish full-text effect sizes or reliably compare methods unless task, metric, and conditions are reported.")
    return {"answer": "\n\n".join(lines), "sources": citations,
            "verified_claim_count": len(claims), "verified_comparison_count": len(comparisons)}
