from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_atlas.llm import fetch_open_pdf, read_pdf_pages
from paper_atlas.service import (ask_collection, discover, discover_research_pack,
                                 download_selected_open_pdfs, expand_collection, extract_selected,
                                 get_graph, import_bibtex)
from paper_atlas.store import PaperStore
from paper_atlas.paths import output_folder

RESEARCH_PRESETS = {
    "qasa": [
        "scientific question answering evidence retrieval reasoning QASA",
        "scientific RAG retrieval reranking evidence coverage scientific papers",
        "adaptive query decomposition multi-intent retrieval scientific question answering",
        "long context RAG scientific QA evidence ordering source structure",
        "table and numerical retrieval augmented generation scientific papers",
        "faithful answer generation evidence verification retrieval augmented generation",
    ],
    "qasa_transfer": [
        "biomedical question answering retrieval augmented generation evidence selection rationale retrieval benchmark",
        "biology literature question answering RAG evidence coverage multi hop retrieval evaluation",
        "chemistry scientific question answering RAG retrieval benchmark PubChem chemical literature",
        "molecular property prediction retrieval augmented generation molecular structure descriptors similar molecules",
        "physics engineering technical question answering literature RAG retrieval evidence tables figures",
        "nuclear physics accelerator scientific literature retrieval augmented question answering system evaluation",
        "Electron-Ion Collider retrieval augmented question answering scientific literature system",
        "cross-domain scientific RAG transfer study domain adaptation retriever evidence grounding ablation",
        "scientific QA retrieval query refinement evidence sufficiency early stopping biomedical chemistry physics",
        "scientific RAG retrieval distractor filtering corpus diversity source bias answer grounding",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="paper-atlas", description="Search papers and explore a cited, evidence-linked graph")
    parser.add_argument("--db", default=None, help="SQLite library path (defaults to data/paper_atlas/library.sqlite3)")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="Search OpenAlex, arXiv, and Semantic Scholar")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--from-year", type=int, default=None)
    search.add_argument("--scope", choices=("cs_methods", "cs_applied"), default="cs_methods")
    search.add_argument("--providers", nargs="+", choices=("openalex", "arxiv", "semanticscholar"),
                         default=("openalex", "arxiv", "semanticscholar"))
    search.add_argument("--semantic", action="store_true",
                        help="Also retrieve OpenAlex embedding neighbors of the query")

    expand = sub.add_parser("expand", help="Add citing, recommended, and semantically similar papers to a collection")
    expand.add_argument("collection_id")
    expand.add_argument("paper_id")
    expand.add_argument("--limit", type=int, default=12)

    research = sub.add_parser("research-pack", help="Run a multi-query, CS-scoped literature search")
    research.add_argument("--preset", choices=tuple(RESEARCH_PRESETS), default="qasa",
                          help="qasa focuses on computer-science methods; qasa_transfer searches applied systems in science domains")
    research.add_argument("--query", action="append", default=[], help="Add or override with a query; repeatable")
    research.add_argument("--name", default="QASA RAG improvement literature")
    research.add_argument("--limit-per-query", type=int, default=12)
    research.add_argument("--from-year", type=int, default=2023)
    research.add_argument("--scope", choices=("cs_methods", "cs_applied"), default="cs_methods")
    research.add_argument("--providers", nargs="+", choices=("openalex", "arxiv", "semanticscholar"),
                          default=("openalex", "arxiv", "semanticscholar"))
    research.add_argument("--out", default="", help="Also save the full JSON result to this path")

    bibtex = sub.add_parser("bibtex", help="Import a BibTeX file into a local collection")
    bibtex.add_argument("file")
    bibtex.add_argument("--name", default="Imported BibTeX")

    graph = sub.add_parser("graph", help="Export a saved collection as graph JSON")
    graph.add_argument("collection_id")
    graph.add_argument("--limit", type=int, default=40)
    graph.add_argument("--out", default="")

    ask = sub.add_parser("ask", help="Ask a saved collection using local Ollama")
    ask.add_argument("collection_id")
    ask.add_argument("question")
    ask.add_argument("--model", default="olmo-3:7b-instruct")
    ask.add_argument("--ollama-url", default="http://127.0.0.1:11434")

    extract = sub.add_parser("analyze", help="Extract methods and claims with quote validation")
    extract.add_argument("paper_id")
    extract.add_argument("--model", default="olmo-3:7b-instruct")
    extract.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    extract.add_argument("--pdf-dir", default=None, help="Folder containing/saving PDFs (OneDrive sync folder is supported)")

    download = sub.add_parser("download", help="Download open PDFs for selected papers into a local or OneDrive folder")
    download.add_argument("paper_ids", nargs="+", help="One or more saved paper IDs")
    download.add_argument("--out-dir", default=None, help="Output folder; defaults to saved folder or Downloads")

    read = sub.add_parser("read", help="Download if needed and extract full-text PDF pages without using a local model")
    read.add_argument("paper_ids", nargs="+", help="One or more saved paper IDs")
    read.add_argument("--pdf-dir", default=None, help="PDF folder; defaults to saved folder or Downloads")
    read.add_argument("--out", default="", help="Write extracted page text to a UTF-8 file instead of the terminal")

    args = parser.parse_args()
    store = PaperStore(args.db)
    if args.command == "search":
        result = discover(args.query, limit=args.limit, from_year=args.from_year,
                          providers=tuple(args.providers), scope=args.scope,
                          semantic=args.semantic, store=store)
    elif args.command == "expand":
        result = expand_collection(args.paper_id, collection_id=args.collection_id,
                                   limit=args.limit, store=store)
    elif args.command == "research-pack":
        queries = args.query or RESEARCH_PRESETS[args.preset]
        result = discover_research_pack(
            queries, limit_per_query=args.limit_per_query, from_year=args.from_year,
            providers=tuple(args.providers), scope=args.scope, name=args.name, store=store,
        )
        if args.out:
            output = Path(args.out).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({
                "collection_id": result["collection_id"],
                "count": result["count"],
                "successful_queries": result["successful_queries"],
                "query_counts": result["query_counts"],
                "warnings": result["warnings"],
                "saved_json": str(output),
            }, ensure_ascii=True, indent=2))
            return
    elif args.command == "bibtex":
        result = import_bibtex(Path(args.file).read_text(encoding="utf-8"),
                               store=store, collection_name=args.name)
    elif args.command == "graph":
        result = get_graph(args.collection_id, limit=args.limit, store=store)
        if args.out:
            Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote {args.out}")
            return
    elif args.command == "ask":
        result = ask_collection(args.question, args.collection_id, model=args.model,
                                base_url=args.ollama_url, store=store)
    elif args.command == "download":
        result = download_selected_open_pdfs(args.paper_ids, output_dir=args.out_dir, store=store)
    elif args.command == "read":
        documents = []
        for paper_id in dict.fromkeys(args.paper_ids):
            paper = store.get_paper(paper_id)
            if not paper:
                documents.append(f"## {paper_id}\nERROR: paper not found in the local library")
                continue
            try:
                pdf = fetch_open_pdf(paper, destination_dir=output_folder(args.pdf_dir, store))
                if not pdf:
                    documents.append(f"## {paper.title} [{paper_id}]\nERROR: no provider-listed open PDF")
                    continue
                pages = read_pdf_pages(pdf)
                documents.append(f"## {paper.title} [{paper_id}]\nPDF: {pdf}\n" + "\n".join(pages))
            except Exception as exc:
                documents.append(f"## {paper.title} [{paper_id}]\nERROR: {exc}")
        text = "\n\n".join(documents)
        if args.out:
            output = Path(args.out).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
            print(f"Wrote {output.resolve()}")
            return
        print(text)
    else:
        result = extract_selected(args.paper_id, model=args.model, base_url=args.ollama_url,
                                  pdf_dir=args.pdf_dir, store=store)
    # Windows terminals often use cp1252 and cannot encode mathematical symbols in titles.
    # ASCII-safe JSON remains valid and avoids losing a successful search at print time.
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
