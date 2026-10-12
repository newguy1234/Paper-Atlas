"""Solara desktop-local research app for Paper Atlas."""

from __future__ import annotations

import html
import os
from pathlib import Path
from urllib.parse import urlparse

import networkx as nx
import solara
from matplotlib.figure import Figure
from solara.lab import use_task

from paper_atlas.exports import collection_bibtex, collection_csv, collection_json
from paper_atlas.graph import build_graph, explain_relationships
from paper_atlas.llm import DEFAULT_MODEL, DEFAULT_OLLAMA_URL, PDF_DIR
from paper_atlas.service import (ask_collection, compare_papers, discover,
                                 download_selected_open_pdfs, expand_collection, extract_selected_many,
                                 import_bibtex)
from paper_atlas.store import PaperStore

STORE = PaperStore()


def _search_worker(query: str, limit: int, from_year: int | None,
                   providers: tuple[str, ...], scope: str, semantic: bool):
    return discover(query, limit=limit, from_year=from_year, providers=providers,
                    scope=scope, semantic=semantic, store=STORE)


def _expand_worker(paper_id: str, collection_id: str):
    return expand_collection(paper_id, collection_id=collection_id, store=STORE)


def _download_worker(paper_ids: list[str], pdf_dir: str):
    return download_selected_open_pdfs(paper_ids, output_dir=pdf_dir, store=STORE)


def _extract_worker(paper_ids: list[str], model: str, base_url: str, pdf_dir: str):
    return extract_selected_many(paper_ids, model=model, base_url=base_url,
                                 pdf_dir=pdf_dir, store=STORE)


def _bibtex_worker(text: str):
    return import_bibtex(text, store=STORE)


def _compare_worker(left_id: str, right_id: str, model: str, base_url: str):
    return compare_papers(left_id, right_id, model=model, base_url=base_url, store=STORE)


def _ask_worker(question: str, collection_id: str, model: str, base_url: str):
    return ask_collection(question, collection_id, model=model, base_url=base_url, store=STORE)


def _figure(graph: nx.MultiDiGraph, selected_id: str | None) -> Figure:
    figure = Figure(figsize=(11, 6.5), layout="constrained")
    ax = figure.subplots()
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Search a topic to build your first paper map", ha="center", va="center")
        ax.axis("off")
        return figure
    positions = nx.spring_layout(graph, seed=17, k=0.8, iterations=80)
    kind_colors = {"paper": "#4776a8", "topic": "#55a99b", "method": "#e8b94e", "claim": "#d86568"}
    for kind, shape in (("paper", "o"), ("topic", "s"), ("method", "D"), ("claim", "d")):
        nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("kind") == kind]
        if not nodes:
            continue
        colors = ["#f2cf64" if graph.nodes[node].get("paper_id") == selected_id else kind_colors[kind]
                  for node in nodes]
        sizes = [1000 if graph.nodes[node].get("paper_id") == selected_id else (680 if kind == "paper" else 420)
                 for node in nodes]
        nx.draw_networkx_nodes(graph, positions, nodelist=nodes, node_color=colors, node_shape=shape,
                               node_size=sizes, edgecolors="#30343b", linewidths=0.7, ax=ax)
    edge_styles = {"cites": ("#424a55", "solid"), "tagged-with": ("#4776a8", "dashed"),
                   "uses-method": ("#a98112", "solid"), "reports-claim": ("#c44e52", "dotted")}
    for relation in set(data.get("relation", "related") for _, _, data in graph.edges(data=True)):
        edges = [(source, target) for source, target, data in graph.edges(data=True)
                 if data.get("relation", "related") == relation]
        color, style = edge_styles.get(relation, ("#8172b3", "dotted"))
        edge_options = {"edge_color": color, "style": style, "alpha": 0.65, "width": 1.3, "ax": ax}
        if relation == "cites":
            edge_options.update(arrows=True, arrowsize=12, connectionstyle="arc3,rad=0.04")
        else:
            edge_options["arrows"] = False
        nx.draw_networkx_edges(graph, positions, edgelist=edges, **edge_options)
    labels = {}
    for node, attrs in graph.nodes(data=True):
        label = attrs.get("title") or attrs.get("label") or node
        labels[node] = label[:36] + ("…" if len(label) > 36 else "")
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=7, font_color="#20242a",
                            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.5}, ax=ax)
    ax.set_title("How the papers connect", fontsize=14, pad=12)
    ax.axis("off")
    return figure


def _safe(text: str) -> str:
    return html.escape(text or "", quote=False)


def _safe_url(url: str) -> str:
    parsed = urlparse(url)
    return url if parsed.scheme == "https" and parsed.netloc else ""


def _onedrive_root() -> Path | None:
    for name in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        value = os.getenv(name, "").strip()
        if value and Path(value).is_dir():
            return Path(value).expanduser().resolve()
    try:
        return next((path.resolve() for path in Path.home().glob("OneDrive*") if path.is_dir()), None)
    except OSError:
        return None


def _default_pdf_dir() -> str:
    return str(PDF_DIR)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


@solara.component
def Page():
    recent = STORE.list_collections(limit=1)
    initial_collection = recent[0]["id"] if recent else ""
    initial_papers = STORE.get_papers(initial_collection, limit=100) if initial_collection else []

    query, set_query = solara.use_state("agentic RAG scientific question answering")
    collection_id, set_collection_id = solara.use_state(initial_collection)
    selected_ids, set_selected_ids = solara.use_state([p.id for p in initial_papers])
    detail_id, set_detail_id = solara.use_state(initial_papers[0].id if initial_papers else "")
    compare_id, set_compare_id = solara.use_state(initial_papers[1].id if len(initial_papers) > 1 else "")
    limit, set_limit = solara.use_state(24)
    from_year, set_from_year = solara.use_state(2023)
    provider_openalex, set_provider_openalex = solara.use_state(True)
    provider_arxiv, set_provider_arxiv = solara.use_state(True)
    provider_s2, set_provider_s2 = solara.use_state(True)
    semantic_search, set_semantic_search = solara.use_state(False)
    # The current research workflow is deliberately cross-domain: look for
    # CS retrieval/reasoning methods demonstrated in scientific applications.
    scope, set_scope = solara.use_state("cs_applied")
    model, set_model = solara.use_state(DEFAULT_MODEL)
    ollama_url, set_ollama_url = solara.use_state(DEFAULT_OLLAMA_URL)
    default_pdf_dir = STORE.get_setting("pdf_output_dir", _default_pdf_dir())
    pdf_dir, set_pdf_dir = solara.use_state(default_pdf_dir)
    folder_notice, set_folder_notice = solara.use_state("")
    bibtex_text, set_bibtex_text = solara.use_state("")
    question, set_question = solara.use_state("Compare the evidence across papers: where do findings agree or conflict? Weigh benefits, costs, limitations, and trade-offs, and say when evidence is insufficient.")

    search_task = use_task(_search_worker, dependencies=None, raise_error=False)
    expand_task = use_task(_expand_worker, dependencies=None, raise_error=False)
    download_task = use_task(_download_worker, dependencies=None, raise_error=False)
    extract_task = use_task(_extract_worker, dependencies=None, raise_error=False)
    compare_task = use_task(_compare_worker, dependencies=None, raise_error=False)
    ask_task = use_task(_ask_worker, dependencies=None, raise_error=False)
    bibtex_task = use_task(_bibtex_worker, dependencies=None, raise_error=False)

    def apply_search_result():
        if search_task.finished and search_task.value:
            result = search_task.value
            if result["collection_id"] != collection_id:
                ids = [paper["id"] for paper in result["papers"]]
                set_collection_id(result["collection_id"])
                set_selected_ids(ids)
                set_detail_id(ids[0] if ids else "")
                set_compare_id(ids[1] if len(ids) > 1 else "")

    solara.use_effect(apply_search_result, [search_task.value])

    def apply_bibtex_result():
        if bibtex_task.finished and bibtex_task.value:
            result = bibtex_task.value
            ids = [paper["id"] for paper in result["papers"]]
            set_collection_id(result["collection_id"])
            set_selected_ids(ids)
            set_detail_id(ids[0] if ids else "")
            set_compare_id(ids[1] if len(ids) > 1 else "")

    solara.use_effect(apply_bibtex_result, [bibtex_task.value])

    papers = STORE.get_papers(collection_id, limit=100) if collection_id else []
    paper_by_id = {paper.id: paper for paper in papers}
    selected_ids = [paper_id for paper_id in selected_ids if paper_id in paper_by_id]
    if detail_id not in paper_by_id and papers:
        detail_id = selected_ids[0] if selected_ids else papers[0].id
    if compare_id not in paper_by_id and len(papers) > 1:
        compare_id = papers[1].id
    selected_id = detail_id if detail_id in paper_by_id else (selected_ids[0] if selected_ids else "")
    selected_paper = paper_by_id.get(selected_id)

    graph = build_graph(STORE, collection_id, limit=40, selected_paper_id=selected_id) if collection_id else nx.MultiDiGraph()
    relationships = explain_relationships(graph, STORE) if collection_id else []
    facts = STORE.get_facts([selected_id]) if selected_id else []
    warnings = (STORE.get_collection(collection_id) or {}).get("warnings", []) if collection_id else []
    options = [f"{p.year or 'n.d.'} · {p.title} · {p.id[-8:]}" for p in papers]
    option_to_id = {label: paper.id for label, paper in zip(options, papers)}
    id_to_option = {paper.id: label for label, paper in zip(options, papers)}
    selected_options = [id_to_option[paper_id] for paper_id in selected_ids if paper_id in id_to_option]

    def start_search():
        providers = tuple(name for name, enabled in (("openalex", provider_openalex), ("arxiv", provider_arxiv),
                                                     ("semanticscholar", provider_s2)) if enabled)
        if providers:
            search_task(query.strip(), int(limit), int(from_year) if from_year else None, providers, scope,
                        semantic_search)

    def start_expand():
        if collection_id and selected_id:
            expand_task(selected_id, collection_id)

    def start_download():
        if selected_ids:
            download_task(list(selected_ids), pdf_dir)

    def start_extract():
        if selected_ids:
            extract_task(list(selected_ids), model.strip(), ollama_url.strip(), pdf_dir)

    def start_compare():
        if selected_id and compare_id and selected_id != compare_id:
            compare_task(selected_id, compare_id, model.strip(), ollama_url.strip())

    def start_ask():
        if collection_id and question.strip():
            ask_task(question.strip(), collection_id, model.strip(), ollama_url.strip())

    def save_pdf_folder():
        from paper_atlas.paths import output_folder
        candidate = output_folder(pdf_dir)
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            set_folder_notice(f"Could not use that folder: {exc}")
            return
        resolved = str(candidate.resolve())
        STORE.set_setting("pdf_output_dir", resolved)
        set_pdf_dir(resolved)
        set_folder_notice("Saved. New PDFs, extraction downloads and local exports will use this folder.")

    def save_exports():
        from paper_atlas.paths import output_folder
        try:
            folder = output_folder(pdf_dir)
            folder.mkdir(parents=True, exist_ok=True)
            for suffix, export in (("json", collection_json), ("csv", collection_csv), ("bib", collection_bibtex)):
                content = export(collection_id, store=STORE)
                target = folder / f"paper_atlas_{collection_id}.{suffix}"
                if isinstance(content, bytes):
                    target.write_bytes(content)
                else:
                    target.write_text(content, encoding="utf-8")
            set_folder_notice(f"Saved JSON, CSV and BibTeX to {folder}")
        except OSError as exc:
            set_folder_notice(f"Could not save exports: {exc}")

    def reset_save_folder():
        resolved = _default_pdf_dir()
        STORE.set_setting("pdf_output_dir", resolved)
        set_pdf_dir(resolved)
        set_folder_notice("Save location reset to Downloads.")

    def start_bibtex_import():
        if bibtex_text.strip():
            bibtex_task(bibtex_text)

    with solara.AppLayout(title="Paper Atlas"):
        with solara.Column(gap="18px", style={"max-width": "1500px", "margin": "0 auto", "padding": "8px 12px 28px"}):
            solara.Markdown("# ☀️ Paper Atlas\n**A personal research desk for finding papers, tracing ideas, and keeping evidence attached.**")
            with solara.Card():
                solara.Markdown("### Save location")
                solara.InputText(label="Folder for papers and exports (blank = Downloads)",
                                 value=pdf_dir, on_value=set_pdf_dir)
                with solara.Row():
                    solara.Button("Save folder", on_click=save_pdf_folder)
                    solara.Button("Use Downloads", on_click=reset_save_folder)
                solara.Text("Paste a local or OneDrive folder. Downloads is used when no folder is chosen.")
                if folder_notice:
                    solara.Text(folder_notice)
            with solara.Card():
                solara.Markdown("### 1 · Discover literature")
                with solara.Row(style={"align-items": "end", "gap": "12px", "flex-wrap": "wrap"}):
                    solara.InputText(label="Topic or research question", value=query, on_value=set_query,
                                     continuous_update=False, style={"min-width": "320px", "flex": "1"})
                    scope_options = {
                        "Computer science methods · strict primary-field filter": "cs_methods",
                        "Computer science methods used in other fields · broader search": "cs_applied",
                    }
                    if scope not in scope_options.values():
                        scope = "cs_methods"
                        set_scope(scope)
                    scope_label = next(label for label, value in scope_options.items() if value == scope)
                    solara.Select(label="Research scope", values=list(scope_options), value=scope_label,
                                  on_value=lambda label: set_scope(scope_options.get(label, "cs_methods")),
                                  style={"min-width": "360px"})
                    if scope == "cs_applied":
                        solara.Text("Find computer-science retrieval and reasoning methods tested in biology, medicine, chemistry, physics, and engineering systems. This broader scope is for transfer ideas; results still need checking against QASA.")
                    solara.SliderInt(label="Maximum papers", value=limit, on_value=set_limit, min=5, max=50, step=1)
                    solara.InputInt(label="From year", value=from_year, on_value=set_from_year, continuous_update=False)
                    solara.Button("Search indexes", on_click=start_search, color="primary", disabled=search_task.pending)
                with solara.Row(style={"flex-wrap": "wrap"}):
                    solara.Checkbox(label="OpenAlex", value=provider_openalex, on_value=set_provider_openalex)
                    solara.Checkbox(label="arXiv", value=provider_arxiv, on_value=set_provider_arxiv)
                    solara.Checkbox(label="Semantic Scholar", value=provider_s2, on_value=set_provider_s2)
                    solara.Checkbox(label="OpenAlex similar-meaning neighbors", value=semantic_search,
                                    on_value=set_semantic_search)
                    if search_task.pending:
                        solara.ProgressLinear(True)
                        solara.Text("Searching scholarly indexes…")
                if search_task.error:
                    solara.Error(str(search_task.exception))
                if search_task.finished and search_task.value:
                    result = search_task.value
                    solara.Success(f"Saved {result['count']} papers in collection {result['collection_id']}.")
                    if result["warnings"]:
                        solara.Warning("Some indexes did not respond: " + " | ".join(result["warnings"]))

            with solara.Card():
                solara.Markdown("### Or import a BibTeX bibliography")
                solara.InputTextArea(label="Paste BibTeX entries", value=bibtex_text, on_value=set_bibtex_text,
                                     rows=7, auto_grow=False,
                                     placeholder="@article{key, title={...}, author={...}, year={...}, doi={...}}")
                solara.Button("Import into a new collection", on_click=start_bibtex_import,
                              color="primary", disabled=not bibtex_text.strip() or bibtex_task.pending)
                if bibtex_task.pending:
                    solara.ProgressLinear(True)
                if bibtex_task.error:
                    solara.Error(str(bibtex_task.exception))
                if bibtex_task.finished and bibtex_task.value:
                    solara.Success(f"Imported {bibtex_task.value['count']} papers into collection {bibtex_task.value['collection_id']}.")

            if collection_id and papers:
                with solara.Card():
                    solara.Markdown("### 2 · Select and collect papers")
                    solara.Text(f"Collection: {collection_id} · {len(papers)} papers · {len(selected_ids)} selected")
                    with solara.Row(style={"align-items": "end", "gap": "10px", "flex-wrap": "wrap"}):
                        solara.InputText(label="PDF save folder · OneDrive sync folder works here",
                                         value=pdf_dir, on_value=set_pdf_dir, continuous_update=False,
                                         style={"min-width": "420px", "flex": "1"})
                        solara.Button("Save folder", on_click=save_pdf_folder)
                    if folder_notice:
                        solara.Text(folder_notice)
                    detected_onedrive = _onedrive_root()
                    if detected_onedrive and _is_within(Path(pdf_dir).expanduser(), detected_onedrive):
                        solara.Info(f"This destination is inside the detected OneDrive folder. OneDrive will sync downloaded PDFs when its desktop client is running: {detected_onedrive}")
                    elif detected_onedrive:
                        solara.Warning(f"Current PDF folder is outside the detected OneDrive folder ({detected_onedrive}). Change it above to save PDFs into OneDrive.")
                    else:
                        solara.Info("OneDrive was not detected automatically. Paste its synced folder path above; the library database stays local.")
                    if warnings:
                        solara.Warning("Some provider results may be missing; run the search again to retry those indexes.")
                    solara.SelectMultiple(label="Papers to act on", values=selected_options, all_values=options,
                                          on_value=lambda labels: set_selected_ids([option_to_id[x] for x in labels if x in option_to_id]),
                                          style={"min-width": "min(850px, 90vw)"})
                    with solara.Row(style={"align-items": "center", "gap": "10px", "flex-wrap": "wrap"}):
                        solara.Button("Select all", on_click=lambda: set_selected_ids([paper.id for paper in papers]))
                        solara.Button("Clear selection", on_click=lambda: set_selected_ids([]))
                        solara.Button("Download selected open PDFs", on_click=start_download,
                                      color="primary", disabled=not selected_ids or download_task.pending)
                        solara.Button("Scrape & extract methods / claims", on_click=start_extract,
                                      disabled=not selected_ids or extract_task.pending)
                        solara.Button("Save exports to folder", on_click=save_exports)
                        solara.FileDownload(lambda: collection_json(collection_id, store=STORE),
                                            filename=f"paper_atlas_{collection_id}.json",
                                            label="Export JSON + graph", mime_type="application/json")
                        solara.FileDownload(lambda: collection_csv(collection_id, store=STORE),
                                            filename=f"paper_atlas_{collection_id}.csv",
                                            label="Export CSV", mime_type="text/csv")
                        solara.FileDownload(lambda: collection_bibtex(collection_id, store=STORE),
                                            filename=f"paper_atlas_{collection_id}.bib",
                                            label="Export BibTeX", mime_type="application/x-bibtex")
                    if download_task.pending:
                        solara.ProgressLinear(True)
                        solara.Text("Downloading only provider-listed open-access PDFs…")
                    if download_task.error:
                        solara.Error(str(download_task.exception))
                    if download_task.finished and download_task.value:
                        results = download_task.value["results"]
                        downloaded = sum(item["status"] == "downloaded" for item in results)
                        solara.Success(f"Downloaded {downloaded} of {len(results)} selected PDFs.")
                        for item in results:
                            if item.get("status") == "downloaded":
                                solara.Text(f"✓ {item.get('title', item['paper_id'])} · {item.get('path', '')}")
                            else:
                                solara.Warning(f"{item.get('title', item['paper_id'])}: {item.get('message', item['status'])}")
                    if extract_task.pending:
                        solara.ProgressLinear(True)
                        solara.Text("Reading open full text or abstracts and checking every extracted quote…")
                    if extract_task.error:
                        solara.Error(str(extract_task.exception))
                    if extract_task.finished and extract_task.value:
                        result = extract_task.value
                        solara.Success(f"Saved {result['fact_count']} quote-checked facts across {result['count']} papers.")
                        for item in result["results"]:
                            if item.get("error"):
                                solara.Warning(f"{item['paper_id']}: {item['error']}")
                            else:
                                paper = paper_by_id.get(item["paper_id"])
                                title = paper.title if paper else item["paper_id"]
                                solara.Text(f"{title} · {len(item['facts'])} facts · source: {item['source_kind']}")
                                if item.get("download_error"):
                                    solara.Text("Full-text download note: " + item["download_error"])

            if collection_id and papers:
                with solara.Card():
                    solara.Markdown("### 3 · Explore the evidence map")
                    with solara.Row(style={"align-items": "end", "gap": "12px", "flex-wrap": "wrap"}):
                        detail_options = [id_to_option[p.id] for p in papers]
                        detail_value = id_to_option.get(selected_id, detail_options[0])
                        solara.Select(label="Focus paper", values=detail_options, value=detail_value,
                                      on_value=lambda label: set_detail_id(option_to_id.get(label, "")),
                                      style={"min-width": "360px"})
                        compare_value = id_to_option.get(compare_id, detail_options[min(1, len(detail_options)-1)])
                        solara.Select(label="Compare with", values=detail_options, value=compare_value,
                                      on_value=lambda label: set_compare_id(option_to_id.get(label, "")),
                                      style={"min-width": "360px"})
                        solara.InputText(label="Local Ollama model", value=model, on_value=set_model,
                                         continuous_update=False)
                        solara.InputText(label="Ollama address", value=ollama_url, on_value=set_ollama_url,
                                         continuous_update=False)
                        solara.Button("Compare abstracts", on_click=start_compare,
                                      disabled=not compare_id or compare_id == selected_id or compare_task.pending)
                        solara.Button("Expand related papers", on_click=start_expand,
                                      disabled=not selected_id or expand_task.pending)
                    if expand_task.pending:
                        solara.ProgressLinear(True)
                        solara.Text("Looking up citations, recommendations, and similar abstracts…")
                    if expand_task.error:
                        solara.Error(str(expand_task.exception))
                    if expand_task.finished and expand_task.value:
                        expanded = expand_task.value
                        solara.Success(
                            f"Added {expanded['added']} related papers to the collection "
                            f"({expanded['count']} total)."
                        )
                        if expanded.get("warnings"):
                            solara.Warning(" | ".join(expanded["warnings"]))
                    if compare_task.pending:
                        solara.ProgressLinear(True)
                    if compare_task.error:
                        solara.Error(str(compare_task.exception))
                    if compare_task.finished and compare_task.value:
                        relation = compare_task.value
                        solara.Markdown(f"**{_safe(relation['relation'].title())}:** {_safe(relation['explanation'])}  \n"
                                        f"Paper A evidence: “{_safe(relation['evidence_source'])}”  \n"
                                        f"Paper B evidence: “{_safe(relation['evidence_target'])}”  \n"
                                        f"Model confidence: {relation['confidence']:.2f} · inferred relation, not citation metadata")

                    graph_nodes = graph.number_of_nodes()
                    graph_edges = graph.number_of_edges()
                    with solara.Row(style={"align-items": "flex-start", "gap": "24px"}):
                        with solara.Column(style={"flex": "1.5", "min-width": "520px"}):
                            solara.Text(f"{len(papers)} papers · {graph_nodes} visible graph nodes · {graph_edges} links")
                            solara.FigureMatplotlib(_figure(graph, selected_id),
                                                    dependencies=[collection_id, selected_id, graph_nodes, graph_edges])
                            solara.Markdown("**Map key:** circles are papers · squares are shared topics · gold diamonds are extracted methods · red diamonds are claims · arrows record citations · dotted purple links are model-inferred relations")
                        with solara.Column(style={"flex": "1", "min-width": "320px"}):
                            if selected_paper:
                                solara.Markdown(f"## {_safe(selected_paper.title)}")
                                solara.Text(f"{selected_paper.year or 'Year unknown'} · {', '.join(selected_paper.authors[:5]) or 'Authors unavailable'} · {selected_paper.venue or 'Venue unavailable'}")
                                url = _safe_url(selected_paper.url)
                                if url:
                                    solara.Markdown(f"[Open paper]({url})")
                                solara.Markdown(_safe(selected_paper.abstract) if selected_paper.abstract else "No abstract is available from the selected indexes.")
                                if selected_paper.open_access_url or selected_paper.arxiv_id:
                                    solara.Text("An open full-text link is available. Download starts only when you select it.")
                                if facts:
                                    solara.Markdown("### Verified methods and claims")
                                    for fact in facts[:10]:
                                        page = f" · page {fact['page']}" if fact.get("page") else ""
                                        solara.Markdown(f"**{_safe(fact['kind'].title())}: {_safe(fact['label'])}**  \n"
                                                        f"{_safe(fact['statement'])}  \n> {_safe(fact['evidence'])}  \n"
                                                        f"Source: {_safe(fact['model'])}{page}")
                    if relationships:
                        solara.Markdown("### Why these papers are connected")
                        for item in relationships[:14]:
                            evidence = f" · evidence: {item['evidence']}" if item["evidence"] else ""
                            solara.Markdown(f"- **{_safe(item['relation'])}** · {_safe(item['explanation'])} "
                                            f"({_safe(item['basis'])}){_safe(evidence)}")

            if collection_id:
                with solara.Card():
                    solara.Markdown("### 4 · Ask across this collection")
                    solara.InputText(label="Research question", value=question, on_value=set_question,
                                     continuous_update=False, style={"width": "100%"})
                    solara.Text("Atlas searches separately for main findings, benefits, limitations, and mixed or conflicting results. It distinguishes direct contradictions from differences in task, data, metric, or conditions.")
                    solara.Button("Compare evidence, disagreements & trade-offs", on_click=start_ask,
                                  color="primary", disabled=not papers or ask_task.pending)
                    if ask_task.pending:
                        solara.ProgressLinear(True)
                        solara.Text("Reading the saved records with your local Ollama model…")
                    if ask_task.error:
                        solara.Error(str(ask_task.exception))
                    if ask_task.finished and ask_task.value:
                        result = ask_task.value
                        if result.get("evidence_views"):
                            view_summary = " · ".join(
                                f"{_safe(label.title())}: {len(ids)} candidates"
                                for label, ids in result["evidence_views"].items()
                            )
                            solara.Text(f"Evidence scan — {view_summary}")
                        solara.Markdown(result["answer"])
                        solara.Markdown("**Sources**")
                        for source in result["sources"].values():
                            url = _safe_url(source.get("url") or "")
                            title = _safe(source["title"])
                            label = f"[{source.get('year') or 'n.d.'}] {title}"
                            basis = _safe(source.get("evidence_basis", "abstract only"))
                            solara.Markdown(f"- [{label}]({url}) · `{_safe(source['paper_id'])}` · {basis}" if url
                                            else f"- {label} · `{_safe(source['paper_id'])}` · {basis}")
            else:
                solara.Info("Search above to create a local paper collection. The library and downloaded PDFs stay on this computer.")

            solara.Markdown(f"---  \nLocal library: `{STORE.path}`  ·  Open PDFs: `{Path(PDF_DIR)}`  ·  Search metadata comes from OpenAlex, arXiv, and Semantic Scholar.")
