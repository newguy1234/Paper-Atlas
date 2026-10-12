from __future__ import annotations

import re
from collections import Counter
from typing import Any

import networkx as nx

from paper_atlas.models import Paper
from paper_atlas.store import PaperStore


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:90]


def build_graph(store: PaperStore, collection_id: str | None = None, *, limit: int = 40,
                selected_paper_id: str | None = None, include_facts: bool = True) -> nx.MultiDiGraph:
    """Build a traceable paper/topic/method/claim graph from locally stored evidence."""
    papers = store.get_papers(collection_id, limit=max(1, min(limit, 150)))
    ids = {paper.id for paper in papers}
    graph = nx.MultiDiGraph()
    openalex_to_id = {paper.openalex_id: paper.id for paper in papers if paper.openalex_id}

    for paper in papers:
        node_id = f"paper:{paper.id}"
        graph.add_node(node_id, kind="paper", paper_id=paper.id, title=paper.title,
                       year=paper.year, citation_count=paper.citation_count,
                       authors=paper.authors[:5], url=paper.url)

    topic_papers: dict[str, list[str]] = {}
    for paper in papers:
        for topic in paper.topics:
            label = topic.strip()
            if len(label) < 3:
                continue
            topic_id = f"topic:{_slug(label)}"
            topic_papers.setdefault(topic_id, []).append(paper.id)
            graph.add_node(topic_id, kind="topic", label=label)
            graph.add_edge(f"paper:{paper.id}", topic_id, relation="tagged-with",
                           provenance="provider-metadata", evidence=label, confidence=1.0)
        for reference in paper.references:
            target_id = openalex_to_id.get(reference)
            if target_id and target_id != paper.id:
                graph.add_edge(f"paper:{paper.id}", f"paper:{target_id}", relation="cites",
                               provenance="OpenAlex referenced_works", evidence=reference, confidence=1.0)

    # Remove one-off topic nodes so the map emphasizes relationships that connect papers.
    for topic_id, connected in topic_papers.items():
        if len(set(connected)) < 2:
            if topic_id in graph:
                graph.remove_node(topic_id)

    if include_facts:
        facts = store.get_facts(list(ids))
        for fact in facts:
            paper_node = f"paper:{fact['paper_id']}"
            if paper_node not in graph:
                continue
            if fact["kind"] == "method":
                fact_node = f"method:{_slug(fact['label'])}"
                graph.add_node(fact_node, kind="method", label=fact["label"])
                relation = "uses-method"
            else:
                fact_node = f"claim:{fact['id']}"
                graph.add_node(fact_node, kind="claim", label=fact["label"], statement=fact["statement"],
                               evidence=fact["evidence"], page=fact["page"], confidence=fact["confidence"])
                relation = "reports-claim"
            provenance = ("manual full-text review" if fact.get("model") == "codex-fulltext-review"
                          else f"extracted by {fact.get('model') or 'local model'}")
            graph.add_edge(paper_node, fact_node, relation=relation, provenance=provenance,
                           evidence=fact["evidence"], confidence=fact["confidence"])
        for relation in store.get_relations(list(ids)):
            source = f"paper:{relation['source_paper_id']}"
            target = f"paper:{relation['target_paper_id']}"
            if source in graph and target in graph:
                graph.add_edge(source, target, relation=relation["relation"],
                               provenance=relation["provenance"],
                               evidence=f"A: {relation['evidence_source']} | B: {relation['evidence_target']}",
                               confidence=relation["confidence"])

    if selected_paper_id:
        center = f"paper:{selected_paper_id}"
        if center in graph:
            neighbors = set(graph.predecessors(center)) | set(graph.successors(center))
            keep = {center} | neighbors
            graph = graph.subgraph(keep).copy()
    graph.graph["selected_paper_id"] = selected_paper_id or ""
    graph.graph["collection_id"] = collection_id or ""
    return graph


def graph_payload(graph: nx.MultiDiGraph) -> dict[str, Any]:
    nodes = [{"id": node_id, **attrs} for node_id, attrs in graph.nodes(data=True)]
    edges = [{"source": source, "target": target, **attrs}
             for source, target, attrs in graph.edges(data=True)]
    return {"nodes": nodes, "edges": edges, "summary": graph_summary(graph)}


def graph_summary(graph: nx.MultiDiGraph) -> dict[str, Any]:
    kinds = Counter(data.get("kind", "unknown") for _, data in graph.nodes(data=True))
    relations = Counter(data.get("relation", "unknown") for _, _, data in graph.edges(data=True))
    papers = [node for node, data in graph.nodes(data=True) if data.get("kind") == "paper"]
    shared_topics = [node for node, data in graph.nodes(data=True) if data.get("kind") == "topic"]
    paper_graph = nx.Graph()
    paper_graph.add_nodes_from(papers)
    for source, target, data in graph.edges(data=True):
        if graph.nodes[source].get("kind") == "paper" and graph.nodes[target].get("kind") == "paper":
            paper_graph.add_edge(source, target, relation=data.get("relation"))
    communities: list[list[str]] = []
    if paper_graph.number_of_nodes() > 1 and paper_graph.number_of_edges() > 0:
        try:
            communities = [sorted(group) for group in nx.community.greedy_modularity_communities(paper_graph)]
        except ZeroDivisionError:
            communities = []
    return {"node_count": graph.number_of_nodes(), "edge_count": graph.number_of_edges(),
            "node_types": dict(kinds), "edge_types": dict(relations),
            "paper_count": len(papers), "shared_topic_count": len(shared_topics),
            "citation_components": communities}


def explain_relationships(graph: nx.MultiDiGraph, store: PaperStore) -> list[dict[str, Any]]:
    """Return human-readable relationship records with their source and evidence class."""
    out = []
    for source, target, data in graph.edges(data=True):
        source_data, target_data = graph.nodes[source], graph.nodes[target]
        relation = data.get("relation", "related")
        if source_data.get("kind") == "paper":
            left = store.get_paper(source_data.get("paper_id", ""))
            left_title = left.title if left else source_data.get("title", source)
        else:
            left_title = source_data.get("label", source)
        if target_data.get("kind") == "paper":
            right = store.get_paper(target_data.get("paper_id", ""))
            right_title = right.title if right else target_data.get("title", target)
        else:
            right_title = target_data.get("label", target)
        if relation == "cites":
            text = f"{left_title} cites {right_title}."
            basis = "provider citation record"
        elif relation in ("tagged-with", "uses-method", "reports-claim"):
            text = f"{left_title} is linked to {relation.replace('-', ' ')}: {right_title}."
            basis = "extracted from paper text" if relation != "tagged-with" else "provider topic metadata"
        else:
            text = f"{left_title} — {relation} → {right_title}."
            basis = "LLM-inferred; inspect the quoted evidence"
        out.append({"source": left_title, "relation": relation, "target": right_title,
                    "explanation": text, "basis": basis, "evidence": data.get("evidence", ""),
                    "confidence": data.get("confidence", 1.0), "provenance": data.get("provenance", "")})
    return out

