"""
entity_analysis.py

Named entity extraction across the corpus (organisations, people,
technologies mentioned as products/works, etc. — whatever spaCy's
default NER model tags), frequency counting, a co-occurrence network
(entities that appear in the same paper are linked), and PNG chart /
GEXF exports.

BR-003: chart PNGs are saved into Implementation/static/ (not the
Implementation/ root) so the Flask web app in web_app.py can serve
them directly as static assets.
"""

import os
from collections import Counter, defaultdict

import spacy

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

SCAN_CHAR_LIMIT = 5000
RELEVANT_ENTITY_TYPES = {"ORG", "PERSON", "PRODUCT", "GPE", "WORK_OF_ART", "NORP"}

_nlp_cache = {}


def _load_nlp():
    if "nlp" not in _nlp_cache:
        _nlp_cache["nlp"] = spacy.load("en_core_web_sm")
    return _nlp_cache["nlp"]


def extract_entities_per_paper(conn):
    nlp = _load_nlp()
    rows = conn.execute("SELECT id, title, full_text FROM papers").fetchall()

    paper_entities = {}  # paper_id -> set of entity strings
    global_counter = Counter()

    for pid, title, text in rows:
        doc = nlp(text[:SCAN_CHAR_LIMIT])
        ents = {
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ in RELEVANT_ENTITY_TYPES and len(ent.text.strip()) > 2
        }
        paper_entities[pid] = ents
        global_counter.update(ents)

    return paper_entities, global_counter


def build_cooccurrence_network(paper_entities, min_shared_papers=1):
    if not HAS_NETWORKX:
        return None

    g = nx.Graph()
    pair_counts = defaultdict(int)

    for entities in paper_entities.values():
        entity_list = sorted(entities)
        for i in range(len(entity_list)):
            g.add_node(entity_list[i])
            for j in range(i + 1, len(entity_list)):
                pair = (entity_list[i], entity_list[j])
                pair_counts[pair] += 1

    for (a, b), count in pair_counts.items():
        if count >= min_shared_papers:
            g.add_edge(a, b, weight=count)

    return g


def plot_top_entities(counter, top_n=15, out_path=None):
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed — skipping chart export.")
        return

    top = counter.most_common(top_n)
    if not top:
        print("No entities found to plot.")
        return

    labels = [t[0][:25] for t in top]
    counts = [t[1] for t in top]

    plt.figure(figsize=(8, max(4, len(labels) * 0.3)))
    plt.barh(labels[::-1], counts[::-1], color="#4c72b0")
    plt.xlabel("Mentions across corpus")
    plt.title(f"Top {len(top)} named entities")
    plt.tight_layout()

    if out_path is None:
        os.makedirs(STATIC_DIR, exist_ok=True)
        out_path = os.path.join(STATIC_DIR, "entity_top_entities.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved entity frequency chart to {out_path}")
    return out_path


def plot_cooccurrence_network(graph, out_path=None, max_nodes=40):
    if not HAS_MATPLOTLIB or not HAS_NETWORKX or graph is None:
        print("matplotlib/networkx not installed — skipping network chart.")
        return

    if graph.number_of_nodes() == 0:
        print("No co-occurrence network to plot (no shared entities found).")
        return

    # Trim to the most-connected nodes if the graph is large, for legibility
    if graph.number_of_nodes() > max_nodes:
        degrees = dict(graph.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        graph = graph.subgraph(top_nodes)

    plt.figure(figsize=(9, 9))
    pos = nx.spring_layout(graph, seed=42, k=0.6)
    weights = [graph[u][v].get("weight", 1) for u, v in graph.edges()]
    nx.draw_networkx_nodes(graph, pos, node_size=200, node_color="#55a868")
    nx.draw_networkx_edges(graph, pos, width=[0.5 + 0.5 * w for w in weights], alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=7)
    plt.title("Entity co-occurrence network (entities sharing a paper)")
    plt.axis("off")
    plt.tight_layout()

    if out_path is None:
        os.makedirs(STATIC_DIR, exist_ok=True)
        out_path = os.path.join(STATIC_DIR, "entity_cooccurrence_network.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved co-occurrence network chart to {out_path}")
    return out_path


def run_entity_analysis(conn, quiet=False):
    paper_entities, global_counter = extract_entities_per_paper(conn)

    if not quiet:
        print(f"Extracted entities from {len(paper_entities)} paper(s).")
        print(f"Top 10 entities overall:")
        for entity, count in global_counter.most_common(10):
            print(f"  {count:3d}  {entity}")

    graph = build_cooccurrence_network(paper_entities)
    plot_top_entities(global_counter)
    plot_cooccurrence_network(graph)

    if HAS_NETWORKX and graph is not None:
        gexf_path = os.path.join(os.path.dirname(BASE_DIR), "results", "entity_network.gexf")
        os.makedirs(os.path.dirname(gexf_path), exist_ok=True)
        try:
            import networkx as nx
            nx.write_gexf(graph, gexf_path)
            if not quiet:
                print(f"Exported entity network to {gexf_path}")
        except Exception as e:
            print(f"Could not export GEXF: {e}")

    return paper_entities, global_counter
