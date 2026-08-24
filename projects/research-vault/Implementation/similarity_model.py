"""
similarity_model.py

Two similarity engines over the indexed paper corpus:

1. Heuristic ("similar" command) — keyword + title overlap, filtering
   out words that appear in more than 12% of all documents (crude
   stopword-like filtering) so common academic boilerplate ("results",
   "method", "paper") doesn't dominate the overlap score.

2. Semantic ("semantic" command) — cosine similarity between spaCy
   document vectors.

BR-001: `en_core_web_sm` has no word vectors at all — every document's
.vector is a zero vector, so every cosine similarity comes out as 0
(or NaN) with no error raised. This was a genuinely confusing failure
mode (the command runs, prints a table, the table is just all zeros)
until it was made an explicit warning below. Real semantic similarity
needs `en_core_web_md` or larger.
"""

import itertools
import os
import sqlite3

import numpy as np
import spacy

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

_nlp_cache = {}


def load_nlp():
    """
    Loads en_core_web_md if available (has real word vectors); falls
    back to en_core_web_sm with an explicit warning if not, rather
    than silently producing all-zero similarity scores (BR-001).
    """
    if "nlp" in _nlp_cache:
        return _nlp_cache["nlp"], _nlp_cache["has_vectors"]

    try:
        nlp = spacy.load("en_core_web_md")
        has_vectors = True
    except OSError:
        nlp = spacy.load("en_core_web_sm")
        has_vectors = False
        print(
            "WARNING: en_core_web_md is not installed — falling back to "
            "en_core_web_sm, which has NO word vectors. Every semantic "
            "similarity score will come out as 0.0. Run:\n"
            "    python -m spacy download en_core_web_md\n"
            "to get meaningful semantic similarity scores."
        )

    _nlp_cache["nlp"] = nlp
    _nlp_cache["has_vectors"] = has_vectors
    return nlp, has_vectors


STOPWORDISH_THRESHOLD = 0.12  # words in >12% of docs are filtered from heuristic overlap
MIN_DOCS_FOR_PCT_FILTER = 8   # below this corpus size, use an absolute floor instead (see below)


def _common_word_threshold_count(n_docs):
    """
    The 12%-of-documents rule only makes sense once the corpus is big
    enough that 12% corresponds to more than a handful of documents.
    On a small corpus (e.g. 4 papers), any word that appears in even
    1 document is already 25% >  12%, so a naive percentage filter
    wrongly strips out genuinely rare, distinguishing words — leaving
    both token sets empty and every pair scoring 0.0 similarity. Below
    MIN_DOCS_FOR_PCT_FILTER papers, fall back to an absolute count
    (a word must appear in at least 3 papers to be filtered as
    "common"), which behaves sensibly at both small and large scale.
    """
    if n_docs < MIN_DOCS_FOR_PCT_FILTER:
        return max(3, n_docs)  # effectively: only filter words in EVERY doc, for tiny corpora
    return int(STOPWORDISH_THRESHOLD * n_docs)


def _tokenise_for_heuristic(text):
    words = [w.strip(".,;:()[]\"'").lower() for w in text.split()]
    return set(w for w in words if len(w) > 3)


def run_heuristic_similarity(conn, top_k=10, min_sim=0.0, quiet=False):
    rows = conn.execute("SELECT id, title, full_text FROM papers").fetchall()
    if len(rows) < 2:
        if not quiet:
            print("Need at least 2 papers indexed to compute similarity.")
        return []

    doc_tokens = {pid: _tokenise_for_heuristic(text[:5000]) for pid, _, text in rows}

    # word document-frequency for the common-word filter (see
    # _common_word_threshold_count for why this isn't a flat 12% cut)
    n_docs = len(rows)
    df = {}
    for tokens in doc_tokens.values():
        for w in tokens:
            df[w] = df.get(w, 0) + 1
    threshold_count = _common_word_threshold_count(n_docs)
    common_words = {w for w, count in df.items() if count > threshold_count}

    filtered_tokens = {
        pid: tokens - common_words for pid, tokens in doc_tokens.items()
    }

    titles = {pid: title for pid, title, _ in rows}
    results = []
    for (id_a, id_b) in itertools.combinations(filtered_tokens.keys(), 2):
        a, b = filtered_tokens[id_a], filtered_tokens[id_b]
        if not a or not b:
            continue
        overlap = len(a & b) / len(a | b)  # Jaccard similarity
        if overlap >= min_sim:
            results.append((id_a, id_b, overlap))

    results.sort(key=lambda r: r[2], reverse=True)

    conn.execute("DELETE FROM paper_similarities WHERE method = 'heuristic'")
    for id_a, id_b, score in results:
        conn.execute(
            "INSERT INTO paper_similarities (paper_a_id, paper_b_id, score, method) VALUES (?, ?, ?, 'heuristic')",
            (id_a, id_b, score),
        )
    conn.commit()

    if not quiet:
        print(f"Top {min(top_k, len(results))} heuristic links (rare-keyword Jaccard overlap):")
        for id_a, id_b, score in results[:top_k]:
            print(f"  {score:.3f}  {titles[id_a][:40]}  <->  {titles[id_b][:40]}")

    return results


def run_semantic_similarity(conn, top_k=10, quiet=False):
    nlp, has_vectors = load_nlp()
    rows = conn.execute("SELECT id, title, full_text FROM papers").fetchall()
    if len(rows) < 2:
        if not quiet:
            print("Need at least 2 papers indexed to compute similarity.")
        return []

    vectors = {}
    titles = {}
    for pid, title, text in rows:
        doc = nlp(text[:5000])
        vectors[pid] = doc.vector
        titles[pid] = title

    results = []
    for id_a, id_b in itertools.combinations(vectors.keys(), 2):
        va, vb = vectors[id_a], vectors[id_b]
        norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(va, vb) / (norm_a * norm_b))
        results.append((id_a, id_b, cos_sim))

    results.sort(key=lambda r: r[2], reverse=True)

    conn.execute("DELETE FROM paper_similarities WHERE method = 'semantic'")
    for id_a, id_b, score in results:
        conn.execute(
            "INSERT INTO paper_similarities (paper_a_id, paper_b_id, score, method) VALUES (?, ?, ?, 'semantic')",
            (id_a, id_b, score),
        )
    conn.commit()

    if not quiet:
        if not has_vectors:
            print("(scores below are meaningless — see warning above)")
        print(f"Top {min(top_k, len(results))} semantic links (cosine similarity):")
        for id_a, id_b, score in results[:top_k]:
            print(f"  {score:.3f}  {titles[id_a][:40]}  <->  {titles[id_b][:40]}")

    return results


def export_similarity_graph(conn, method, out_path):
    """
    Exports the similarity edges for a given method ('heuristic' or
    'semantic') as a GEXF graph file, viewable in Gephi.
    """
    if not HAS_NETWORKX:
        print("networkx not installed — skipping GEXF export. pip install networkx")
        return

    rows = conn.execute(
        "SELECT paper_a_id, paper_b_id, score FROM paper_similarities WHERE method = ?",
        (method,),
    ).fetchall()
    titles = dict(conn.execute("SELECT id, title FROM papers").fetchall())

    g = nx.Graph()
    for pid, title in titles.items():
        g.add_node(pid, label=title[:60])
    for a, b, score in rows:
        g.add_edge(a, b, weight=score)

    nx.write_gexf(g, out_path)
    print(f"Exported {method} similarity graph ({len(rows)} edges) to {out_path}")
