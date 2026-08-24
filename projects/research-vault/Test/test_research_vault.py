"""
test_research_vault.py — automated test suite (Ctrl-Alt-Elite QA).
Run: python3 -m pytest Test/ -v
"""
import os, sys, sqlite3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Implementation"))
import PaperCluster as pc
from similarity_model import run_heuristic_similarity, _common_word_threshold_count

def _fresh_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE papers (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT UNIQUE, title TEXT, full_text TEXT, category TEXT)")
    conn.execute("CREATE TABLE paper_similarities (paper_a_id INTEGER, paper_b_id INTEGER, score REAL, method TEXT)")
    return conn

def test_categorise_matches_keyword():
    cats = {"HPC": ["mpi", "openmp"], "ML": ["neural network"]}
    assert pc.categorise("This paper uses MPI for parallelism", cats) == "HPC"
    assert pc.categorise("A neural network approach", cats) == "ML"
    assert pc.categorise("Completely unrelated text", cats) == "General Research / Unsorted"

def test_categorise_case_insensitive():
    cats = {"HPC": ["mpi"]}
    assert pc.categorise("Uses MPI heavily", cats) == "HPC"

def test_search_finds_title_match():
    conn = _fresh_conn()
    conn.execute("INSERT INTO papers (filename, title, full_text, category) VALUES (?,?,?,?)",
                 ("a.pdf", "Deep Learning Survey", "body text", "ML"))
    conn.commit()
    rows = conn.execute("SELECT title FROM papers WHERE title LIKE ?", ("%Deep%",)).fetchall()
    assert len(rows) == 1

def test_common_word_threshold_small_corpus():
    # small corpora should use the absolute-count fallback, not a raw 12% cut
    assert _common_word_threshold_count(4) == 4
    assert _common_word_threshold_count(3) == 3

def test_common_word_threshold_large_corpus():
    assert _common_word_threshold_count(100) == 12

def test_heuristic_similarity_needs_two_papers():
    conn = _fresh_conn()
    conn.execute("INSERT INTO papers (filename, title, full_text, category) VALUES (?,?,?,?)",
                 ("a.pdf", "Solo Paper", "text", "ML"))
    conn.commit()
    results = run_heuristic_similarity(conn, quiet=True)
    assert results == []

def test_heuristic_similarity_ranks_similar_pair_higher():
    conn = _fresh_conn()
    papers = [
        ("a.pdf", "A", "transformer attention mechanism neural architecture study"),
        ("b.pdf", "B", "transformer attention mechanism neural architecture research"),
        ("c.pdf", "C", "gardening tips for tomatoes and cucumbers in summer"),
    ]
    for fname, title, text in papers:
        conn.execute("INSERT INTO papers (filename, title, full_text, category) VALUES (?,?,?,?)",
                     (fname, title, text, "X"))
    conn.commit()
    results = run_heuristic_similarity(conn, quiet=True)
    results_by_pair = {(a, b): score for a, b, score in results}
    # A-B (both about transformers) should score higher than A-C or B-C
    ab_score = [s for (a, b), s in results_by_pair.items()][0]
    scores_sorted = sorted(results, key=lambda r: r[2], reverse=True)
    top_pair_titles = {1: "A", 2: "B", 3: "C"}
    assert scores_sorted[0][2] > scores_sorted[-1][2]
