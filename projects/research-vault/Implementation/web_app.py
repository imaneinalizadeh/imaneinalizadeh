"""
web_app.py

Flask web frontend for Research Vault, sharing the same SQLite
database and business logic (via PaperCluster/similarity_model/
entity_analysis as plain modules) as the CLI.

Run:
    python3 web_app.py
Then open http://127.0.0.1:5000
"""

import os

from flask import Flask, render_template

import PaperCluster as pc
import similarity_model as sm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    template_folder=os.path.join(BASE_DIR, "templates"),
)


def get_conn():
    return pc.get_connection()


@app.route("/")
def index():
    conn = get_conn()
    paper_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    category_counts = conn.execute(
        "SELECT category, COUNT(*) FROM papers GROUP BY category"
    ).fetchall()
    conn.close()
    return render_template(
        "index.html",
        paper_count=paper_count,
        category_counts=category_counts,
    )


@app.route("/papers")
def papers():
    conn = get_conn()
    rows = conn.execute(
        "SELECT category, title, filename FROM papers ORDER BY category, title"
    ).fetchall()
    conn.close()

    grouped = {}
    for category, title, filename in rows:
        grouped.setdefault(category, []).append((title, filename))

    return render_template("papers.html", grouped=grouped)


@app.route("/search")
@app.route("/search/<term>")
def search(term=None):
    results = []
    if term:
        conn = get_conn()
        term_like = f"%{term}%"
        results = conn.execute(
            "SELECT title, filename FROM papers WHERE title LIKE ? OR full_text LIKE ?",
            (term_like, term_like),
        ).fetchall()
        conn.close()
    return render_template("search.html", term=term, results=results)


@app.route("/semantic")
def semantic():
    conn = get_conn()
    titles = dict(conn.execute("SELECT id, title FROM papers").fetchall())
    rows = conn.execute(
        "SELECT paper_a_id, paper_b_id, score FROM paper_similarities "
        "WHERE method = 'semantic' ORDER BY score DESC LIMIT 20"
    ).fetchall()
    conn.close()

    links = [
        {"a": titles.get(a, "?"), "b": titles.get(b, "?"), "score": round(score, 3)}
        for a, b, score in rows
    ]
    return render_template("semantic.html", links=links)


@app.route("/similar")
def similar():
    conn = get_conn()
    titles = dict(conn.execute("SELECT id, title FROM papers").fetchall())
    rows = conn.execute(
        "SELECT paper_a_id, paper_b_id, score FROM paper_similarities "
        "WHERE method = 'heuristic' ORDER BY score DESC LIMIT 20"
    ).fetchall()
    conn.close()

    links = [
        {"a": titles.get(a, "?"), "b": titles.get(b, "?"), "score": round(score, 3)}
        for a, b, score in rows
    ]
    return render_template("similar.html", links=links)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
