"""
PaperCluster.py

Main CLI entry point and orchestrator for Research Vault. Ingests PDFs
from Papers/, categorises them by keyword rules in categories.json,
stores everything in a local SQLite database, and exposes an
interactive prompt for search/similarity/entity commands (which are
implemented in similarity_model.py and entity_analysis.py and simply
called from here).

Built by Ctrl-Alt-Elite — University of Edinburgh, EPCC MSc Practical
Software Development, 2025-26.

Run:
    python3 PaperCluster.py
"""

import json
import os
import sqlite3
import sys

import fitz  # PyMuPDF
from prompt_toolkit import prompt

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "research_vault.sqlite")
PAPERS_DIR = os.path.join(os.path.dirname(__file__), "Papers")
CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "categories.json")

# Only the first N characters of a paper are scanned for entities/
# keyword categorisation — long papers' acknowledgements/references
# sections rarely change the category, and this keeps ingestion fast.
SCAN_CHAR_LIMIT = 5000


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            title TEXT,
            full_text TEXT,
            category TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_similarities (
            paper_a_id INTEGER,
            paper_b_id INTEGER,
            score REAL,
            method TEXT
        )
    """)
    conn.commit()
    return conn


def load_categories():
    if not os.path.exists(CATEGORIES_PATH):
        return {}
    with open(CATEGORIES_PATH) as f:
        return json.load(f)


def categorise(text, categories):
    text_lower = text.lower()
    for category, keywords in categories.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                return category
    return "General Research / Unsorted"


def extract_title(doc, filename):
    """
    Best-effort title extraction: use PDF metadata if present, else
    fall back to the first non-empty line of the first page, else the
    filename.
    """
    meta_title = doc.metadata.get("title", "").strip()
    if meta_title:
        return meta_title

    if len(doc) > 0:
        first_page_text = doc[0].get_text()
        for line in first_page_text.splitlines():
            line = line.strip()
            if len(line) > 5:
                return line

    return os.path.splitext(filename)[0]


def ingest_pdf(conn, filepath, categories):
    filename = os.path.basename(filepath)

    existing = conn.execute(
        "SELECT id FROM papers WHERE filename = ?", (filename,)
    ).fetchone()
    if existing:
        return None  # already indexed

    try:
        doc = fitz.open(filepath)
    except Exception as e:
        print(f"  [skip] Could not open {filename}: {e}")
        return None

    full_text = ""
    for page in doc:
        full_text += page.get_text()

    title = extract_title(doc, filename)
    category = categorise(full_text[:SCAN_CHAR_LIMIT], categories)
    doc.close()

    conn.execute(
        "INSERT INTO papers (filename, title, full_text, category) VALUES (?, ?, ?, ?)",
        (filename, title, full_text, category),
    )
    conn.commit()
    return title, category


def index_all_papers(conn, categories, quiet=False):
    if not os.path.exists(PAPERS_DIR):
        os.makedirs(PAPERS_DIR, exist_ok=True)
        if not quiet:
            print(f"Created {PAPERS_DIR}/ — drop PDF files there and run 'reindex'.")
        return

    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf")]
    added = 0
    for fname in pdf_files:
        result = ingest_pdf(conn, os.path.join(PAPERS_DIR, fname), categories)
        if result:
            title, category = result
            added += 1
            if not quiet:
                print(f"  + {fname}  ->  [{category}] {title[:60]}")
    if not quiet:
        print(f"Indexed {added} new paper(s). {len(pdf_files)} PDF(s) found in Papers/.")


def cmd_get_all_papers(conn):
    rows = conn.execute(
        "SELECT category, title, filename FROM papers ORDER BY category, title"
    ).fetchall()
    if not rows:
        print("No papers indexed yet. Add PDFs to Papers/ and run 'reindex'.")
        return
    current_category = None
    for category, title, filename in rows:
        if category != current_category:
            print(f"\n== {category} ==")
            current_category = category
        print(f"  - {title}  ({filename})")


def cmd_search(conn, term):
    term_like = f"%{term}%"
    rows = conn.execute(
        "SELECT title, filename FROM papers WHERE title LIKE ? OR full_text LIKE ?",
        (term_like, term_like),
    ).fetchall()
    if not rows:
        print(f"No papers matched '{term}'.")
        return
    print(f"{len(rows)} paper(s) matched '{term}':")
    for title, filename in rows:
        print(f"  - {title}  ({filename})")


def cmd_reindex(conn, categories):
    conn.execute("DELETE FROM papers")
    conn.execute("DELETE FROM paper_similarities")
    conn.commit()
    print("Cleared index. Re-scanning Papers/ ...")
    index_all_papers(conn, categories)


HELP_TEXT = """
Available commands:
  Get_all_papers   List all indexed papers grouped by category
  search <term>    Search paper titles and body text
  similar          Heuristic keyword + title similarity scan
  semantic         Semantic similarity using spaCy document vectors
  entities         Named entity frequency analysis and co-occurrence network
  reload           Reload categories.json without restarting
  reindex          Wipe the database and re-ingest all PDFs from scratch
  help             Show this message
  q                Exit
"""


def main():
    conn = get_connection()
    categories = load_categories()
    index_all_papers(conn, categories, quiet=True)

    print("Research Vault — Ctrl-Alt-Elite")
    print(HELP_TEXT)

    while True:
        try:
            cmd = prompt("Vault > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "" or cmd.lower() == "help":
            print(HELP_TEXT)
        elif cmd == "q":
            break
        elif cmd == "Get_all_papers":
            cmd_get_all_papers(conn)
        elif cmd.startswith("search "):
            cmd_search(conn, cmd[len("search "):].strip())
        elif cmd == "similar":
            from similarity_model import run_heuristic_similarity
            run_heuristic_similarity(conn)
        elif cmd == "semantic":
            from similarity_model import run_semantic_similarity
            run_semantic_similarity(conn)
        elif cmd == "entities":
            from entity_analysis import run_entity_analysis
            run_entity_analysis(conn)
        elif cmd == "reload":
            categories = load_categories()
            print("Reloaded categories.json")
        elif cmd == "reindex":
            cmd_reindex(conn, categories)
        else:
            print(f"Unknown command: '{cmd}'. Type 'help' for the command list.")

    conn.close()


if __name__ == "__main__":
    sys.exit(main())
