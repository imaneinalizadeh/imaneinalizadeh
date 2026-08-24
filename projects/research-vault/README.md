# Research Vault

A local research paper management system: ingest PDFs, categorise
them by keyword rules, compute similarity between papers (two ways),
extract named entities and their co-occurrence network, and browse
everything through both a CLI and a Flask web interface.

Built by **Ctrl-Alt-Elite** — University of Edinburgh, EPCC MSc
Practical Software Development, 2025-26.

## Two similarity engines, two real bugs found and fixed

- **Heuristic** (`similar` command) — Jaccard overlap of "rare"
  keywords (words filtered out if they're too common across the
  corpus). **Bug fixed:** the original 12%-of-documents filter broke
  on small corpora — with only 4 papers, any word appearing in just 1
  of them is already 25% > 12%, so it got wrongly stripped as "too
  common", leaving every pair with zero shared words and zero
  similarity. Fixed with an absolute-count fallback below 8 documents
  (see `_common_word_threshold_count` in `similarity_model.py`).
- **Semantic** (`semantic` command) — cosine similarity of spaCy
  document vectors. **Known limitation surfaced, not hidden:**
  `en_core_web_sm` has no word vectors at all, so every score silently
  comes out as 0.0 unless you also install `en_core_web_md`. The code
  now prints an explicit warning rather than failing silently (BR-001).

## Repository structure

```
research-vault/
├── Implementation/
│   ├── PaperCluster.py       CLI orchestrator — PDF ingestion, categorisation, SQLite
│   ├── similarity_model.py   Heuristic + semantic similarity engines
│   ├── entity_analysis.py    NER, co-occurrence network, chart export
│   ├── web_app.py            Flask frontend (shares the CLI's DB and logic)
│   ├── categories.json       Keyword rules for paper categorisation
│   ├── templates/            Flask Jinja2 templates
│   ├── static/               Generated entity charts (gitignored, regenerate via CLI)
│   ├── Papers/                Drop PDFs here
│   └── data/                 SQLite DB (auto-created, gitignored)
├── Test/
│   └── test_research_vault.py   7 automated tests (pytest)
├── Docs/
│   └── (test plan / bug reports)
├── results/                   Exported GEXF graphs
├── requirements.txt
├── CHANGELOG.md
└── CONTRIBUTING.md
```

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md   # for real (non-zero) semantic scores

cd Implementation
python3 PaperCluster.py
```

At the `Vault >` prompt:

| Command | What it does |
|---|---|
| `Get_all_papers` | List all indexed papers grouped by category |
| `search <term>` | Search paper titles and body text |
| `similar` | Heuristic keyword + title similarity scan |
| `semantic` | Semantic similarity using spaCy document vectors |
| `entities` | Named entity frequency + co-occurrence network |
| `reload` | Reload categories.json without restarting |
| `reindex` | Wipe the database and re-ingest all PDFs |
| `q` | Exit |

Web interface: `python3 web_app.py`, then open http://127.0.0.1:5000
(`/`, `/papers`, `/search`, `/semantic`, `/similar`).

## Running the tests

```bash
python3 -m pytest Test/ -v
```

7 tests: keyword categorisation, search, the small-corpus threshold
fix (regression test for the bug above), and a similarity-ranking
sanity check (two papers about the same topic must score higher than
either does against an unrelated third paper).

## Author

Iman Ein Alizadeh, as part of the Ctrl-Alt-Elite team.
