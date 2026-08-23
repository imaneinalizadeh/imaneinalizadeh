# Research Vault — Academic Paper Management System

A full-stack academic paper ingestion, categorisation, and similarity search system. Built as a group software engineering project at the University of Edinburgh. Ingests PDFs, extracts metadata and keywords using NLP, stores papers in a SQLite database, and serves a Flask web interface with semantic similarity search.

**Team:** Iman (Product Manager), Aman (QA Lead), Zheng (Dev Lead)

---

## What It Does

```
PDF Upload
    │
    ▼
Text Extraction (PyMuPDF)
    │
    ▼
NLP Pipeline (spaCy)
  ├── Keyword extraction
  ├── Abstract detection
  └── Category classification
    │
    ▼
SQLite Database
    │
    ▼
Flask Web Interface
  ├── Browse by category
  ├── Full-text search
  └── Semantic similarity (/similar)
```

---

## Repository Structure

```
research-vault/
├── Implementation/
│   ├── app.py                     # Flask web application
│   ├── paper_ingester.py          # PDF parsing and NLP pipeline
│   ├── database.py                # SQLite ORM layer
│   ├── paper_cluster.py           # Semantic similarity (spaCy vectors)
│   ├── category_config.json       # Keyword-to-category mapping
│   ├── Papers/                    # Drop PDFs here for ingestion
│   ├── static/                    # CSS, JS assets
│   └── templates/                 # Jinja2 HTML templates
├── Test/
│   └── test_research_vault.py     # Pytest test suite
├── docs/
│   ├── architecture.md            # System design decisions
│   └── api.md                     # REST endpoint reference
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

### 2. Add papers

Drop PDF files into `Implementation/Papers/`.

### 3. Ingest and run

```bash
cd Implementation
python paper_ingester.py     # Ingest all PDFs in Papers/
python app.py                # Start Flask server at localhost:5000
```

### 4. Run tests

```bash
python -m pytest Test/test_research_vault.py -v
```

---

## Category Configuration

Edit `Implementation/category_config.json` to customise categories:

```json
{
  "Machine Learning & Deep Learning": ["neural", "deep learning", "transformer", "bert"],
  "High Performance Computing": ["mpi", "openmp", "cuda", "speedup", "parallel"],
  "Computer Vision": ["image", "detection", "segmentation", "cnn", "mediapipe"]
}
```

Papers that do not match any keyword go to **General Research / Unsorted**.

---

## Known Issues

| ID | Issue | Severity |
|----|-------|----------|
| BR-001 | `en_core_web_sm` has no word vectors — semantic scan silently returns no results | S2 |
| BR-002 | `/similar` route recomputes O(n²) on every page load — slow with 30+ papers | S3 |
| BR-003 | `static_folder` exposes entire `Implementation/` directory including SQLite DB | S2 |
| BR-004 | Importing `PaperCluster` triggers spaCy model load on Flask startup | S3 |

---

## Dependencies

```
flask>=2.3.0
pymupdf>=1.23.0
spacy>=3.7.0
pytest>=7.4.0
```

---

## Academic Context

**Course:** Software Engineering group project
**Institution:** University of Edinburgh
**Team size:** 3
**Stack:** Python, Flask, SQLite, spaCy, PyMuPDF
