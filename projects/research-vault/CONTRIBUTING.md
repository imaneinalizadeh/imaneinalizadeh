# Contributing

Built by **Ctrl-Alt-Elite** — EPCC MSc Practical Software Development, 2025-26.

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md   # optional but needed for real semantic scores
```

## Running
```bash
cd Implementation
python3 PaperCluster.py   # CLI
python3 web_app.py        # web (needs the CLI run at least once, or Papers/ populated + reindex)
```

## Tests
```bash
python3 -m pytest Test/ -v
```

## Code style
PEP 8, docstrings on all public functions. Keep `Implementation/` modules
importable standalone (no side effects on import) so `web_app.py` and
`Test/` can both import them cleanly.
