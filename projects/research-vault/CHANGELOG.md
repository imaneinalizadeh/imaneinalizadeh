# Changelog

## [1.2.0]
- Fixed heuristic similarity breaking on small corpora: the 12%-of-documents
  common-word filter wrongly stripped every word when run on fewer than ~8
  papers (any word in 1 doc already exceeds 12% at small N), producing zero
  similarity links. Now falls back to an absolute document-count threshold
  below `MIN_DOCS_FOR_PCT_FILTER` papers.
- Added automated test suite (`Test/test_research_vault.py`, 7 tests).

## [1.1.0]
- Added Flask web frontend (`web_app.py`) sharing the CLI's SQLite DB.
- Added named entity co-occurrence network + chart export (`entity_analysis.py`).
- BR-003: entity chart PNGs now save to `Implementation/static/` instead of
  the Implementation root, so Flask can serve them directly.

## [1.0.0]
- Initial CLI: PDF ingestion, keyword categorisation, search.
- Heuristic ("similar") and semantic ("semantic") similarity commands.
- BR-001: added explicit warning when `en_core_web_md` isn't installed,
  since `en_core_web_sm` silently produces all-zero semantic scores.
