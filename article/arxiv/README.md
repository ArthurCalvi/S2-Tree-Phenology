# Article Literature Cache

This directory centralizes reference material used while drafting the manuscript.

## Contents
- `papers/`: Markdown summaries and PDFs of foundational work (AlphaEarth, SatMAE, SSL4Eco, etc.).
- `papers/*md`: Lightly edited extracts highlighting methodology details relevant to our experiments.
- `papers/*pdf`: Original publications for full context when refining citations.
- `index.json`: literature ledger maintained by Claude Research (title, authors, year, arXiv/DOI, filename base, bibkey, rationale, target sections).

## Usage
- When citing or interpreting a result, cross-check the corresponding Markdown summary before referencing the PDF.
- Keep new literature in the same naming convention (`Title_With_Underscores.{md,pdf}`) so retrieval scripts stay consistent.
- Update `article/manuscript/references.bib` after incorporating new citations.
- Keep `index.json` in sync with new downloads and bibkeys; note missing/failed fetches so the next run can fix them.
- The TypeScript loop runner (`loop-runner/loop.ts`) writes `research_loop*.json` in `article/artifacts/` and streams readable logs while populating this cache.
