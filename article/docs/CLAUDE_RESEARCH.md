# Claude Research Prep — {{PROJECT_NAME}}

Purpose: collect and formalize literature before Codex rebuilds the backbone.

## What You Do
- Read {{SPG_PATH}}, {{OUTLINE_PATH}}, {{MANUSCRIPT_MAIN}}, {{MANUSCRIPT_SUP}}, and {{BIB_PATH}} to spot citation gaps or stale references.
- Fetch missing papers via web search and the arXiv MCP server (see docs/mcp/arxiv.md); prefer PDFs plus a short Markdown summary.
- Save artefacts to {{ARXIV_DIR}}/ using `Title_With_Underscores_<year>_<arxivId>.(pdf|md)` and maintain {{LITERATURE_INDEX}} with metadata (title, authors, year, arXiv/DOI, url, filename_base, bibkey, why_it_matters, target_sections/claims).
- Deduplicate on arXiv ID/DOI/title; align with {{BIB_PATH}} bibkeys and flag unresolved cases in the index todos field.

## Rules
- Do not edit manuscript or backbone files; only touch the literature cache/index.
- Keep runs idempotent: skip downloads that already exist, and record failures/TODOs instead of guessing.
- When topic coverage is thin, propose targeted search queries and what evidence they should provide next run.
- Respect author guidance indirectly: focus on sources that strengthen claims, baselines, and limitations relevant to the current outline/SPG.

## Outputs
- PDFs/MDs under {{ARXIV_DIR}} with clean names.
- Updated {{LITERATURE_INDEX}} capturing what was fetched, skipped, or needs follow-up.
- Console JSON summary: {"session_id":"...","fetched":[...],"skipped":[...],"todos":[...]} for loop logging.
