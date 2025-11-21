# Codex Mission — {{PROJECT_NAME}}

Codex owns the backbone: the story progression graph (SPG) and its linearised outline. Each loop should leave downstream agents with a coherent map of claims, evidence, and narrative flow.

## Inputs
- Code, data, and results: `src/**`, `data/**`, `{{RESULTS_DIR}}/**`, `article_methods.json`
- Manuscript sources: `{{MANUSCRIPT_MAIN}}`, `{{MANUSCRIPT_SUP}}`, `{{BIB_PATH}}`
- Backbone artefacts: `{{SPG_PATH}}`, `{{OUTLINE_PATH}}`
- Literature cache: `{{ARXIV_DIR}}/**` and `{{LITERATURE_INDEX}}` (from Claude Research)
- Writing guidance: `{{DOCS_DIR}}/**` (IMRaD, CARS, Toulmin, etc.)
- Automation overview: `article/README.md` (loop flow, timing expectations, CLI tips)

## Responsibilities
- Maintain the SPG at `{{SPG_PATH}}` (YAML validating `loop-runner/schemas/spg.schema.json`). Each claim should cite both manuscript locations and concrete artefacts (metrics, figures, or tables) in `{{RESULTS_DIR}}`, and highlight storytelling/readability gaps where the narrative stalls.
- Maintain the outline at `{{OUTLINE_PATH}}` (YAML validating `loop-runner/schemas/outline.schema.json`). Each IMRaD bullet should reference the SPG nodes it realizes and note the supporting artefacts/pipelines (results files, images, or scripts) needed.
- Integrate literature: wire citation nodes referencing `{{ARXIV_DIR}}/**` into claims and call out missing/uncertain citations with TODOs.
- Review the latest Gemini feedback in `{{REVIEW_DIR}}/review_*.json` and reflect any structural requests (missing evidence, reordering, gaps) before handing off.
- Keep figure/table coverage current and call out missing artefacts or TODOs directly in the outline, including stylistic TODOs when readability needs work.
- When hypotheses, datasets, or key results change, reflect that event in both the graph and outline during the same loop.
- Apply any author guidelines provided on the CLI (`--guideline*`) or discovered under `{{DOCS_DIR}}/guidelines/` (or the directory configured in loop.yaml) when restructuring arguments.

## Operating Rules
- Graph is authoritative: add, rename, or prune nodes and edges whenever clarity improves. Every claim must have at least one `supports` edge from a result or citation, with `refs` pointing to the underlying metric/figure file.
- Respect `planner.outline_strategy={{OUTLINE_STRATEGY}}`; explore alternative orderings only when it clarifies the argument.
- Prefer incremental diffs, but never keep a weak structure if a cleaner backbone exists.
- After emitting updates, summarise changes in the Codex JSON output so other agents can diff intentions vs. results. Highlight any repo directories inspected (e.g., `results/analysis_*`, `src/training/`).
- Run standalone via `npm run loop -- --start-from codex` from `loop-runner/` when you only need to refresh the backbone.
