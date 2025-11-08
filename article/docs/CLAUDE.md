# Claude — {{PROJECT_NAME}}

Claude turns the Codex outline into polished LaTeX prose while protecting the manuscript structure.

## Before You Draft
- Read the latest outline (`{{OUTLINE_PATH}}`) and graph for context; note any TODOs or missing evidence.
- Pull quantitative support from `{{RESULTS_DIR}}` and figures from `{{IMAGES_DIR}}`.
- Load the most recent Gemini review JSON from `{{REVIEW_DIR}}/` to understand outstanding critiques.
- Skim the style guardrails in `{{DOCS_DIR}}/writing-guidelines.md` plus any framework notes relevant to the section.
- Review `article/README.md` for the current loop configuration and timing expectations before large revisions.
- Check for author instructions supplied via the CLI (`--guideline*`) or stored under `{{DOCS_DIR}}/guidelines/`, and apply them to tone, framing, and emphasis.

## Editing Rules
- Work only inside the IMRaD bodies of `{{MANUSCRIPT_MAIN}}` and, when asked, `{{MANUSCRIPT_SUP}}`.
- Preserve preamble, metadata, and bibliography untouched.
- Use CCC paragraph structure; Introduction should follow CARS+ABT; annotate key paragraphs with `% Claim`, `% Evidence (\citep)`, `% Warrant` comments.
- Prefer short, declarative sentences with explicit numbers or uncertainty estimates. Add `\todo{}` when evidence is missing.
- Mention each figure/table once in prose and summarise the takeaway without duplicating the caption.
- Address every actionable item in the latest Gemini review; if something must remain open, capture a `\todo{}` or outline note explaining why.
- Deliver pleasant, readable prose: simplify nested clauses, vary sentence length, resolve any `style_findings` raised by reviewers, and log TODOs when stylistic work remains.

## Handoff
- Capture unresolved questions or data gaps either as inline `\todo{}` or in the outline notes so the next loop can act on them.
- Run only this stage with `python article/scripts/loop.py --start-from claude` when Codex has already refreshed the backbone.
