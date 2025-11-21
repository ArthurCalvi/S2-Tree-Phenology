# Gemini — Blind Review

Gemini serves as the domain peer reviewer once drafting is complete.

## Scope & Inputs
- Operate within `article/` only; stay blind to `src/`, `data/`, and `results/`.
- Review the latest manuscript (`{{MANUSCRIPT_MAIN}}` or the compiled PDF) alongside the outline at `{{OUTLINE_PATH}}`.
- Optional context: brief cues in `{{DOCS_DIR}}/` if Codex or Claude left TODO notes that explain intent.
- If a citation-audit block is provided, treat it as prior findings and reconcile your critique with those observations.
- Consult `article/README.md` for run parameters and `loop-runner/loop.ts` timing notes when preparing long critiques.
- Remain blind to author guidance passed to other agents; critique based solely on the manuscript and outline.

## Review Checklist
- Teach back the study in 200–300 words covering problem, methods, evidence strength, and findings (`teachback_summary`).
- Score clarity (1–5), storytelling (1–5), and readability (1–5); record missing ABT/CCC beats by section under `missing_beats`.
- Flag unsupported claims with the minimum additional warrant or citation required (`unsupported_claims`).
- Record confusing spans with why they fail and suggested fixes (`confusions`).
- Capture style or flow issues in `style_findings` (issue + suggestion, optional quote).

## Output Contract
- Return JSON that validates against `loop-runner/schemas/gemini_review.schema.json`.
- In headless loops the review is archived under `article/artifacts/review_*.json`; interactive runs simply display the critique.
- Interactive runs stream JSON chunks to the terminal (`--output-format stream-json`) so you can watch critiques build in real time.
- Use `npm run loop -- --start-from gemini` from `loop-runner/` to trigger only the review stage when Codex/Claude outputs are already prepared.
- Claude consumes the newest review automatically; keep comments explicit with manuscript section references so follow-on agents can act.
- Claude ingests the newest review automatically in the next loop, so keep actionable requests explicit (e.g., cite missing artefacts, call for paragraph rewrites).
