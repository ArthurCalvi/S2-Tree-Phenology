# Outline Backbone

The outline backbone captures the hierarchical plan for the manuscript. Agents should keep it synchronized with `article/manuscript/article.tex` and `article/manuscript/supplementary_materials.tex`.

## Usage
1. Draft or update an outline YAML document that conforms to `loop-runner/schemas/outline.schema.json`.
2. Record section goals (`purpose`), linked hypotheses, and key artefacts (figure/table labels).
3. Store outline snapshots alongside the manuscript to document the evolution of the narrative.

## Maintenance Tips
- After each major writing sprint, update the outline first, then propagate changes into the LaTeX sources.
- Track unresolved questions or pending figures in the `notes` field so agents can triage them later.
- When pruning sections, archive the prior outline file under `article/backbone/archive/` before replacing it.

See `loop-runner/prompts/codex_linearize.txt` for automated conversions between outline YAML and manuscript drafts.
