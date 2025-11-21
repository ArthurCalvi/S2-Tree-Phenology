# Story Progression Graph

Automation agents use the Story Progression Graph (SPG) to coordinate narrative revisions. The graph captures which ideas must appear, how they connect, and which evidence supports each step.

## Workflow
1. Populate `nodes` with every narrative checkpoint (hypothesis framing, dataset description, key result, limitation).
2. Connect nodes with directed `edges` that describe how the story flows (`supports`, `depends_on`, `tests`, etc.).
3. Store the graph as YAML or JSON that validates against `loop-runner/schemas/spg.schema.json`.

## Tips
- Keep node identifiers short but stable so prompts and automation steps can reference them reliably.
- Update the graph before major rewrites; agents can diff successive graphs to understand structural changes.
- Link supporting evidence using relative paths (e.g., `results/final_model/phenology_metrics_selected_features.json`).

See `loop-runner/prompts/codex_graph.txt` for automated procedures that generate or validate SPG files from the codebase.
