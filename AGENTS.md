# Repository Guidelines

## Project Structure & Module Organization
Source lives under `src/` by pipeline stage (e.g., `src/features/`, `src/training/`, `src/inference/`). Tests in `tests/` as `test_*.py` target inference utilities and tiling. Bash entrypoints under `src/**/run_*.sh`, `ops/jobs/`, `ops/qa/jobs/`, plus helper scripts like `ops/bash/run_sequential_geefetch.sh`. Configs and artifacts: `ops/geefetch/configs/` for GEE YAML, LaTeX/QGIS/DOT assets in their dedicated folders.

## Build, Test, and Development Commands
- `pytest -q`: execute Python tests quietly.
- `bash src/features/run_prepare_inference_feature.sh`: generate feature inputs before inference.
- `bash src/features/run_inference_feature.sh`: run feature inference once prepped data exists.
- `bash src/training/run_train_rf_selected_features.sh`: train random forest on selected features (use genus variant when needed).
- `python src/training/train_unet_phenology_indice.py --help`: inspect UNet training arguments before launching.
- `bash src/inference/run_inference_rf.sh`: apply RF models to produce predictions.
- `bash ops/qa/jobs/qa_corsica.sh`: launch QA checks for the Corsica workflow.
- `bash ops/bash/run_sequential_geefetch.sh`: trigger GEE fetch pipeline; choose configs from `ops/geefetch/configs/`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, `snake_case` for functions/variables, `PascalCase` for classes. Modules use descriptive names (`train_rf_selected_features.py`). Keep utilities in `src/utils.py` and constants in `src/constants.py`. Prefer explicit relative imports within `src/`.

## Testing Guidelines
Use `pytest`; keep new tests under `tests/` and name them `test_*.py`. Mirror inference scenarios with minimal fixtures or temp paths. Run `pytest -q` locally before commits. Aim to cover new utilities and data-path logic.

## Commit & Pull Request Guidelines
Use concise imperative commit subjects (e.g., `add inference rf`). For PRs, include purpose, reproduction commands, linked issues, and relevant screenshots/plots (avoid heavy outputs). Confirm default configs still run and tests pass before requesting review.

## Security & Configuration Tips
Never commit secrets or machine-specific paths. Duplicate templates in `ops/geefetch/configs/` for environment overrides instead of editing shared files. Parameterize scripts via CLI flags or config files to keep deployments reproducible.

## Helpful Artifacts
- `todo.md`: living, high-level checklist of next operational steps (e.g., data downloads, parquet regeneration/rescaling, training/inference runs). Check here first to align work with the current plan.
- `article_methods.json`: mapping from article method components to concrete scripts, entry commands, inputs/outputs, and expected results. It links the manuscript sources (`article/manuscript/article.tex`, `article/manuscript/supplementary_materials.tex`, `article/manuscript/references.bib`) to each workflow in this repo so you always know where supporting information lives. Consult and update this file frequently so it continues to reflect how the repository operates.
- `article/manuscript/`: canonical LaTeX sources for the paper and supplements.
- `article/docs/`: writing frameworks (IMRaD, CARS, Toulmin, CCC/ABT, Gopen & Swan) plus deeper research notes under `article/docs/research/`. Agent-specific briefs live in `article/docs/AGENTS.codex.md`, `article/docs/CLAUDE.md`, and `article/docs/GEMINI.md`.
- `article/backbone/`: active story graph (`spg.yml`) and outline (`outline.yml`) plus README guides for each structure.
- `article/scripts/`: orchestration prompts (`prompts/`), schemas (`schemas/`), and the loop runner (`loop.py`).
- `article/arxiv/`: cached PDFs/markdown summaries of cited foundation-model and remote-sensing papers (e.g., AlphaEarth Foundations, SatMAE, SSL4Eco). Use these when refining literature context or detailing methodology inputs.
- `article/artifacts/review_*.json`: latest Gemini blind-review output. Claude must address these comments in the next drafting pass.
- `article/artifacts/review_loopX_<timestamp>.json`: time-stamped history of Gemini reviews for each loop iteration.
- Author guidelines: supplied via CLI (`--guideline`, `--guideline-file`) and/or markdown snippets under `article/docs/guidelines/`; these influence Codex and Claude prompts automatically.

### Documentation

Agents (Codex, Claude, Gemini) docs lives here : /docs, visit it whenever you are modifying code of one agent. 


## Manuscript Revision Workflow
Rewrite `article/manuscript/article.tex` and `article/manuscript/supplementary_materials.tex` iteratively, aligning with the outline and graph definitions in `article/backbone/`. Maintain a concise scientific style: analyse experimental results without extrapolation, avoid references to repository scripts, and favour clear prose over bullet points. Prior to edits, review `article/docs/writing-guidelines.md` alongside supporting frameworks in `article/docs/` and literature in `article/arxiv/` or `article/docs/research/`.

Keep an up-to-date narrative map by enriching the notes fields in `article/backbone/outline.yml` and `article/backbone/spg.yml` whenever hypotheses, figures, or section priorities shift. Automation agents can run the full loop via `python article/scripts/loop.py` once `article/config/loop.yaml` tokens are filled.
- Use `--start-from claude` or `--start-from gemini` to resume a loop midstream, and append `--build-pdf` when you want `latexmk` to run; add `--mode interactive` to supervise each agent manually.
- `--start-from` only affects the first iteration of a run; later loops execute the full Codex → Claude → Gemini order automatically.

### Article Writing Guidelines
- Use simple sentences and keep paragraphs focused on one idea.
- Do not cite repository paths or filenames in the manuscript; reference methods conceptually.
- Aim for a clean, lean tone that prioritises clarity over flourish.
- Ensure the storytelling stays easy to follow—state the context, the action, and the outcome explicitly in each section.

## Multi-agent Coordination
- Assume other automation agents or collaborators might have active changes in the working tree. Never delete or overwrite files created by teammates unless you have confirmed they are no longer needed.
- When you need to modify shared helpers (e.g., under `ops/gee/` or `src/visualization/`), review `git status` and existing scripts first, then extend them in place rather than replacing or removing them.
- If you notice conflicting edits, communicate the collision in the task log or leave a note instead of forcefully resetting files. This keeps figure-generation scripts and credential helpers available for everyone.
