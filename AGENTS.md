# Repository Guidelines

## Project Structure & Module Organization
Source lives under `src/` by pipeline stage (e.g., `src/features/`, `src/training/`, `src/inference/`). Tests in `tests/` as `test_*.py` target inference utilities and tiling. Bash entrypoints under `src/**/run_*.sh`, `jobs/`, `qa/jobs/`, plus top-level helpers like `run_sequential_geefetch.sh`. Configs and artifacts: `geefetch/configs/` for GEE YAML, LaTeX/QGIS/DOT assets in their dedicated folders.

## Build, Test, and Development Commands
- `pytest -q`: execute Python tests quietly.
- `bash src/features/run_prepare_inference_feature.sh`: generate feature inputs before inference.
- `bash src/features/run_inference_feature.sh`: run feature inference once prepped data exists.
- `bash src/training/run_train_rf_selected_features.sh`: train random forest on selected features (use genus variant when needed).
- `python src/training/train_unet_phenology_indice.py --help`: inspect UNet training arguments before launching.
- `bash src/inference/run_inference_rf.sh`: apply RF models to produce predictions.
- `bash qa/jobs/qa_corsica.sh`: launch QA checks for Corsica workflow.
- `bash run_sequential_geefetch.sh`: trigger GEE fetch pipeline; choose configs from `geefetch/configs/`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, `snake_case` for functions/variables, `PascalCase` for classes. Modules use descriptive names (`train_rf_selected_features.py`). Keep utilities in `src/utils.py` and constants in `src/constants.py`. Prefer explicit relative imports within `src/`.

## Testing Guidelines
Use `pytest`; keep new tests under `tests/` and name them `test_*.py`. Mirror inference scenarios with minimal fixtures or temp paths. Run `pytest -q` locally before commits. Aim to cover new utilities and data-path logic.

## Commit & Pull Request Guidelines
Use concise imperative commit subjects (e.g., `add inference rf`). For PRs, include purpose, reproduction commands, linked issues, and relevant screenshots/plots (avoid heavy outputs). Confirm default configs still run and tests pass before requesting review.

## Security & Configuration Tips
Never commit secrets or machine-specific paths. Duplicate templates in `geefetch/configs/` for environment overrides instead of editing shared files. Parameterize scripts via CLI flags or config files to keep deployments reproducible.

## Helpful Artifacts
- `todo.md`: living, high-level checklist of next operational steps (e.g., data downloads, parquet regeneration/rescaling, training/inference runs). Check here first to align work with the current plan.
- `article_methods.json`: mapping from article method components to concrete scripts, entry commands, inputs/outputs, and expected results. It links the manuscript sources (`article/article.tex`, `article/supplementary_materials.tex`, `article/phenology.bib`) to each workflow in this repo so you always know where supporting information lives. Consult and update this file frequently so it continues to reflect how the repository operates.
- `article/` & `article/review/`: LaTeX sources for the manuscript plus supervisor feedback (`review_supervisors.md`), historical notes, and writing guidelines (`writing_a_good_scientific_paper.md`, `article_revision_guidelines.md`). Start here before editing text or responding to reviewer comments.
- `arxiv/`: locally cached PDFs/markdown summaries of cited foundation-model and remote-sensing papers (e.g., AlphaEarth Foundations, SatMAE, SSL4Eco). Use these when refining literature context or detailing methodology inputs.

## Manuscript Revision Workflow
Rewrite `article/article.tex` and `article/supplementary_materials.tex` iteratively, following the roadmap in `article/review/article_revision_guidelines.md`. Preserve earlier drafts under `article/previous_article.tex` and `article/previous_supplementary_materials.tex`; they contain supervisor remarks marked with `# commentary` and should guide revisions. Maintain a concise scientific style: analyse experimental results without extrapolation, avoid references to repository scripts, and favour clear prose over bullet points. Prior to edits, review `article/review/writing_a_good_scientific_paper.md` alongside supporting resources (`article/review/more_info_dataset.tex`, `arxiv/`, and `deepresearch/`) to ensure consistency with the intended scholarly tone.

Maintain a living narrative map in `scratchpad.md`. Use it to capture the storyline, hypotheses, figure/table intents, and cross-section themes as described in `article/review/writing_a_good_scientific_paper.md`, then propagate those points across Methods, Results, Discussion, and Conclusion draft iterations.

## Multi-agent Coordination
- Assume other automation agents or collaborators might have active changes in the working tree. Never delete or overwrite files created by teammates unless you have confirmed they are no longer needed.
- When you need to modify shared helpers (e.g., under `gee/` or `src/visualization/`), review `git status` and existing scripts first, then extend them in place rather than replacing or removing them.
- If you notice conflicting edits, communicate the collision in the task log or leave a note instead of forcefully resetting files. This keeps figure-generation scripts and credential helpers available for everyone.
