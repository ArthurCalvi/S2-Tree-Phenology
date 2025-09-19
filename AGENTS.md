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
