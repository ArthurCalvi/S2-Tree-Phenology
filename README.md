# S2-Tree-Phenology

France-wide deciduous vs evergreen mapping at 10 m resolution. The project compares two pipelines—physics-informed Sentinel-2 harmonics and AlphaEarth foundation-model embeddings—sharing the same sampling, eco-region folds, and evaluation protocol. This repository contains data-prep utilities, training scripts, HPC batch jobs, QA workflows, and article assets needed to reproduce the experiments reported in `article/article_v3.tex`.

## Quick Start

1. **Set up the environment**: create a Python ≥3.10 env and `pip install -r requirements.txt`. Configure Google Earth Engine (GEE) credentials and install `geefetch` for automated downloads.
2. **Acquire inputs**: download Sentinel-2 monthly mosaics and AlphaEarth embeddings for France using the YAMLs under `geefetch/configs/` (see [Data Acquisition](#data-acquisition)).
3. **Build training tables**: run the parquet builders for harmonics and embeddings in `src/sampling/` to extract per-pixel training records.
4. **Train models**: launch `bash src/training/run_train_rf_selected_features.sh` (harmonics) and the embedding variants (`run_train_rf_embeddings_all.sh`, `run_train_rf_embeddings_topk.sh`). Each job writes `.joblib` weights, metrics, and feature manifests to `results/final_model/`.
5. **Run national inference**: use the feature or embedding inference CLIs (`src/inference/inference_rf_on_tiles.py`, `src/inference/inference_rf_embeddings.py`) from Jean Zay job arrays to tile through France.
6. **Evaluate & report**: QA scripts in `qa/jobs/` and analysis notebooks under `src/analysis/` reproduce the figures/statistics referenced in the article.

## Repository Layout Highlights

- `src/`: project code organised by processing stage (see `architecture.md`).
  - `sampling/`: tile selection and parquet builders for harmonics (`dataset_creation.py`) and embeddings (`dataset_preparation_embedding.py`, `convert_embeddings_to_dataframe.py`).
  - `features/`: Sentinel-2 harmonic feature extraction (`prepare_inference_feature.py`, `inference_feature.py`).
  - `training/`: Random Forest training scripts plus Slurm wrappers (`train_rf_selected_features.py`, `train_rf_embeddings.py`).
  - `inference/`: Tile-based RF inference for harmonics (`inference_rf_on_tiles.py`) and embeddings (`inference_rf_embeddings.py`).
  - `analysis/`, `qa/`, `reporting/`: downstream similarity studies, QA jobs, and reporting utilities.
- `geefetch/configs/`: GEE download configs including yearly AlphaEarth YAMLs (`alphaearth_embeddings/`).
- `jobs/`: Slurm launchers for data acquisition and QA, e.g. `jobs/alphaearth_embeddings/job_geefetch_alphaearth_embeddings_*.sh`.
- `article/`: LaTeX manuscript, figures, and tables referenced in the README.

## Environment Setup

### Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Earth Engine & Geefetch

- Authenticate with `earthengine authenticate` using the `phdforest` project (update `satellite_default.gee.ee_project_id` if needed).
- Install the lab's Geefetch fork (Python ≥3.10):

```bash
pip install git+https://github.com/gbelouze/geefetch.git
```

### Jean Zay HPC Modules

Job scripts assume the standard environment:

```bash
module purge
module load pytorch-gpu/py3/2.5.0 gdal/3.10.0 google-cloud-cli
export PYTHONUSERBASE=$ALL_CCFRWORK/envs/gf-0
export PATH=$ALL_CCFRWORK/envs/gf-0/bin:$PATH
```

Tune module versions to match system availability.

## Data Acquisition

### Sentinel-2 Monthly Mosaics (Harmonic Pipeline)

1. Copy an existing YAML (e.g. `geefetch/configs/config_corsica_20230415.yaml`) and adjust `data_dir`, temporal range, and AOI as needed.
2. Submit the Geefetch job using a template such as `jobs/job_geeftech_20230115.sh`:

```bash
sbatch jobs/job_geeftech_20230115.sh
```

Tiles are written per month under the specified `data_dir` and later stacked by `src/features/prepare_inference_feature.py`.

### AlphaEarth Embeddings (2017–2024)

1. Choose the yearly config under `geefetch/configs/alphaearth_embeddings/`.
2. Stage the Slurm wrapper from `jobs/alphaearth_embeddings/` and adjust output paths if required:

```bash
sbatch jobs/alphaearth_embeddings/job_geefetch_alphaearth_embeddings_2021.sh
```

Each job creates `/lustre/.../alphaearth_embeddings/<year>/` containing yearly 64-band GeoTIFF chips for the France AOI.

## Build Training Tables

### Harmonix Dataset

1. Assemble labelled pixels with eco-region metadata:

```bash
python src/sampling/dataset_creation.py \
  --tiles results/datasets/tiles_2_5_km_final.parquet \
  --out results/datasets/training_datasets_pixels.parquet
```

2. (Optional) Apply weighting or quality filters using utilities in `src/sampling/`.

### Embedding Dataset

1. Generate per-tile embedding stacks (if not already done) using `src/sampling/dataset_preparation_embedding.py`:

```bash
python src/sampling/dataset_preparation_embedding.py \
  --tiles_path results/datasets/tiles_2_5_km_final.parquet \
  --dataset_path results/datasets/final_dataset.parquet \
  --features_vrt /path/to/alphaearth_features.vrt \
  --output_dir data/training/embeddings
```

2. Convert the tiles to a parquet training table:

```bash
python src/dataset_creation/convert_embeddings_to_dataframe.py \
  --input_dir data/training/embeddings \
  --final_dataset results/datasets/final_dataset.parquet \
  --output_path results/datasets/training_datasets_pixels_embedding.parquet
```

The parquet includes columns `embedding_0`–`embedding_63`, categorical labels, eco-region, and optional weights.

## Train Random Forest Baselines

### Harmonic 14-Feature Model

```bash
bash src/training/run_train_rf_selected_features.sh
```

Outputs:
- `results/final_model/phenology_model_selected_features_<date>.joblib`
- `results/final_model/phenology_metrics_selected_features.json`
- `results/final_model/phenology_eco_metrics_selected_features.csv`

### Embedding Models (All Bands & Top-14)

All-band baseline:

```bash
bash src/training/run_train_rf_embeddings_all.sh
```

Top-K (default K=14, using pre-computed feature lists when available):

```bash
bash src/training/run_train_rf_embeddings_topk.sh
```

Both commands call `src/training/train_rf_embeddings.py`, which now serialises the fitted estimator and metadata:
- `rf_embeddings_<tag>_<timestamp>.joblib`
- `model_metadata_<tag>.json` (lists features, classes, RF hyperparameters)
- `features_<tag>.txt`
- `metrics_<tag>.json`, `eco_metrics_<tag>.csv`

These artefacts feed inference and reporting scripts.

## Large-Scale Inference

### Harmonic Feature Pipeline

1. Prepare inference tiles (harmonic features) from Sentinel-2 mosaics:

```bash
python src/features/prepare_inference_feature.py --help
```

2. Optionally compute features per date using `src/features/inference_feature.py`.

3. Run RF inference per tile via the job array `src/inference/run_inference_rf.sh` (edit the paths inside before submitting).

### Embedding RF Inference

Use the dedicated CLI to score embedding tiles:

```bash
python src/inference/inference_rf_embeddings.py \
  --input-dir /path/to/embedding_tiles \
  --output-dir /path/to/output_probs \
  --model results/final_model/rf_embeddings_embeddings_topk_k14_<timestamp>.joblib \
  --features-file results/final_model/features_embeddings_topk_k14.txt \
  --block-size 2048 --save-classes
```

For Jean Zay batch execution, adapt and submit the array job:

```bash
sbatch src/inference/run_inference_rf_embeddings.sh
```

The script validates paths, sets optional `--rf-n-jobs`, and writes probability rasters (uint8 0–255) plus an optional class map.

## Evaluation, QA, and Analysis

- **National QA checks**: `bash qa/jobs/qa_corsica.sh` runs the standard QA workflow on benchmark tiles.
- **Embedding vs harmonic similarity**: `bash src/analysis/run_compute_embedding_harmonic_similarity.sh` computes ridge-based alignments; plotting utilities under `src/analysis/` reproduce the article heatmaps.
- **Map comparison**: `src/comparison/compare_maps.py` evaluates RF outputs against reference layers (BD Forêt, DLT) and writes summaries for reporting.
- **Post-processing**: `src/post-processing/classify_forest_types.py` and `compress_rf_probabilities.py` turn probability tiles into labelled or compressed products.

All metric CSV/JSON outputs land under `results/`, while diagnostic plots go to `results/analysis_*` or `runs/` (TensorBoard).

## Reproducing the Article

- `article/article_v3.tex` is the latest manuscript. The supporting commands and artefacts are enumerated in `article_methods.json`, mapping each experiment to scripts, inputs, and outputs.
- Figures referenced in the paper (national map, Corsica vignette, similarity charts) are generated by the pipelines above and stored under `article/images/` or `results/analysis_*`.

## Additional Resources

- `AGENTS.md`: high-level contributor guidelines generated for coding agents.
- `architecture.md`: detailed overview of the `src/` modules and workflow.
- `PROGRESS.md`: running log of project milestones.

For issues or clarifications, open a GitHub issue or contact the maintainers listed in the manuscript.
