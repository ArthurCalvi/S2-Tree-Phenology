# Tree Phenology Classification

This repository contains code for training and evaluating machine learning models to classify tree phenology (deciduous vs evergreen) using Sentinel-2 time series features.

## Dataset

The dataset used contains pixel-level features derived from Sentinel-2 time series over one year, with the following key properties:

- Each row represents one pixel
- Features include amplitude, phase, and offset from harmonic fitting of various indices (NDVI, EVI, NBR, CRSWIR)
- Reference data includes phenology (deciduous/evergreen), genus, and species
- The dataset includes metadata like eco-region, tile_id, and sample weight

## Training Script

The main training script `train_phenology.py` implements a 5-fold cross-validation approach with the following characteristics:

1. **Spatial Cross-Validation**: 
   - Folds are created based on tile_id to avoid spatial data leakage
   - Pixels from the same tile cannot appear in both training and validation sets

2. **Eco-Region Distribution**:
   - The script maintains an approximate 80/20% distribution of eco-regions between training and validation sets
   - For each fold, it displays the distribution of pixels across eco-regions

3. **Feature Sets**:
   - The script tests different vegetation indices (NDVI, EVI, NBR, CRSWIR) independently
   - For each index, it uses features derived from harmonic fitting: amplitude_h1, amplitude_h2, offset, var_residual
   - **Circular Features Handling**: Phase features (phase_h1, phase_h2) are transformed using cos/sin to properly represent their circular nature, resulting in phase_h1_cos, phase_h1_sin, phase_h2_cos, phase_h2_sin

4. **Evaluation**:
   - Metrics calculated include F1-score, confusion matrix, true positives, false positives, true negatives, false negatives
   - Results are presented per eco-region and overall
   - Feature importance plots show the relative importance of each feature

5. **Class Balancing**:
   - Uses `class_weight='balanced'` option in RandomForest to handle class imbalance
   - Applies sample weights from the dataset if available

6. **Progress Tracking and Reporting**:
   - Utilizes progress bars (tqdm) to track processing of indices and cross-validation folds
   - Comprehensive logging to both console and log file for easy monitoring
   - Generates a detailed PDF report with all training results and metrics

## PDF Report

The script automatically generates a comprehensive PDF report (`results/models/phenology_report.pdf`) that includes:

1. **Summary Comparison**: Performance metrics across all indices
2. **Detailed Results per Index**:
   - Confusion matrix with class breakdown
   - Performance metrics (accuracy, precision, recall, F1-score, TP, FP, TN, FN)
   - Results broken down by eco-region
   - Feature importance ranking

This report provides a convenient way to review and share results without needing to analyze individual CSV files.

## Usage

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the training script:
   ```
   python train_phenology.py
   ```

3. Results will be saved in the `results/models/` directory, including:
   - CSV files with metrics per fold and per eco-region
   - Feature importance plots
   - Comparison of performance across indices
   - PDF report with comprehensive results
   - Log file with training progress and metrics

## Output Structure

The script generates the following outputs in the `results/models/` directory:

- `phenology_<index>_overall_results.csv`: Metrics for each fold for the specified index
- `phenology_<index>_ecoregion_results.csv`: Metrics per eco-region for the specified index
- `phenology_<index>_feature_importance.png`: Feature importance visualization
- `phenology_indices_comparison.csv`: Summary comparison of all indices
- `phenology_indices_comparison.png`: Bar chart comparing F1-scores across indices
- `phenology_report.pdf`: Comprehensive PDF report with all results
- `phenology_training.log`: Detailed log of the training process

## Project Structure

```
├── jobs/           # Job scripts and configuration files
├── tests/          # Unit tests
├── src/            # Source code
│   ├── sampling/   # Data sampling and preprocessing
│   ├── training/   # Model training code
│   ├── features/   # Feature engineering
│   ├── inference/  # Model inference and predictions
│   ├── utils.py    # Utility functions
│   └── constants.py# Project constants and configurations
└── results/        # Output results and model artifacts
    └── models/     # Trained models and evaluation results
```

## Setup

[Instructions for setting up the project will go here]

## Contributing

[Contributing guidelines will go here]

## License

[License information will go here] 