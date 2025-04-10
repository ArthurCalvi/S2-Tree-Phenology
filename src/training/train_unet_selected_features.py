"""
Tree Phenology Classification with U-Net (Selected Features)

This script trains a small U-Net convolutional neural network (< 100k parameters)
to classify forest pixels into deciduous or evergreen trees using spatial context
from satellite imagery, allowing selection of specific input features.

Key features:
- Uses a small U-Net architecture.
- Allows training with a user-defined set of input features.
- Performs 5-fold cross-validation based on geographic distribution.
- Applies appropriate data transformations for circular phase features.
- Logs training metrics to TensorBoard for visualization.
- Handles imbalanced classes with weighted loss functions.
- Saves trained model, normalization statistics, and configuration.

Usage:
    python train_phenology_unet_selected_features.py --features FEATURE1,FEATURE2,... [--output_dir DIR] [--model_name NAME] [...]

Author: Arthur Calvi
Date: November 2023 / Modified: July 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
import logging
from tqdm import tqdm
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import rasterio
from collections import defaultdict
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from sklearn.utils import shuffle
import torch.nn.functional as F
import json
import datetime # Added
from pathlib import Path # Added
import sys # Added
import math # Good practice

# Add project root for imports
try:
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
except NameError:
    project_root = str(Path('.').resolve())
    if project_root not in sys.path:
        sys.path.append(project_root)

# Import utility functions
try:
    from src.utils import (
        unscale_feature, # Needed indirectly by PhenologyTileDataset
        SmallUNet, 
        SpatialTransforms, 
        PhenologyTileDataset, 
        create_eco_balanced_folds_tiles, # Renamed function
        scaled_masked_loss, 
        evaluate_unet_model, # Renamed function
        visualize_unet_predictions, # Renamed function
        prepare_unet_images_for_tensorboard, # Renamed function
        get_tile_eco_region_mapping, # Needed by create_eco_balanced_folds_tiles
        # Need these for feature selection handling in PhenologyTileDataset
        BASE_FEATURE_NAMES_PER_INDEX,
        OUTPUT_FEATURE_NAMES_PER_INDEX,
        ALL_FEATURE_BAND_INDICES, 
        FEATURE_SUFFIX_TO_TYPE,
        PHASE_FEATURE_SUFFIXES, # Added in utils
        PHASE_TRANSFORM_SUFFIXES, # Added in utils
        format_confusion_matrix # Added import
    )
    # Import constants (assuming they are now needed here or in utils)
    from src.constants import PHENOLOGY_MAPPING
except ImportError as e:
    logger.critical(f"Failed to import from src.utils or src.constants: {e}. Check paths and __init__.py files.", exc_info=True)
    sys.exit(1)

# Set up logging
os.makedirs('logs', exist_ok=True)
# Modified log file name
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phenology_unet_selected_features_training.log', mode='w'), # Overwrite log each run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default output directory (can be overridden by CLI)
DEFAULT_OUTPUT_DIR = 'results/models/unet_selected'
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Constants (can be overridden by CLI)
PATCH_SIZE = 64
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50

# --- Training Function ---
def train_and_evaluate_unet(
        selected_features, tile_paths, output_dir, model_base_name,
        n_splits=5, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LEARNING_RATE,
        use_augmentation=False, use_sample_weights=True, weight_scale=1.0, test_mode=False):
    """Trains and evaluates U-Net using CV with selected features."""
    logger.info(f"Starting U-Net CV Training: {len(selected_features)} features, {n_splits} folds.")
    logger.info(f"Params: Epochs={epochs}, Batch={batch_size}, LR={lr}, Patch={patch_size}, Augment={use_augmentation}, Weights={use_sample_weights} (Scale={weight_scale})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    num_workers = 4 if device.type == 'cuda' else 0
    actual_batch_size = batch_size if device.type == 'cuda' else min(batch_size, 4) # Smaller batch for CPU

    # Compute global stats using a subset of tiles
    logger.info("Computing global statistics for normalization...")
    dummy_dataset = PhenologyTileDataset(
        tile_paths=tile_paths[:min(20, len(tile_paths))],  # Use subset of tiles for efficiency
        selected_features=selected_features,
        compute_global_stats=True,
        use_sample_weights=False,
        logger=logger
    )
    global_stats = dummy_dataset.global_stats
    output_feature_names = dummy_dataset.output_feature_names
    num_output_features = dummy_dataset.num_output_features
    logger.info(f"Training with {num_output_features} processed features: {output_feature_names}")

    # Save global stats
    stats_filename = f"{model_base_name}_normalization_stats.json"
    stats_path = os.path.join(output_dir, stats_filename)
    try:
        with open(stats_path, 'w') as f: json.dump(global_stats, f, indent=4)
        logger.info(f"Saved normalization statistics to {stats_path}")
    except Exception as e: 
        logger.error(f"Error saving normalization stats: {e}")
    
    # Create folds
    fold_splits = create_eco_balanced_folds_tiles(tile_paths, n_splits=n_splits, logger=logger)
    all_fold_metrics = []
    tb_log_dir = os.path.join(output_dir, 'tensorboard_logs')
    tb_writer = SummaryWriter(tb_log_dir)
    logger.info(f"TensorBoard logs: {tb_log_dir}")

    # Store results per fold
    results_per_fold = []

    # --- Cross-Validation Loop ---
    for fold, (train_tiles, val_tiles) in enumerate(fold_splits):
        fold_num = fold + 1
        logger.info(f"\n=== Fold {fold_num}/{n_splits} ===")
        if not val_tiles: logger.warning(f"Skipping fold {fold_num}: No validation tiles."); continue

        try:
            train_transforms = SpatialTransforms(p_flip=0.5) if use_augmentation else None
            train_dataset = PhenologyTileDataset(
                train_tiles,
                selected_features=selected_features,
                patch_size=patch_size,
                transform=train_transforms,
                global_stats=global_stats,
                use_sample_weights=use_sample_weights,
                logger=logger
            )
            val_dataset = PhenologyTileDataset(
                val_tiles,
                selected_features=selected_features,
                patch_size=patch_size,
                global_stats=global_stats,
                use_sample_weights=use_sample_weights,
                logger=logger
            )
        except Exception as e: 
            logger.error(f"Error creating datasets for fold {fold_num}: {e}", exc_info=True)
            continue

        if len(train_dataset) == 0 or len(val_dataset) == 0: logger.warning(f"Skipping fold {fold_num}: Empty dataset(s)."); continue

        train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == 'cuda')
        val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
        logger.info(f"Fold {fold_num}: Train patches={len(train_dataset)}, Val patches={len(val_dataset)}")

        model = SmallUNet(in_channels=num_output_features, n_classes=3).to(device)
        logger.info(f"Fold {fold_num}: Model params={model.count_parameters():,}")

        # Class weights for Deciduous[0], Evergreen[1] (after label shift in loss)
        class_counts = np.zeros(3)
        for _, label, _ in train_dataset: unique, counts = np.unique(label.numpy(), return_counts=True); class_counts[unique] += counts
        total_valid = class_counts[1] + class_counts[2]
        if total_valid > 0 and class_counts[1] > 0 and class_counts[2] > 0:
            weights = [total_valid / class_counts[1], total_valid / class_counts[2]]
            logger.info(f"Fold {fold_num}: Class weights (D/E): [{weights[0]:.2f}, {weights[1]:.2f}]")
        else: weights = [1.0, 1.0]; logger.warning(f"Fold {fold_num}: Using equal class weights.")
        class_weights_tensor = torch.tensor(weights, device=device, dtype=torch.float32)

        criterion = lambda out, targ, s_w: scaled_masked_loss(out, targ, weight=class_weights_tensor, sample_weights=s_w, weight_scale=weight_scale)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.3, div_factor=25.0, final_div_factor=1000.0)

        best_val_f1 = -1.0
        best_model_path = os.path.join(output_dir, f"{model_base_name}_fold{fold_num}_best.pt")
        start_fold_time = time.time()
        logger.info(f"Fold {fold_num}: Starting training...")

        # --- Epoch Loop ---
        for epoch in range(epochs):
            model.train()
            running_train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train", leave=False)
            for inputs, labels, sample_weights in pbar:
                inputs, labels, sample_weights = inputs.to(device), labels.to(device), sample_weights.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels, sample_weights)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            epoch_train_loss = running_train_loss / len(train_loader) if train_loader else 0.0
            val_metrics = evaluate_unet_model(model, val_loader, device, criterion, weight_scale, class_weights_tensor)
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1_score']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")

            # TensorBoard Logging
            tb_writer.add_scalar(f'Loss/Train/Fold_{fold_num}', epoch_train_loss, epoch)
            tb_writer.add_scalar(f'Loss/Val/Fold_{fold_num}', val_metrics['loss'], epoch)
            tb_writer.add_scalar(f'Metrics/Val_F1/Fold_{fold_num}', val_metrics['f1_score'], epoch)
            tb_writer.add_scalar(f'Metrics/Val_Accuracy/Fold_{fold_num}', val_metrics['accuracy'], epoch)
            tb_writer.add_scalar(f'Metrics/Val_Precision/Fold_{fold_num}', val_metrics['precision'], epoch)
            tb_writer.add_scalar(f'Metrics/Val_Recall/Fold_{fold_num}', val_metrics['recall'], epoch)
            tb_writer.add_scalar(f'LearningRate/Fold_{fold_num}', optimizer.param_groups[0]['lr'], epoch)

            # Image Logging
            if epoch % 5 == 0 or epoch == epochs - 1:
                try:
                    vis_batch = next(iter(val_loader))
                    vis_inputs, vis_labels, _ = vis_batch
                    model.eval()
                    with torch.no_grad(): 
                        _, vis_preds = torch.max(model(vis_inputs.to(device)), 1)
                    model.train()
                    tb_images = prepare_unet_images_for_tensorboard(vis_inputs, vis_labels, vis_preds.cpu(), output_feature_names, f"fold_{fold_num}")
                    for img_tag, img_tensor in tb_images.items(): 
                        tb_writer.add_image(img_tag, img_tensor, global_step=epoch)
                except Exception as e_vis: 
                    logger.error(f"TB Image Error: {e_vis}", exc_info=False)

            # Save Best Model
            current_val_f1 = val_metrics['f1_score']
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                try: torch.save(model.state_dict(), best_model_path); logger.info(f"Epoch {epoch+1}: Best model saved (Val F1: {best_val_f1:.4f})")
                except Exception as e_save: logger.error(f"Error saving model: {e_save}")
        # --- End Epoch Loop ---

        fold_time = time.time() - start_fold_time
        logger.info(f"Fold {fold_num} training finished in {fold_time:.2f}s.")

        # Final Evaluation with Best Model
        if os.path.exists(best_model_path):
            try: model.load_state_dict(torch.load(best_model_path, map_location=device)); logger.info("Loaded best model for final evaluation.")
            except Exception as e: logger.error(f"Error loading best model: {e}. Evaluating last epoch model.")
        else: logger.warning("Best model file not found. Evaluating last epoch model.")

        final_metrics = evaluate_unet_model(model, val_loader, device)
        logger.info(f"Fold {fold_num} Final Val F1: {final_metrics['f1_score']:.4f}, Acc: {final_metrics['accuracy']:.4f}")
        logger.info("\n" + format_confusion_matrix(final_metrics['confusion_matrix']))

        # Store fold results
        final_metrics['fold'] = fold_num
        final_metrics['training_time_seconds'] = round(fold_time, 2)
        final_metrics['confusion_matrix'] = final_metrics['confusion_matrix'].tolist() # For JSON
        all_fold_metrics.append(final_metrics)

        # Visualize predictions
        try: visualize_unet_predictions(model, val_loader, device, output_feature_names, output_dir, f"fold_{fold_num}")
        except Exception as e: logger.error(f"Prediction visualization error: {e}", exc_info=False)

    # --- End CV Loop ---

    if not all_fold_metrics:
        logger.error("No folds completed successfully.")
        tb_writer.close()
        return None, None

    # Aggregate and Save Results
    results_df = pd.DataFrame(all_fold_metrics)
    avg_metrics = {f"mean_{k}": results_df[k].mean() for k in ['f1_score', 'precision', 'recall', 'accuracy']}
    std_metrics = {f"std_{k}": results_df[k].std() for k in ['f1_score', 'precision', 'recall', 'accuracy']}
    overall_summary = {**avg_metrics, **std_metrics}

    logger.info("\n=== Overall CV Results ===")
    for k, v in overall_summary.items(): logger.info(f"{k.replace('_', ' ').title()}: {v:.4f}")

    fold_metrics_path = os.path.join(output_dir, f"{model_base_name}_fold_metrics.json")
    summary_csv_path = os.path.join(output_dir, f"{model_base_name}_cv_summary.csv")
    try:
        with open(fold_metrics_path, 'w') as f: json.dump(all_fold_metrics, f, indent=4)
        logger.info(f"Saved detailed fold metrics to {fold_metrics_path}")
        # Add summary stats to fold results df for saving
        for k, v in overall_summary.items(): results_df[k] = v
        results_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
        logger.info(f"Saved CV summary results to {summary_csv_path}")
    except Exception as e: logger.error(f"Error saving results files: {e}")

    tb_writer.close()
    return overall_summary, stats_path, output_feature_names, num_output_features # Return more info for config
    
# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Train U-Net for phenology classification with selected features.')
    parser.add_argument('--features', '-f', type=str, required=True, help='Comma-separated feature names (e.g., "ndvi_amplitude_h1,ndvi_phase_h1_cos").')
    parser.add_argument('--output_dir', '-o', type=str, default=DEFAULT_OUTPUT_DIR, help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}).')
    parser.add_argument('--model_name', '-m', type=str, default='phenology_unet_selected', help='Base name for saved files (default: phenology_unet_selected).')
    parser.add_argument('--tile_dir', type=str, default='data/training/training_tiles2023_w_corsica/training_tiles2023', help='Directory with training TIFF tiles.')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds (default: 5).')
    parser.add_argument('--epochs', '-e', type=int, default=EPOCHS, help=f'Number of epochs (default: {EPOCHS}).')
    parser.add_argument('--batch_size', '-b', type=int, default=BATCH_SIZE, help=f'Batch size (default: {BATCH_SIZE}).')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help=f'Learning rate (default: {LEARNING_RATE}).')
    parser.add_argument('--patch_size', type=int, default=PATCH_SIZE, help=f'Patch size (default: {PATCH_SIZE}).')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation (flips).')
    parser.add_argument('--no_weights', action='store_true', help='Disable eco-region sample weights.')
    parser.add_argument('--weight_scale', type=float, default=1.0, help='Eco-region weight scaling factor (default: 1.0).')
    parser.add_argument('--test', '-t', action='store_true', help='Run in test mode (fewer tiles).')
    parser.add_argument('--test_tiles', type=int, default=10, help='Number of tiles for test mode (default: 10).')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Starting U-Net training script (Selected Features)")
    logger.info(f"Cmd: {' '.join(sys.argv)}")
    logger.info(f"Args: {vars(args)}")

    # Get tiles
    if not os.path.isdir(args.tile_dir): logger.error(f"Tile directory not found: {args.tile_dir}"); return
    tile_paths = glob.glob(os.path.join(args.tile_dir, "tile_*_training.tif"))
    if not tile_paths: logger.error(f"No training tiles found in {args.tile_dir}"); return
    logger.info(f"Found {len(tile_paths)} training tiles.")

    if args.test:
        if len(tile_paths) > args.test_tiles:
            tile_paths = list(np.random.choice(tile_paths, args.test_tiles, replace=False))
            logger.info(f"TEST MODE: Using {len(tile_paths)} tiles.")
        else: logger.warning(f"Test mode: Using all {len(tile_paths)} found tiles.")

    # Parse and validate features
    selected_features = [f.strip() for f in args.features.split(',')]
    logger.info(f"Selected features requested: {selected_features}")

    # Train
    start_time = time.time()
    train_result = train_and_evaluate_unet(
        selected_features=selected_features, tile_paths=tile_paths, output_dir=args.output_dir,
        model_base_name=args.model_name, n_splits=args.folds, patch_size=args.patch_size,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, use_augmentation=args.augment,
        use_sample_weights=(not args.no_weights), weight_scale=args.weight_scale, test_mode=args.test
    )
    elapsed_time = time.time() - start_time
    
    if train_result is None:
        logger.error("Training function failed.")
        return

    avg_metrics, norm_stats_path, output_feature_names, num_output_features = train_result
    logger.info(f"CV training finished in {elapsed_time:.2f} seconds.")

    # Save Config File
    config_filename = f"{args.model_name}_config.json"
    config_path = os.path.join(args.output_dir, config_filename)
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    config_data = {
        'model_base_name': args.model_name, 'creation_date': current_date, 'model_type': 'SmallUNet',
        'selected_features': selected_features, 'feature_count_input': len(selected_features),
        'feature_count_processed': num_output_features, 'processed_feature_names': output_feature_names,
        'normalization_stats_file': os.path.basename(norm_stats_path) if norm_stats_path else None,
        'training_arguments': vars(args), 'cross_validation_summary': avg_metrics,
    }
    try:
        with open(config_path, 'w') as f: json.dump(config_data, f, indent=4, default=str)
        logger.info(f"Training config saved to: {config_path}")
    except Exception as e: logger.error(f"Error saving config: {e}")

    logger.info("Script completed successfully.")

if __name__ == "__main__":
    main() 