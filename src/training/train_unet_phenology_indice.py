"""
Tree Phenology Classification with U-Net

This script trains a small U-Net convolutional neural network (< 100k parameters)
to classify forest pixels into deciduous or evergreen trees using spatial context
from satellite imagery.

Key features:
- Uses only 58k parameters while maintaining good performance
- Trains separate models for each vegetation index (ndvi, evi, nbr, crswir)
- Performs 5-fold cross-validation based on geographic distribution
- Applies appropriate data transformations for circular phase features
- Logs training metrics to TensorBoard for visualization
- Handles imbalanced classes with weighted loss functions

Usage:
    python train_phenology_unet.py [--test] [--index INDEX] [--folds N]

Author: Arthur Calvi
Date: November 2023
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils import apply_science_style
apply_science_style()
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
import sys
from pathlib import Path
import math

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
        unscale_feature,
        SmallUNet,
        SpatialTransforms,
        PhenologyTileDataset,
        create_eco_balanced_folds_tiles,
        scaled_masked_loss,
        evaluate_unet_model,
        visualize_unet_predictions,
        prepare_unet_images_for_tensorboard,
        get_tile_eco_region_mapping,
        format_confusion_matrix
    )
    # Import constants (assuming they are now needed here or in utils)
    from src.constants import PHENOLOGY_MAPPING
except ImportError as e:
    logger.critical(f"Failed to import from src.utils or src.constants: {e}. Check paths and __init__.py files.", exc_info=True)
    sys.exit(1)

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phenology_unet_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create results directory if it doesn't exist
os.makedirs('results/models/unet', exist_ok=True)

# Define the indices to test
INDICES = ['ndvi', 'evi', 'nbr', 'crswir']

# Map feature suffixes to unscaling types
FEATURE_SUFFIX_TO_TYPE = {
    '_amplitude_h1': 'amplitude',
    '_amplitude_h2': 'amplitude',
    '_phase_h1': 'phase',
    '_phase_h2': 'phase',
    '_offset': 'offset',
    '_var_residual': 'variance'
}

# Base feature names for each index (before cos/sin transform)
BASE_FEATURE_NAMES = {
    'ndvi': ['ndvi_amplitude_h1', 'ndvi_amplitude_h2', 'ndvi_phase_h1', 'ndvi_phase_h2', 'ndvi_offset', 'ndvi_var_residual'],
    'evi': ['evi_amplitude_h1', 'evi_amplitude_h2', 'evi_phase_h1', 'evi_phase_h2', 'evi_offset', 'evi_var_residual'],
    'nbr': ['nbr_amplitude_h1', 'nbr_amplitude_h2', 'nbr_phase_h1', 'nbr_phase_h2', 'nbr_offset', 'nbr_var_residual'],
    'crswir': ['crswir_amplitude_h1', 'crswir_amplitude_h2', 'crswir_phase_h1', 'crswir_phase_h2', 'crswir_offset', 'crswir_var_residual']
}

# Output feature names (after cos/sin transform)
OUTPUT_FEATURE_NAMES = {
    'ndvi': ['ndvi_amplitude_h1', 'ndvi_amplitude_h2', 'ndvi_phase_h1_cos', 'ndvi_phase_h1_sin', 'ndvi_phase_h2_cos', 'ndvi_phase_h2_sin', 'ndvi_offset', 'ndvi_var_residual'],
    'evi': ['evi_amplitude_h1', 'evi_amplitude_h2', 'evi_phase_h1_cos', 'evi_phase_h1_sin', 'evi_phase_h2_cos', 'evi_phase_h2_sin', 'evi_offset', 'evi_var_residual'],
    'nbr': ['nbr_amplitude_h1', 'nbr_amplitude_h2', 'nbr_phase_h1_cos', 'nbr_phase_h1_sin', 'nbr_phase_h2_cos', 'nbr_phase_h2_sin', 'nbr_offset', 'nbr_var_residual'],
    'crswir': ['crswir_amplitude_h1', 'crswir_amplitude_h2', 'crswir_phase_h1_cos', 'crswir_phase_h1_sin', 'crswir_phase_h2_cos', 'crswir_phase_h2_sin', 'crswir_offset', 'crswir_var_residual']
}

# Constants
PATCH_SIZE = 64  # Size of spatial patches to extract
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 30

def train_and_evaluate_unet(index, tile_paths, n_splits=5, patch_size=PATCH_SIZE, 
                           batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LEARNING_RATE,
                           use_augmentation=False, use_sample_weights=True, weight_scale=1.0):
    """
    Train and evaluate U-Net model for phenology classification using spatial context.
    Uses cross-validation based on tiles.
    
    Args:
        index: Vegetation index to use ('ndvi', 'evi', 'nbr', or 'crswir')
        tile_paths: List of paths to training tiles
        n_splits: Number of cross-validation folds
        patch_size: Size of spatial patches to extract
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        use_augmentation: Whether to use spatial data augmentation
        use_sample_weights: Whether to use eco-region based sample weights
        weight_scale: Scaling factor for eco-region weights (1.0 = full effect, 0.0 = no effect)
    """
    # Log parameters
    logger.info(f"Starting training for {index.upper()} with U-Net model")
    if use_sample_weights:
        logger.info(f"Using eco-region weights with scaling factor: {weight_scale}")
    else:
        logger.info("Eco-region based sample weighting is DISABLED")
        
    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Adjust batch size and workers based on device
    num_workers = 0 if device.type == 'cpu' else 4  # No parallel workers on CPU to avoid issues
    actual_batch_size = batch_size
    if device.type == 'cpu':
        # Use a smaller batch size on CPU
        actual_batch_size = min(batch_size, 8)
        logger.info(f"Reduced batch size to {actual_batch_size} for CPU training")
    
    # Compute global statistics once for the entire dataset
    logger.info("Computing global statistics for normalization...")
    dummy_dataset = PhenologyTileDataset(
        tile_paths[:min(20, len(tile_paths))],  # Use subset of tiles for efficiency
        index=index,
        compute_global_stats=True,
        use_sample_weights=False,  # Don't compute weights when just getting stats
        logger=logger
    )
    global_stats = dummy_dataset.global_stats
    
    # Save global statistics to disk for later use in inference
    stats_path = f"results/models/unet/phenology_{index}_normalization_stats.json"
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    
    with open(stats_path, 'w') as f:
        json.dump(global_stats, f, indent=2)
    
    logger.info(f"Saved normalization statistics to {stats_path}")
    
    # Create folds for cross-validation
    fold_splits = create_eco_balanced_folds_tiles(tile_paths, n_splits=n_splits, logger=logger)
    
    # Store results
    results_per_fold = []
    
    # Setup TensorBoard writer
    tb_writer = SummaryWriter(f'runs/phenology_unet_{index}')
    
    # Train and evaluate for each fold
    for fold, (train_tiles, val_tiles) in enumerate(fold_splits):
        logger.info(f"\n=== Fold {fold+1}/{n_splits} ===")
        logger.info(f"Training on {len(train_tiles)} tiles, validating on {len(val_tiles)} tiles")
        
        # Skip folds with no validation data
        if len(val_tiles) == 0:
            logger.warning(f"Skipping fold {fold+1} because it has no validation tiles")
            continue
        
        # Create datasets with transforms for training if augmentation is enabled
        train_transforms = SpatialTransforms(p_flip=0.5) if use_augmentation else None
        
        train_dataset = PhenologyTileDataset(
            train_tiles, 
            index=index, 
            patch_size=patch_size,
            transform=train_transforms,  # Pass the transforms
            global_stats=global_stats,
            use_sample_weights=use_sample_weights,
            logger=logger
        )
        
        val_dataset = PhenologyTileDataset(
            val_tiles, 
            index=index, 
            patch_size=patch_size,
            global_stats=global_stats,  # No transforms for validation
            use_sample_weights=use_sample_weights,
            logger=logger
        )
        
        # Skip if validation dataset is empty
        if len(val_dataset) == 0:
            logger.warning(f"Skipping fold {fold+1} because validation dataset has 0 patches")
            continue
            
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False, num_workers=num_workers)
        
        logger.info(f"Training dataset: {len(train_dataset)} patches")
        logger.info(f"Validation dataset: {len(val_dataset)} patches")
        
        # Define model
        num_features = train_dataset.num_output_features
        model = SmallUNet(in_channels=num_features, n_classes=3)  # 3 classes: NoData, Deciduous, Evergreen
        
        # Count parameters
        num_params = model.count_parameters()
        logger.info(f"Model has {num_params} trainable parameters")
        
        # Move model to device
        model.to(device)
        
        # Define loss function and optimizer
        # Use weighted cross-entropy to handle class imbalance
        # Get class counts from training dataset
        class_counts = np.zeros(3)  # NoData, Deciduous, Evergreen
        for _, label, _ in train_dataset:
            unique, counts = np.unique(label.numpy(), return_counts=True)
            for u, c in zip(unique, counts):
                if u < 3:
                    class_counts[u] += c
        
        # Calculate class weights (ignore NoData)
        if class_counts[1] > 0 and class_counts[2] > 0:
            # Calculate weights only for Deciduous (1) and Evergreen (2)
            total_valid_pixels = class_counts[1] + class_counts[2]
            weight_deciduous = total_valid_pixels / class_counts[1]
            weight_evergreen = total_valid_pixels / class_counts[2]
            # Create a tensor of size [2] for the two valid classes
            class_weights_list = [weight_deciduous, weight_evergreen]
            class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(device)
        else:
            logger.warning("No valid pixels found for calculating class weights. Using equal weights.")
            class_weights_tensor = torch.tensor([0.5, 0.5], dtype=torch.float32).to(device) # Default equal weights

        # Add debug print to check the shape
        logger.info(f"DEBUG: Calculated class_weights_tensor shape: {class_weights_tensor.shape}")

        # Use the custom loss function
        criterion = lambda outputs, targets, sample_weights: scaled_masked_loss(
            outputs, targets, weight=class_weights_tensor, sample_weights=sample_weights, weight_scale=weight_scale
        )
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,  # Peak learning rate after warmup
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # Spend 30% of training time warming up
            div_factor=25.0,  # Initial lr = max_lr/25
            final_div_factor=10000.0,  # Final lr = max_lr/10000 (corrected typo)
            anneal_strategy='cos'  # Use cosine annealing for both phases
        )
        
        # Training loop
        best_f1 = 0.0
        best_model_path = f"results/models/unet/phenology_{index}_fold{fold+1}.pt"
        
        logger.info("Starting training...")
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            
            for inputs, labels, sample_weights in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                sample_weights = sample_weights.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels, sample_weights)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            # Calculate epoch loss
            epoch_loss = running_loss / len(train_loader.dataset)
            
            # Validation phase
            val_metrics = evaluate_unet_model(model, val_loader, device, criterion)
            
            # Log results
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Loss: {epoch_loss:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val F1: {val_metrics['f1_score']:.4f}")
            
            # Write to TensorBoard
            tb_writer.add_scalar('Loss/train', epoch_loss, epoch)
            tb_writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            tb_writer.add_scalar('F1/val', val_metrics['f1_score'], epoch)
            tb_writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            
            # Add image visualizations to TensorBoard periodically
            if epoch % 5 == 0 or epoch == epochs - 1:  # Every 5 epochs and last epoch
                # Get a batch of validation data
                val_inputs, val_labels, _ = next(iter(val_loader))
                val_inputs = val_inputs.to(device)
                
                # Get predictions
                model.eval()
                with torch.no_grad():
                    val_outputs = model(val_inputs)
                    _, val_preds = torch.max(val_outputs, 1)
                model.train()
                
                # Prepare images for TensorBoard
                tb_images = prepare_unet_images_for_tensorboard(
                    batch_inputs=val_inputs, 
                    batch_labels=val_labels, 
                    batch_preds=val_preds, 
                    output_feature_names=train_dataset.output_feature_names,
                    tag_prefix=f"Fold_{fold+1}"
                )
                
                # Add images to TensorBoard
                for img_tag, img_tensor in tb_images.items():
                    tb_writer.add_image(img_tag, img_tensor, global_step=epoch)
            
            # Save best model
            if val_metrics['f1_score'] > best_f1:
                best_f1 = val_metrics['f1_score']
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
            
            # Update scheduler
            scheduler.step()
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(best_model_path))
        
        # Final evaluation
        final_metrics = evaluate_unet_model(model, val_loader, device)
        
        # Log confusion matrix
        cm_text = format_confusion_matrix(
            final_metrics['confusion_matrix'], 
            labels=[f'{PHENOLOGY_MAPPING[1]} (1)', f'{PHENOLOGY_MAPPING[2]} (2)']
        )
        logger.info("\n" + cm_text)
        
        # Store results
        fold_results = {
            'fold': fold + 1,
            'f1_score': final_metrics['f1_score'],
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'accuracy': final_metrics['accuracy']
        }
        results_per_fold.append(fold_results)
        
        # Visualize some predictions
        vis_output_dir = f"results/visualizations/unet/{index}" # Define output dir
        os.makedirs(vis_output_dir, exist_ok=True) # Ensure it exists
        visualize_unet_predictions(
            model=model, 
            dataloader=val_loader, 
            device=device, 
            output_feature_names=train_dataset.output_feature_names, # Pass correct feature names
            output_dir=vis_output_dir, # Pass output directory
            fold_tag=f"fold_{fold+1}", # Pass fold tag
            num_samples=5,
            logger=logger # Pass logger
        )
    
    # Aggregate results
    results_df = pd.DataFrame(results_per_fold)
    
    logger.info("\n=== Overall Results ===")
    logger.info(f"Average F1 Score: {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")
    logger.info(f"Average Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    logger.info(f"Average Recall: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    
    # Save results
    results_df.to_csv(f"results/models/unet/phenology_{index}_results.csv", index=False)
    
    # Close TensorBoard writer
    tb_writer.close()
    
    return results_df

def main():
    """Main function to train phenology U-Net models for all indices"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train small U-Net for phenology classification')
    parser.add_argument('--test', '-t', action='store_true', 
                        help='Run in test mode with fewer tiles')
    parser.add_argument('--test_tiles', type=int, default=10,
                        help='Number of tiles to use in test mode (default: 10)')
    parser.add_argument('--index', '-i', type=str, default=None,
                        help='Specific index to train (default: all indices)')
    parser.add_argument('--folds', '-f', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--epochs', '-e', type=int, default=EPOCHS,
                        help=f'Number of epochs to train (default: {EPOCHS})')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation (flips only)')
    parser.add_argument('--no_weights', action='store_true',
                        help='Disable eco-region based sample weights')
    parser.add_argument('--weight_scale', type=float, default=1.0,
                        help='Scaling factor for eco-region weights (1.0 = full effect, 0.0 = no effect)')
    args = parser.parse_args()
    
    logger.info("Starting phenology U-Net training script")
    
    # Get all tile paths
    tile_dir = 'data/training/training_tiles2023_w_corsica/training_tiles2023'
    tile_paths = glob.glob(f"{tile_dir}/tile_*_training.tif")
    logger.info(f"Found {len(tile_paths)} training tiles")
    
    # If in test mode, use only a subset of tiles
    if args.test:
        if len(tile_paths) > args.test_tiles:
            tile_paths = np.random.choice(tile_paths, args.test_tiles, replace=False)
        logger.info(f"TEST MODE: Using {len(tile_paths)} tiles")
    
    # Summary results across indices
    summary_results = []
    
    # Use specific index or process all indices
    indices_to_process = [args.index] if args.index else INDICES
    
    # Process each index
    for index in indices_to_process:
        logger.info(f"\n\n{'='*50}")
        logger.info(f"Training U-Net with {index.upper()} features")
        logger.info(f"{'='*50}")
        
        # Log whether eco-region weights are enabled
        use_weights = not args.no_weights
        logger.info(f"Eco-region based sample weights: {'ENABLED' if use_weights else 'DISABLED'}")
        
        # Train and evaluate
        start_time = time.time()
        results_df = train_and_evaluate_unet(
            index, 
            tile_paths, 
            n_splits=args.folds, 
            epochs=args.epochs,
            use_augmentation=args.augment,
            use_sample_weights=use_weights,
            weight_scale=args.weight_scale
        )
        elapsed_time = time.time() - start_time
        
        logger.info(f"\nTraining completed in {elapsed_time:.2f} seconds")
        
        # Add to summary
        summary_results.append({
            'index': index,
            'f1_score': results_df['f1_score'].mean(),
            'f1_std': results_df['f1_score'].std(),
            'precision': results_df['precision'].mean(),
            'recall': results_df['recall'].mean(),
            'training_time': elapsed_time
        })
    
    # Create summary comparison if we processed multiple indices
    if len(indices_to_process) > 1:
        logger.info("Creating summary comparison report...")
        summary_df = pd.DataFrame(summary_results).sort_values('f1_score', ascending=False)
        summary_df.to_csv("results/models/unet/phenology_unet_indices_comparison.csv", index=False)
        
        logger.info("\n=== Indices Comparison ===")
        logger.info("\n" + summary_df[['index', 'f1_score', 'f1_std', 'precision', 'recall', 'training_time']].round(4).to_string())
        
        # Plot comparison of F1 scores
        plt.figure(figsize=(10, 6))
        plt.bar(summary_df['index'], summary_df['f1_score'], yerr=summary_df['f1_std'])
        plt.xlabel('Vegetation Index')
        plt.ylabel('F1 Score')
        plt.title('U-Net Performance Comparison Across Indices')
        plt.ylim(0, 1)
        plt.savefig("results/models/unet/phenology_unet_indices_comparison.png")
        plt.close()
    
    logger.info("U-Net training script completed successfully")

if __name__ == "__main__":
    main() 