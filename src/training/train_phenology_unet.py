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

# Define the phenology mapping
PHENOLOGY_MAPPING = {0: 'No Data', 1: 'Deciduous', 2: 'Evergreen'}

# Constants
PATCH_SIZE = 64  # Size of spatial patches to extract
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 30

class SmallUNet(nn.Module):
    """
    Small U-Net architecture for phenology classification
    """
    def __init__(self, in_channels, n_classes=3):
        super(SmallUNet, self).__init__()
        
        # Reduce number of filters to keep model small
        filters = [16, 32, 64]
        
        # Define encoder path
        self.enc1 = self._conv_block(in_channels, filters[0])
        self.enc2 = self._conv_block(filters[0], filters[1])
        self.enc3 = self._conv_block(filters[1], filters[2])
        
        # Bottleneck
        self.bottleneck = self._conv_block(filters[2], filters[2])
        
        # Define decoder path
        self.dec3 = self._conv_block(filters[2] * 2, filters[1])
        self.dec2 = self._conv_block(filters[1] * 2, filters[0])
        self.dec1 = self._conv_block(filters[0] * 2, filters[0])
        
        # Final layer
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Initialize weights
        self._initialize_weights()
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming initialization for convolutions
        and proper initialization for batch normalization layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # He initialization suitable for ReLU
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.BatchNorm2d):
                # Standard initialization for batch norm
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
        # Special handling for final classification layer
        # Initialize with lower variance for better initial stability
        nn.init.normal_(self.final.weight, mean=0, std=0.01)
        nn.init.constant_(self.final.bias, 0)
    
    def forward(self, x):
        # Encoder path with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))
        
        # Decoder path with skip connections
        dec3 = self.dec3(torch.cat([self.upsample(bottleneck), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upsample(dec2), enc1], dim=1))
        
        # Final classification layer
        return self.final(dec1)
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpatialTransforms:
    """Performs conservative data augmentation suitable for satellite imagery"""
    def __init__(self, p_flip=0.5):
        self.p_flip = p_flip
    
    def __call__(self, features, label, weight=None):
        # Only horizontal/vertical flips, no rotations
        # Forest patterns have no inherent directionality from above
        
        # Random horizontal flip
        if np.random.random() < self.p_flip:
            features = torch.flip(features, dims=[-1])
            label = torch.flip(label, dims=[-1])
        
        # Random vertical flip
        if np.random.random() < self.p_flip:
            features = torch.flip(features, dims=[-2])
            label = torch.flip(label, dims=[-2])
            
        if weight is not None:
            return features, label, weight
        else:
            return features, label


class PhenologyTileDataset(Dataset):
    """Dataset for loading and processing phenology TIFF tiles with global normalization"""
    
    def __init__(self, tile_paths, index='ndvi', transform=None, patch_size=64, random_state=42, 
                 global_stats=None, compute_global_stats=False, use_sample_weights=True):
        self.tile_paths = tile_paths
        self.index = index
        self.transform = transform
        self.patch_size = patch_size
        self.random_state = random_state
        self.global_stats = global_stats  # Dictionary with precomputed global statistics
        self.use_sample_weights = use_sample_weights
        
        # Try to load eco-region weights if enabled
        self.eco_weights = None
        self.tile_to_eco = None
        self.pixel_weights = {}
        
        if use_sample_weights:
            # Try to load the tile to eco-region mapping
            self.tile_to_eco = get_tile_eco_region_mapping()
            # Load the weights from the parquet file
            self._load_eco_region_weights()
        
        # Find feature band indices for the given index
        # Based on the GDALInfo results
        # Adjust indices to be 1-indexed for rasterio
        index_mappings = {
            'ndvi': [1, 2, 3, 4, 5, 6],  # amplitude_h1, amplitude_h2, phase_h1, phase_h2, offset, var_residual
            'evi': [7, 8, 9, 10, 11, 12],
            'nbr': [13, 14, 15, 16, 17, 18],
            'crswir': [19, 20, 21, 22, 23, 24]
        }
        
        # Verify index is valid
        if index not in index_mappings:
            raise ValueError(f"Index {index} not found. Available indices: {list(index_mappings.keys())}")
        
        self.feature_bands = index_mappings[index]
        self.phenology_band = 25  # phenology is the 25th band (1-indexed)
        
        # Compute global statistics if requested and not provided
        if compute_global_stats and global_stats is None:
            logger.info(f"Computing global statistics for {index} index features...")
            self.global_stats = self._compute_global_statistics()
            logger.info(f"Global statistics computed: {self.global_stats}")
        
        # Extract patches and cache them
        logger.info(f"Preparing dataset for {index} index")
        self.patches = self._extract_patches()
        logger.info(f"Dataset prepared with {len(self.patches)} patches")
    
    def _load_eco_region_weights(self):
        """Load eco-region weights from the weighted parquet file"""
        try:
            weights_file = 'results/datasets/training_datasets_pixels.parquet'
            logger.info(f"Loading eco-region weights from {weights_file}")
            
            if os.path.exists(weights_file):
                # Load the parquet file with weights
                weights_df = pd.read_parquet(weights_file)
                logger.info(f"Loaded weights for {len(weights_df)} pixels")
                
                # Check if 'weight' column exists
                if 'weight' in weights_df.columns and 'eco_region' in weights_df.columns:
                    # Create mapping from eco-region to weight
                    self.eco_weights = {}
                    for eco, group in weights_df.groupby('eco_region'):
                        self.eco_weights[eco] = group['weight'].iloc[0]
                    
                    logger.info(f"Loaded weights for {len(self.eco_weights)} eco-regions")
                    
                    # Log the weights
                    logger.info("Eco-region weights:")
                    for eco, weight in self.eco_weights.items():
                        logger.info(f"  {eco}: {weight:.4f}")
                    
                    logger.info("Eco-region based sample weighting is ENABLED for training")
                else:
                    logger.warning("Weight or eco_region column not found in weights file")
            else:
                logger.warning(f"Weights file {weights_file} not found, proceeding without eco-region weights")
        except Exception as e:
            logger.error(f"Error loading eco-region weights: {e}")
            logger.warning("Proceeding without eco-region weights")

    def _compute_global_statistics(self):
        """Compute global statistics for feature normalization across all tiles"""
        # Define which features need min/max or percentile statistics
        amplitude_indices = [0, 1, 4, 5]  # 0-indexed positions in feature array
        
        # Initialize arrays to collect samples for statistics calculation
        feature_samples = {idx: [] for idx in amplitude_indices}
        
        # Sample data from tiles to compute statistics
        # We don't need to process every pixel, just enough for stable statistics
        rng = np.random.RandomState(self.random_state)
        max_samples_per_tile = 1000
        
        for tile_path in tqdm(self.tile_paths, desc="Computing global statistics"):
            try:
                with rasterio.open(tile_path) as src:
                    # Read feature bands
                    feature_data = src.read(self.feature_bands)
                    
                    # Get valid mask (where data is finite)
                    valid_mask = np.all(np.isfinite(feature_data), axis=0)
                    valid_indices = np.where(valid_mask)
                    
                    if len(valid_indices[0]) > 0:
                        # Randomly sample pixels from this tile
                        num_samples = min(max_samples_per_tile, len(valid_indices[0]))
                        sample_indices = rng.choice(len(valid_indices[0]), num_samples, replace=False)
                        
                        for feat_idx in amplitude_indices:
                            # Collect samples for this feature
                            samples = feature_data[feat_idx, 
                                                  valid_indices[0][sample_indices], 
                                                  valid_indices[1][sample_indices]]
                            feature_samples[feat_idx].extend(samples.flatten().tolist())
            except Exception as e:
                logger.warning(f"Error sampling tile {tile_path} for statistics: {e}")
        
        # Compute statistics from collected samples
        stats = {}
        for feat_idx, samples in feature_samples.items():
            if len(samples) > 0:
                # Convert to numpy array for calculation
                samples_array = np.array(samples)
                
                # Calculate percentiles for robust normalization
                q_low = np.percentile(samples_array, 2)
                q_high = np.percentile(samples_array, 98)
                
                # Use string keys for JSON compatibility
                stats[str(feat_idx)] = {
                    'q_low': float(q_low),
                    'q_high': float(q_high)
                }
            else:
                # Fallback if no valid samples found
                stats[str(feat_idx)] = {
                    'q_low': 0.0,
                    'q_high': 1.0
                }
        
        return stats
    
    def _extract_patches(self):
        """Extract patches from tiles using consistent global normalization"""
        all_patches = []
        rng = np.random.RandomState(self.random_state)
        
        for tile_path in tqdm(self.tile_paths, desc="Extracting patches"):
            try:
                with rasterio.open(tile_path) as src:
                    # Read feature bands and phenology band
                    feature_data = src.read(self.feature_bands)
                    phenology_data = src.read(self.phenology_band)
                    
                    # Get tile dimensions
                    _, height, width = feature_data.shape
                    
                    # Get tile id from path for eco-region lookup
                    tile_id = os.path.basename(tile_path).split('_')[1]
                    
                    # Get the eco-region weight if available
                    eco_weight = 1.0
                    if self.use_sample_weights and self.tile_to_eco and self.eco_weights:
                        eco_region = self.tile_to_eco.get(tile_id, None)
                        if eco_region and eco_region in self.eco_weights:
                            eco_weight = self.eco_weights[eco_region]
                    
                    # Ensure patches will be fully within the tile
                    # We need to restrict the valid center locations to account for patch radius
                    half_size = self.patch_size // 2
                    
                    # For a patch to be fully within the tile, its center must be at least half_size
                    # away from any edge
                    valid_y_min = half_size
                    valid_y_max = height - half_size
                    valid_x_min = half_size
                    valid_x_max = width - half_size
                    
                    # Create mask of valid pixel positions (both in bounds and with valid phenology)
                    valid_mask = np.zeros((height, width), dtype=bool)
                    valid_mask[valid_y_min:valid_y_max, valid_x_min:valid_x_max] = True
                    
                    # Find coordinates with valid phenology
                    valid_phenology = phenology_data > 0
                    
                    # Combine with boundary constraints
                    valid_locations = np.logical_and(valid_mask, valid_phenology)
                    valid_yx = np.column_stack(np.where(valid_locations))
                    
                    # Also sample random locations within valid bounds (regardless of phenology)
                    num_random_patches = 50
                    random_y = rng.randint(valid_y_min, valid_y_max, num_random_patches)
                    random_x = rng.randint(valid_x_min, valid_x_max, num_random_patches)
                    random_yx = np.column_stack((random_y, random_x))
                    
                    # Combine and shuffle coordinates
                    all_yx = np.vstack((valid_yx, random_yx)) if len(valid_yx) > 0 else random_yx
                    all_yx = shuffle(all_yx, random_state=self.random_state)
                    
                    # Extract patches centered on selected pixels
                    max_patches_per_tile = 100
                    patch_count = 0
                    
                    for y, x in all_yx:
                        # Double-check that the patch is fully within bounds
                        # This is redundant but ensures safety
                        if (y >= half_size and 
                            y + half_size <= height and
                            x >= half_size and 
                            x + half_size <= width):
                            
                            # Extract feature patch
                            feature_patch = feature_data[:, 
                                                         y-half_size:y+half_size, 
                                                         x-half_size:x+half_size]
                            
                            # Extract label patch
                            label_patch = phenology_data[y-half_size:y+half_size, 
                                                         x-half_size:x+half_size]
                            
                            # Check if the patch contains at least some valid reference data
                            if np.any(label_patch > 0) and np.all(np.isfinite(feature_patch)):
                                # Normalize features
                                normalized_features = np.zeros_like(feature_patch, dtype=np.float32)
                                
                                # Handle amplitude features with global statistics
                                amplitude_indices = [0, 1, 4, 5]  # Positions in the feature array (0-indexed)
                                for feat_idx in amplitude_indices:
                                    if feat_idx < len(normalized_features):
                                        if self.global_stats and str(feat_idx) in self.global_stats:
                                            # Use global statistics for normalization
                                            q_low = self.global_stats[str(feat_idx)]['q_low']
                                            q_high = self.global_stats[str(feat_idx)]['q_high']
                                        else:
                                            # Fallback to per-tile normalization if no global stats
                                            q_low = np.percentile(feature_patch[feat_idx], 2)
                                            q_high = np.percentile(feature_patch[feat_idx], 98)
                                        
                                        if q_high > q_low:
                                            normalized_features[feat_idx] = np.clip(
                                                (feature_patch[feat_idx] - q_low) / (q_high - q_low),
                                                0, 1
                                            )
                                
                                # Handle phase features with sin/cos transformation
                                phase_indices = [2, 3]  # Positions in the feature array (0-indexed)
                                for i, feat_idx in enumerate(phase_indices):
                                    if feat_idx < len(normalized_features):
                                        # Convert to radians and apply sin/cos
                                        phase_rad = feature_patch[feat_idx] * (2 * np.pi / 65535)  # Assuming phase stored as 0-65535
                                        
                                        # We're replacing a single phase band with sin/cos components
                                        normalized_features[feat_idx] = np.cos(phase_rad)
                                        if feat_idx+1 < len(normalized_features):
                                            normalized_features[feat_idx+1] = np.sin(phase_rad)
                                
                                all_patches.append({
                                    'features': normalized_features,
                                    'label': label_patch,
                                    'tile_id': tile_id,
                                    'weight': eco_weight  # Add the eco-region weight
                                })
                                
                                patch_count += 1
                                if patch_count >= max_patches_per_tile:
                                    break
            
            except Exception as e:
                logger.error(f"Error processing tile {tile_path}: {e}")
        
        logger.info(f"Extracted a total of {len(all_patches)} patches from {len(self.tile_paths)} tiles")
        return all_patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        
        # Convert to torch tensors
        features = torch.from_numpy(patch['features']).float()
        
        # Convert label to int64 (supported by PyTorch) before creating tensor
        # The error was due to uint32 type not being supported by PyTorch
        label_np = patch['label'].astype(np.int64)
        label = torch.from_numpy(label_np).long()
        
        # Get the sample weight
        weight = torch.tensor(patch['weight'], dtype=torch.float32)
        
        # Apply transformations if any
        if self.transform:
            features, label, weight = self.transform(features, label, weight)
        
        return features, label, weight

def get_tile_eco_region_mapping(mapping_file='results/datasets/training_datasets_pixels.parquet'):
    """
    Load mapping of tiles to eco-regions from parquet file.
    
    Returns a dictionary mapping tile IDs to eco-region names.
    """
    try:
        # Load the parquet file
        logger.info(f"Loading tile to eco-region mapping from {mapping_file}")
        df = pd.read_parquet(mapping_file)
        logger.info(f"Mapping file loaded with {len(df)} entries")
        
        # Verify that required columns exist
        if 'tile_id' not in df.columns:
            raise ValueError("'tile_id' column not found in mapping file")
        
        if 'eco_region' not in df.columns:
            raise ValueError("'eco_region' column not found in mapping file")
        
        # Create dictionary mapping tile IDs to eco-regions
        # Group by tile_id and take the first eco_region for each tile
        tile_to_eco = {}
        
        # Convert IDs to strings to ensure matching
        df['tile_id'] = df['tile_id'].astype(str)
        
        # Get unique tile-eco_region combinations
        tile_eco_df = df[['tile_id', 'eco_region']].drop_duplicates()
        
        # Create mapping dictionary
        for _, row in tile_eco_df.iterrows():
            tile_id = row['tile_id']
            eco_region = row['eco_region']
            tile_to_eco[tile_id] = eco_region
        
        # Log unique eco-regions in the mapping file
        eco_regions = df['eco_region'].unique()
        logger.info(f"Found {len(eco_regions)} unique eco-regions")
        
        # Report the results
        eco_counts = {}
        for eco in tile_to_eco.values():
            eco_counts[eco] = eco_counts.get(eco, 0) + 1
            
        logger.info(f"Created mapping for {len(tile_to_eco)} tiles across {len(eco_counts)} eco-regions")
        for eco, count in sorted(eco_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {eco}: {count} tiles")
            
        return tile_to_eco
    
    except Exception as e:
        logger.error(f"Error loading eco-region mapping: {e}")
        raise

def create_eco_balanced_folds(tiles, n_splits=5, random_state=42):
    """
    Create folds that balance eco-region distribution for TIFF tiles.
    Ensures that each eco-region is properly represented in training and validation.
    
    Args:
        tiles: List of tile paths
        n_splits: Number of cross-validation folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of (train_tiles, val_tiles) tuples for each fold
    """
    logger.info("Creating eco-region balanced folds...")
    
    # Get the tile to eco-region mapping
    tile_to_eco = get_tile_eco_region_mapping()
    
    # Group tiles by eco-region
    eco_to_tiles = defaultdict(list)
    for tile_path in tiles:
        tile_id = os.path.basename(tile_path).split('_')[1]
        eco_region = tile_to_eco.get(tile_id, "Unknown")
        eco_to_tiles[eco_region].append(tile_path)
    
    # Log the distribution of tiles by eco-region
    logger.info("Distribution of tiles by eco-region:")
    for eco, eco_tiles in eco_to_tiles.items():
        logger.info(f"  {eco}: {len(eco_tiles)} tiles")
    
    # Initialize fold assignments
    fold_splits = [{"train": [], "val": []} for _ in range(n_splits)]
    
    # For each eco-region, distribute tiles evenly across folds
    for eco_region, eco_tiles in eco_to_tiles.items():
        # Shuffle tiles within each eco-region
        eco_tiles = shuffle(eco_tiles, random_state=random_state)
        
        # Create stratified folds for this eco-region
        eco_folds = np.array_split(eco_tiles, n_splits)
        
        # Assign to validation for each fold, rest to training
        for fold_idx in range(n_splits):
            val_tiles = eco_folds[fold_idx]
            
            # Training tiles are all other folds for this eco-region
            train_tiles = []
            for i in range(n_splits):
                if i != fold_idx:
                    train_tiles.extend(eco_folds[i])
            
            # Add to the fold assignments
            fold_splits[fold_idx]["train"].extend(train_tiles)
            fold_splits[fold_idx]["val"].extend(val_tiles)
    
    # Convert to final format
    final_fold_splits = []
    for fold in fold_splits:
        final_fold_splits.append((fold["train"], fold["val"]))
    
    # Log fold information
    for fold_idx, (train_tiles, val_tiles) in enumerate(final_fold_splits):
        logger.info(f"Fold {fold_idx + 1}/{n_splits}: {len(train_tiles)} training tiles, {len(val_tiles)} validation tiles")
        
        # Also log eco-region distribution in each fold
        train_eco_counts = {}
        for tile_path in train_tiles:
            tile_id = os.path.basename(tile_path).split('_')[1]
            eco = tile_to_eco.get(tile_id, "Unknown")
            train_eco_counts[eco] = train_eco_counts.get(eco, 0) + 1
            
        val_eco_counts = {}
        for tile_path in val_tiles:
            tile_id = os.path.basename(tile_path).split('_')[1]
            eco = tile_to_eco.get(tile_id, "Unknown")
            val_eco_counts[eco] = val_eco_counts.get(eco, 0) + 1
        
        logger.info(f"  Training eco-region distribution:")
        for eco, count in train_eco_counts.items():
            logger.info(f"    {eco}: {count} tiles ({count/len(train_tiles)*100:.1f}%)")
            
        logger.info(f"  Validation eco-region distribution:")
        for eco, count in val_eco_counts.items():
            logger.info(f"    {eco}: {count} tiles ({count/len(val_tiles)*100:.1f}%)")
    
    return final_fold_splits

def scaled_masked_loss(outputs, targets, weight=None, sample_weights=None, weight_scale=1.0):
    """
    Custom loss function that applies scaled eco-region weights
    """
    # Create mask for valid pixels (non-zero)
    valid_mask = (targets > 0)
    
    # If no valid pixels, return zero loss
    if not torch.any(valid_mask):
        return torch.tensor(0.0, device=outputs.device, requires_grad=True, dtype=torch.float32)
    
    # Extract valid predictions and targets
    outputs_masked = outputs.permute(0, 2, 3, 1)[valid_mask]  # Shape: (N, C)
    targets_masked = targets[valid_mask]  # Shape: (N,)
    
    # Apply per-pixel loss
    if weight is not None:
        # Apply weighted cross-entropy loss for class weights
        pixel_losses = F.cross_entropy(outputs_masked, targets_masked, weight=weight, reduction='none')
    else:
        pixel_losses = F.cross_entropy(outputs_masked, targets_masked, reduction='none')
    
    # Apply sample weights if provided and enabled
    if sample_weights is not None and weight_scale > 0.0:
        # Create a mask mapping from pixel position back to its batch index
        batch_indices = torch.zeros_like(targets, dtype=torch.long)
        for i in range(targets.shape[0]):
            batch_indices[i, :, :] = i
        
        # Extract batch indices for valid pixels
        batch_indices_masked = batch_indices[valid_mask]
        
        # Map each pixel loss to its corresponding sample weight, with scaling
        # Scale the weights: w_scaled = 1 + (w_original - 1) * scale_factor
        # This makes weights approach 1.0 (no effect) as scale_factor approaches 0
        pixel_weights = 1.0 + (sample_weights[batch_indices_masked] - 1.0) * weight_scale
        
        # Apply weights and compute mean
        weighted_loss = (pixel_losses * pixel_weights).sum() / pixel_weights.sum().clamp(min=1e-8)
    else:
        # Regular mean if no sample weights or weight_scale is 0
        weighted_loss = pixel_losses.mean()
    
    return weighted_loss

def evaluate_model(model, dataloader, device, criterion=None):
    """Evaluate model performance on dataloader"""
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []
    
    # Check if dataset is empty and return default metrics if so
    if len(dataloader.dataset) == 0:
        return {
            'loss': 0.0,
            'accuracy': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'confusion_matrix': np.zeros((3, 3))  # 3x3 matrix for NoData, Deciduous, Evergreen
        }
    
    with torch.no_grad():
        for inputs, labels, sample_weights in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            sample_weights = sample_weights.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss if criterion provided
            if criterion:
                loss = criterion(outputs, labels, sample_weights)
                running_loss += loss.item() * inputs.size(0)
            
            # For metrics
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store true labels and predictions
            y_true.extend(labels.flatten().cpu().numpy())
            y_pred.extend(preds.flatten().cpu().numpy())
    
    # Calculate metrics
    # Filter out NoData (0) pixels
    valid_indices = [i for i, label in enumerate(y_true) if label > 0]
    y_true_filtered = [y_true[i] for i in valid_indices]
    y_pred_filtered = [y_pred[i] for i in valid_indices]
    
    metrics = {
        'loss': running_loss / len(dataloader.dataset) if criterion else 0,
        'accuracy': np.mean([y_t == y_p for y_t, y_p in zip(y_true_filtered, y_pred_filtered)]),
        'f1_score': f1_score(y_true_filtered, y_pred_filtered, average='weighted'),
        'precision': precision_score(y_true_filtered, y_pred_filtered, average='weighted'),
        'recall': recall_score(y_true_filtered, y_pred_filtered, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true_filtered, y_pred_filtered)
    }
    
    return metrics

def format_confusion_matrix(cm, labels=None):
    """Format confusion matrix for logging"""
    if labels is None:
        labels = [f"{PHENOLOGY_MAPPING[i]} ({i})" for i in range(1, 3)]  # Exclude NoData
    
    # Assume binary classification (1: Deciduous, 2: Evergreen)
    try:
        tn, fp, fn, tp = cm.ravel()
        
        cm_text = f"""
Confusion Matrix:
---------------------------
              |  Predicted
    Actual    | {labels[0]:^10} | {labels[1]:^10}
---------------------------
{labels[0]:^12} | {tn:^10} | {fp:^10}
{labels[1]:^12} | {fn:^10} | {tp:^10}
---------------------------
"""
    except ValueError:
        # If we can't unpack into tn,fp,fn,tp, just return the raw matrix
        cm_text = f"Confusion Matrix:\n{cm}"
    
    return cm_text

def visualize_predictions(model, dataloader, device, index_name, num_samples=5):
    """Visualize model predictions for a few examples"""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            if len(samples) >= num_samples:
                break
                
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Store samples
            for i in range(min(inputs.size(0), num_samples - len(samples))):
                samples.append({
                    'input': inputs[i].cpu().numpy(),
                    'true': labels[i].cpu().numpy(),
                    'pred': preds[i].cpu().numpy()
                })
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i, sample in enumerate(samples):
        # Show input features (first channel as example)
        axes[i, 0].imshow(sample['input'][0], cmap='viridis')
        axes[i, 0].set_title(f'Input ({index_name}_amplitude_h1)')
        
        # Show true labels
        axes[i, 1].imshow(sample['true'], cmap='tab10', vmin=0, vmax=2)
        axes[i, 1].set_title('True Labels')
        
        # Show predicted labels
        axes[i, 2].imshow(sample['pred'], cmap='tab10', vmin=0, vmax=2)
        axes[i, 2].set_title('Predicted Labels')
    
    plt.tight_layout()
    plt.savefig(f"results/models/unet/phenology_{index_name}_predictions.png")
    plt.close()

def prepare_images_for_tensorboard(batch_inputs, batch_labels, batch_preds, phenology_mapping):
    """
    Prepare input features, true labels, and predictions for TensorBoard visualization
    
    Args:
        batch_inputs: Input tensor with shape [B, C, H, W]
        batch_labels: Label tensor with shape [B, H, W]
        batch_preds: Prediction tensor with shape [B, H, W]
        phenology_mapping: Dictionary mapping label indices to class names
        
    Returns:
        Dict of image tensors ready for TensorBoard visualization
    """
    # Convert to numpy for easier manipulation
    inputs_np = batch_inputs.cpu().numpy()
    labels_np = batch_labels.cpu().numpy()
    preds_np = batch_preds.cpu().numpy()
    
    batch_size = inputs_np.shape[0]
    
    # Create a color map for visualization
    # 0: No Data (black), 1: Deciduous (green), 2: Evergreen (blue)
    cmap = np.array([
        [0, 0, 0],        # Black for No Data
        [0, 255, 0],      # Green for Deciduous
        [0, 0, 255]       # Blue for Evergreen
    ], dtype=np.uint8)
    
    # Prepare visualization dictionary
    vis_images = {}
    
    # Process each sample in the batch
    for i in range(min(batch_size, 8)):  # Limit to 8 samples to avoid too many images
        # Visualize amplitude_h1 (first channel)
        amp_h1 = inputs_np[i, 0]
        # Normalize to [0, 1]
        amp_h1_norm = (amp_h1 - amp_h1.min()) / (amp_h1.max() - amp_h1.min() + 1e-8)
        # Convert to RGB with viridis colormap
        amp_h1_rgb = plt.cm.viridis(amp_h1_norm)[:, :, :3]  # Remove alpha channel
        amp_h1_rgb = (amp_h1_rgb * 255).astype(np.uint8)
        vis_images[f'input/amplitude_h1_{i}'] = torch.from_numpy(amp_h1_rgb).permute(2, 0, 1)
        
        # Visualize ground truth
        label_rgb = cmap[labels_np[i]]
        vis_images[f'ground_truth/{i}'] = torch.from_numpy(label_rgb).permute(2, 0, 1)
        
        # Visualize prediction
        pred_rgb = cmap[preds_np[i]]
        vis_images[f'prediction/{i}'] = torch.from_numpy(pred_rgb).permute(2, 0, 1)
    
    return vis_images

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
        use_sample_weights=False  # Don't compute weights when just getting stats
    )
    global_stats = dummy_dataset.global_stats
    
    # Save global statistics to disk for later use in inference
    stats_path = f"results/models/unet/phenology_{index}_normalization_stats.json"
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    
    with open(stats_path, 'w') as f:
        json.dump(global_stats, f, indent=2)
    
    logger.info(f"Saved normalization statistics to {stats_path}")
    
    # Create folds for cross-validation
    fold_splits = create_eco_balanced_folds(tile_paths, n_splits=n_splits)
    
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
            use_sample_weights=use_sample_weights
        )
        
        val_dataset = PhenologyTileDataset(
            val_tiles, 
            index=index, 
            patch_size=patch_size,
            global_stats=global_stats,  # No transforms for validation
            use_sample_weights=use_sample_weights
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
        num_features = len(train_dataset.feature_bands)
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
            class_weights = torch.tensor(
                [0.0, class_counts.sum() / class_counts[1], class_counts.sum() / class_counts[2]],
                device=device, dtype=torch.float32  # Explicitly use float32 instead of default float64
            )
        else:
            class_weights = torch.tensor([0.0, 1.0, 1.0], device=device, dtype=torch.float32)
        
        # Use the custom loss function
        criterion = lambda outputs, targets, sample_weights: scaled_masked_loss(
            outputs, targets, weight=class_weights, sample_weights=sample_weights, weight_scale=weight_scale
        )
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,  # Peak learning rate after warmup
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # Spend 30% of training time warming up
            div_factor=25.0,  # Initial lr = max_lr/25
            final_div_factor=1000.0,  # Final lr = max_lr/25000
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
            val_metrics = evaluate_model(model, val_loader, device, criterion)
            
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
                tb_images = prepare_images_for_tensorboard(
                    val_inputs, 
                    val_labels, 
                    val_preds, 
                    PHENOLOGY_MAPPING
                )
                
                # Add images to TensorBoard
                for img_name, img_tensor in tb_images.items():
                    tb_writer.add_image(img_name, img_tensor, epoch)
            
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
        final_metrics = evaluate_model(model, val_loader, device)
        
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
        visualize_predictions(model, val_loader, device, index, num_samples=5)
    
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