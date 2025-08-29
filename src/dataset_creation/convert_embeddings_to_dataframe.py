#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Embedding Training Tiles to DataFrame

Reads embedding-based training tiles (GeoTIFF) produced by
src/sampling/dataset_preparation.py when given an embeddings VRT.

Assumptions:
- Each file is named like: data/training/embeddings/tile_###_training.tif
- Bands 1..64: embedding_0..embedding_63
- Bands 65..69: phenology, genus, species, source, year (uint32)

Outputs a parquet with:
- Columns: tile_id, row, col, embedding_0..embedding_63, phenology, eco_region
- Optionally includes genus, species, source, year if --include-meta is set
- Adds eco_region by merging tile_id with final_dataset.parquet (NomSER -> eco_region)

Usage:
  python src/dataset_creation/convert_embeddings_to_dataframe.py \
    --input_dir data/training/embeddings \
    --final_dataset results/datasets/final_dataset.parquet \
    --output_path results/datasets/training_datasets_pixels_embedding.parquet
"""

import os
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from typing import Optional
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
# Note: simple merge-based matching is used by default per user request.
# PyArrow imports previously used for per-tile matching were removed to keep it simple.


def setup_logging(level: str = "INFO") -> str:
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/convert_embeddings_to_dataframe_{ts}.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return log_file


def extract_tile_id_from_name(name: str) -> Optional[int]:
    # Expecting tile_###_training.tif
    m = re.search(r'tile_(\d+)_training', name)
    if m:
        return int(m.group(1))
    return None


def load_tile_to_ecoregion_map(final_dataset_path: str) -> dict:
    """Build tile_id -> eco_region mapping from final_dataset.parquet (NomSER)."""
    logging.info(f"Loading eco-region mapping from {final_dataset_path}")
    df = pd.read_parquet(final_dataset_path)
    if 'tile_id' not in df.columns:
        raise ValueError("final_dataset.parquet missing 'tile_id'")
    if 'NomSER' not in df.columns:
        raise ValueError("final_dataset.parquet missing 'NomSER' (eco-region)")
    # Most common NomSER per tile
    mapping = (df[['tile_id', 'NomSER']]
                .groupby('tile_id')['NomSER']
                .agg(lambda s: s.value_counts().index[0]))
    mapping = mapping.rename('eco_region').to_dict()
    logging.info(f"Eco-region mapping: {len(mapping)} tiles")
    return mapping


def main():
    ap = argparse.ArgumentParser(description="Convert embedding tiles to parquet for training.")
    ap.add_argument('--input_dir', default='data/training/embeddings', help='Folder with tile_###_training.tif')
    ap.add_argument('--final_dataset', default='results/datasets/final_dataset.parquet', help='Path to final_dataset.parquet for eco-region mapping (NomSER)')
    ap.add_argument('--output_path', default='results/datasets/training_datasets_pixels_embedding.parquet', help='Output parquet path')
    ap.add_argument('--match_baseline', action='store_true', help='Inner-join with baseline parquet to keep exactly matching pixels and import weight')
    ap.add_argument('--baseline_path', default='results/datasets/training_datasets_pixels.parquet', help='Path to baseline parquet for matching and weights')
    ap.add_argument('--sanitize', action='store_true', help='Clip embedding_* to [-1,1] and replace non-finite with 0.0 before sampling (note: defeats non-finite dropping)')
    ap.add_argument('--no-drop-nonfinite', dest='drop_nonfinite', action='store_false', help='Do not drop rows with any non-finite embedding values')
    ap.set_defaults(drop_nonfinite=True)
    ap.add_argument('--chunked', action='store_true', help='Stream merge per-tile (memory-friendly)')
    ap.add_argument('--include-meta', action='store_true', help='Include genus, species, source, year in output')
    ap.add_argument('--limit', type=int, default=None, help='Limit number of tiles (debug)')
    ap.add_argument('--loglevel', default='INFO')
    args = ap.parse_args()

    log_file = setup_logging(args.loglevel)
    logging.info("=== Convert Embeddings to DataFrame ===")
    logging.info(f"Input dir: {args.input_dir}")
    logging.info(f"Final dataset: {args.final_dataset}")
    logging.info(f"Output: {args.output_path}")
    if args.match_baseline:
        logging.info(f"Match baseline: True (baseline_path={args.baseline_path})")

    # Build tile_id -> eco_region mapping
    tile_to_eco = load_tile_to_ecoregion_map(args.final_dataset)

    # Collect tiles
    inp = Path(args.input_dir)
    files = sorted(inp.glob('tile_*_training.tif'))
    if args.limit:
        files = files[:args.limit]
    logging.info(f"Found {len(files)} training tiles")
    if not files:
        raise SystemExit("No training tiles found. Ensure dataset_preparation was run with embeddings.")

    rows = []
    baseline_ds = ds.dataset(args.baseline_path, format='parquet') if args.match_baseline and args.chunked else None
    writer = None
    before_total = 0  # before merge with baseline (when chunked) or final concat (when not)
    after_total = 0
    extracted_pixels_total = 0  # before non-finite filtering
    kept_pixels_total = 0       # after non-finite filtering
    for f in tqdm(files, desc='Tiles'):
        tile_id = extract_tile_id_from_name(f.name)
        if tile_id is None:
            logging.warning(f"Skip {f.name} (no tile_id)")
            continue
        with rasterio.open(f) as src:
            cnt = src.count
            if cnt < 69:
                logging.warning(f"{f.name}: expected >=69 bands (64 emb + 5 meta), got {cnt}")
            data = src.read()  # (bands, H, W)
            H, W = data.shape[1], data.shape[2]

            # Assume band layout: 0..63 -> embeddings, 64..68 -> phenology,genus,species,source,year
            emb = data[0:64].astype(np.float32)  # (64,H,W)
            if args.sanitize:
                # Sanitize embeddings: clip to [-1,1], replace non-finite with 0.0
                np.clip(emb, -1.0, 1.0, out=emb)
                emb[~np.isfinite(emb)] = 0.0
            phenology = data[64] if cnt >= 65 else None
            genus = data[65] if (args.include_meta and cnt >= 66) else None
            species = data[66] if (args.include_meta and cnt >= 67) else None
            source = data[67] if (args.include_meta and cnt >= 68) else None
            year = data[68] if (args.include_meta and cnt >= 69) else None

            if phenology is None:
                logging.warning(f"{f.name}: phenology band missing; skipping")
                continue

            # valid pixels where phenology != 0
            yy, xx = np.where(phenology != 0)
            if len(yy) == 0:
                continue

            # Check non-finite across all 64 embeddings at the selected pixels
            emb_pix = emb[:, yy, xx]  # shape (64, N)
            finite_mask = np.isfinite(emb_pix).all(axis=0)
            n_total = emb_pix.shape[1]
            n_keep = int(finite_mask.sum())
            n_drop = n_total - n_keep
            extracted_pixels_total += n_total
            kept_pixels_total += n_keep
            if n_drop > 0 and args.drop_nonfinite and not args.sanitize:
                logging.warning(
                    f"{f.name}: dropping {n_drop:,}/{n_total:,} pixels with non-finite embeddings ({n_drop/n_total*100:.2f}%)")

            # apply mask if dropping is enabled (sanitize would have replaced non-finite already)
            if args.drop_nonfinite and not args.sanitize:
                yy, xx = yy[finite_mask], xx[finite_mask]
                emb_pix = emb[:, yy, xx]

            if len(yy) == 0:
                continue

            rowd = {
                'tile_id': np.full(len(yy), tile_id, dtype=np.int32),
                'row': yy.astype(np.int32),
                'col': xx.astype(np.int32),
                'phenology': phenology[yy, xx].astype(np.uint16)
            }
            # embeddings (use emb_pix if computed, else slice again)
            if 'emb_pix' in locals() and emb_pix.shape[1] == len(yy):
                for i in range(64):
                    rowd[f'embedding_{i}'] = emb_pix[i]
            else:
                for i in range(64):
                    rowd[f'embedding_{i}'] = emb[i, yy, xx]

            if args.include_meta:
                if genus is not None: rowd['genus'] = genus[yy, xx].astype(np.uint32)
                if species is not None: rowd['species'] = species[yy, xx].astype(np.uint32)
                if source is not None: rowd['source'] = source[yy, xx].astype(np.uint32)
                if year is not None: rowd['year'] = year[yy, xx].astype(np.uint32)

            df = pd.DataFrame(rowd)
            # add eco_region by tile_id
            df['eco_region'] = df['tile_id'].map(tile_to_eco).fillna('Unknown')

            if args.match_baseline and args.chunked:
                before_total += len(df)
                # Filter baseline by tile_id to reduce memory
                tbl = baseline_ds.to_table(filter=ds.field('tile_id') == tile_id, columns=['row','col','weight'])
                base_df = tbl.to_pandas() if tbl.num_rows > 0 else pd.DataFrame(columns=['row','col','weight'])
                merged = df.merge(base_df, on=['row','col'], how='left')
                if 'weight' in merged.columns:
                    merged = merged.dropna(subset=['weight'])
                after_total += len(merged)
                if not merged.empty:
                    t = pa.Table.from_pandas(merged, preserve_index=False)
                    if writer is None:
                        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                        writer = pq.ParquetWriter(args.output_path, t.schema)
                    writer.write_table(t)
            else:
                rows.append(df)

    if not rows:
        raise SystemExit("No valid pixels extracted.")

    # Optional merge with baseline to import weights and then drop rows without matches
    if args.match_baseline and args.chunked:
        # close writer and report
        if writer is not None:
            writer.close()
        removed = before_total - after_total
        pct_removed = (removed / before_total * 100) if before_total > 0 else 0.0
        logging.info(f"Embeddings rows before merge: {before_total:,}")
        logging.info(f"Embeddings rows after  merge: {after_total:,}")
        logging.info(f"Removed rows (no baseline match): {removed:,} ({pct_removed:.2f}%)")
        logging.info(f"Saved {after_total:,} rows to {args.output_path}")
    elif args.match_baseline:
        logging.info("Merging with baseline weights and dropping non-matching rows")
        out = pd.concat(rows, ignore_index=True)
        before = len(out)
        logging.info(f"Embeddings rows before merge: {before:,}")
        try:
            base = pd.read_parquet(args.baseline_path, columns=['tile_id','row','col','weight'])
        except Exception as e:
            logging.error(f"Failed to read baseline parquet: {e}")
            raise
        logging.info(f"Baseline rows loaded: {len(base):,}")
        merged = out.merge(base, on=['tile_id','row','col'], how='left')
        missing = merged['weight'].isna().sum()
        pct_missing = (missing / before * 100) if before > 0 else 0.0
        if missing:
            logging.warning(f"Rows without baseline weight (to be removed): {missing:,} ({pct_missing:.2f}%)")
        merged = merged.dropna(subset=['weight'])
        after = len(merged)
        removed = before - after
        pct_removed = (removed / before * 100) if before > 0 else 0.0
        logging.info(f"Embeddings rows after dropping non-matches: {after:,} (removed {removed:,}, {pct_removed:.2f}%)")
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        merged.to_parquet(args.output_path, index=False)
        logging.info(f"Saved {len(merged):,} rows to {args.output_path}")
    else:
        out = pd.concat(rows, ignore_index=True)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        out.to_parquet(args.output_path, index=False)
        logging.info(f"Saved {len(out):,} rows to {args.output_path}")
    # Report global non-finite filtering summary
    if extracted_pixels_total > 0 and (args.drop_nonfinite and not args.sanitize):
        dropped = extracted_pixels_total - kept_pixels_total
        logging.info(
            f"Non-finite filtering: kept {kept_pixels_total:,} / {extracted_pixels_total:,} "
            f"pixels (dropped {dropped:,}, {(dropped/extracted_pixels_total*100):.2f}%)")
    logging.info(f"Log: {log_file}")

if __name__ == '__main__':
    main()
