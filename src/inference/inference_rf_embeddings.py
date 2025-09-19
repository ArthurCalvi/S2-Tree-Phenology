#!/usr/bin/env python3
"""Run RandomForest inference on AlphaEarth embedding tiles.

This script reads GeoTIFF tiles that contain 64 AlphaEarth embedding bands
(e.g. A00..A63), selects the subset of embedding features used to train the
RandomForest model, and writes class probabilities (and optionally class
labels) to GeoTIFF tiles.
"""

import argparse
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# Ensure project root is available on PYTHONPATH (useful on HPC nodes)
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

logger = logging.getLogger("rf_embeddings_inference")


def load_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    features = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not features:
        raise ValueError(f"Features file {path} is empty")
    for feat in features:
        if not feat.startswith("embedding_"):
            raise ValueError(f"Unexpected feature name '{feat}'. Expected 'embedding_<idx>'.")
    return features


def embedding_to_band_index(feature_name: str) -> int:
    try:
        idx = int(feature_name.split("_")[1])
    except (IndexError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Malformed embedding feature '{feature_name}'") from exc
    return idx + 1  # rasterio bands are 1-based


def generate_windows(width: int, height: int, block_size: int) -> Iterable[Window]:
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            win_width = min(block_size, width - col)
            win_height = min(block_size, height - row)
            yield Window(col, row, win_width, win_height)


def process_tile(tile_path: Path,
                 output_dir: Path,
                 model,
                 feature_names: list[str],
                 block_size: int,
                 save_classes: bool) -> None:
    band_indices = [embedding_to_band_index(f) for f in feature_names]
    logger.info("Processing %s with %d embedding bands", tile_path.name, len(band_indices))

    with rasterio.open(tile_path) as src:
        if max(band_indices) > src.count:
            raise ValueError(
                f"Tile {tile_path} only has {src.count} bands, but feature index {max(band_indices)} is required"
            )

        profile = src.profile.copy()
        profile.update({
            "count": len(model.classes_),
            "dtype": "uint8",
            "nodata": None,
            "compress": "lzw",
            "predictor": 2,
            "BIGTIFF": "IF_SAFER",
        })

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / tile_path.name

        class_profile = profile.copy()
        class_profile.update({
            "count": 1,
            "nodata": 0,
        })
        class_output_path = output_dir / f"{tile_path.stem}_classes.tif"

        windows = list(generate_windows(src.width, src.height, block_size))

        with rasterio.open(output_path, "w", **profile) as dst_prob:
            class_context = rasterio.open(class_output_path, "w", **class_profile) if save_classes else nullcontext()
            with class_context as dst_cls:
                for window in tqdm(windows, desc=f"{tile_path.name}"):
                    block = src.read(band_indices, window=window, out_dtype="float32")
                    flat = block.reshape(len(band_indices), -1).T  # (N_pixels, N_features)
                    if flat.size == 0:
                        continue
                    np.clip(flat, -1.0, 1.0, out=flat)
                    flat = np.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=-1.0)

                    probs = model.predict_proba(flat)
                    probs_map = probs.T.reshape(len(model.classes_), window.height, window.width)
                    prob_uint8 = np.clip(np.round(probs_map * 255), 0, 255).astype(np.uint8)
                    dst_prob.write(prob_uint8, window=window)

                    if dst_cls is not None:
                        class_idx = np.argmax(probs, axis=1)
                        class_vals = model.classes_[class_idx]
                        class_raster = class_vals.reshape(1, window.height, window.width).astype(np.uint8, copy=False)
                        dst_cls.write(class_raster, window=window)

    logger.info("Finished %s", tile_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RandomForest inference on AlphaEarth embedding tiles")
    parser.add_argument("--input-dir", required=True, help="Directory containing embedding GeoTIFF tiles")
    parser.add_argument("--output-dir", required=True, help="Directory to store RF probability tiles")
    parser.add_argument("--model", required=True, help="Path to the trained RandomForest .joblib file")
    parser.add_argument("--features-file", required=True, help="Text file listing embedding features (one per line)")
    parser.add_argument("--block-size", type=int, default=2048, help="Window size for block-wise inference")
    parser.add_argument("--tile-idx", type=int, default=None, help="Optional tile index (for SLURM array jobs)")
    parser.add_argument("--rf-n-jobs", type=int, default=None, help="Override RandomForest n_jobs before inference")
    parser.add_argument("--save-classes", action="store_true", help="Also write a class map GeoTIFF")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    model = joblib.load(args.model)
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not implement predict_proba; cannot run RF inference")
    if args.rf_n_jobs is not None and hasattr(model, "n_jobs"):
        logger.info("Setting model n_jobs to %d", args.rf_n_jobs)
        model.n_jobs = args.rf_n_jobs

    feature_names = load_feature_list(Path(args.features_file))
    band_indices = [embedding_to_band_index(f) for f in feature_names]
    logger.info("Model expects %d embedding bands (max index %d)", len(feature_names), max(band_indices))

    tile_paths = sorted([p for p in input_dir.glob("*.tif") if p.is_file()])
    if not tile_paths:
        raise SystemExit(f"No GeoTIFF tiles found in {input_dir}")

    if args.tile_idx is not None:
        if args.tile_idx < 0 or args.tile_idx >= len(tile_paths):
            raise IndexError(f"tile-idx {args.tile_idx} is outside [0, {len(tile_paths) - 1}]")
        tile_subset = [tile_paths[args.tile_idx]]
        logger.info("Running single-tile mode for index %d: %s", args.tile_idx, tile_subset[0].name)
    else:
        tile_subset = tile_paths
        logger.info("Processing %d tiles sequentially", len(tile_subset))

    for tile_path in tile_subset:
        process_tile(
            tile_path=tile_path,
            output_dir=output_dir,
            model=model,
            feature_names=feature_names,
            block_size=args.block_size,
            save_classes=args.save_classes,
        )

    logger.info("Inference complete")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
