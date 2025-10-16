#!/usr/bin/env python3
"""Run RandomForest inference on AlphaEarth embedding tiles.

This script reads GeoTIFF tiles that contain 64 AlphaEarth embedding bands
(e.g. A00..A63), selects the subset of embedding features used to train the
RandomForest model, and writes class probabilities (and optionally class
labels) to GeoTIFF tiles.
"""

import argparse
import logging
import math
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import joblib
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

try:  # pragma: no cover - optional during static analysis
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:
    InconsistentVersionWarning = None  # type: ignore

# Ensure project root is available on PYTHONPATH (useful on HPC nodes)
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

logger = logging.getLogger("rf_embeddings_inference")


UINT16_MAX = np.float32(65535.0)


def configure_warning_filters() -> None:
    """Silence expected sklearn warnings during inference."""
    if InconsistentVersionWarning is not None:
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names",
        category=UserWarning,
    )


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


def feature_to_band_labels(feature_names: list[str]) -> dict[str, str]:
    """Return mapping from embedding feature (embedding_XX) to raster band label (Axx)."""
    mapping: dict[str, str] = {}
    for feat in feature_names:
        try:
            idx = int(feat.split("_")[1])
        except (IndexError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Malformed embedding feature '{feat}'") from exc
        mapping[feat] = f"A{idx:02d}"
    return mapping


def build_band_lookup(descriptions: Iterable[str]) -> Mapping[str, int]:
    """Create lookup dictionary from band description (A00) to 1-based band index."""
    lookup: dict[str, int] = {}
    for idx, desc in enumerate(descriptions, start=1):
        key = f"A{idx-1:02d}" if not desc else desc.strip()
        if not key:
            key = f"A{idx-1:02d}"
        lookup[key] = idx
    return lookup


def rescale_uint16_block(block: np.ndarray,
                         nodata_values: Tuple[Optional[float], ...],
                         band_min: float,
                         band_max: float) -> np.ndarray:
    """Convert uint16-scaled embeddings back to float range [band_min, band_max]."""
    scale = band_max - band_min
    if scale == 0:
        raise ValueError("band_max must differ from band_min for rescaling")

    # Work on a copy to avoid mutating the caller's array unexpectedly
    out = (block / UINT16_MAX) * scale + band_min

    if nodata_values:
        for band_idx, nodata in enumerate(nodata_values):
            if nodata is None:
                continue
            mask = block[band_idx] == nodata
            if np.any(mask):
                out[band_idx][mask] = np.nan

    np.clip(out, band_min, band_max, out=out)
    return out


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
                 save_classes: bool,
                 band_min: float,
                 band_max: float,
                 missing_fill: float,
                 fail_on_missing: bool) -> None:
    logger.info("Processing %s with window size %d", tile_path.name, block_size)

    with rasterio.open(tile_path) as src:
        feature_to_label = feature_to_band_labels(feature_names)
        band_lookup = build_band_lookup(src.descriptions or [])

        # Map requested features to available raster bands
        feature_band_indices: Dict[str, Optional[int]] = {}
        missing_features: list[str] = []
        for feat in feature_names:
            label = feature_to_label[feat]
            band_idx = band_lookup.get(label)
            if band_idx is None:
                missing_features.append(feat)
                feature_band_indices[feat] = None
            else:
                feature_band_indices[feat] = band_idx

        if missing_features:
            msg = ("Missing embedding bands in tile {tile}: "
                   ", ".join(missing_features)).format(tile=tile_path.name)
            if fail_on_missing:
                raise ValueError(msg)
            logger.warning("%s - filling with %.3f", msg, missing_fill)

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

        nodata_values = src.nodatavals if src.nodatavals else tuple([src.nodata] * src.count)
        is_uint16 = np.dtype(src.dtypes[0]) == np.uint16
        requested_band_indices = sorted({idx for idx in feature_band_indices.values() if idx is not None})
        if not requested_band_indices:
            logger.warning("No matching bands found in tile %s; skipping", tile_path.name)
            return

        nodata_subset = tuple(
            nodata_values[idx - 1] if nodata_values else None
            for idx in requested_band_indices
        )
        band_to_position = {band_idx: pos for pos, band_idx in enumerate(requested_band_indices)}

        tiles_x = math.ceil(src.width / block_size)
        tiles_y = math.ceil(src.height / block_size)
        total_windows = tiles_x * tiles_y

        with rasterio.open(output_path, "w", **profile) as dst_prob:
            class_context = rasterio.open(class_output_path, "w", **class_profile) if save_classes else nullcontext()
            with class_context as dst_cls:
                with tqdm(
                    total=total_windows,
                    desc=tile_path.name,
                    unit="window",
                    dynamic_ncols=True,
                ) as progress:
                    for window in generate_windows(src.width, src.height, block_size):
                        block = src.read(requested_band_indices, window=window, out_dtype="float32")

                        if is_uint16:
                            block = rescale_uint16_block(block, nodata_subset, band_min, band_max)

                        feature_stack = np.empty((len(feature_names), window.height, window.width), dtype=np.float32)
                        for feat_idx, feat in enumerate(feature_names):
                            band_idx = feature_band_indices[feat]
                            if band_idx is None:
                                feature_stack[feat_idx].fill(missing_fill)
                                continue
                            array_idx = band_to_position[band_idx]
                            feature_stack[feat_idx] = block[array_idx]

                        flat = feature_stack.reshape(len(feature_names), -1).T  # (N_pixels, N_features)
                        if flat.size == 0:
                            progress.update(1)
                            continue
                        np.clip(flat, band_min, band_max, out=flat)
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

                        progress.update(1)

    logger.info("Finished %s", tile_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RandomForest inference on AlphaEarth embedding tiles")
    parser.add_argument("--input-dir", required=True, help="Directory containing embedding GeoTIFF tiles")
    parser.add_argument("--output-dir", required=True, help="Directory to store RF probability tiles")
    parser.add_argument("--model", required=True, help="Path to the trained RandomForest .joblib file")
    parser.add_argument("--features-file", required=True, help="Text file listing embedding features (one per line)")
    parser.add_argument("--block-size", type=int, default=1024,
                        help="Window size for block-wise inference (default: 1024)")
    parser.add_argument("--tile-idx", type=int, default=None, help="Optional tile index (for SLURM array jobs)")
    parser.add_argument("--rf-n-jobs", type=int, default=None, help="Override RandomForest n_jobs before inference")
    parser.add_argument("--save-classes", action="store_true", help="Also write a class map GeoTIFF")
    parser.add_argument("--embedding-min", type=float, default=-1.0, help="Minimum embedding value prior to uint16 scaling")
    parser.add_argument("--embedding-max", type=float, default=1.0, help="Maximum embedding value prior to uint16 scaling")
    parser.add_argument("--missing-fill", type=float, default=0.0, help="Fill value to use when a requested feature band is absent")
    parser.add_argument("--fail-on-missing", action="store_true", help="Raise an error if any requested feature band is absent in a tile")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    configure_warning_filters()

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
    band_labels = feature_to_band_labels(feature_names)
    logger.info("Model expects %d embedding bands (%s)",
                len(feature_names),
                ", ".join(sorted(band_labels.values())))

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
            band_min=args.embedding_min,
            band_max=args.embedding_max,
            missing_fill=args.missing_fill,
            fail_on_missing=args.fail_on_missing,
        )

    logger.info("Inference complete")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
