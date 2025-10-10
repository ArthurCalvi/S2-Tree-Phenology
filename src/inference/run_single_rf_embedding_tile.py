#!/usr/bin/env python3
"""Convenience wrapper to run RF embedding inference on a single tile."""

import argparse
import logging
import sys
from pathlib import Path

import joblib

# Ensure project root is on sys.path so we can import the inference helpers
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.inference.inference_rf_embeddings import (
    configure_warning_filters,
    load_feature_list,
    process_tile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RF embedding inference on a single GeoTIFF tile")
    parser.add_argument("--input-dir", help="Directory with embedding tiles (optional if --tile-path is given)")
    parser.add_argument("--tile-path", help="Explicit path to the tile to process")
    parser.add_argument("--tile-idx", type=int, default=0,
                        help="Index within input-dir tiles to process when --tile-path is not provided")
    parser.add_argument("--output-dir", required=True, help="Directory where outputs will be written")
    parser.add_argument("--model", required=True, help="Path to the trained RandomForest .joblib file")
    parser.add_argument("--features-file", required=True, help="Text file listing embedding features")
    parser.add_argument("--block-size", type=int, default=1024,
                        help="Window size for block-wise inference (default: 1024)")
    parser.add_argument("--embedding-min", type=float, default=-1.0,
                        help="Minimum embedding value prior to uint16 scaling")
    parser.add_argument("--embedding-max", type=float, default=1.0,
                        help="Maximum embedding value prior to uint16 scaling")
    parser.add_argument("--missing-fill", type=float, default=0.0,
                        help="Fill value used when a requested feature band is absent")
    parser.add_argument("--fail-on-missing", action="store_true",
                        help="Raise an error if any requested feature band is absent in the tile")
    parser.add_argument("--save-classes", action="store_true", help="Also write a class raster")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    configure_warning_filters()

    tile_path: Path
    if args.tile_path:
        tile_path = Path(args.tile_path)
        if not tile_path.exists():
            raise FileNotFoundError(f"Tile not found: {tile_path}")
    else:
        if not args.input_dir:
            raise SystemExit("Provide either --tile-path or both --input-dir and --tile-idx")
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        tiles = sorted(p for p in input_dir.glob("*.tif") if p.is_file())
        if not tiles:
            raise SystemExit(f"No tiles found under {input_dir}")
        if args.tile_idx < 0 or args.tile_idx >= len(tiles):
            raise IndexError(f"tile-idx {args.tile_idx} outside [0, {len(tiles) - 1}]")
        tile_path = tiles[args.tile_idx]
        logging.info("Selected tile %s (index %d)", tile_path.name, args.tile_idx)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model)
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not implement predict_proba")

    if args.fail_on_missing and getattr(model, "n_features_in_", None) is not None:
        expected = getattr(model, "n_features_in_")
        logging.debug("Model expects %d features", expected)

    feature_names = load_feature_list(Path(args.features_file))

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


if __name__ == "__main__":  # pragma: no cover
    main()
