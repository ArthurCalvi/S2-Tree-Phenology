#!/usr/bin/env python3
"""
Export autumn Sentinel-2 composites for the manuscript figure tiles.

For each tile, the script filters COPERNICUS/S2_SR_HARMONIZED to September-November,
applies QA60/SCL cloud masking, computes a median composite, and saves the result as
an EPSG:2154 GeoTIFF at 10 m resolution.

Authentication mirrors `gee/export_embeddings_per_tile.py`: credentials are pulled from
environment variables (e.g., loaded via `.env`) or passed explicitly using CLI flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import ee
import geemap
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.visualization.figure_tile_selection import H2_H3_TILE_SELECTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download autumn Sentinel-2 composites for selected tiles."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2018, 2020, 2022, 2023],
        help="Years to export (default: 2018 2020 2022 2023).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/figure_s2_tiles"),
        help="Directory where GeoTIFFs are written.",
    )
    parser.add_argument(
        "--tile-selection",
        type=Path,
        default=None,
        help="Optional JSON file overriding the default tile selection.",
    )
    parser.add_argument(
        "--tile-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of tile IDs to export.",
    )
    parser.add_argument(
        "--cloud-threshold",
        type=float,
        default=20.0,
        help="Maximum CLOUDY_PIXEL_PERCENTAGE for Sentinel-2 scenes (default: 20).",
    )
    parser.add_argument(
        "--season-start-month",
        type=int,
        default=9,
        help="Start month for the seasonal window (default: September).",
    )
    parser.add_argument(
        "--season-end-month",
        type=int,
        default=11,
        help="End month for the seasonal window (default: November).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="Target resolution in metres (default: 10).",
    )
    parser.add_argument(
        "--label",
        default="S2",
        help="Label prefix added to output filenames (default: S2).",
    )
    parser.add_argument(
        "--service-account",
        default=None,
        help="Optional service account email for Earth Engine auth.",
    )
    parser.add_argument(
        "--private-key",
        default=None,
        help="Path to the service account key JSON.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Earth Engine project / quota project.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load EARTHENGINE/GOOGLE credentials from (defaults to ./.env if present).",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=int,
        default=5,
        help="Delay between export attempts to avoid hammering the EE backend.",
    )
    return parser.parse_args()


def load_env_file(env_path: Optional[Path], strict: bool = False) -> None:
    if env_path is None:
        return
    if not env_path.exists():
        if strict:
            raise FileNotFoundError(f".env file not found: {env_path}")
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass
    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def resolve_auth_params(args: argparse.Namespace) -> tuple[Optional[str], Optional[str], Optional[str]]:
    service_account = (
        args.service_account
        or os.getenv("EE_SERVICE_ACCOUNT")
        or os.getenv("EARTHENGINE_SERVICE_ACCOUNT")
    )
    private_key = (
        args.private_key
        or os.getenv("EE_PRIVATE_KEY")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )
    project = args.project or os.getenv("EE_PROJECT") or os.getenv("EARTHENGINE_PROJECT")
    return service_account, private_key, project


def initialize_earth_engine(service_account: Optional[str], private_key: Optional[str], project: Optional[str]) -> None:
    if service_account:
        if not private_key:
            raise ValueError("--private-key must be provided when --service-account is set")
        credentials = ee.ServiceAccountCredentials(service_account, private_key)
        ee.Initialize(credentials, project=project)
    else:
        try:
            ee.Initialize(project=project)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=project)


def load_tile_selection(path: Optional[Path]) -> list[dict]:
    if path is None:
        return [dict(tile) for tile in H2_H3_TILE_SELECTION]
    if not path.exists():
        raise FileNotFoundError(f"Tile selection JSON not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Tile selection file must contain a list of tile entries")
    return data


def filter_tiles(tiles: Iterable[dict], tile_ids: Optional[Iterable[int]]) -> list[dict]:
    if tile_ids is None:
        return list(tiles)
    id_set = {int(tid) for tid in tile_ids}
    return [tile for tile in tiles if int(tile["tile_id"]) in id_set]


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus_mask = qa.bitwiseAnd(1 << 11).eq(0)
    scl = image.select("SCL")
    scl_mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    mask = cloud_mask.And(cirrus_mask).And(scl_mask)
    return image.updateMask(mask)


def build_autumn_composite(
    region: ee.Geometry,
    year: int,
    cloud_threshold: float,
    start_month: int,
    end_month: int,
) -> ee.Image:
    start = ee.Date.fromYMD(year, start_month, 1)
    end = ee.Date.fromYMD(year, end_month, 1).advance(1, "month")
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(region)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .map(mask_s2_clouds)
        .select(["B2", "B3", "B4", "B8", "B11", "B12"])
    )
    composite = collection.median().clip(region)
    return composite.set(
        {
            "acquisition_year": year,
            "season_start": start.format("YYYY-MM-dd"),
            "season_end": end.advance(-1, "day").format("YYYY-MM-dd"),
        }
    )


def export_tile(
    composite: ee.Image,
    tile: dict,
    year: int,
    output_dir: Path,
    scale: float,
    poll_interval: int,
    label: str,
) -> None:
    tile_id = int(tile["tile_id"])
    bbox = tile.get("bbox_wgs84")
    if bbox is None:
        raise ValueError(f"Tile {tile_id:03d} missing 'bbox_wgs84' entry")
    lambert_bbox = tile.get("bbox_lambert")
    if lambert_bbox is None:
        raise ValueError(f"Tile {tile_id:03d} missing 'bbox_lambert' entry")

    region = ee.Geometry.Rectangle(bbox)
    xmin, ymin, xmax, ymax = lambert_bbox

    crs_transform = [
        float(scale),
        0.0,
        float(xmin),
        0.0,
        -float(scale),
        float(ymax),
    ]

    filename = output_dir / f"{label}_S2_tile_{tile_id:03d}_{year}.tif"
    geemap.ee_export_image(
        composite,
        str(filename),
        region=region,
        scale=scale,
        crs="EPSG:2154",
        crs_transform=crs_transform,
        file_per_band=False,
        verbose=False,
    )
    time.sleep(max(1, poll_interval))


def main() -> None:
    args = parse_args()
    env_path = args.env_file or Path(".env")
    load_env_file(env_path, strict=args.env_file is not None)
    service_account, private_key, project = resolve_auth_params(args)
    initialize_earth_engine(service_account, private_key, project)

    tiles = filter_tiles(load_tile_selection(args.tile_selection), args.tile_ids)
    if not tiles:
        raise ValueError("No tiles selected for export. Provide valid --tile-ids or selection file.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tqdm(total=len(tiles) * len(args.years), unit="export") as progress:
        for year in args.years:
            for tile in tiles:
                composite = build_autumn_composite(
                    region=ee.Geometry.Rectangle(tile["bbox_wgs84"]),
                    year=year,
                    cloud_threshold=args.cloud_threshold,
                    start_month=args.season_start_month,
                    end_month=args.season_end_month,
                )
                try:
                    export_tile(
                        composite=composite,
                        tile=tile,
                        year=year,
                        output_dir=args.output_dir,
                        scale=args.scale,
                        poll_interval=args.poll_interval_sec,
                        label=args.label,
                    )
                except Exception as exc:
                    progress.set_postfix_str(f"tile {tile['tile_id']} year {year} failed: {exc}")
                finally:
                    progress.update(1)


if __name__ == "__main__":
    main()
