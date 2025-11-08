import argparse
import os
import sys
import time
from typing import List, Optional

import ee
import geemap
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL embeddings per tile to local GeoTIFFs"
    )
    parser.add_argument(
        "--tiles_asset",
        required=True,
        help="Earth Engine FeatureCollection asset path with a 'tile_id' property",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Year to export (calendar year summarized by the embedding)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/embeddings",
        help="Local directory to save GeoTIFFs (default: data/embeddings)",
    )
    parser.add_argument(
        "--crs",
        default="EPSG:2154",
        help="CRS for export (default: EPSG:2154)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="Export scale in meters (default: 10)",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index of tiles to export (after sorting by tile_id)",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index (exclusive) of tiles to export. If None, export all after start_index",
    )
    parser.add_argument(
        "--poll_interval_sec",
        type=int,
        default=20,
        help="Polling interval in seconds for task status updates (default: 20)",
    )
    parser.add_argument(
        "--env_file",
        default=None,
        help="Optional .env file to load environment variables from",
    )
    parser.add_argument(
        "--service_account",
        default=None,
        help="Optional service account email for EE auth",
    )
    parser.add_argument(
        "--private_key",
        default=None,
        help="Path to service account private key JSON (required if service_account set)",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Optional Earth Engine project to bill/associate requests",
    )
    return parser.parse_args()


def initialize_earth_engine(service_account: Optional[str], private_key: Optional[str], project: Optional[str]) -> None:
    # Auth: service account if provided, else default auth flow
    if service_account:
        if not private_key:
            print("--private_key is required when --service_account is set", file=sys.stderr)
            sys.exit(1)
        creds = ee.ServiceAccountCredentials(service_account, private_key)
        ee.Initialize(creds, project=project)
    else:
        # If first time on this machine, uncomment next line to run interactive auth
        # ee.Authenticate()
        ee.Initialize(project=project)


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


def load_env_file(env_file: Optional[str]) -> None:
    if not env_file:
        return
    # Try python-dotenv
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(env_file, override=False)
        return
    except Exception:
        pass
    # Try pydotenv
    try:
        import pydotenv  # type: ignore
        pydotenv.load_dotenv(env_file)
        return
    except Exception:
        pass
    # Minimal fallback parser KEY=VALUE lines
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                key, value = s.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
    except Exception:
        # Ignore if file can't be read
        pass


def load_tiles(tiles_asset: str, start_index: int, end_index: int) -> ee.FeatureCollection:
    fc = (
        ee.FeatureCollection(tiles_asset)
        .filter(ee.Filter.notNull(["tile_id"]))
        .sort("tile_id")
    )
    # Materialize to a list to slice deterministically, then back to FC
    tiles_list = ee.List(fc.toList(100000))
    size = tiles_list.size()
    if end_index is None:
        end_index = size
    subset = tiles_list.slice(start_index, end_index)
    return ee.FeatureCollection(subset)


def get_embedding_image(year: int) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    band_list = [f"A{i:02d}" for i in range(64)]
    img = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(start, end)
        .mosaic()
        .select(band_list)
    )
    return img


def download_tiles_locally(
    tiles_fc: ee.FeatureCollection,
    base_image: ee.Image,
    year: int,
    output_dir: str,
    crs: str,
    scale: float,
    poll_interval_sec: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    # Reproject geometry to WGS84 for stable region parameter
    tiles = tiles_fc.toList(100000)
    count = int(tiles_fc.size().getInfo())

    with tqdm(total=count, unit="tile") as pbar:
        for i in range(count):
            ft = ee.Feature(tiles.get(i))
            # Fetch tile_id value
            tile_id = int(ft.get("tile_id").getInfo())
            region = ft.geometry()

            filename = os.path.join(output_dir, f"emb_tile_{tile_id:03d}_{year}.tif")
            # Compute a simple north-up affine transform from the region bounds
            bounds = ee.Geometry(region).bounds()
            coords = ee.List(ee.List(bounds.coordinates()).get(0))
            ll = ee.List(coords.get(0))  # lower-left (xmin, ymin)
            ur = ee.List(coords.get(2))  # upper-right (xmax, ymax)
            xmin = ee.Number(ll.get(0))
            ymin = ee.Number(ll.get(1))
            xmax = ee.Number(ur.get(0))
            ymax = ee.Number(ur.get(1))
            # Use the requested scale and top-left origin (xmin, ymax)
            crs_transform = [float(scale), 0.0, float(xmin.getInfo()), 0.0, -float(scale), float(ymax.getInfo())]

            image = base_image.clip(region)

            try:
                # geemap accepts crs and crs_transform through kwargs
                geemap.ee_export_image(
                    image=image,
                    filename=filename,
                    region=region,
                    crs=crs,
                    crs_transform=crs_transform,
                    file_per_band=False,
                )
            except Exception as exc:  # keep going on failures
                pbar.set_postfix_str(f"tile {tile_id} failed: {exc}")
            finally:
                # Small delay between requests to be gentle on quotas
                time.sleep(max(1, poll_interval_sec // 4))
                pbar.update(1)


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    sa, key, proj = resolve_auth_params(args)
    initialize_earth_engine(sa, key, proj)

    tiles_fc = load_tiles(args.tiles_asset, args.start_index, args.end_index)
    base_image = get_embedding_image(args.year)

    download_tiles_locally(
        tiles_fc=tiles_fc,
        base_image=base_image,
        year=args.year,
        output_dir=args.output_dir,
        crs=args.crs,
        scale=args.scale,
        poll_interval_sec=args.poll_interval_sec,
    )


if __name__ == "__main__":
    main()


