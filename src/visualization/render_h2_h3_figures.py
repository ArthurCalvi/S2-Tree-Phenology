#!/usr/bin/env python3
"""Render manuscript figures for spatial (H2) and temporal (H3) comparisons."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import to_rgb
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.visualization.figure_tile_selection import H2_H3_TILE_SELECTION
from src.utils import science_style, apply_science_style

apply_science_style()

LOGGER = logging.getLogger("render_h2_h3_figures")


DEFAULT_H2_TILES: Sequence[int] = (274, 605, 59, 562, 420)
DEFAULT_H3_TILES: Sequence[int] = (274, 562, 605)
H3_YEARS: Sequence[int] = (2018, 2020, 2022, 2023)

CLASS_COLORS = {
    0: to_rgb("#d0d0d0"),   # background / masked
    1: to_rgb("#e3712c"),   # deciduous
    2: to_rgb("#2693c1"),   # evergreen
}


def load_s2_rgb(tile_id: int, year: int = 2023) -> np.ndarray:
    """Load Sentinel-2 composite and return stretched RGB array in [0,1]."""
    # Prefer H2 label, fall back to H3 exports
    candidates = [
        Path(f"results/figures/figure_s2_tiles/H2_S2_tile_{tile_id:03d}_{year}.tif"),
        Path(f"results/figures/figure_s2_tiles/H3_S2_tile_{tile_id:03d}_{year}.tif"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Sentinel-2 composite not found for tile {tile_id} year {year}")

    with rasterio.open(path) as src:
        # Bands: B2, B3, B4, B8, B11, B12
        rgb = src.read([3, 2, 1]).astype(np.float32)

    stretched = np.zeros_like(rgb, dtype=np.float32)
    for band_idx in range(3):
        band = rgb[band_idx]
        mask = band > 0
        if not mask.any():
            continue
        low, high = np.percentile(band[mask], (2, 98))
        if high <= low:
            high = low + 1.0
        norm = np.clip((band - low) / (high - low), 0, 1)
        stretched[band_idx] = norm

    return np.moveaxis(stretched, 0, -1)


def load_class_map(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        data = src.read(1)
    return data


def colorize_classes(classes: np.ndarray) -> np.ndarray:
    rgb = np.empty((*classes.shape, 3), dtype=np.float32)
    rgb[...] = np.array(CLASS_COLORS[0], dtype=np.float32)
    for label, color in CLASS_COLORS.items():
        rgb[classes == label] = color
    return rgb


def eco_region_lookup() -> dict[int, str]:
    letters = {
        "Greater Semi-Continental East": "G",
        "Oceanic Southwest": "O",
        "Corsica": "C",
        "Alps": "A",
    }
    mapping: dict[int, str] = {}
    for tile in H2_H3_TILE_SELECTION:
        name = tile["eco_region"]
        mapping[int(tile["tile_id"])] = letters.get(name, name)
    return mapping


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def draw_class_legend(fig: plt.Figure) -> None:
    legend_ax = fig.add_axes([0.35, 0.015, 0.3, 0.045])
    legend_ax.axis("off")
    entries = [("Deciduous", CLASS_COLORS[1]), ("Evergreen", CLASS_COLORS[2])]
    for idx, (label, color) in enumerate(entries):
        xpos = 0.05 + idx * 0.5
        legend_ax.add_patch(
            mpatches.Rectangle(
                (xpos, 0.2),
                0.15,
                0.6,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
            )
        )
        legend_ax.text(
            xpos + 0.18,
            0.5,
            label,
            va="center",
            ha="left",
            fontsize=11,
            fontweight="normal",
        )


def render_h2_figure(
    tiles: Sequence[int],
    output_path: Path,
) -> None:
    eco_names = eco_region_lookup()
    n_rows = len(tiles)
    with science_style():
        fig, axes = plt.subplots(
            n_rows,
            3,
            figsize=(9, 3 * n_rows),
            dpi=200,
            constrained_layout=False,
        )
        axes = np.atleast_2d(axes)

        for row, tile_id in enumerate(tiles):
            LOGGER.info("Rendering H2 tile %03d", tile_id)
            s2_rgb = load_s2_rgb(tile_id, year=2023)
            emb_path = Path(f"results/figures/figure_h2/embedding/H2_emb14_tile_{tile_id:03d}_2023.tif")
            harm_path = Path(f"results/figures/figure_h2/harmonic/H2_harm14_tile_{tile_id:03d}_2023.tif")
            emb_classes = load_class_map(emb_path)
            harm_classes = load_class_map(harm_path)

            panels = [
                ("Sentinel-2 (2023)", s2_rgb),
                ("EMB-14 (2023)", colorize_classes(emb_classes)),
                ("HARM-14 (2023)", colorize_classes(harm_classes)),
            ]

            for col, (title, image) in enumerate(panels):
                ax = axes[row, col]
                ax.imshow(image)
                if row == 0:
                    ax.set_title(title, fontsize=11)
                ax.axis("off")

        fig.subplots_adjust(bottom=0.10, left=0.02, right=0.99, top=0.97, hspace=0.02, wspace=0.02)
        draw_class_legend(fig)

        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    LOGGER.info("Saved H2 figure to %s", output_path)


def render_h3_figure(
    tiles: Sequence[int],
    years: Sequence[int],
    output_path: Path,
) -> None:
    eco_names = eco_region_lookup()
    n_tile_rows = len(tiles)
    n_rows = n_tile_rows * 2
    n_cols = len(years)
    with science_style():
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(2.8 * n_cols, 2.8 * n_rows),
            dpi=200,
            constrained_layout=False,
        )
        axes = np.atleast_2d(axes)

        for tile_idx, tile_id in enumerate(tiles):
            LOGGER.info("Rendering H3 tile %03d", tile_id)
            sentinel_row = tile_idx * 2
            emb_row = sentinel_row + 1

            for col, year in enumerate(years):
                s2_rgb = load_s2_rgb(tile_id, year=year)
                emb_path = Path(
                    f"results/figures/figure_h3/{year}/emb_tile_{tile_id:03d}_{year}_classes.tif"
                )
                emb_classes = load_class_map(emb_path)

                ax_sentinel = axes[sentinel_row, col]
                ax_emb = axes[emb_row, col]

                ax_sentinel.imshow(s2_rgb)
                if tile_idx == 0:
                    ax_sentinel.set_title(f"{year}", fontsize=12, fontweight="bold", pad=8)
                ax_sentinel.axis("off")

                ax_emb.imshow(colorize_classes(emb_classes))
                ax_emb.axis("off")

        fig.subplots_adjust(bottom=0.08, left=0.003, right=0.999, top=0.96, hspace=0.008, wspace=0.006)
        draw_class_legend(fig)

        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    LOGGER.info("Saved H3 figure to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render manuscript figures for H2 and H3.")
    parser.add_argument(
        "--h2-tiles",
        nargs="+",
        type=int,
        default=list(DEFAULT_H2_TILES),
        help="Tile IDs to include in Figure 4/H2 (default: %(default)s).",
    )
    parser.add_argument(
        "--h3-tiles",
        nargs="+",
        type=int,
        default=list(DEFAULT_H3_TILES),
        help="Tile IDs to include in Figure 5/H3 (default: %(default)s).",
    )
    parser.add_argument(
        "--h2-output",
        type=Path,
        default=Path("results/figures/figure4_h2_panel.png"),
        help="Output path for Figure 4/H2 panel.",
    )
    parser.add_argument(
        "--h3-output",
        type=Path,
        default=Path("results/figures/figure5_h3_panel.png"),
        help="Output path for Figure 5/H3 panel.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    render_h2_figure(args.h2_tiles, args.h2_output)
    render_h3_figure(args.h3_tiles, H3_YEARS, args.h3_output)


if __name__ == "__main__":
    main()
