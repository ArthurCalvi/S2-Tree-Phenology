import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_r2_tile_map(per_tile_path: str, out_dir: str, title: str = "Tile similarity (R^2)"):
    df = pd.read_parquet(per_tile_path)
    if df.empty:
        print("No tile similarity data found.")
        return
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 10))
    sc = ax.scatter(df["x2154"], df["y2154"], c=df["r2_linear_mean"], s=20, cmap="viridis", vmin=0, vmax=1)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("R^2 (mean across Top-14)")
    ax.set_title(title)
    ax.set_xlabel("X (EPSG:2154)")
    ax.set_ylabel("Y (EPSG:2154)")
    ax.set_aspect('equal', adjustable='box')
    out_path = os.path.join(out_dir, "r2_tile_map.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_r2_hist(per_tile_path: str, out_dir: str):
    df = pd.read_parquet(per_tile_path)
    if df.empty:
        return
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["r2_linear_mean"].dropna(), bins=30, color="#377eb8")
    ax.set_xlabel("R^2 (mean across Top-14)")
    ax.set_ylabel("Tile count")
    ax.set_title("Distribution of tile-level similarity")
    out_path = os.path.join(out_dir, "r2_tile_hist.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tile", default="results/analysis_similarity/embedding_harmonic_similarity_tile.parquet")
    ap.add_argument("--out", default="results/analysis_similarity/maps")
    args = ap.parse_args()
    plot_r2_tile_map(args.per_tile, args.out)
    plot_r2_hist(args.per_tile, args.out)


if __name__ == "__main__":
    main()

