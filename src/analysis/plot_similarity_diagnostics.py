import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils import apply_science_style, science_style

apply_science_style()


def parse_group(group: str):
    # Expect format index:component, e.g., ndvi:amplitude_h1
    if ":" in group:
        idx, comp = group.split(":", 1)
        return idx.upper(), comp
    return "OTHER", group


def plot_coefficients_heatmap(coef_csv: str, out_dir: str):
    df = pd.read_csv(coef_csv)
    if df.empty:
        print("Coefficient CSV is empty.")
        return
    df[["index", "component"]] = df.apply(lambda r: pd.Series(parse_group(r["group"])), axis=1)

    for eco, dfe in df.groupby("eco_region"):
        pivot = dfe.pivot_table(index="index", columns="component", values="coef_norm_mean", aggfunc="mean")
        pivot = pivot.reindex(index=["NDVI", "EVI", "NBR", "CRSWIR"])
        pivot = pivot.sort_index(axis=1)
        os.makedirs(out_dir, exist_ok=True)
        with science_style():
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(pivot.values, cmap="magma", aspect="auto", vmin=0, vmax=max(0.3, np.nanmax(pivot.values)))
            ax.set_xticks(np.arange(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(np.arange(pivot.shape[0]))
            ax.set_yticklabels(pivot.index)
            ax.set_title(f"Normalized coefficient contributions — {eco}")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Mean normalized |coef|")
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"coef_heatmap_{eco.replace(' ', '_')}.png")
            plt.savefig(out_path, dpi=200)
            plt.close(fig)
        print(f"Saved {out_path}")


def plot_closest_harmonic_bar(closest_csv: str, out_dir: str):
    df = pd.read_csv(closest_csv)
    if df.empty:
        print("Closest-harmonic CSV is empty.")
        return
    # Map feature to index:component group for compact bars
    def to_group_name(feat: str):
        parts = feat.split("_")
        if len(parts) >= 2:
            idx = parts[0].upper()
            comp = "_".join(parts[1:])
            return f"{idx}:{comp}"
        return feat

    df["group"] = df["best_feature"].apply(to_group_name)

    for eco, dfe in df.groupby("eco_region"):
        counts = dfe["group"].value_counts().sort_values(ascending=False)
        os.makedirs(out_dir, exist_ok=True)
        with science_style():
            fig, ax = plt.subplots(figsize=(10, 4))
            counts.plot(kind="bar", color="#4daf4a", ax=ax)
            ax.set_ylabel("Embeddings (Top-14) count")
            ax.set_title(f"Most similar harmonic group — {eco}")
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"closest_harmonic_bar_{eco.replace(' ', '_')}.png")
            plt.savefig(out_path, dpi=200)
            plt.close(fig)
        print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coef", default="results/analysis_similarity/embedding_harmonic_coefficients_region.csv")
    ap.add_argument("--closest", default="results/analysis_similarity/closest_harmonic_region.csv")
    ap.add_argument("--out", default="results/analysis_similarity/diagnostics")
    args = ap.parse_args()
    plot_coefficients_heatmap(args.coef, os.path.join(args.out, "coefficients"))
    plot_closest_harmonic_bar(args.closest, os.path.join(args.out, "closest"))


if __name__ == "__main__":
    main()
