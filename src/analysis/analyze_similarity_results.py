"""Utility script to post-process embedding vs harmonic similarity outputs.

Usage::

    python -m src.analysis.analyze_similarity_results \
        --input-dir results/analysis_similarity \
        --output-dir results/analysis_similarity/summary

Outputs
=======
- ``summary_region.csv``: per eco-region metrics (mean R², tile count, top harmonic groups).
- ``summary_group_counts.csv``: aggregate statistics per harmonic group across regions.
- ``summary_component_counts.csv``: aggregate by harmonic component (e.g. amplitude_h1, offset).
- ``summary_report.md``: human readable highlights for quick reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

REGION_SIM_FILE = "embedding_harmonic_similarity_region.csv"
REGION_TOPGROUPS_FILE = "embedding_harmonic_region_topgroups.csv"
REGION_COEF_FILE = "embedding_harmonic_coefficients_region.csv"


def parse_group(group: str) -> Tuple[str, str]:
    if ":" not in group:
        return group, ""
    idx, component = group.split(":", maxsplit=1)
    return idx, component


def load_inputs(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    region_sim = pd.read_csv(input_dir / REGION_SIM_FILE)
    top_groups = pd.read_csv(input_dir / REGION_TOPGROUPS_FILE)
    coef = pd.read_csv(input_dir / REGION_COEF_FILE)
    return region_sim, top_groups, coef


def build_region_summary(region_sim: pd.DataFrame, top_groups: pd.DataFrame) -> pd.DataFrame:
    top_sorted = (
        top_groups
        .assign(order=lambda df: df.groupby("eco_region").corr_abs_mean.rank(ascending=False, method="first"))
        .sort_values(["eco_region", "order"])
    )
    top3 = top_sorted[top_sorted["order"] <= 3].copy()
    top3["index"], top3["component"] = zip(*top3["group"].map(parse_group))

    records = []
    for eco, df in top3.groupby("eco_region"):
        entry = {"eco_region": eco}
        for _, row in df.iterrows():
            rank = int(row["order"])
            entry[f"top{rank}_group"] = row["group"]
            entry[f"top{rank}_corr_mean"] = row["corr_abs_mean"]
            entry[f"top{rank}_corr_min"] = row["corr_abs_min"]
            entry[f"top{rank}_corr_max"] = row["corr_abs_max"]
            entry[f"top{rank}_n_pairs"] = row["n_pairs"]
        records.append(entry)
    top_pivot = pd.DataFrame(records)
    return region_sim.merge(top_pivot, on="eco_region", how="left")


def build_group_summary(top_groups: pd.DataFrame) -> pd.DataFrame:
    top_groups = top_groups.copy()
    top_groups["index"], top_groups["component"] = zip(*top_groups["group"].map(parse_group))
    agg = (
        top_groups
        .groupby("group")
        .agg(
            n_regions=("eco_region", "nunique"),
            corr_mean=("corr_abs_mean", "mean"),
            corr_min=("corr_abs_min", "min"),
            corr_max=("corr_abs_max", "max"),
            component=("component", "first"),
            index_name=("index", "first"),
        )
        .reset_index()
        .sort_values("corr_mean", ascending=False)
    )
    return agg


def build_component_summary(group_summary: pd.DataFrame) -> pd.DataFrame:
    comp = (
        group_summary
        .groupby("component")
        .agg(
            n_groups=("group", "count"),
            n_regions_total=("n_regions", "sum"),
            corr_mean_avg=("corr_mean", "mean"),
            corr_mean_max=("corr_mean", "max"),
        )
        .reset_index()
        .sort_values("corr_mean_avg", ascending=False)
    )
    return comp


def write_markdown_report(output_path: Path, region_summary: pd.DataFrame,
                          group_summary: pd.DataFrame, component_summary: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# Embedding–harmonic similarity summary\n")

    mean_r2 = region_summary["r2_linear_mean"].mean()
    lines.append(f"* Average mean $R^2$ across regions: {mean_r2:.3f} (all regions < 0).\n")

    if not component_summary.empty:
        top_comp = component_summary.iloc[0]
        lines.append(
            "* Dominant component: ``{component}`` (mean |r| ≈ {corr:.3f}, total region uses: {regions})."
            .format(component=top_comp["component"], corr=top_comp["corr_mean_avg"], regions=int(top_comp["n_regions_total"]))
        )

    if not group_summary.empty:
        lines.append("\n## Top harmonic groups\n")
        for _, row in group_summary.head(5).iterrows():
            lines.append(
                "* ``{group}``: mean |r| = {corr:.3f} across {n_regions} regions."
                .format(group=row["group"], corr=row["corr_mean"], n_regions=int(row["n_regions"]))
            )

    lines.append("\n## Per-region highlights\n")
    for _, row in region_summary.sort_values("r2_linear_mean").iterrows():
        eco = row["eco_region"]
        r2 = row["r2_linear_mean"]
        entries = []
        for rank in range(1, 4):
            group = row.get(f"top{rank}_group")
            corr = row.get(f"top{rank}_corr_mean")
            if isinstance(group, str) and pd.notna(corr):
                entries.append(f"top{rank}: {group} (|r|≈{corr:.3f})")
        if entries:
            lines.append(f"* **{eco}** — mean $R^2$ = {r2:.3f}; " + "; ".join(entries))
        else:
            lines.append(f"* **{eco}** — mean $R^2$ = {r2:.3f}; no groups available")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize embedding/harmonic similarity outputs")
    parser.add_argument("--input-dir", type=Path, default=Path("results/analysis_similarity"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis_similarity/summary"))
    args = parser.parse_args()

    region_sim, top_groups, coef = load_inputs(args.input_dir)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    region_summary = build_region_summary(region_sim, top_groups)
    group_summary = build_group_summary(top_groups)
    component_summary = build_component_summary(group_summary)

    region_summary.to_csv(output_dir / "summary_region.csv", index=False)
    group_summary.to_csv(output_dir / "summary_group_counts.csv", index=False)
    component_summary.to_csv(output_dir / "summary_component_counts.csv", index=False)

    write_markdown_report(output_dir / "summary_report.md", region_summary, group_summary, component_summary)

    print(f"Wrote summaries to {output_dir}")


if __name__ == "__main__":
    main()
