#!/usr/bin/env python3
"""Aggregate map comparison parquet outputs to national and eco-region metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.comparison.compare_maps import ECO_REGION_CLASSES

DLT_CONFUSION_KEYS = (
    'cm_Deciduous_vs_Broadleaved',
    'cm_Deciduous_vs_Coniferous',
    'cm_Evergreen_vs_Broadleaved',
    'cm_Evergreen_vs_Coniferous',
)

BDFORET_CONFUSION_KEYS = (
    'cm_Deciduous_vs_Deciduous',
    'cm_Deciduous_vs_Evergreen',
    'cm_Evergreen_vs_Deciduous',
    'cm_Evergreen_vs_Evergreen',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate comparison parquet files by eco-region.")
    parser.add_argument('--input', required=True, type=Path, help='Comparison parquet produced by compare_maps.py')
    parser.add_argument('--eco-map', required=True, type=Path, help='Eco-region raster (greco.tif) used during comparison')
    parser.add_argument('--ref-type', choices=['DLT', 'BDForet'], required=True, help='Reference map type')
    parser.add_argument('--output', required=True, type=Path, help='Output CSV summarising national and per eco-region metrics')
    return parser.parse_args()


def _sample_eco_id(eco_ds: rasterio.io.DatasetReader, row_off: int, col_off: int, height: int, width: int) -> int:
    center_row = row_off + height // 2
    center_col = col_off + width // 2
    window = Window(center_col - 1, center_row - 1, 3, 3)
    data = eco_ds.read(1, window=window, boundless=True, fill_value=0)
    values, counts = np.unique(data[data > 0], return_counts=True)
    if len(values) == 0:
        return 0
    return int(values[counts.argmax()])


def _load_confusion_columns(df: pd.DataFrame, ref_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if ref_type == 'DLT':
        keys = DLT_CONFUSION_KEYS
    else:
        keys = BDFORET_CONFUSION_KEYS
    try:
        a = df[keys[0]].to_numpy(dtype=np.int64)
        b = df[keys[1]].to_numpy(dtype=np.int64)
        c = df[keys[2]].to_numpy(dtype=np.int64)
        d = df[keys[3]].to_numpy(dtype=np.int64)
    except KeyError as exc:
        raise KeyError(f"Missing confusion column {exc} in input parquet")
    return a, b, c, d


def _metrics_from_counts(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> Dict[str, float]:
    A = a.sum()
    B = b.sum()
    C = c.sum()
    D = d.sum()
    total = A + B + C + D
    po = (A + D) / total if total else np.nan
    row1 = A + B
    row2 = C + D
    col1 = A + C
    col2 = B + D
    pe = ((row1 * col1) + (row2 * col2)) / (total ** 2) if total else np.nan
    kappa = (po - pe) / (1 - pe) if total and (1 - pe) else np.nan

    precision_decid = A / (A + C) if (A + C) else np.nan
    recall_decid = A / (A + B) if (A + B) else np.nan
    precision_ever = D / (B + D) if (B + D) else np.nan
    recall_ever = D / (C + D) if (C + D) else np.nan

    f1_decid = (2 * precision_decid * recall_decid / (precision_decid + recall_decid)
                if np.isfinite(precision_decid + recall_decid) and (precision_decid + recall_decid) else np.nan)
    f1_ever = (2 * precision_ever * recall_ever / (precision_ever + recall_ever)
               if np.isfinite(precision_ever + recall_ever) and (precision_ever + recall_ever) else np.nan)

    return {
        'total_pixels': int(total),
        'overall_accuracy': po,
        'kappa': kappa,
        'precision_deciduous': precision_decid,
        'recall_deciduous': recall_decid,
        'f1_deciduous': f1_decid,
        'precision_evergreen': precision_ever,
        'recall_evergreen': recall_ever,
        'f1_evergreen': f1_ever,
        'f1_macro': np.nanmean([f1_decid, f1_ever]),
    }


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)

    eco_ids = []
    with rasterio.open(args.eco_map) as eco_ds:
        for row_off, col_off, height, width in tqdm(df[['row_off', 'col_off', 'height', 'width']].itertuples(index=False),
                                                    total=len(df), desc='Assign eco-region'):
            eco_ids.append(_sample_eco_id(eco_ds, row_off, col_off, height, width))
    df = df.assign(eco_id=eco_ids)
    df['eco_region'] = df['eco_id'].map(ECO_REGION_CLASSES)

    a, b, c, d = _load_confusion_columns(df, args.ref_type)

    national_metrics = _metrics_from_counts(a, b, c, d)
    national_metrics.update({'eco_region': 'national', 'ref_type': args.ref_type})

    rows = [national_metrics]

    for eco_id, group in df.groupby('eco_id'):
        if eco_id == 0 or group.empty:
            continue
        eco_name = ECO_REGION_CLASSES.get(int(eco_id), f'eco_{eco_id}')
        ga, gb, gc, gd = _load_confusion_columns(group, args.ref_type)
        metrics = _metrics_from_counts(ga, gb, gc, gd)
        metrics.update({'eco_region': eco_name, 'ref_type': args.ref_type})
        rows.append(metrics)

    output_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
