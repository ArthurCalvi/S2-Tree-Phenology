#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanitize embeddings parquet in-place-like: clip embedding_* to [-1,1] and
replace non-finite (inf, -inf, NaN) with 0.0, writing to a new parquet.

Usage:
  python src/dataset_creation/sanitize_embeddings_parquet.py \
    --input results/datasets/training_datasets_pixels_embedding.parquet \
    --output results/datasets/training_datasets_pixels_embedding.parquet

If input == output, a temporary file is written and then moved into place.
"""

import argparse
import os
import tempfile
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    emb_cols = [c for c in df.columns if c.startswith('embedding_')]
    if not emb_cols:
        return df
    df[emb_cols] = df[emb_cols].replace([float('inf'), float('-inf')], 0.0)
    df[emb_cols] = df[emb_cols].clip(-1.0, 1.0)
    df[emb_cols] = df[emb_cols].fillna(0.0)
    return df


def main():
    ap = argparse.ArgumentParser(description='Sanitize embeddings parquet values')
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    same_target = os.path.abspath(args.input) == os.path.abspath(args.output)
    out_path = args.output
    if same_target:
        # write to a temporary file in same directory for atomic replace
        tmp_dir = os.path.dirname(args.output)
        fd, tmp_path = tempfile.mkstemp(prefix='embeddings_clean_', suffix='.parquet', dir=tmp_dir)
        os.close(fd)
        out_path = tmp_path

    dataset = ds.dataset(args.input, format='parquet')
    writer = None
    for batch in dataset.to_batches():
        pdf = batch.to_pandas()
        pdf = sanitize_df(pdf)
        table = pa.Table.from_pandas(pdf, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()

    if same_target:
        os.replace(out_path, args.output)


if __name__ == '__main__':
    main()

