#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check a GEE embeddings VRT for non-finite values and value range.

Scans the mosaic by a grid of windows and reports, per band:
- count of non-finite values (inf, -inf, NaN)
- finite min/max
- fraction outside [-1, 1]

Usage:
  python src/analysis/check_embeddings_vrt.py \
    --vrt data/embeddings/features.vrt \
    --window 1024 --stride 1024 --max-windows 200
"""

import argparse
import numpy as np
import rasterio
from rasterio.windows import Window


def scan_vrt(path: str, window_size: int, stride: int, max_windows: int):
    with rasterio.open(path) as src:
        W = src.width
        H = src.height
        nb = src.count
        stats = {
            i: {
                'nonfinite': 0,
                'finite_min': float('inf'),
                'finite_max': float('-inf'),
                'outside_unit': 0,
                'total': 0,
                'nodata': src.nodata
            } for i in range(1, nb + 1)
        }
        n_scanned = 0
        for y in range(0, H, stride):
            if n_scanned >= max_windows:
                break
            for x in range(0, W, stride):
                if n_scanned >= max_windows:
                    break
                w = Window(x, y, min(window_size, W - x), min(window_size, H - y))
                arr = src.read(window=w).astype(np.float32)  # (bands, h, w)
                h, w_ = arr.shape[1], arr.shape[2]
                if h == 0 or w_ == 0:
                    continue
                n_scanned += 1
                # per band
                for b in range(arr.shape[0]):
                    a = arr[b]
                    total = a.size
                    nonfinite = np.count_nonzero(~np.isfinite(a))
                    finite = a[np.isfinite(a)]
                    outside = np.count_nonzero((finite < -1.0) | (finite > 1.0))
                    st = stats[b + 1]
                    st['nonfinite'] += int(nonfinite)
                    st['outside_unit'] += int(outside)
                    st['total'] += int(total)
                    if finite.size > 0:
                        st['finite_min'] = float(min(st['finite_min'], finite.min()))
                        st['finite_max'] = float(max(st['finite_max'], finite.max()))

        return stats


def main():
    ap = argparse.ArgumentParser(description='Scan embeddings VRT for non-finite values and ranges')
    ap.add_argument('--vrt', default='data/embeddings/features.vrt')
    ap.add_argument('--window', type=int, default=1024)
    ap.add_argument('--stride', type=int, default=1024)
    ap.add_argument('--max-windows', type=int, default=200)
    args = ap.parse_args()

    stats = scan_vrt(args.vrt, args.window, args.stride, args.max_windows)
    print(f"Scanned up to {args.max_windows} windows of size {args.window} with stride {args.stride}")
    for b in sorted(stats.keys()):
        st = stats[b]
        tot = st['total']
        nf = st['nonfinite']
        outside = st['outside_unit']
        nf_pct = (nf / tot * 100) if tot else 0.0
        out_pct = (outside / max(1, (tot - nf)) * 100) if tot else 0.0
        print(f"Band {b:02d}: nonfinite {nf} ({nf_pct:.4f}%), finite_min {st['finite_min']:.6f}, finite_max {st['finite_max']:.6f}, outside[-1,1] {outside} ({out_pct:.4f}%), nodata={st['nodata']}")


if __name__ == '__main__':
    main()

