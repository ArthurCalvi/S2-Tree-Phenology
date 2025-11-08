import pandas as pd
import sys
import os

path = 'results/datasets/training_datasets_pixels.parquet'
print(f"Checking {path}...")

if not os.path.exists(path):
    print(f"Error: File not found at {path}", file=sys.stderr)
    sys.exit(1)

try:
    df = pd.read_parquet(path)
    print("Columns found:", list(df.columns))
    cols_to_check = ['genus', 'species']
    for col in cols_to_check:
        if col in df.columns:
            # Handle potential NaN values before calling unique()
            col_series = df[col].dropna()
            unique_vals = col_series.unique()
            n_unique = len(unique_vals)
            print(f"\nColumn '{col}': Exists")
            print(f"  Number of unique values: {n_unique}")
            # Safely handle potential non-string unique values for display
            sample_vals_str = [str(v) for v in unique_vals[:10]]
            print(f"  Sample unique values: {sample_vals_str}" + ('...' if n_unique > 10 else ''))
        else:
            print(f"\nColumn '{col}': NOT FOUND")
except Exception as e:
    print(f"Error loading or processing {path}: {e}", file=sys.stderr)
    sys.exit(1) 