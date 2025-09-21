# -*- coding: utf-8 -*-
"""
Concatenate all Zscore_traceAll.csv files from SyncRecording* subfolders
into a single one-column CSV called 'Zscore_traceAll_combined.csv'
stored in the parent folder. No extra columns added.

Parent: G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\FiringRateExample
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np

# --------- user setting ---------
parent = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\FiringRateExample")
out_csv = parent / "Zscore_traceAll_combined.csv"
# --------------------------------

def natural_sync_key(p: Path):
    """Sort SyncRecording* folders by trailing number (SyncRecording3 < SyncRecording12)."""
    m = re.search(r"(\d+)$", p.name)
    return (int(m.group(1)) if m else 10**9, p.name)

def main():
    series_list = []
    found = 0

    folders = sorted(parent.glob("SyncRecording*"), key=natural_sync_key)
    if not folders:
        print("No SyncRecording* folders found.")
        return

    for f in folders:
        csv_path = f / "Zscore_traceAll.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path} (read error: {e})")
            continue

        # Pick 'zscore_raw' if present, otherwise the first numeric column
        if "zscore_raw" in df.columns:
            s = df["zscore_raw"]
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                print(f"Skipping {csv_path} (no numeric columns).")
                continue
            s = df[num_cols[0]]

        # Ensure 1D float series; keep NaNs if you want exact concatenation
        s = pd.Series(s.to_numpy(dtype=float), name="zscore_raw")
        series_list.append(s)
        found += 1
        print(f"Added: {csv_path}  ({len(s)} rows)")

    if not series_list:
        print("Nothing to concatenate.")
        return

    big = pd.concat(series_list, ignore_index=True).to_frame(name="zscore_raw")
    big.to_csv(out_csv, index=False)
    print(f"\nConcatenated {found} files → {out_csv}  (total rows: {len(big)})")

if __name__ == "__main__":
    main()
