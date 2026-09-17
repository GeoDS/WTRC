"""Combine the split .npy flow files into the single CSV the analysis reads.

The flow dataset is committed in chunks to stay under file size limits. Run this
once to rebuild `data/WICTs_allyears.csv`, which the example notebooks expect.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "flows"
OUTPUT_CSV = HERE / "WICTs_allyears.csv"
COLUMNS = ["geoid_o", "geoid_d", "weight", "t", "i", "j"]

chunks = []
for file_name in sorted(os.listdir(DATA_DIR)):
    if not file_name.endswith(".npy"):
        continue
    file_path = os.path.join(DATA_DIR, file_name)
    print(f"Loading {file_path}...")
    chunks.append(pd.DataFrame(np.load(file_path), columns=COLUMNS))

combined = pd.concat(chunks, ignore_index=True)
combined.to_csv(OUTPUT_CSV, index=False)
print(f"Combined {len(chunks)} files into {OUTPUT_CSV} ({len(combined):,} rows)")
