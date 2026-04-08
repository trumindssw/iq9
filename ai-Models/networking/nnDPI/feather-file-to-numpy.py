#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

feather_dir = './ProcessedPackets/'
out_dir = './ProcessedNumpy/'

# Create the output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# Define the byte columns we want to keep
BYTE_COLS = [str(i) for i in range(1500)]

for fname in os.listdir(feather_dir):
    if not fname.endswith('.feather'):
        continue
        
    # Read the feather file
    df = pd.read_feather(os.path.join(feather_dir, fname))
    
    # Filter columns and convert to numpy array
    cols = [c for c in BYTE_COLS if c in df.columns]
    arr = df[cols].values.astype(np.float32)
    
    # Save as .npy
    np.save(os.path.join(out_dir, fname + '.npy'), arr)
    print(f'Saved {fname} -> shape {arr.shape}')
