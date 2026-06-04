#!/usr/bin/env python3
"""
Generate Pseudo-Labels for Benign Packets using pretrained nndpi model
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Configuration
PRETRAINED_MODEL_PATH = "nndpi.h5"
FEATHER_PATH = "../nnDPI/CombinedPackets/allpkts.feather"   # ← Update if path different
OUTPUT_DIR = "data/benign_pseudo"
BATCH_SIZE = 2048

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_pretrained_model():
    print("Loading pretrained model...")
    model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
    print(f"Model loaded. Outputs: {len(model.outputs)}")
    return model

def load_benign_packets():
    print(f"Loading benign packets from: {FEATHER_PATH}")
    # Load only packet bytes
    cols = [str(i) for i in range(1500)]
    X_df = pd.read_feather(FEATHER_PATH, columns=cols)
    X = X_df.to_numpy().astype(np.uint8)
    
    print(f"Loaded {len(X):,} benign packets")
    return X

def predict_in_batches(model, X, batch_size=BATCH_SIZE):
    n_samples = len(X)
    tag1_list, tag2_list, tag3_list, tag4_list = [], [], [], []
    
    print(f"Predicting tags for {n_samples:,} benign packets...")
    
    for i in tqdm(range(0, n_samples, batch_size)):
        batch = X[i:i+batch_size]
        preds = model.predict(batch, verbose=0)
        
        tag1_list.append(np.argmax(preds[0], axis=1))
        tag2_list.append(np.argmax(preds[1], axis=1))
        tag3_list.append(np.argmax(preds[2], axis=1))
        tag4_list.append(np.argmax(preds[3], axis=1))
    
    return (np.concatenate(tag1_list), np.concatenate(tag2_list),
            np.concatenate(tag3_list), np.concatenate(tag4_list))

def main():
    model = load_pretrained_model()
    X_benign = load_benign_packets()
    
    tag1, tag2, tag3, tag4 = predict_in_batches(model, X_benign)
    
    # tag_5 = 0 for all benign
    tag5 = np.zeros(len(X_benign), dtype=np.int32)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, "benign_pseudo_labeled.npz")
    np.savez_compressed(output_path,
                       X=X_benign,
                       tag_1=tag1,
                       tag_2=tag2,
                       tag_3=tag3,
                       tag_4=tag4,
                       tag_5=tag5)
    
    print("\n✅ Benign pseudo-labels generated successfully!")
    print(f"Saved to: {output_path}")
    print(f"tag_1 unique: {np.unique(tag1)} → {len(np.unique(tag1))} classes")
    print(f"tag_2 unique: {len(np.unique(tag2))} classes")
    print(f"tag_3 unique: {len(np.unique(tag3))} classes")
    print(f"tag_4 unique: {len(np.unique(tag4))} classes")
    
    print("\nNext step: Run the new balancing script.")

if __name__ == "__main__":
    main()
