#!/usr/bin/env python3
"""
Step 2: Split Dataset
Split preprocessed dataset into train/validation/test sets
"""

import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

import config

def load_full_dataset():
    """Load the full preprocessed dataset"""
    dataset_path = os.path.join(config.PREPROCESSED_DIR, config.FULL_DATASET_NAME)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Run 1_preprocess_cicddos.py first!")
        sys.exit(1)
    
    print(f"📂 Loading dataset from: {dataset_path}")
    data = np.load(dataset_path)
    
    X = data['X']
    y = data['y']
    
    print(f"✅ Loaded: X={X.shape}, y={y.shape}")
    
    return X, y

def split_dataset(X, y):
    """Split into train/val/test sets"""
    print("\n" + "="*70)
    print("🔪 Splitting Dataset")
    print("="*70)
    
    print(f"\nSplit ratios:")
    print(f"  Train: {config.TRAIN_RATIO*100:.0f}%")
    print(f"  Val:   {config.VAL_RATIO*100:.0f}%")
    print(f"  Test:  {config.TEST_RATIO*100:.0f}%")
    
    # First split: train vs (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED,
        stratify=y
    )
    
    # Second split: val vs test
    val_ratio_adjusted = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio_adjusted),
        random_state=config.RANDOM_SEED,
        stratify=y_temp
    )
    
    print(f"\n📊 Split Results:")
    print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    - Benign: {np.sum(y_train==0):,}")
    print(f"    - Attack: {np.sum(y_train==1):,}")
    
    print(f"  Val:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    - Benign: {np.sum(y_val==0):,}")
    print(f"    - Attack: {np.sum(y_val==1):,}")
    
    print(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"    - Benign: {np.sum(y_test==0):,}")
    print(f"    - Attack: {np.sum(y_test==1):,}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_splits(train, val, test):
    """Save train/val/test splits"""
    print("\n" + "="*70)
    print("💾 Saving Splits")
    print("="*70)
    
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    
    # Save train
    train_path = os.path.join(config.SPLITS_DIR, config.TRAIN_DATASET_NAME)
    np.savez_compressed(train_path, X=X_train, y=y_train)
    print(f"✅ Train: {train_path} ({os.path.getsize(train_path)/(1024**3):.2f} GB)")
    
    # Save val
    val_path = os.path.join(config.SPLITS_DIR, config.VAL_DATASET_NAME)
    np.savez_compressed(val_path, X=X_val, y=y_val)
    print(f"✅ Val:   {val_path} ({os.path.getsize(val_path)/(1024**3):.2f} GB)")
    
    # Save test
    test_path = os.path.join(config.SPLITS_DIR, config.TEST_DATASET_NAME)
    np.savez_compressed(test_path, X=X_test, y=y_test)
    print(f"✅ Test:  {test_path} ({os.path.getsize(test_path)/(1024**3):.2f} GB)")
    
    # Save split info
    split_info = {
        'timestamp': datetime.now().isoformat(),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'test_samples': int(len(X_test)),
        'train_ratio': config.TRAIN_RATIO,
        'val_ratio': config.VAL_RATIO,
        'test_ratio': config.TEST_RATIO,
        'random_seed': config.RANDOM_SEED
    }
    
    info_path = os.path.join(config.LOGS_DIR, 'split_info.json')
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n📝 Split info saved: {info_path}")

def main():
    print("="*70)
    print("Step 2: Split Dataset into Train/Val/Test")
    print("="*70)
    
    # Load full dataset
    X, y = load_full_dataset()
    
    # Split
    train, val, test = split_dataset(X, y)
    
    # Save
    save_splits(train, val, test)
    
    print("\n" + "="*70)
    print("✅ Step 2 complete! Next run: python 3_analyze_dataset.py")
    print("="*70)

if __name__ == '__main__':
    main()
