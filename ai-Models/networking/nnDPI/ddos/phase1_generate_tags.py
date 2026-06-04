#!/usr/bin/env python3
"""
Phase 1: Generate Pseudo-Labels for DDoS Dataset
Use pretrained nndpi model to predict tag_1 to tag_4 for all DDoS packets
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import json

# Configuration
PRETRAINED_MODEL_PATH = "nndpi.h5"  # Your pretrained nndpi model
SPLITS_DIR = "data/splits"
OUTPUT_DIR = "data/augmented"
BATCH_SIZE = 2048  # Adjust based on your GPU memory

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_pretrained_model():
    """Load the pretrained nndpi model"""
    print("="*70)
    print("📦 Loading Pretrained nndpi Model")
    print("="*70)
    
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"❌ Pretrained model not found: {PRETRAINED_MODEL_PATH}")
        print("   Please place your trained dpi.h5 file in the current directory")
        sys.exit(1)
    
    model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
    print(f"✅ Model loaded successfully")
    print(f"   Inputs: {model.input_shape}")
    print(f"   Outputs: {[out.shape for out in model.output]}")
    
    return model

def load_split_data(split_name):
    """Load train/val/test split"""
    split_path = os.path.join(SPLITS_DIR, f"{split_name}.npz")
    
    if not os.path.exists(split_path):
        print(f"❌ Split not found: {split_path}")
        sys.exit(1)
    
    data = np.load(split_path)
    X = data['X']
    y = data['y']  # This is the benign/attack label
    
    print(f"✅ Loaded {split_name}: X={X.shape}, y={y.shape}")
    
    return X, y

def predict_tags_in_batches(model, X, batch_size=BATCH_SIZE):
    """
    Predict tag_1 to tag_4 for all packets in batches
    Returns: tag_1, tag_2, tag_3, tag_4 arrays (predicted classes)
    """
    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Initialize arrays to store predictions
    tag_1_preds = []
    tag_2_preds = []
    tag_3_preds = []
    tag_4_preds = []
    
    print(f"\n🔮 Predicting tags for {n_samples:,} packets in {n_batches} batches...")
    
    for i in tqdm(range(0, n_samples, batch_size), desc="Inference"):
        batch_X = X[i:i+batch_size]
        
        # Model outputs: [tag_1, tag_2, tag_3, tag_4] (softmax probabilities)
        predictions = model.predict(batch_X, verbose=0)
        
        # Get predicted class (argmax) for each tag
        tag_1_preds.append(np.argmax(predictions[0], axis=1))
        tag_2_preds.append(np.argmax(predictions[1], axis=1))
        tag_3_preds.append(np.argmax(predictions[2], axis=1))
        tag_4_preds.append(np.argmax(predictions[3], axis=1))
    
    # Concatenate all batches
    tag_1 = np.concatenate(tag_1_preds)
    tag_2 = np.concatenate(tag_2_preds)
    tag_3 = np.concatenate(tag_3_preds)
    tag_4 = np.concatenate(tag_4_preds)
    
    return tag_1, tag_2, tag_3, tag_4

def save_augmented_data(X, y, tag_1, tag_2, tag_3, tag_4, split_name):
    """
    Save augmented dataset with all 5 labels
    X: packet bytes
    y: attack/benign (tag_5)
    tag_1 to tag_4: predicted from pretrained model
    """
    output_path = os.path.join(OUTPUT_DIR, f"{split_name}_augmented.npz")
    
    np.savez_compressed(
        output_path,
        X=X,
        tag_1=tag_1,
        tag_2=tag_2,
        tag_3=tag_3,
        tag_4=tag_4,
        tag_5=y  # attack/benign label
    )
    
    file_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"✅ Saved augmented {split_name}: {output_path} ({file_size_gb:.2f} GB)")
    
    return output_path

def analyze_predicted_tags(tag_1, tag_2, tag_3, tag_4, y, split_name):
    """Analyze the distribution of predicted tags"""
    print(f"\n📊 Tag Distribution Analysis for {split_name}:")
    print(f"  tag_1: {len(np.unique(tag_1))} unique classes")
    print(f"  tag_2: {len(np.unique(tag_2))} unique classes")
    print(f"  tag_3: {len(np.unique(tag_3))} unique classes")
    print(f"  tag_4: {len(np.unique(tag_4))} unique classes")
    
    # Show distribution per attack/benign
    print(f"\n  Attack packets: {np.sum(y==1):,}")
    print(f"    tag_1 distribution: {np.bincount(tag_1[y==1])[:5]}")
    print(f"  Benign packets: {np.sum(y==0):,}")
    print(f"    tag_1 distribution: {np.bincount(tag_1[y==0])[:5]}")

def save_metadata(model, splits_info):
    """Save metadata about the augmentation process"""
    metadata = {
        'pretrained_model': PRETRAINED_MODEL_PATH,
        'model_architecture': {
            'input_shape': str(model.input_shape),
            'output_shapes': [str(out.shape) for out in model.output]
        },
        'splits': splits_info
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, 'augmentation_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📝 Metadata saved: {metadata_path}")

def main():
    print("="*70)
    print("Phase 1: Generate Pseudo-Labels (tag_1 to tag_4)")
    print("="*70)
    
    # Load pretrained model
    model = load_pretrained_model()
    
    splits_info = {}
    
    # Process each split: train, val, test
    for split_name in ['train', 'val', 'test']:
        print("\n" + "="*70)
        print(f"Processing {split_name.upper()} split")
        print("="*70)
        
        # Load split data
        X, y = load_split_data(split_name)
        
        # Predict tags 1-4 using pretrained model
        tag_1, tag_2, tag_3, tag_4 = predict_tags_in_batches(model, X)
        
        # Analyze predictions
        analyze_predicted_tags(tag_1, tag_2, tag_3, tag_4, y, split_name)
        
        # Save augmented data
        output_path = save_augmented_data(X, y, tag_1, tag_2, tag_3, tag_4, split_name)
        
        splits_info[split_name] = {
            'samples': len(X),
            'benign': int(np.sum(y==0)),
            'attack': int(np.sum(y==1)),
            'output_file': output_path
        }
    
    # Save metadata
    save_metadata(model, splits_info)
    
    print("\n" + "="*70)
    print("✅ Phase 1 Complete!")
    print("="*70)
    print("\n📦 Augmented datasets created:")
    print(f"   Location: {OUTPUT_DIR}/")
    print(f"   Files: train_augmented.npz, val_augmented.npz, test_augmented.npz")
    print("\n📊 Each file contains:")
    print("   - X: packet bytes (1500 bytes)")
    print("   - tag_1 to tag_4: predicted from pretrained nndpi")
    print("   - tag_5: ground truth attack/benign label")
    print("\n➡️  Next: Run phase2_retrain_5tags.py to train the 5-output model")
    print("="*70)

if __name__ == '__main__':
    main()