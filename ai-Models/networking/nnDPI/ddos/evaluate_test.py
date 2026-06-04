#!/usr/bin/env python3
"""
Evaluate Trained Model on Test Dataset
"""

import os
import numpy as np
import tensorflow as tf
import json
from datetime import datetime

# ========================= CONFIG =========================
MODEL_PATH = "models/ddos_5tag_transfer_final.h5"   # Change to final.h5 if you prefer
TEST_DATA_DIR = "data/balanced_final_38_corrected"
TEST_FILE = "test_balanced.npz"
BATCH_SIZE = 1024

# =========================================================

def load_test_data():
    print("="*70)
    print("📂 Loading Test Dataset")
    print("="*70)
    
    path = os.path.join(TEST_DATA_DIR, TEST_FILE)
    
    if not os.path.exists(path):
        print(f"❌ Test file not found: {path}")
        print("   Make sure you ran the balancing script correctly.")
        exit(1)
    
    data = np.load(path)
    
    print(f"Test samples: {len(data['X']):,}")
    print(f"Benign (0): {np.sum(data['tag_5']==0):,}")
    print(f"Attack (1): {np.sum(data['tag_5']==1):,}")
    
    return {
        'X': data['X'].astype(np.uint8),
        'tag_1': data['tag_1'],
        'tag_2': data['tag_2'],
        'tag_3': data['tag_3'],
        'tag_4': data['tag_4'],
        'tag_5': data['tag_5']
    }

def main():
    # Load test data
    test_data = load_test_data()
    
    # Load trained model
    print("\n" + "="*70)
    print("🏷️  Loading Trained Model")
    print("="*70)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Train the model first using phase2_retrain_5tags.py")
        exit(1)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully: {MODEL_PATH}")
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("🚀 Evaluating on Test Set")
    print("="*70)
    
    test_results = model.evaluate(
        x=test_data['X'],
        y={
            'tag_1': test_data['tag_1'],
            'tag_2': test_data['tag_2'],
            'tag_3': test_data['tag_3'],
            'tag_4': test_data['tag_4'],
            'tag_5': test_data['tag_5']
        },
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Print results nicely
    metric_names = model.metrics_names
    print("\n" + "="*70)
    print("📊 FINAL TEST RESULTS")
    print("="*70)
    
    results_dict = {}
    for name, value in zip(metric_names, test_results):
        print(f"   {name:35} : {value:.4f}")
        results_dict[name] = float(value)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"logs/test_results_{timestamp}.json"
    
    with open(result_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✅ Test results saved to: {result_path}")
    print("\n🎯 Key Metrics to Check:")
    print(f"   tag_5_accuracy     : {results_dict.get('tag_5_accuracy', 'N/A'):.4f}")
    print(f"   val_tag_5_accuracy : Compare with your validation score")

if __name__ == "__main__":
    main()
