#!/usr/bin/env python3
"""
Phase 2: Transfer Learning - Add tag_5 to Pretrained nnDPI (FIXED)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import json
from datetime import datetime

# ========================= CONFIG =========================
# ========================= CONFIG =========================
PRETRAINED_MODEL_PATH = "nndpi.h5"
AUGMENTED_DIR = "data/balanced_final_38_corrected"
TRAIN_FILE = "train_balanced.npz"
VAL_FILE = "val_balanced.npz"
TEST_FILE = "test_balanced.npz"

OUTPUT_DIR = "models"
LOGS_DIR = "logs"
MAX_LEN = 1500
BATCH_SIZE = 1024
EPOCHS = 30
INITIAL_LR = 3e-5
DROPOUT_RATE = 0.35

# NEW: Force correct number of classes (Very Important)
EXPECTED_CLASSES = [7, 18, 28, 13]   # tag_1, tag_2, tag_3, tag_4

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
# =========================================================

def load_balanced_data():
    print("="*70)
    print("📂 Loading Balanced Data")
    print("="*70)
    
    data = {}
    for split_name, fname in [('train', TRAIN_FILE), ('val', VAL_FILE), ('test', TEST_FILE)]:
        path = os.path.join(AUGMENTED_DIR, fname)
        d = np.load(path)
        
        data[split_name] = {
            'X': d['X'].astype(np.uint8),
            # Clip labels to prevent "out of range" error
            'tag_1': np.clip(d['tag_1'], 0, EXPECTED_CLASSES[0]-1),
            'tag_2': np.clip(d['tag_2'], 0, EXPECTED_CLASSES[1]-1),
            'tag_3': np.clip(d['tag_3'], 0, EXPECTED_CLASSES[2]-1),
            'tag_4': np.clip(d['tag_4'], 0, EXPECTED_CLASSES[3]-1),
            'tag_5': d['tag_5']
        }
        
        print(f"   {split_name}: {d['X'].shape} | tag_5: {np.bincount(d['tag_5'])}")
    
    print(f"\n✅ Using fixed class counts: {EXPECTED_CLASSES}")
    return data   # No need for label_mappings anymore
def create_transfer_model(pretrained_path, n_classes):
    """Load pretrained model and add new tag_5 head"""
    print("\n" + "="*70)
    print("🏗️  Creating Transfer Learning Model")
    print("="*70)
    
    base_model = tf.keras.models.load_model(pretrained_path)
    print(f"✅ Loaded pretrained nnDPI with {len(base_model.outputs)} outputs")
    
    # Print layer names to help debugging (optional)
    # for i, layer in enumerate(base_model.layers[-10:]):
    #     print(i, layer.name)
    
    # Take the feature vector (before original output heads)
    # Usually the layer before the first Dense/Output
    features = base_model.layers[-5].output  
    
    # New shared dense layers
    x = Dense(128, activation='relu', name='transfer_dense_1')(features)
    x = BatchNormalization(name='transfer_bn_1')(x)
    x = Dropout(DROPOUT_RATE, name='transfer_dropout_1')(x)
    
    # New heads with unique names
    out1 = Dense(n_classes[0], activation='softmax', name='tag_1')(x)
    out2 = Dense(n_classes[1], activation='softmax', name='tag_2')(x)
    out3 = Dense(n_classes[2], activation='softmax', name='tag_3')(x)
    out4 = Dense(n_classes[3], activation='softmax', name='tag_4')(x)
    out5 = Dense(2, activation='softmax', name='tag_5')(x)
    
    model = Model(inputs=base_model.input, outputs=[out1, out2, out3, out4, out5])
    
# Freeze pretrained backbone, train all new heads
    for layer in model.layers:

    # Train newly added transfer layers
        if layer.name.startswith('transfer_'):
            layer.trainable = True

    # Train all output heads
        elif layer.name.startswith('tag_'):
            layer.trainable = True

    # Freeze pretrained nnDPI feature extractor
        else:
            layer.trainable = False
    
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    return model

def main():
    data = load_balanced_data()   # ← No more mappings needed
    
    n_classes = EXPECTED_CLASSES[:]   # Use fixed values
    
    model = create_transfer_model(PRETRAINED_MODEL_PATH, n_classes)
    
    loss_weights = {
        'tag_1': 0.15, 'tag_2': 0.15, 'tag_3': 0.15, 
        'tag_4': 0.15, 'tag_5': 0.40
    }
    # ... rest of the code remains same

    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='sparse_categorical_crossentropy',
        loss_weights=loss_weights,
        metrics=['accuracy']
    )
    
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_tag_5_accuracy', patience=10, restore_best_weights=True, mode='max', verbose=1),
        callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, "ddos_5tag_transfer_best.h5"),
                                monitor='val_tag_5_accuracy', save_best_only=True, mode='max', verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)
    ]
    
    print("\n🚀 Starting Transfer Learning...")
    history = model.fit(
        x=data['train']['X'],
        y={
            'tag_1': data['train']['tag_1'],
            'tag_2': data['train']['tag_2'],
            'tag_3': data['train']['tag_3'],
            'tag_4': data['train']['tag_4'],
            'tag_5': data['train']['tag_5']
        },
        validation_data=(
            data['val']['X'],
            {
                'tag_1': data['val']['tag_1'],
                'tag_2': data['val']['tag_2'],
                'tag_3': data['val']['tag_3'],
                'tag_4': data['val']['tag_4'],
                'tag_5': data['val']['tag_5']
            }
        ),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "ddos_5tag_transfer_final.h5")
    model.save(final_path)
    print(f"\n✅ Model saved: {final_path}")
    
    print("\n🎯 Next step: Convert to SNPE .dlc for Qualcomm board")

if __name__ == "__main__":
    main()
