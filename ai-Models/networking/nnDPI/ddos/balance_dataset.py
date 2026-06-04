#!/usr/bin/env python3
"""
FINAL BALANCING - Accounting for benign in augmented + 38% target + 70/20/10 split
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

# ========================= CONFIG =========================
ATTACK_DIR = "data/augmented"
BENIGN_PSEUDO_PATH = "data/benign_pseudo/benign_pseudo_labeled.npz"
OUTPUT_DIR = "data/balanced_final_38_corrected"

TARGET_BENIGN_RATIO = 0.38      # 38% benign overall
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

os.makedirs(OUTPUT_DIR, exist_ok=True)
EXPECTED_CLASSES = [7, 18, 28, 13]

def standardize_labels(tag, max_classes):
    return np.clip(tag, 0, max_classes - 1)

def main():
    print("="*100)
    print("🏗️ FINAL CORRECTED BALANCING (Considering benign in augmented)")
    print("="*100)

    # === Load Benign from Pseudo (33k+) ===
    b_pseudo = np.load(BENIGN_PSEUDO_PATH)
    X_b_pseudo = b_pseudo['X']
    t1_b_p = standardize_labels(b_pseudo['tag_1'], EXPECTED_CLASSES[0])
    t2_b_p = standardize_labels(b_pseudo['tag_2'], EXPECTED_CLASSES[1])
    t3_b_p = standardize_labels(b_pseudo['tag_3'], EXPECTED_CLASSES[2])
    t4_b_p = standardize_labels(b_pseudo['tag_4'], EXPECTED_CLASSES[3])
    t5_b_p = b_pseudo['tag_5']

    print(f"Benign from pseudo: {len(X_b_pseudo):,}")

    # === Load ALL data from augmented (contains ~3800 benign + many attacks) ===
    X_a_list, t1_a_list, t2_a_list, t3_a_list, t4_a_list, t5_a_list = [], [], [], [], [], []

    total_benign_in_aug = 0

    for split in ['train', 'val', 'test']:
        path = os.path.join(ATTACK_DIR, f"{split}_augmented.npz")
        d = np.load(path)
        X_a_list.append(d['X'])
        t1_a_list.append(standardize_labels(d['tag_1'], EXPECTED_CLASSES[0]))
        t2_a_list.append(standardize_labels(d['tag_2'], EXPECTED_CLASSES[1]))
        t3_a_list.append(standardize_labels(d['tag_3'], EXPECTED_CLASSES[2]))
        t4_a_list.append(standardize_labels(d['tag_4'], EXPECTED_CLASSES[3]))
        t5_a_list.append(d['tag_5'])
        
        benign_in_this = np.sum(d['tag_5'] == 0)
        total_benign_in_aug += benign_in_this
        print(f"  {split}: {len(d['X']):,} total | Benign: {benign_in_this:,}")

    # Combine all augmented data
    X_aug_all = np.vstack(X_a_list)
    t1_aug = np.concatenate(t1_a_list)
    t2_aug = np.concatenate(t2_a_list)
    t3_aug = np.concatenate(t3_a_list)
    t4_aug = np.concatenate(t4_a_list)
    t5_aug = np.concatenate(t5_a_list)

    print(f"\nTotal in augmented: {len(X_aug_all):,}")
    print(f"Benign in augmented: {total_benign_in_aug:,}")

    # === Combine ALL benign ===
    X_b_all = np.vstack([X_aug_all[t5_aug == 0], X_b_pseudo])
    t1_b_all = np.concatenate([t1_aug[t5_aug == 0], t1_b_p])
    t2_b_all = np.concatenate([t2_aug[t5_aug == 0], t2_b_p])
    t3_b_all = np.concatenate([t3_aug[t5_aug == 0], t3_b_p])
    t4_b_all = np.concatenate([t4_aug[t5_aug == 0], t4_b_p])
    t5_b_all = np.concatenate([t5_aug[t5_aug == 0], t5_b_p])

    print(f"Total Benign (both sources): {len(X_b_all):,}")

    # === Sample Attacks ===
    X_attack_all = X_aug_all[t5_aug == 1]
    t1_attack = t1_aug[t5_aug == 1]
    t2_attack = t2_aug[t5_aug == 1]
    t3_attack = t3_aug[t5_aug == 1]
    t4_attack = t4_aug[t5_aug == 1]
    t5_attack = t5_aug[t5_aug == 1]

    n_benign_total = len(X_b_all)
    n_attack_needed = int(n_benign_total * (1 - TARGET_BENIGN_RATIO) / TARGET_BENIGN_RATIO)

    print(f"\nAttacks needed for ~{TARGET_BENIGN_RATIO*100:.0f}% benign: {n_attack_needed:,}")

    np.random.seed(42)
    attack_idx = np.random.choice(len(X_attack_all), n_attack_needed, replace=False)

    X_a = X_attack_all[attack_idx]
    t1_a = t1_attack[attack_idx]
    t2_a = t2_attack[attack_idx]
    t3_a = t3_attack[attack_idx]
    t4_a = t4_attack[attack_idx]
    t5_a = t5_attack[attack_idx]

    # === Final Combined Dataset ===
    X_all = np.vstack([X_a, X_b_all])
    tag1_all = np.concatenate([t1_a, t1_b_all])
    tag2_all = np.concatenate([t2_a, t2_b_all])
    tag3_all = np.concatenate([t3_a, t3_b_all])
    tag4_all = np.concatenate([t4_a, t4_b_all])
    tag5_all = np.concatenate([t5_a, t5_b_all])

    print(f"\nFinal Dataset Size: {len(X_all):,} packets")
    print(f"Final Benign Ratio: {np.sum(tag5_all==0)/len(tag5_all)*100:.2f}%")

    # === 70 / 20 / 10 Split ===
    print("\n🔀 Creating 70% Train | 20% Val | 10% Test split...")

    X_train, X_temp, *labels = train_test_split(
        X_all, tag1_all, tag2_all, tag3_all, tag4_all, tag5_all,
        test_size=(1 - TRAIN_RATIO), random_state=42, stratify=tag5_all
    )

    X_val, X_test, *labels_val_test = train_test_split(
        X_temp, *labels[1::2],   # take the temp labels
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), 
        random_state=42, stratify=labels[-1]   # stratify on tag5
    )

    # Reconstruct labels for val and test (this part is a bit tricky, better to use indices or restructure)
    # Simpler way below:

    # Better implementation:
    indices = np.arange(len(X_all))
    train_idx, temp_idx = train_test_split(indices, test_size=(1-TRAIN_RATIO), random_state=42, stratify=tag5_all)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=tag5_all[temp_idx])

    splits = {
        'train': (X_all[train_idx], tag1_all[train_idx], tag2_all[train_idx], tag3_all[train_idx], tag4_all[train_idx], tag5_all[train_idx]),
        'val':   (X_all[val_idx],   tag1_all[val_idx],   tag2_all[val_idx],   tag3_all[val_idx],   tag4_all[val_idx],   tag5_all[val_idx]),
        'test':  (X_all[test_idx],  tag1_all[test_idx],  tag2_all[test_idx],  tag3_all[test_idx],  tag4_all[test_idx],  tag5_all[test_idx])
    }

    for name, (X, t1, t2, t3, t4, t5) in splits.items():
        benign_pct = np.sum(t5 == 0) / len(t5) * 100
        path = os.path.join(OUTPUT_DIR, f"{name}_balanced.npz")
        np.savez_compressed(path, X=X, tag_1=t1, tag_2=t2, tag_3=t3, tag_4=t4, tag_5=t5)
        print(f"✅ {name.upper():5} : {len(X):,} packets | Benign: {benign_pct:.1f}%")

    print("\n🎉 Done! Dataset ready.")

if __name__ == "__main__":
    main()
