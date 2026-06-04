#!/usr/bin/env python3
"""
Step 3: Analyze Dataset - FAST VERSION
Generate statistics and visualizations
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import config

def load_splits_info_only():
    """Load only metadata, not full arrays"""
    print("="*70)
    print("📊 Loading Dataset Metadata")
    print("="*70)
    
    train_path = os.path.join(config.SPLITS_DIR, config.TRAIN_DATASET_NAME)
    val_path = os.path.join(config.SPLITS_DIR, config.VAL_DATASET_NAME)
    test_path = os.path.join(config.SPLITS_DIR, config.TEST_DATASET_NAME)
    
    # Check if files exist
    for name, path in [("Train", train_path), ("Val", val_path), ("Test", test_path)]:
        if not os.path.exists(path):
            print(f"❌ {name} dataset not found: {path}")
            print("   Run 2_split_dataset.py first!")
            sys.exit(1)
    
    # Load only labels (much faster than loading full X arrays)
    print("\n📂 Loading labels only (fast)...")
    train = np.load(train_path)
    val = np.load(val_path)
    test = np.load(test_path)
    
    print("✅ Loaded!")
    
    return train, val, test

def analyze_labels(train, val, test):
    """Analyze label distribution"""
    print("\n" + "="*70)
    print("📊 Label Distribution Analysis")
    print("="*70)
    
    splits = [
        ('Train', train['y']),
        ('Val', val['y']),
        ('Test', test['y'])
    ]
    
    for name, y in splits:
        benign = np.sum(y == 0)
        attack = np.sum(y == 1)
        total = len(y)
        
        print(f"\n{name}:")
        print(f"  Total: {total:,}")
        print(f"  Benign: {benign:,} ({benign/total*100:.2f}%)")
        print(f"  Attack: {attack:,} ({attack/total*100:.2f}%)")
        print(f"  Class balance ratio: {benign/attack:.2f}:1")

def analyze_packet_sizes_fast(train):
    """Analyze actual packet sizes (sample efficiently)"""
    print("\n" + "="*70)
    print("📏 Packet Size Analysis")
    print("="*70)
    
    # Sample efficiently - load only what we need
    X = train['X']
    total_packets = len(X)
    
    # Sample at most 1000 packets for size analysis
    sample_size = min(1000, total_packets)
    
    print(f"\nAnalyzing {sample_size} random packets...")
    
    # Random indices
    np.random.seed(42)
    indices = np.random.choice(total_packets, sample_size, replace=False)
    
    actual_sizes = []
    for idx in indices:
        packet = X[idx]
        # Find actual size (before padding)
        non_zero = np.where(packet != 0)[0]
        if len(non_zero) > 0:
            actual_size = non_zero[-1] + 1
        else:
            actual_size = 0
        actual_sizes.append(actual_size)
    
    actual_sizes = np.array(actual_sizes)
    
    print(f"\nActual packet sizes (before padding to {config.PACKET_SIZE}):")
    print(f"  Min:     {np.min(actual_sizes)} bytes")
    print(f"  Max:     {np.max(actual_sizes)} bytes")
    print(f"  Mean:    {np.mean(actual_sizes):.1f} bytes")
    print(f"  Median:  {np.median(actual_sizes):.1f} bytes")
    print(f"  Std Dev: {np.std(actual_sizes):.1f} bytes")

def plot_distributions(train, val, test):
    """Create visualization plots"""
    print("\n" + "="*70)
    print("📈 Creating Visualizations")
    print("="*70)
    
    try:
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with 1 row, 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        splits = [
            ('Train', train['y'], axes[0]),
            ('Val', val['y'], axes[1]),
            ('Test', test['y'], axes[2])
        ]
        
        colors = ['#2ecc71', '#e74c3c']  # Green for benign, Red for attack
        
        for name, y, ax in splits:
            benign = np.sum(y == 0)
            attack = np.sum(y == 1)
            
            bars = ax.bar(['Benign', 'Attack'], [benign, attack], 
                         color=colors, alpha=0.8, edgecolor='black')
            
            ax.set_title(f'{name} Set', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Packets', fontsize=12)
            ax.ticklabel_format(axis='y', style='plain')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('CIC-DDoS2019 Dataset - Label Distribution', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(config.LOGS_DIR, 'label_distribution.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved plot: {plot_path}")
        
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")

def print_summary(train, val, test):
    """Print final summary"""
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    
    total_packets = len(train['y']) + len(val['y']) + len(test['y'])
    total_benign = np.sum(train['y']==0) + np.sum(val['y']==0) + np.sum(test['y']==0)
    total_attack = np.sum(train['y']==1) + np.sum(val['y']==1) + np.sum(test['y']==1)
    
    print(f"\n📦 Overall Dataset:")
    print(f"   Total packets: {total_packets:,}")
    print(f"   Benign: {total_benign:,} ({total_benign/total_packets*100:.1f}%)")
    print(f"   Attack: {total_attack:,} ({total_attack/total_packets*100:.1f}%)")
    
    print(f"\n💾 Dataset Files:")
    train_path = os.path.join(config.SPLITS_DIR, config.TRAIN_DATASET_NAME)
    val_path = os.path.join(config.SPLITS_DIR, config.VAL_DATASET_NAME)
    test_path = os.path.join(config.SPLITS_DIR, config.TEST_DATASET_NAME)
    
    train_size = os.path.getsize(train_path) / (1024**3)
    val_size = os.path.getsize(val_path) / (1024**3)
    test_size = os.path.getsize(test_path) / (1024**3)
    
    print(f"   Train: {train_size:.2f} GB")
    print(f"   Val:   {val_size:.2f} GB")
    print(f"   Test:  {test_size:.2f} GB")
    print(f"   Total: {train_size+val_size+test_size:.2f} GB")
    
    print(f"\n📊 Files ready for training!")
    print(f"   Location: {config.SPLITS_DIR}")

def main():
    print("="*70)
    print("Step 3: Dataset Analysis (Fast Version)")
    print("="*70)
    
    # Load splits (labels only for speed)
    print("\n⚡ Loading data efficiently...")
    train, val, test = load_splits_info_only()
    
    # Analyze labels
    analyze_labels(train, val, test)
    
    # Analyze packet sizes (small sample)
    print("\n⚡ Sampling packets for size analysis...")
    analyze_packet_sizes_fast(train)
    
    # Create plots
    print("\n⚡ Generating plots...")
    plot_distributions(train, val, test)
    
    # Print summary
    print_summary(train, val, test)
    
    print("\n" + "="*70)
    print("✅ Step 3 complete! Dataset is ready for training.")
    print("   Next: Implement training script (4_train_nndpi.py)")
    print("="*70)

if __name__ == '__main__':
    main()
