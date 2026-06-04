"""
Configuration file for CIC-DDoS2019 preprocessing pipeline
Edit paths according to your setup
"""

import os

# ========================================
# Dataset Paths - UPDATE THESE!
# ========================================

# Where your downloaded dataset is
RAW_DATA_DIR = "dataset/"
PCAP_DIR = os.path.join(RAW_DATA_DIR, "PCAP-01-12_0-0249")
CSV_DIR = os.path.join(RAW_DATA_DIR, "CSV-01-12")

# Where to save processed data
OUTPUT_BASE_DIR = "./data"
PREPROCESSED_DIR = os.path.join(OUTPUT_BASE_DIR, "preprocessed")
SPLITS_DIR = os.path.join(OUTPUT_BASE_DIR, "splits")
LOGS_DIR = "./logs"

# Create directories
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ========================================
# Preprocessing Parameters
# ========================================

# Packet preprocessing
PACKET_SIZE = 1500              # Target packet size
REMOVE_ETHERNET = True          # Remove Ethernet header
MASK_IPS = True                 # Mask IP addresses

# Processing limits (None = process all)
PCAP_LIMIT = 2               # Set to 2 for testing, None for full run

# ========================================
# Dataset Split Ratios
# ========================================

TRAIN_RATIO = 0.7               # 70% training
VAL_RATIO = 0.15                # 15% validation
TEST_RATIO = 0.15               # 15% testing

# ========================================
# Output Filenames
# ========================================

FULL_DATASET_NAME = "cicddos2019_full.npz"
TRAIN_DATASET_NAME = "train.npz"
VAL_DATASET_NAME = "val.npz"
TEST_DATASET_NAME = "test.npz"

# ========================================
# Training Parameters (for step 4)
# ========================================

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
EPOCHS = 50
RANDOM_SEED = 42
