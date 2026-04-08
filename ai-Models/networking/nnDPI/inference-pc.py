import numpy as np
import pandas as pd
import tensorflow as tf
import os
from nndpi_train import create_model
 
N_TAG_1 = 7
N_TAG_2 = 18
N_TAG_3 = 28
N_TAG_4 = 13
 
print("Building model architecture...")
model = create_model(max_len=1500, n_tag_1=N_TAG_1, n_tag_2=N_TAG_2, n_tag_3=N_TAG_3, n_tag_4=N_TAG_4)
 
print("Loading weights...")
model.load_weights('./Model/nndpi.h5', by_name=True)
print("Model loaded successfully!\n")
 
# These are the metadata columns to exclude - only keep byte columns (0-1499)
META_COLS = ['tag_1', 'tag_2', 'tag_3', 'tag_4', 'protocol', 'filename', 'ix']
BYTE_COLS = [str(i) for i in range(1500)]
 
feather_dir = './ProcessedPackets/'
for fname in sorted(os.listdir(feather_dir)):
    if not fname.endswith('.feather'):
        continue
    fpath = os.path.join(feather_dir, fname)
    df = pd.read_feather(fpath)
 
    # Drop metadata columns if present, keep only byte columns that exist
    cols_to_use = [c for c in BYTE_COLS if c in df.columns]
    df = df[cols_to_use]
 
    print(f"File: {fname} — {len(df)} packets, {len(cols_to_use)} byte columns")
 
    input_array = np.array(df.values, dtype=np.float32)
    preds = model.predict(input_array, verbose=0)
 
    t1 = np.argmax(preds[0], axis=1)
    t2 = np.argmax(preds[1], axis=1)
    t3 = np.argmax(preds[2], axis=1)
    t4 = np.argmax(preds[3], axis=1)
 
    for i in range(min(3, len(t1))):
        print(f"  Packet {i+1}:")
        print(f"    tag_1 (traffic type) : class {t1[i]} ({preds[0][i][t1[i]]*100:.1f}%)")
        print(f"    tag_2 (application)  : class {t2[i]} ({preds[1][i][t2[i]]*100:.1f}%)")
        print(f"    tag_3 (flow type)    : class {t3[i]} ({preds[2][i][t3[i]]*100:.1f}%)")
        print(f"    tag_4 (vpn status)   : class {t4[i]} ({preds[3][i][t4[i]]*100:.1f}%)")
    print()

