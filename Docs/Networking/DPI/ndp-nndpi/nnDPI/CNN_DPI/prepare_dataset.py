import os
import numpy as np

DATA_DIR = "ProcessedNumpy"

label_map = {
    "aim":0,
    "facebook":1,
    "gmail":2,
    "icq":3,
    "skype":4,
    "spotify":5,
    "scp":6,
    "email":7,
    "vpn":8
}

X = []
y = []

for fname in os.listdir(DATA_DIR):

    if not fname.endswith(".npy"):
        continue

    path = os.path.join(DATA_DIR, fname)

    data = np.load(path)

    label = None
    for key in label_map:
        if key in fname.lower():
            label = label_map[key]
            break

    if label is None:
        continue

    labels = np.full(data.shape[0], label)

    X.append(data)
    y.append(labels)

X = np.concatenate(X)
y = np.concatenate(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

np.save("X.npy", X)
np.save("y.npy", y)
