import numpy as np
import tensorflow as tf
import os

print("Loading TFLite model...")
                                                                     
interpreter = tf.lite.Interpreter(model_path="cnn_dpi_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded!")
print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])
print()

npy_dir = "nnDPI/ProcessedNumpy/"

for fname in sorted(os.listdir(npy_dir)):

    if not fname.endswith(".npy"):
        continue

    arr = np.load(os.path.join(npy_dir, fname))

    print(f"File: {fname} — {arr.shape[0]} packets")

    for i in range(min(10, arr.shape[0])):

        packet = arr[i].reshape(1,1500,1).astype("float32")

        interpreter.set_tensor(input_details[0]['index'], packet)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]

        cls = np.argmax(output)
        conf = output[cls] * 100

        print(f"  Packet {i+1}: class {cls} ({conf:.1f}%)")

    print()

