import numpy as np
import tensorflow as tf
import os

print("Loading INT8 TFLite model...")

interpreter = tf.lite.Interpreter(model_path="cnn_dpi_model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])

# quantization parameters
input_scale, input_zero = input_details[0]['quantization']

npy_dir = "nnDPI/ProcessedNumpy"

for fname in os.listdir(npy_dir):

    if not fname.endswith(".npy"):
        continue

    path = os.path.join(npy_dir, fname)

    arr = np.load(path)
    print(f"File: {fname} — {arr.shape[0]} packets")

    for i in range(min(10, arr.shape[0])):

        packet = arr[i].reshape(1,1500,1).astype("float32")

        # convert float32 → int8
        packet_q = (packet / input_scale + input_zero).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], packet_q)

        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # dequantize output
        out_scale, out_zero = output_details[0]['quantization']
        output = (output.astype(np.float32) - out_zero) * out_scale

        cls = np.argmax(output)
        conf = output[cls] * 100

        print(f"  Packet {i+1}: class {cls} ({conf:.1f}%)")

    print()
