import tensorflow as tf
import numpy as np

# Load representative dataset
X = np.load("X.npy")

X = X[:500]                 # calibration samples
X = X.reshape(-1,1500,1).astype("float32")

def representative_dataset():
    for i in range(len(X)):
        yield [X[i:i+1]]

# Convert directly from SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model("cnn_dpi_model")

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("cnn_dpi_model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("INT8 model created successfully")
