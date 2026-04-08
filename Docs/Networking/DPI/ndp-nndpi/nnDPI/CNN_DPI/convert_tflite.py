import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("cnn_dpi_model")

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

open("cnn_dpi_model.tflite","wb").write(tflite_model)

print("TFLite model saved")
