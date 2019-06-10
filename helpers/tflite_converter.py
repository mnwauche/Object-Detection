import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_frozen_graph('ssdlite_mnet.pb')
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)