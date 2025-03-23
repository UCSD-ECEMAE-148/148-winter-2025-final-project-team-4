# General imports used throughout the tutorial
import tensorflow as tf
from IPython.display import SVG
# import the ClientRunner class from the hailo_sdk_client package
from hailo_sdk_client import ClientRunner

chosen_hw_arch = "hailo8"

model_name = "suarez"
model_path = "suarez.tflite"
runner = ClientRunner(hw_arch=chosen_hw_arch)
hn, npz = runner.translate_tf_model(model_path, model_name)

hailo_model_har_name = f"{model_name}_hailo_model.har"
runner.save_har(hailo_model_har_name)
