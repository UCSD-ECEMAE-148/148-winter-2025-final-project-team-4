# General imports used throughout the tutorial
# file operations
import json
import os

import numpy as np
import tensorflow as tf
from IPython.display import SVG
from matplotlib import patches
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python.eager.context import eager_mode

# import the hailo sdk client relevant classes
from hailo_sdk_client import ClientRunner, InferenceContext
IMAGES_TO_VISUALIZE = 5

# First, we will prepare the calibration set. Resize the images to the correct size and crop them.
def preproc(image, output_height=120, output_width=160, resize_side=160):
    """imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px"""
    with eager_mode():
        h, w = image.shape[0], image.shape[1]
        scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
        resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [int(h * scale), int(w * scale)])
        cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
        return tf.squeeze(cropped_image)

images_path = "data/data/images/set1"
images_list = [img_name for img_name in os.listdir(images_path) if os.path.splitext(img_name)[1] == ".jpg"]
calib_dataset = np.zeros((len(images_list), 120, 160, 3))
for idx, img_name in enumerate(sorted(images_list)):
    img = np.array(Image.open(os.path.join(images_path, img_name)))
    img_preproc = preproc(img)
    calib_dataset[idx, :, :, :] = img_preproc.numpy()
np.save("calib_set.npy", calib_dataset)





model_name = "suarez"
hailo_model_har_name = f"{model_name}_hailo_model.har"
assert os.path.isfile(hailo_model_har_name), "Please provide valid path for HAR file"
runner = ClientRunner(har=hailo_model_har_name)


alls = "normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])\n"

runner.load_model_script(alls)
# Call Optimize to perform the optimization process
runner.optimize(calib_dataset)
# Save the result state to a Quantized HAR file
quantized_model_har_path = f"{model_name}_quantized_model.har"
runner.save_har(quantized_model_har_path)

