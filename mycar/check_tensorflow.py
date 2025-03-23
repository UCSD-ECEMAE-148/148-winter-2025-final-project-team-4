import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

# List available devices
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available: ", gpus)
else:
    print("No GPU found")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
