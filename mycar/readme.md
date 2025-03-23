# Changes to Donkey Car's files in our project 

## manage.py
DonkeyCar’s manage.py file is not programmed to consider .hef files and is built strictly for Keras and PyTorch based models. Since the Hailo framework is relatively new, the hailo_runner.py is used as an intermediary between DonkeyCar and the Hailo-8 NPU on the AI-Hat+. manage.py now checks for a .hef file extension for the model and creates a model of type HailoRunner.

## profile.py
DonkeyCar’s profile.py file is not programmed to consider .hef files and is built strictly for Keras and PyTorch based models. Since the Hailo framework is relatively new, the hailo_runner.py is used as an intermediary between DonkeyCar and the Hailo-8 NPU on the AI-Hat+. profile.py now checks for a .hef file extension for the model and creates a model of type HailoRunner.

## hailo_runner.py
In the unoptimized version of this script, the runner processes the image input, calls the Hailo-built run function, and runs inference on the model to return the throttle and steering values to manage.py. The problem with this approach is that a new instance of the model is created with every new input, greatly underutilizing the performance capabilities of the NPU. The /hailoconversion subdirectory should contain the necessary file converter from .h5 to .hef.

## WIP_hailo_runner_persistent.py
The work-in-progress version of this script (WIP_hailorunner_persistent.py) intends to create a persistent instance of the model, eliminating the need for a new instance upon every input, and optimizing the process. This version is untested, but likely works and should see a significant performance boost over hailo_runner.py. The /hailoconversion subdirectory should contain the necessary file converter from .h5 to .hef.

## train.py
Slightly optimized memory management during training. Allowed us to train a higher resolution model. Added dynamic memory allocation and tested mixed precision training.

## tflite_profile.py
Added support for .tflite models to DonkeyCar