#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow.
Basic usage should feel familiar: train.py --tubs data/ --model models/mypilot.h5

Usage:
    train.py [--tubs=tubs] (--model=<model>)
    [--type=(linear|inferred|tensorrt_linear|tflite_linear)]
    [--comment=<comment>]

Options:
    -h --help              Show this screen.
"""

from docopt import docopt
import donkeycar as dk
from donkeycar.pipeline.training import train
import tensorflow as tf
from tensorflow.keras import mixed_precision
import os


def main():
    args = docopt(__doc__)
    cfg = dk.load_config()
    tubs = args['--tubs']
    model = args['--model']
    model_type = args['--type']
    comment = args['--comment']
    train(cfg, tubs, model, model_type, comment)


if __name__ == "__main__":
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # If using mixed precision, ensure you lower your learning rate for stability
    # mixed_precision.set_global_policy('mixed_float16')
    
    main()
