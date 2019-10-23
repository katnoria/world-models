import argparse
import pickle
import logging
import os
import uuid
from datetime import datetime
from glob import glob
from time import time
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Load images from the dataset and run it through
# VAE model to generate compressed Z representation

def load_npz_data(fname):
    """
    Loads npz data into array
    """
    with np.load(fname) as data:
        np_data = data['arr_0']
    return np_data

def load_preprocess_image(record):
    import pdb; pdb.set_trace();
    image = tf.io.read_file(fname)
    import pdb; pdb.set_trace()
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [64, 64])
    image /= 255.0
    return image    

def load_resized_image(items):
    obs_img, action, next_obs_img = items    
    obs_img /= 255.
    next_obs_img /= 255.

    return obs_img, action, next_obs_img

def load_data(dirname, max_items=None):
    fnames = glob("{}/*.pkl".format(dirname))
    size = len(fnames) if not max_items else max_items
    fnames = np.random.choice(fnames, size)
    ds = tf.data.Dataset.from_tensor_slices(fnames)    
    dataset = ds.map(load_resized_image, num_parallel_calls=AUTOTUNE)
    items = dataset.take(1)
    print(items)

    

def process():
    # 1. Load data into dataset
    data = load_npz_data('processed_with_actions/50rollouts.npz')
    # data = load_data('dataset/', 10)
    ds = tf.data.Dataset.from_tensor_slices(data)
    # 2. Preprocess dataset
    dataset = ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
    # 3. Load VAE Model
    # 4. Generate z vectors and store them



if __name__ == "__main__":
    process()