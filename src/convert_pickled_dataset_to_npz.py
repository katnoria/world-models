import argparse
import pickle
import logging
import os
import uuid
from datetime import datetime
from glob import glob
from time import time
import numpy as np
from PIL import Image

import psutil
import ray

SIZE = 64, 64

@ray.remote
def resize_image(img_array):    
    im = Image.fromarray(img_array)
    im.thumbnail(SIZE)
    return np.array(im)

@ray.remote
def load_pickle(fname):
    arraylist = pickle.load(open(fname, 'rb'))
    output = []
    for obs, action, next_obs in arraylist:
        output.append((
            ray.get(resize_image.remote(obs)),
            action,
            ray.get(resize_image.remote(next_obs))
        ))
    return output


def convert_states(input_fnames, out_dirname='processed/'):
    if not os.path.exists(out_dirname):
        os.makedirs(out_dirname)

    available_cpus = psutil.cpu_count()
    ray.init(num_cpus=available_cpus)
    start = time()
    dataset = ray.get([load_pickle.remote(fname) for fname in input_fnames])
    dataset = np.asarray([subitem[0] for item in dataset for subitem in item])    
    out_fname = '{}/{}rollouts.npz'.format(out_dirname, len(input_fnames))
    np.savez(out_fname, dataset)
    print('saved {}'.format(out_fname))
    print('Took {} seconds'.format(time() - start))

def resize_and_compress_buffer(input_fnames, out_dirname='processed_with_actions/'):
    if not os.path.exists(out_dirname):
        os.makedirs(out_dirname)

    available_cpus = psutil.cpu_count()
    ray.init(num_cpus=available_cpus)
    start = time()
    dataset = ray.get([load_pickle.remote(fname) for fname in input_fnames])
    dataset = np.asarray([subitem for item in dataset for subitem in item])
    out_fname = '{}/{}rollouts.npz'.format(out_dirname, len(input_fnames))
    np.savez(out_fname, dataset)
    print('saved {}'.format(out_fname))
    print('Took {} seconds'.format(time() - start))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default=10, type=int, help="Number of rollouts to process into a single npz file")
    args = parser.parse_args()

    fnames = glob('dataset/*.pkl')
    process_fnames = fnames[:args.rollouts]
    # convert_states(process_fnames)
    resize_and_compress_buffer(process_fnames)