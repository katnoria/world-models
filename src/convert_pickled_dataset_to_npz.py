import argparse
import pickle
import logging
import os
import uuid
from datetime import datetime
from glob import glob
from time import time
import numpy as np

import psutil
import ray

@ray.remote
def load_pickle(fname):
    array = pickle.load(open(fname, 'rb'))
    return array


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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default=10, type=int, help="Number of rollouts to process into a single npz file")
    args = parser.parse_args()

    fnames = glob('dataset/*.pkl')
    process_fnames = fnames[:args.rollouts]
    convert_states(process_fnames)