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

fnames = glob('dataset/*.pkl')
num_files_to_process = 10

process_fnames = fnames[:num_files_to_process]
available_cpus = psutil.cpu_count()
ray.init(num_cpus=available_cpus)
start = time()
dataset = ray.get([load_pickle.remote(fname) for fname in process_fnames])
dataset = [subitem for item in dataset for subitem in item]
np.savez('test.npz', dataset)
print('Took {} seconds'.format(time() - start))

