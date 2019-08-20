import argparse
import pickle
import logging
import os
from datetime import datetime
from time import time

import gym
import psutil
import ray

# This script will generate the dataset of state, action, new state tuple
# and store it locally as dataset.pkl. 
# We make use of ray to generate the datast in parallel
if not os.path.exists("logs"):
    os.makedirs("logs")

today = datetime.now().strftime('%Y%m%d')

logger = logging.getLogger('worldmodels')
logger.setLevel(logging.DEBUG)

# Create logger
logger = logging.getLogger("worldmodels")
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s') 
logger.setLevel(logging.DEBUG)

filehandler = logging.FileHandler(filename='logs/dataset.{}.log'.format(today))
filehandler.setFormatter(formatter)
filehandler.setLevel(logging.DEBUG)
logger.addHandler(filehandler)
# Uncomment to enable console logger
steamhandler = logging.StreamHandler()
steamhandler.setFormatter(formatter)
steamhandler.setLevel(logging.INFO)
logger.addHandler(steamhandler)



def rollout():
    env = gym.make('CarRacing-v0')
    obs = env.reset()
    total_score = 0
    steps = 0
    buffer = []
    while True:
        action = env.action_space.sample()
        new_obs, reward, done, info = env.step(action)
        total_score += reward
        steps += 1
        buffer.append((obs.copy(), action, new_obs.copy()))
        # buffer.append(obs.copy())
        obs = new_obs
        if done: 
            logger.info('Total reward {} in {} steps'.format(total_score, steps))
            break
    env.close()

    return buffer                

def save_dataset(dataset, fname):
    if not os.path.exists("dataset"):
        os.makedirs("dataset")
        
    with open("dataset/{}".format(fname), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

def collect_samples(num_rollouts):
    for rollout_idx in range(num_rollouts):
        logger.info("Collecting dataset for rollout {}".format(rollout_idx))
        dataset = rollout()
        save_dataset(dataset, "dataset_{}.pkl".format(rollout_idx))        

@ray.remote
def collect():
    # env = gym.make('CarRacing-v0')
    # obs = env.reset()
    # total_score = 0
    # steps = 0
    # buffer = []
    # while True:
    #     action = env.action_space.sample()
    #     new_obs, reward, done, info = env.step(action)
    #     total_score += reward
    #     steps += 1
    #     buffer.append((obs.copy(), action, new_obs.copy()))
    #     # buffer.append(obs.copy())
    #     obs = new_obs
    #     if done: 
    #         print('Total reward {} in {} steps'.format(total_score, steps))
    #         break

    # return buffer
    return rollout()

def collect_samples_using_ray(num_cpus, num_rollouts):
    available_cpus = psutil.cpu_count(logical=False)
    if num_cpus > 0:
        if num_cpus > available_cpus:
            msg = "You have {} cpus available, use a count <= {}".format(available_cpus, available_cpus)
            raise Exception(msg)            
    else:
        num_cpus = available_cpus

    ray.init(num_cpus=num_cpus) 
    logger.info('Starting dataset collection in parallel on {} cpus'.format(num_cpus))
    dataset = ray.get([collect.remote() for _ in range(num_rollouts)])
    logger.debug(len(dataset))
    save_dataset(dataset, "dataset.pkl")
    # with open('dataset.pkl', 'wb') as f:
    #     pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", default=False, help="Whether to use parallel processing powered by Ray.")
    parser.add_argument("--ray", default=-1, type=int, help="Number of CPUs to use in parallel to generate the dataset. Use -1 to use all cpus available")
    parser.add_argument("--num_rollouts", default=5, type=int, help="Number of rollouts to use for dataset collection")
    args = parser.parse_args()
    start = time()

    if args.parallel:
        collect_samples_using_ray(args.ray, args.num_rollouts)
    else:
        collect_samples(args.num_rollouts)

    logger.info('Took {} seconds to generate the dataset'.format(time() - start))
