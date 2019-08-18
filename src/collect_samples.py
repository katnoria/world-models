import gym
from PIL import Image

def save(idx, data):
    img = Image.fromarray(data)
    img.save('{}.jpg'.format(idx))

def collect():
    env = gym.make('CarRacing-v0')
    obs = env.reset()
    save(0, obs)
    total_score = 0
    steps = 0
    while True:
        env.render()
        action = env.action_space.sample()
        print(action)
        obs, reward, done, info = env.step(action)
        total_score += reward
        steps += 1
        print(obs.shape)
        print(obs)
        break
        if done: 
            print('Total reward {} in {} steps'.format(total_score, steps))
            break


if __name__ == "__main__":
    print(gym.__version__)
    collect()