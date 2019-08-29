import gym
from atlas_ml import *
env = gym.make('CartPole-v0')
env.reset()
for _ in range(10000):
    env.render()
    __,__,done,__ = env.step(env.action_space.sample()) # take a random action
    if done==True:
        env.reset()
env.close()

