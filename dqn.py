import gym
import numpy as np

from neural_net import QNetwork

env = gym.make("MountainCar-v0")
cur_state = env.reset()
print(cur_state)

done = False

action = None
i = 0

q_estimator = QNetwork(len(cur_state), len(env.action_space))

while not done:
    action = 2
    state, reward, done, info = env.step(action)
    print(reward, done, state)
    i += 1

print(i)
