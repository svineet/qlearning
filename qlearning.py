import gym
import numpy as np

import math
import pickle

NUM_BINS = 50
RESUME = True
TEST_RUN = True
filename = "qtable.pkl"

def bin_ize(a, high, low):
    width = (high-low)/(NUM_BINS)
    return math.floor((a-low)/width)

def discretize(variables, highlows):
    thing = []
    for var, hl in zip(variables, highlows):
        thing.append(bin_ize(var, *hl))

    return thing

env = gym.make("MountainCar-v0")
cur_state = env.reset()

high = (env.observation_space.high)
low = (env.observation_space.low)
size = list(map(int, (discretize(env.observation_space.high, zip(high, low)))))
size = [a+1 for a in size]
num_actions = env.action_space.n
q = np.zeros(size+[num_actions])
print("Q-table shape is: ", q.shape)

if RESUME:
    q = pickle.load(open(filename, 'rb'))
    print("Resumed state")

# Learning rate
alpha = 0.1
# Reward discounting
gamma = 0.95
# Exploration probability
epsilon = 1
# Exploration probability decay factor
epsgamma = 0.9999
# Number of total iterations to train
num_epochs = 500000

i = 0
eps = 1
action = None
done = False
while i <= num_epochs and not TEST_RUN:
    discur_actionless = discretize(cur_state, zip(high, low))
    if np.random.uniform(0, 1) <= epsilon:
        action = np.random.choice(np.arange(3))
    else:
        action = np.argmax(q[discur_actionless[0], discur_actionless[1], :])

    next_state, reward, done, info = env.step(action)
    disnext = tuple(discretize(next_state, zip(high, low))+[action])
    discur = tuple(discretize(cur_state, zip(high, low))+[action])

    """
        The Q value formula

        q[state] = (1-alpha)*q[state] + alpha*(gamma*reward + q[next_state])
    """

    q[discur] = (1-alpha)*q[discur] + alpha*(gamma*reward + q[disnext])
    if done:
        next_state = env.reset()
        print("Episode ended", i)
        eps = 0

    epsilon *= epsgamma
    cur_state = next_state
    i += 1
    eps += 1

print(i)

# Test episode

cur_state = env.reset()
done = False
action = 0
i = 0
while not done:
    discur_actionless = discretize(cur_state, zip(high, low))
    discur = tuple(discretize(cur_state, zip(high, low))+[action])
    action = np.argmax(q[discur_actionless[0], discur_actionless[1], :])

    cur_state, reward, done, info = env.step(action)

    if not done:
        env.render()
    else:
        break

    print("Current q value", q[discur])
    i += 1

print("Done at", i)

print("Dumping pickle for saved model")
pickle.dump(q, open(filename, "wb"))
print("Dump done.")
