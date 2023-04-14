import gym
import os
from gym.envs.toy_text.frozen_lake import generate_random_map
from matplotlib import pyplot as plt
from cmdscreen import CMDScreen
from Sarsa0 import Sarsa0
import numpy as np
import time
# FROZENLAKE
# (0123->LDRU )
screen = CMDScreen()

env = gym.make("FrozenLake-v1", desc=["SFFF", "FHFH", "FFFH", "HFFG"],
               map_name="4x4", is_slippery=False)

env.reset()

alpha = 0.15
eps = 0.1
gamma = 1.0
n_episodes = 1500

# INITIALIZE Q
# Q is abritrarily initialized for all the states except terminal ones.. where Q(terminalS, :) = 0

Q = np.ones((16, 4))*0.5
# we know 5, 7, 11, 12, 15 are terminal
Q[5][:] = 0
Q[7][:] = 0
Q[11][:] = 0
Q[12][:] = 0
Q[15][:] = 0
Q[:, 0]


def get_action(eps, Q, s):
    r = np.random.random()
    if r < eps:
        a = np.random.randint(0, 4)
    else:
        a = np.argmax(Q[s])
    return a


env.reset()
for e in range(n_episodes):
    s = 0
    done = False
    a = get_action(eps, Q, s)
    while not done:
        s_, r, done, prob = env.step(a)
        a_ = get_action(eps, Q, s_)

        Q[s][a] = Q[s][a] + alpha*(r + gamma*Q[s_][a_] - Q[s][a])
        s = s_
        a = a_
    env.reset()
# LETS DISPLAY

env.reset()

print(f"ran {n_episodes}, results")
done = False
s = 0
while not done:
    os.system("cls")
    print(env.render(mode="ansi"))
    time.sleep(0.5)
    a = get_action(0.0, Q, s)
    s, r, done, pr = env.step(a)

os.system("cls")
print(env.render(mode="ansi"))
