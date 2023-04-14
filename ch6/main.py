import gym
import os
from gym.envs.toy_text.frozen_lake import generate_random_map
from matplotlib import pyplot as plt
from cmdscreen import CMDScreen
from TD0 import TD0
import numpy as np
# FROZENLAKE
# (0123->LDRU )


def run_episode(pi, env, screen):
    env.reset()
    episode_data = []
    s = 0
    while True:
        a = pi[s]
        s_, r, done, prob = env.step(a)

        episode_data.append((s, s_, r))
        # screen.set_value(env.render(mode="ansi"))
        s = s_

        if done:
            break
    return episode_data


def main():
    screen = CMDScreen()

    env = gym.make("FrozenLake-v1", desc=generate_random_map(size=4),
                   map_name="4x4", is_slippery=False)

    env.reset()
    img = env.render(mode="ansi")
    # screen
    obj = screen.register("RL environment", img)
    reward_object = screen.register("reward:", "0")
    VText = screen.register("V", "[]")
    screen.render()
    # screen.render()
    episode_data = []
    td = TD0()
    env.reset()
# PREDICT V
# batch updating

    # get episode
    for i in range(100):
        td.pi = np.random.randint(0, 4, (16, ), dtype=np.int8)
        episode_data = run_episode(td.pi, env, obj)
        # batch update
        for _ in range(100):
            td.evaluate(episode_data)
            VText.set_value(td.V)
        # print(td.V)


if __name__ == "__main__":
    main()
