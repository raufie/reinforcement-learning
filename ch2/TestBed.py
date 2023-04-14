from Environment import ArmedBandit
from Agent import Agent
import time

from torch import argmax
import numpy as np


class TestBed:
    def __init__(self, n_tests=2000, n_steps=1000):
        self.n_steps = n_steps
        self.n_tests = n_tests
        self.agent = Agent(4, "sample_average")
        self.env = ArmedBandit(4)

    def run_experiment(self, k=4, epsilon=0):

        rewards = []
        actions = []
        optimalActions = []
        t = time.time()
        for i in range(self.n_tests):
            rewards.append([])
            optimalActions.append(0)
            self.env.reset_environment()
            self.agent.resetQ()
            print("Run # " + str(i)+"\r", end="")
            for j in range(self.n_steps):
                a = self.agent.get_EGreedy(epsilon)
                r = self.env.step(a)
                rewards[i].append(r)
                actions[i].append(a)
                optimalActions[i] += 1 if a == np.argmax(
                    self.env.Q_Optimal) else 0
                self.agent.train(a, r)
        print("Experiment Finished - time elapsed: ", time.time() - t)
        return rewards, optimalActions
