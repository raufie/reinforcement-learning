import numpy as np


class ArmedBandit:
    def __init__(self, k=4):
        self.k = k
        self.reset_environment()

    def step(self, action):
        return self.getReward(action)

    def getReward(self, action):
        return np.random.normal(self.Q_Optimal[action], 1)

    def reset_environment(self):
        self.Q_Optimal = np.random.randn(self.k)
