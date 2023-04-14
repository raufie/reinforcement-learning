import numpy as np


class Agent:
    def __init__(self, action_space, method="sample_average"):
        self.action_space = action_space
        self.method = method
        self.Q = np.zeros(action_space)
        self.action_count = np.zeros(action_space)
        self.t = 0

    def get_EGreedy(self, epsilon=0.1):

        sample = np.random.rand()
        if sample > epsilon:
            return np.random.choice(np.where(self.Q == np.max(self.Q))[0])
        else:
            return np.random.choice(np.arange(self.action_space))

    def train(self, action, reward):
        self.action_count[action] += 1
        if self.method == "sample_average":
            self.apply_sample_average(action, reward)

    def apply_sample_average(self, action, reward):
        self.t += 1
        self.Q[action] = self.Q[action] + \
            (1/self.action_count[action]) * (reward - self.Q[action])

    def resetQ(self):
        self.Q = np.zeros(self.action_space)
        self.t = 0
        self.action_count = np.zeros(self.action_space)
