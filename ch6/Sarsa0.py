import numpy as np


class Sarsa0:
    def __init__(self, n_states=16, n_actions=4):
        self.Q = np.random.uniform(0, 1, (n_states, n_actions))

        self.Q[15] = np.array([0, 0, 0, 0])

        self.pi = np.random.randint(0, 4, (16, ), dtype=np.int8)
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 0.1

    def evaluate(self, episode):
        for s, a, r, s_, a_ in episode:

            self.Q[s, a] = self.Q[s, a] + self.alpha * \
                (r + self.gamma*self.Q[s_, a_] - self.Q[s, a])

    def get_pi(self, state, eps=0.2):
        if np.random.random() > eps:
            # greedy (LDRU)

            return np.argmax(self.Q[state])
        else:
            # non greedy
            return np.random.randint(0, 4)
