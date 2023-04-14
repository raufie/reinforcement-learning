import numpy as np


class TD0:
    def __init__(self, n_states=16, n_actions=4):
        self.V = np.random.uniform(0, 1, (16,))

        self.V[15] = 0
        self.pi = np.random.randint(0, 4, (16, ), dtype=np.int8)
        self.alpha = 0.2

    def evaluate(self, episode):
        for s, s_, r in episode:

            self.V[s] = self.V[s] + self.alpha*(r + 0.1*self.V[s_] - self.V[s])

    def get_pi(self, state, eps=0.8):
        if np.random.random() <= eps:
            # greedy (LDRU)
            l, d, r, u = state-1, state+4, state+1, state-4
            V = [self.V[l]]
            if state+4 > 15:
                V.append(1e-99)
            else:
                V.append(d)

            # r
            if state-4 < 0:
                V.append(1e-99)
            else:
                V.append(r)

            # u
            V.append(self.V[u])

            next_states = np.array([l, d, r, u])
            return np.argmax(V)
        else:
            # non greedy
            return np.random.randint(0, 4)
