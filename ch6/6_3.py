from matplotlib import pyplot as plt
import numpy as np


class TD:
    def __init__(self, alpha=0.1, gamma=1.0):
        self.V = [0.5, 0.5, 0.5, 0.5, 0.5]
        self.alpha = alpha
        self.gamma = gamma

    def step(self, s, a):
        # a=1, right, a=-1 left... left at 0: r = 0 vs = terminal, right at -1, r = 1, vs = terminal
        r = 0.0
        V_ = 0.5
        s_ = -1.0
        if s == 0 and a == -1:
            # terminal
            V_ = 0.0
        elif s == 4 and a == 1:
            V_ = 0.0
            r = 1.0
        else:
            s_ = s + a
        self.V[s] = self.V[s] + self.alpha*(r + self.gamma*V_ - self.V[s])

        return s_


X_P = []
for _ in range(100):
    X_P.append([0, 1, 2, 3, 4])

td = TD(gamma=0.5)
i = 1
V_arr = []
for iter in range(100):
    s = 0

    while s != -1:
        s = td.step(s, 1)
        # print(f"i={i}", td.V)
        # V_arr.append(td.V)
    print(f"i={i}", td.V)
    V_arr.append(td.V)
    i += 1
plt.ylim(0, 1)

plt.scatter(X_P, V_arr)
plt.show()
