from TestBed import TestBed
import matplotlib.pyplot as plt
import numpy as np
import pickle


def run10ArmedBandits():
    n_tests = 2000
    n_runs = 1000
    testBed = TestBed(n_tests, n_runs)
    rewards1, optimalActions1 = testBed.run_experiment(k=10, epsilon=0.1)
    rewards2, optimalActions2 = testBed.run_experiment(k=10, epsilon=0.01)
    rewards3, optimalActions3 = testBed.run_experiment(k=10, epsilon=0)

    avg1 = getAvgRewards(rewards1, n_tests, n_runs)
    avg2 = getAvgRewards(rewards2, n_tests, n_runs)
    avg3 = getAvgRewards(rewards3, n_tests, n_runs)

    # save_data(avg1, "epsilon0")
    # save_data(avg1, "epsilon0_01")
    # save_data(avg1, "epsilon0_1")
    plt.plot(avg1, 'b', label="$\epsilon = 0.1$")
    plt.plot(avg2, 'r', label="$\epsilon = 0.01$")
    plt.plot(avg3, 'g', label="$\epsilon = 0$")
    plt.xlabel("n steps")
    plt.ylabel("average reward")
    plt.title('Sample Average Method - avg reward')
    plt.legend()
    plt.savefig("2_2 avg.png")
    plt.show()
    plt.clf()
    plt.close()
    # plotting avg reward wrt true r vals in percentage

    plt.plot(optimalActions1 / n_runs, 'b', label="$\epsilon = 0.1$")
    plt.plot(optimalActions2 / n_runs, 'r', label="$\epsilon = 0.01$")
    plt.plot(optimalActions3 / n_runs, 'g', label="$\epsilon = 0$")
    plt.xlabel("n steps")
    plt.ylabel("percetange of the time it got optimal values")
    plt.title('Sample Average Method - avg reward')
    plt.legend()
    plt.savefig("2_2 optimal actions percentage.png")


def getAvgRewards(rewards, n_tests, n_steps):
    avg_rewards = np.zeros(n_steps)
    for i, run in enumerate(rewards):

        for j, reward in enumerate(run):
            avg_rewards[j] += reward

    avg_rewards /= n_tests
    return avg_rewards


def displayResults(results):
    plt.plot(results)

    plt.show()


def save_data(results, name):
    f = open(f"{name}.pkl", "wb")
    pickle.dump(results, f)


if __name__ == "__main__":
    run10ArmedBandits()
