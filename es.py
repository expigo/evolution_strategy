import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pathlib



def es(obj, orig_data, mu_size=200, ratio=5, Nmax=200, eps=10e-5):
    CHROMOSOME_SIZE = 6
    N_PROBLEM_PARAMS = 3
    N_ALG_PARAMS = 3
    rmse = calc_rmse(orig_data)

    parents_x = np.array([np.random.default_rng().uniform(low=-10, high=10, size=(N_PROBLEM_PARAMS,)) for _ in range(mu_size)])
    sigmas = np.array([np.random.default_rng().uniform(low=0, high=10, size=(N_ALG_PARAMS,)) for _ in range(mu_size)])
    zero_mean_gauss = lambda x: np.random.normal(0, x)

    epochs = []
    for epoch in range(Nmax):
        fitnesses = np.apply_along_axis(rmse, 1, parents_x).reshape(-1, 1)
        mu = np.hstack((parents_x, sigmas, fitnesses))

        mutation_operand = np.vectorize(zero_mean_gauss)(sigmas)
        offspring = parents_x + mutation_operand

        plus = np.vstack([parents_x, offspring])

        plus_sorted = sorted(plus, key=lambda x: rmse(x))
        parents_x = np.array(plus_sorted[:200])


        # update sigmas
        for i in range(mu_size):
            tau1 = 1 / np.sqrt(2 * CHROMOSOME_SIZE)
            tau2 = 1 / np.sqrt(2 * np.sqrt(CHROMOSOME_SIZE))
            r1 = np.random.default_rng().normal(0, tau1)
            sigmas[i, :] = np.exp(np.random.default_rng().normal(0, tau2, 3) + r1)

        # parents_x = offspring

    return parents_x, mu, mutation_operand, parents_x, offspring

def calc_rmse(data):
    def fn(x):
        [a, b, c] = x
        f = lambda i: a*(i ** 2 - b*np.cos(c*np.pi*i))
        x = np.arange(-5, 5, 0.1)
        y = f(data[:, 0])

        return mean_squared_error(data[:, 1], y)

    return fn


if __name__ == "__main__":
    f = lambda a, b, c: lambda i: a*(i ** 2 - b*np.cos(c*np.pi*i))

    path_to_data = pathlib.PurePath("data/model3.txt")
    data = np.loadtxt(str(path_to_data))

    res = es(f, data)
    mu = res[3]

    rmse = calc_rmse(data)
    mu_sorted = sorted(mu, key=lambda x: rmse(x))

    results2 = []
    for i in range(len(mu_sorted)):
        # print(*mu[i, :])
        results2.append(rmse(*mu_sorted[i]))

    for i in range(5):
        f_parametrized = f(*np.array(mu_sorted[i]))
        plt.plot(data[:, 0], f_parametrized(data[:, 0]), label=f'{i}')
        plt.plot(data[:, 0], data[:, 1], label="ground truth")
        plt.legend()
        plt.show()



    print()
