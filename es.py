import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error
import pathlib

# matplotlib.use('TkAgg')
plt.style.use('seaborn-white')


def es(obj, orig_data, mu_size=200, phi=5, Nmax=200, eps=10e-5, comma=False):
    CHROMOSOME_SIZE = 6
    N_PROBLEM_PARAMS = 3
    N_ALG_PARAMS = 3
    rmse = calc_rmse(orig_data, obj)

    parents_x = np.array([np.random.default_rng().uniform(low=-10, high=10, size=(N_PROBLEM_PARAMS,))
                          for _ in range(mu_size)])
    sigmas = np.array([np.random.default_rng().uniform(low=0, high=10, size=(N_ALG_PARAMS,))
                       for _ in range(mu_size)])
    zero_mean_gauss = lambda x: np.random.normal(0, x)
    p = np.hstack((parents_x, sigmas))

    curr_rmse = 10e-3
    prev_rmse = 10e-3
    for epoch in range(Nmax):
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(p[:, 0], p[:, 1], p[:, 2])
        # plt.show()

        offspring = numpy.resize(p, (mu_size * phi, CHROMOSOME_SIZE))

        tau1 = 1 / np.sqrt(2 * CHROMOSOME_SIZE)
        tau2 = 1 / np.sqrt(2 * np.sqrt(CHROMOSOME_SIZE))

        for i in range(mu_size * phi):
            r1 = np.random.default_rng().normal(0, tau1)
            r2 = np.random.default_rng().normal(0, tau2, 3)
            offspring[i, 3:] = offspring[i, 3:] * np.exp(r1 + r2)

        if comma:
            population = offspring
        else:
            population = np.vstack([p, offspring])

        mutation_operand = np.vectorize(zero_mean_gauss)(population[:, 3:])
        population[0:mutation_operand.shape[0], 0:mutation_operand.shape[1]] += mutation_operand

        # calc offspring fitness
        fitness = np.apply_along_axis(rmse, 1, population[:, :3]).reshape(-1, 1)

        population = np.hstack([population, fitness])
        population_sorted = np.array(sorted(population, key=lambda x: x[6]))

        curr_rmse = population_sorted[0, 6]
        print(f'epoch: {epoch} | RMSE: {curr_rmse}')
        if np.abs(curr_rmse-prev_rmse) < eps:
            break

        p = population_sorted[:200, :6]

        prev_rmse = curr_rmse

    return population_sorted, curr_rmse


def calc_rmse(data, objective):
    def fn(params):
        f = objective(*params)
        y = f(data[:, 0])
        return mean_squared_error(data[:, 1], y)

    return fn


if __name__ == "__main__":
    f = lambda a, b, c: lambda i: a*(i ** 2 - b*np.cos(c*np.pi*i))

    path_to_data = pathlib.PurePath("data/model3.txt")
    data = np.loadtxt(str(path_to_data))

    result = es(f, data, comma=False)

    f_approximated = f(*np.array(result[0][0, :3]))
    plt.plot(data[:, 0], f_approximated(data[:, 0]), label=f'approx')
    plt.plot(data[:, 0], data[:, 1], label="ground truth")
    plt.title(f'RMSE = {result[1]: .4f}')
    plt.legend()
    plt.show()
