import itertools as it

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def cartesian_dist(x1, x2):
    diff = x1 - x2
    return (diff.dot(diff)) ** 0.5


def draw_tour(mu, X):
    _, nodes = zip(*sorted(zip(mu, range(len(mu)))))
    ax = plt.gca()
    for i in it.chain(range(1, len(mu)), range(1)):
        x1, y1 = X[nodes[i - 1], :]
        x2, y2 = X[nodes[i], :]
        arr = matplotlib.patches.Arrow(x1, y1, x2 - x1, y2 - y1, width=0.05)
        ax.add_patch(arr)

    plt.show()


def get_samples(mu, L, shape):
    samples = np.random.randn(*shape)
    return mu + L.dot(samples.T).T


def distance_table(points):
    N, _ = points.shape
    D = np.zeros((N, N))
    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points):
            D[i, j] = cartesian_dist(point1, point2)
    return D


def main():
    n_problem = 10
    rho = 0.1
    X = np.random.rand(n_problem, 2)

    Sigma = np.eye(n_problem) * 10
    LSigma = np.linalg.cholesky(Sigma)
    mu = np.zeros(n_problem)
    N = int(100 / rho * n_problem**2)

    baseline = float("inf")
    best = None
    D = distance_table(X)

    def tour_distance(mu):
        _, nodes = zip(*sorted(zip(mu, range(len(mu)))))
        return sum(
            D[nodes[i - 1], nodes[i]] for i in it.chain(range(1, len(mu)), range(1))
        )
        # return sum(cartesian_dist(X[nodes[i-1], :], X[nodes[i], :]) for i in it.chain(range(1, len(mu)), range(1)))

    def f(mu, Sigma, baseline, best=None):
        L = np.linalg.cholesky(Sigma)
        S = mu + L.dot(np.random.randn(n_problem, N)).T
        scores = (tour_distance(s) for s in S)
        elites = sorted(
            ((td, s) for td, s in zip(scores, S) if td <= baseline), key=lambda x: x[0]
        )[: int(0.03 * N)]
        elite_arr = np.array([s for td, s in elites])
        if best is None:
            best = elite_arr[0]
        else:
            best = min(elite_arr[0], best, key=tour_distance)

        return (
            np.mean(elite_arr, axis=0),
            np.cov(elite_arr.T),
            tour_distance(elite_arr[-1]),
            best,
        )

    for _ in range(5):
        mu, Sigma, baseline, best = f(mu, Sigma, baseline, best)
        print(tour_distance(mu), tour_distance(best))

    draw_tour(best, X)


if __name__ == "__main__":
    main()
