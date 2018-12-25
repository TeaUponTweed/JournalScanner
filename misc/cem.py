import itertools as it

import numpy as np

from scipy.stats import norm, uniform

import matplotlib.pyplot as plt
import matplotlib


def make_normal(x, P):
    L = np.linalg.cholesky(P)
    def f(nsamples):
        return  x + (L @ np.random.randn(len(x), nsamples)).T
    return f

def infer_normal(X):
    x = np.mean(X, axis=0)
    P = np.cov(X.T)
    return make_normal(x, P)


def make_uniform(mins, maxs):
    scales = maxs - mins
    def f(nsamples):
        return mins + np.random.rand(len(mins), nsamples).T*scales
    return f

def infer_uniform(X):
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    print(mins)
    print(maxs)
    return make_uniform(mins, maxs)



class CEMMinimizer(object):
    def __init__(self, distribution, infer_distribtion, nsamples=10000, relite=0.03, niter=5, verbose=False):
        self.distribution = distribution
        self.infer_distribtion = infer_distribtion
        self.nsamples = nsamples
        self.relite = relite
        self.niter = niter
        self.verbose = verbose

    def minimize(self, f):
        baseline = float('inf')
        best = None
        for _ in range(self.niter):
            S = self.distribution(self.nsamples)
            scores = (f(s) for s in S)
            elites = sorted(((td, s) for td, s in zip(scores, S) if td <= baseline), key=lambda x: x[0])[:int(self.relite*self.nsamples)]
            elite_arr = np.array([s for td, s in elites])
            if best is None:
                best = elite_arr[0]
            else:
                best = min(elite_arr[0], best, key=f)

            if self.verbose:
                print(elites[0][0], best)

            self.distribution = self.infer_distribtion(elite_arr)

        return best


def cartesian_dist(x1, x2):
    diff = x1 - x2
    return (diff.dot(diff))**.5


def distance_table(points):
    N, _ = points.shape
    D = np.zeros((N, N))
    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points):
            D[i, j] = cartesian_dist(point1, point2)
    return D

def draw_tour(mu, X):
    _, nodes = zip(*sorted(zip(mu, range(len(mu)))))
    ax = plt.gca()
    for i in it.chain(range(1, len(mu)), range(1)):
        x1, y1 = X[nodes[i-1], :]
        x2, y2 = X[nodes[i], :]
        arr = matplotlib.patches.Arrow(x1, y1, x2-x1, y2-y1, width=0.05)
        ax.add_patch(arr)

    plt.show()



def test():
    n_problem = 10
    problem = np.random.rand(n_problem, 2)
    Sigma = np.eye(n_problem)*10
    mu = np.zeros(n_problem)
    N = int(1000 * n_problem**2)
    D = distance_table(problem)

    def tour_distance(mu):
        _, nodes = zip(*sorted(zip(mu, range(len(mu)))))
        return sum(D[nodes[i-1], nodes[i]] for i in it.chain(range(1, len(mu)), range(1)))

    x = CEMMinimizer(distribution=make_normal(mu, Sigma), infer_distribtion=infer_normal, nsamples=N, verbose=False).minimize(tour_distance)

    # draw_tour(x, problem)
    print('min distance ', tour_distance(x))
    x = CEMMinimizer(distribution=make_uniform(np.zeros(n_problem), np.ones(n_problem)), infer_distribtion=infer_uniform, nsamples=N, verbose=False).minimize(tour_distance)
    # draw_tour(x, problem)
    print('min distance ', tour_distance(x))


if __name__ == '__main__':
    test()

