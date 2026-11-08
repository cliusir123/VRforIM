import numpy as np
import numpy.random as nr
from scipy.special import logsumexp


def sample_g(size):
    w = nr.beta(1, 1, size)
    return w / (1 - w)


class _GaussianScaleMixture:
    def __init__(self, dim) -> None:
        self.dim = dim

    def _set_design(self, cov0, inputs):
        self.cov0 = cov0
        self.inputs = inputs
        self.chol0 = np.linalg.cholesky(cov0)

    def quad(self, B):
        v = np.atleast_2d(B) @ self.inputs.T
        return np.sum(v * v, axis=1)


class gprior(_GaussianScaleMixture):

    def set_parameters(self, cov0, inputs, L=1000):
        self._set_design(cov0, inputs)
        self.L = L
        self.new_chain()

    def new_chain(self):
        self.G = sample_g(self.L)

    def sampler(self, num):
        g = sample_g(num)
        z = nr.standard_normal((num, self.dim))
        return np.sqrt(g)[:, None] * (z @ self.chol0.T)

    def log_density(self, B):
        S = self.quad(B)
        log_g = np.log(self.G)
        rows = max(1, 2_000_000 // self.L)
        out = np.empty(len(S))
        for s in range(0, len(S), rows):
            out[s:s + rows] = logsumexp(-0.5 * (self.dim * log_g + S[s:s + rows, None] / self.G), axis=1)
        return out - np.log(self.L)


def log_prior_of_S(S, dim, n_grid=6000):
    t = np.linspace(np.log(1e-7), np.log(1e9), n_grid)
    g = np.exp(t)
    S = np.atleast_1d(S)
    return logsumexp(-0.5 * (dim * np.log(g) + S[:, None] / g) - 2 * np.log1p(g) + t, axis=1) + np.log(t[1] - t[0])


class mix_normal(_GaussianScaleMixture):

    def set_parameters(self, weights, gs, cov0, inputs):
        self._set_design(cov0, inputs)
        self.weights = np.asarray(weights)
        self.gs = np.asarray(gs)
        self.n_comps = len(self.weights)

    def sampler(self, num):
        z = nr.choice(self.n_comps, size=num, p=self.weights)
        eps = nr.standard_normal((num, self.dim))
        return np.sqrt(self.gs[z])[:, None] * (eps @ self.chol0.T)

    def logpdf(self, B):
        return self.logpdf_of_S(self.quad(B))

    def logpdf_of_S(self, S):
        S = np.atleast_1d(S)
        return logsumexp(np.log(self.weights) - 0.5 * (self.dim * np.log(self.gs) + S[:, None] / self.gs), axis=1)

    def em_algorithm(self, data, n_comps, inputs, max_itr=1000):
        self.inputs = inputs
        S = self.quad(data)
        weights = np.ones(n_comps) / n_comps
        gs = sample_g(n_comps)
        for _ in range(max_itr):
            logr = np.log(weights) - 0.5 * (self.dim * np.log(gs) + S[:, None] / gs)
            r = np.exp(logr - logsumexp(logr, axis=1, keepdims=True))
            nk = np.maximum(r.sum(axis=0), 1e-12)
            weights = nk / len(S)
            gs = (r * S[:, None]).sum(axis=0) / (self.dim * nk)
        self.n_comps = n_comps
        self.weights = weights
        self.gs = gs
        return weights, gs
