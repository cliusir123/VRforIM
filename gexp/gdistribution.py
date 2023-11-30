import jax.numpy as jnp
import jax
import jax.scipy.stats as jss
import numpy as np
import numpy.random as nr

class gprior():
    def __init__(self,dim) -> None:
        self.dim = dim
    
    def set_parameters(self, cov0):
        self.cov0 = cov0
    
    def sampler(self, num):
        samples = []
        for i in range(num):
            w = nr.beta(1, 1)
            g = w / (1 - w)
            cov = self.cov0 * g
            # cov = cov0
            beta = nr.multivariate_normal(np.zeros(self.dim), cov)
            samples.append(beta)
        return np.array(samples)

    def logpdf(self, x, cov0):
        n = 500
        loglik = 0
        for i in range(n):
            w = nr.beta(1, 1)
            g = w / (1 - w)
            cov = cov0 * g
            # cov = cov0
            loglik += jss.multivariate_normal.pdf(x, jnp.zeros(self.dim), cov)
        return jnp.log(loglik / n)

class mix_normal():
    def __init__(self, dim) -> None:
        self.dim = dim
    
    def set_parameters(self, n_comps, weights, covs):
        self.n_comps = n_comps
        self.weights = weights
        self.covs = covs
    
    def sampler(self, num):
        samples = []
        for i in range(num):
            z = nr.choice(np.arange(self.n_comps), p = self.weights)
            beta = nr.multivariate_normal(np.zeros(self.dim), self.covs[z])
            samples.append(beta)
        return np.array(samples)
    
    def logpdf(self, x):
        lik = 0
        for i in range(self.n_comps):
            lik += jss.multivariate_normal.pdf(x, np.zeros(self.dim), self.covs[i]) * self.weights[i]
        loglik = jnp.log(lik)
        return loglik

    def em_algorithm(self, data, n_comps, cov0, max_itr = 1000):
        dim = self.dim
        n = data.shape[0]
        weights = np.ones(n_comps) / n_comps
        gs = nr.uniform(1,10,n_comps)
        det0 = np.linalg.det(cov0)
        covs = np.array([cov0 * gs[i] for i in range(n_comps)])
        quad = np.array([data[i].T @ np.linalg.inv(cov0) @ data[i] for i in range(n)])
        for itr in range(max_itr):
            if (itr+1) % 10 == 0:
                print(itr+1, weights, gs, weights.sum())
            # E-step
            loglik = np.zeros((n, n_comps))
            for i in range(n_comps):
                for j in range(n):
                    llk = np.log(weights[i]) - 0.5 * (dim * np.log(2*np.pi*gs[i]) + np.log(det0) + quad[j] /gs[i])
                    loglik[j,i] = np.exp(llk)
                    if loglik[j, i] < 1e-10:
                        loglik[j, i] = 1e-10
            # loglik_max = np.max(loglik, axis = 1)
            # loglik = loglik - loglik_max[:, np.newaxis]
            # loglik = loglik + loglik_max[:, np.newaxis]
            loglik = loglik / np.sum(loglik, axis = 1)[:, None]
            # M-step
            weights = np.sum(loglik, axis = 0) / n
            for i in range(n_comps):
                for j in range(n):
                    gs[i] += loglik[j, i] * quad[j]
                gs[i] = gs[i] / (np.sum(loglik[:, i]) * self.dim)
        covs = np.array([cov0 * gs[i] for i in range(n_comps)])
        self.n_comps = n_comps
        self.weights = weights
        self.gs = gs
        self.covs = covs
        return weights, covs, gs