import jax.numpy as jnp
import jax
import jax.scipy.stats as jss
import numpy as np
import numpy.random as nr


class gprior():
    def __init__(self,dim) -> None:
        self.dim = dim
    
    def set_parameters(self, cov0, input, g):
        self.cov0 = cov0
        self.input = input
        self.g = g
    
    def sampler(self, num):
        samples = []
        for i in range(num):
            cov = self.cov0 * self.g[i]
            beta = nr.multivariate_normal(np.zeros(self.dim), cov)
            samples.append(beta)
        return np.array(samples)

    def logpdf(self, x):
        n = 1000
        w = nr.beta(1,1,n)
        g = w/(1-w)
        self.g = g
        v = self.input @ x.T
        S = v.T @ v
        logl = - 0.5 * (self.dim * jnp.log(self.g) + S / self.g)
        logl = jax.scipy.special.logsumexp(logl)
        return logl

class mix_normal():
    def __init__(self, dim) -> None:
        self.dim = dim
        
    
    def set_parameters(self, n_comps, weights, gs, covs, input):
        self.n_comps = n_comps
        self.weights = weights
        self.gs = gs
        self.input  = input
        self.covs = covs
    
    def sampler(self, num):
        samples = []
        for i in range(num):
            z = nr.choice(np.arange(self.n_comps), p = self.weights)
            beta = nr.multivariate_normal(np.zeros(self.dim), self.covs[z])
            samples.append(beta)
        return np.array(samples)
    
    def logpdf(self, x):
        v = self.input @ x.T
        S = v.T @ v
        loglik = jnp.log(self.weights) - 0.5 * (self.dim * jnp.log(self.gs) + S / self.gs)
        loglik = jax.scipy.special.logsumexp(loglik)
        return loglik

    def em_algorithm(self, data, n_comps, input, max_itr = 1000):
        dim = self.dim
        n = data.shape[0]
        weights = np.ones(n_comps) / n_comps
        w = nr.beta(1,1,n_comps)
        gs = w/(1-w)
        ddet = np.linalg.det(np.linalg.inv(input.T @ input))
        quad = np.array([data[i].T @ (input.T @ input) @ data[i] for i in range(n)])
        for itr in range(max_itr):
            if (itr+1) % 1000 == 0:
                print(itr+1, weights, gs)
            # E-step
            loglik = np.zeros((n, n_comps))
            for i in range(n_comps):
                for j in range(n):
                    llk = np.log(weights[i]) - 0.5 * (dim * np.log(2*np.pi*gs[i]) + np.log(ddet)+ quad[j] /gs[i])
                    loglik[j,i] = np.exp(llk)
                    if loglik[j, i] < 1e-10:
                        loglik[j, i] = 1e-10
            loglik = loglik / np.sum(loglik, axis = 1)[:, None]
            # M-step
            weights = np.sum(loglik, axis = 0) / n
            for i in range(n_comps):
                for j in range(n):
                    gs[i] += loglik[j, i] * quad[j]
                gs[i] = gs[i] / (np.sum(loglik[:, i]) * self.dim)
        self.n_comps = n_comps
        self.weights = weights
        self.gs = gs
        return weights, gs

