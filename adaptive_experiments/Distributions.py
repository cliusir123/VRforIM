from pathlib import Path

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import scipy.io

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class DisBase():
    def __init__(self,
                 dim: int = 100,
                 value: int = 10) -> None:
        self.rng_seq = hk.PRNGSequence(value)
        self.dim = dim


class multivariate_normal(DisBase):
    def __init__(self, dim: int = 100, value: int = 10) -> None:
        super().__init__(dim, value)

    def sampler(self, *args):
        epsilon = jax.random.normal(next(self.rng_seq), shape=[self.dim, args[2]])
        x = args[1] @ epsilon + args[0].reshape((self.dim, 1))
        return x.T, epsilon.T

    def logpdf(self, *args):
        v = jax.lax.linalg.triangular_solve(args[2], args[0] - args[1], left_side=True, lower=True)
        logp = - 0.5 * jnp.sum(v * v)
        return logp


class Bayesian_Logistic_Regression(DisBase):
    DATASETS = ('ripley', 'pima', 'heart', 'german')

    def __init__(self, dim: int = 100) -> None:
        super().__init__(dim)

    def load_data(self, dataset):
        if dataset not in self.DATASETS:
            raise ValueError(f"dataset must be one of {self.DATASETS}")
        mat = scipy.io.loadmat(DATA_DIR / f"{dataset}.mat")
        data = mat['X']
        inputs = data[:, 0:-1].astype(np.float64)
        outputs = data[:, -1].astype(np.float64)
        if dataset in ('heart', 'german'):
            outputs = (outputs == 2).astype(np.float64)
            inputs = (inputs - inputs.mean(axis=0)) / inputs.std(axis=0)
        self.inputs = inputs
        self.outputs = outputs

    def logpdf(self, x, *paras, prior_var=1.0):
        logits = self.inputs @ x
        log_lik = distrax.Bernoulli(logits=logits).log_prob(self.outputs).sum()
        log_prior = -(0.5 * x.shape[0]) * jnp.log(2. * jnp.pi) - (0.5 / prior_var) * jnp.sum(x * x)
        return log_lik + log_prior


class GP_Regression(DisBase):
    DATASETS = ('boston', 'pendulum')

    def __init__(self) -> None:
        super().__init__()

    def load_data(self, dataset):
        if dataset not in self.DATASETS:
            raise ValueError(f"dataset must be one of {self.DATASETS}")
        mat = scipy.io.loadmat(DATA_DIR / f"{dataset}.mat")
        if dataset == 'pendulum':
            self.Xtrain = mat['X_tr']
            self.Ytrain = mat['T_tr']
        elif dataset == 'boston':
            self.Xtrain = mat['X']
            self.Ytrain = mat['y']
        mean_inputs = np.mean(np.array(self.Xtrain), axis=0)
        std_inputs = np.std(np.array(self.Xtrain), axis=0)
        self.Xtrain = (self.Xtrain - mean_inputs[None, :]) / std_inputs[None, :]
        self.Ytrain = self.Ytrain - np.mean(self.Ytrain)
        self.dim = 3
        self.n_data = self.Xtrain.shape[0]
        diff = self.Xtrain[:, None, :] - self.Xtrain[None, :, :]
        self.sqdist = np.sum(diff ** 2, axis=2)

    def logpdf(self, x, *paras):
        cov = jnp.exp(- self.sqdist / jnp.exp(x[0]) / 2) * jnp.exp(x[1]) \
            + jnp.exp(x[2]) * jnp.eye(self.n_data)
        L = jnp.linalg.cholesky(cov)
        v = jax.lax.linalg.triangular_solve(L, self.Ytrain, left_side=True, lower=True)
        log_lik = - 0.5 * jnp.sum(v * v) - 0.5 * self.n_data * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L)))
        log_prior = -(0.5 * x.shape[0]) * jnp.log(2. * jnp.pi) - (0.5 / 10) * jnp.sum(x * x)
        return log_lik + log_prior
