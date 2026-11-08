import jax
import jax.numpy as jnp
import haiku as hk
import numpy.random as nr
import jax.scipy.stats as jss
import numpy as np
import pandas as pd
import distrax
import scipy
import scipy.stats as ss

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
        epsilon = jax.random.normal(next(self.rng_seq), shape = [self.dim, args[2]])
        x = args[1] @ epsilon + args[0].reshape((self.dim,1))
        return x.T, epsilon.T

    def logpdf(self, *args):
        v = jax.lax.linalg.triangular_solve(args[2], args[0] - args[1], left_side=True, lower=True)
        logp = - 0.5 * jnp.sum(v * v)
        return logp
