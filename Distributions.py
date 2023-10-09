import jax
import jax.numpy as jnp
import haiku as hk


class DisBase():
    def __init__(self,
                 dim: int = 100,
                 value: int = 10) -> None:
        self.rng_seq = hk.PRNGSequence(value)
        self.dim = dim


class student_t(DisBase):
    def __init__(self, dim: int = 100, value: int = 10, df: int = 3) -> None:
        super().__init__(dim, value)
        self.df = df

    def sampler(self, *args): #args: 0 loc 1 scale 2 number
        epsilon = jax.random.normal(next(self.rng_seq), shape = [self.dim, args[2]])
        y = args[1] @ epsilon
        u = u = jnp.sqrt(self.df / nr.chisquare(self.df))
        x = u * y + args[0].reshape((self.dim,1))
        return x.T, epsilon.T

    def logpdf(self, *args): #args: 0 data 1 loc 2 scale
        v = jax.lax.linalg.triangular_solve(args[2], args[0] - args[1], left_side=True, lower=True)
        logp = - 0.5 * (self.df + self.dim) * jnp.log(1 + jnp.sum(v * v) / self.df)
        return logp


class multivariate_normal(DisBase):
    def __init__(self, dim: int = 100, value: int = 10) -> None:
        super().__init__(dim, value)

    def sampler(self, *args): #args: 0 loc 1 scale 2 number , 3 beta
        epsilon = jax.random.normal(next(self.rng_seq), shape = [self.dim, args[2]])
        x = args[1] @ epsilon + args[0].reshape((self.dim,1))
        return x.T, epsilon.T

    def logpdf(self, *args): #args: 0 data 1 loc 2 scale
        v = jax.lax.linalg.triangular_solve(args[2], args[0] - args[1], left_side=True, lower=True)
        logp = - 0.5 * jnp.sum(v * v)
        return logp