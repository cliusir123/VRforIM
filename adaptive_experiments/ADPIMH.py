import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as nr
import optax
from tqdm import tqdm

from Distributions import multivariate_normal

CHUNK = 256


class Basic_settings():
    def __init__(
        self,
        target_str: str = 'multivariate_normal',
        itrnum: int = 10000,
        dim: int = 16,
        learning_rate: float = 0.001,
        model=None,
        value: int = 10,
        schedule: str = 'exponential',
        cov_scale: float = 1.0,
    ) -> None:
        if schedule not in ('exponential', 'cosine'):
            raise ValueError("schedule must be 'exponential' or 'cosine'")
        self.schedule = schedule
        self.cov_scale = cov_scale
        self.target_str = target_str
        self.itrnum = itrnum
        self.dim = dim
        self.lr_l = learning_rate
        self.lr_m = learning_rate * 10
        if target_str == 'multivariate_normal':
            self.target = multivariate_normal(dim, value)
        else:
            if model is None:
                raise ValueError("a model must be given for a non-Gaussian target")
            self.target = model
        self.proposal = multivariate_normal(dim, value)

    def set_distribution_parameters(self, t_mean, t_L, p_mean, p_L) -> None:
        self.t_mean = t_mean
        self.t_L = t_L
        self.p_mean = jnp.asarray(p_mean, dtype=jnp.float32)
        self.p_L = jnp.asarray(p_L, dtype=jnp.float32)

        def logt(y):
            return self.target.logpdf(y, t_mean, t_L)
        self.logtarget = jax.jit(logt)
        self.logproposal = jax.jit(self.proposal.logpdf)
        self._vlogt = jax.jit(jax.vmap(logt))
        self._vlogq = jax.jit(jax.vmap(self.proposal.logpdf, in_axes=(0, None, None)))
        self._vvalgrad = jax.jit(jax.vmap(jax.value_and_grad(logt)))

    @property
    def im_L(self):
        return self.p_L * jnp.sqrt(self.cov_scale)

    @staticmethod
    def _chunked(fn, Y, *args):
        m = Y.shape[0]
        size = min(CHUNK, m)
        outs = []
        for s in range(0, m, size):
            block = Y[s:s + size]
            k = block.shape[0]
            if k < size:
                block = jnp.concatenate([block, jnp.repeat(block[-1:], size - k, axis=0)])
            out = fn(block, *args)
            outs.append(jax.tree_util.tree_map(lambda a: np.asarray(a)[:k], out))
        if isinstance(outs[0], tuple):
            return tuple(np.concatenate(parts) for parts in zip(*outs))
        return np.concatenate(outs)

    def init_state(self) -> None:
        x0, _ = self.proposal.sampler(self.p_mean, self.im_L, 1)
        self.current = np.asarray(x0).reshape((self.dim,))
        self.log_t = float(self.logtarget(self.current))
        self.log_p = float(self.logproposal(self.current, self.p_mean, self.im_L))
        self.cum_obj = 0

    def IMHsampling(self, num):
        Y, eps = self.proposal.sampler(self.p_mean, self.im_L, num)
        return Y, eps

    def IMHsteps(self, Y, X, alpha, accepted, offset):
        Yj = jnp.asarray(Y)
        log_t = self._chunked(self._vlogt, Yj)
        log_q = self._chunked(self._vlogq, Yj, self.p_mean, self.im_L)
        Y = np.asarray(Y)
        for j in range(Y.shape[0]):
            i = offset + j
            X[i] = self.current
            a = min(1.0, float(np.exp((log_t[j] - self.log_t) - (log_q[j] - self.log_p))))
            alpha[i] = a
            if nr.uniform(0.0, 1.0) <= a:
                self.current = Y[j]
                self.log_t = float(log_t[j])
                self.log_p = float(log_q[j])
                accepted[i] = 1

    def _lr_schedule(self, init_value):
        if self.schedule == 'cosine':
            return optax.cosine_decay_schedule(init_value=init_value, decay_steps=max(1, self.n_batch))
        return optax.exponential_decay(init_value=init_value, transition_steps=4000,
                                       decay_rate=0.2, staircase=True)

    def optimization_init(self):
        schedule_m = self._lr_schedule(self.lr_m)
        schedule_l = self._lr_schedule(self.lr_l)
        self.optimizer_mean = optax.adam(schedule_m)
        self.optimizer_L = optax.adam(schedule_l)
        self.opt_state_mean = self.optimizer_mean.init(self.p_mean)
        self.opt_state_L = self.optimizer_L.init(self.p_L)

    def optimization_collect(self, epsilon):
        eps = np.asarray(epsilon)
        B = eps.shape[0]
        samples = jnp.asarray(self.p_mean)[None, :] + jnp.asarray(eps) @ self.p_L.T
        vals, grads = self._chunked(self._vvalgrad, samples)
        self.grad_val = jnp.asarray(grads.mean(axis=0))
        self.grad_L = jnp.asarray(grads.T @ eps / B) + jnp.diag(1 / jnp.diag(self.p_L))
        self.cum_obj = float(vals.mean() + jnp.sum(jnp.log(jnp.diag(self.p_L))))

    def optimization_update(self):
        updates, self.opt_state_mean = self.optimizer_mean.update(-self.grad_val, self.opt_state_mean)
        self.p_mean = optax.apply_updates(self.p_mean, updates)
        updates, self.opt_state_L = self.optimizer_L.update(-self.grad_L, self.opt_state_L)
        p_L = jnp.tril(optax.apply_updates(self.p_L, updates))
        bounded_diag = jnp.clip(jnp.diag(p_L), 0.001, 1000.)
        self.p_L = p_L - jnp.diag(jnp.diag(p_L)) + jnp.diag(bounded_diag)
        self.log_p = float(self.logproposal(self.current, self.p_mean, self.im_L))

    def proposal_params(self):
        p_L = np.asarray(self.im_L, dtype=np.float64)
        return np.asarray(self.p_mean, dtype=np.float64).reshape(self.dim), p_L @ p_L.T


class IndMH(Basic_settings):

    def IMH(self):
        X = np.zeros((self.itrnum, self.dim))
        alpha = np.zeros(self.itrnum)
        accepted = np.zeros(self.itrnum)

        self.init_state()
        Y, _ = self.IMHsampling(self.itrnum)
        Y = np.asarray(Y)
        self.IMHsteps(Y, X, alpha, accepted, 0)

        mean, cov = self.proposal_params()
        return dict(X=X, Y=Y, alpha=alpha, accepted=accepted,
                    means=mean[None], covs=cov[None],
                    batch_idx=np.zeros(self.itrnum, dtype=int))


class BAIM(Basic_settings):

    def __init__(self, target_str='multivariate_normal', itrnum=10000, batch_size=50, dim=16,
                 learning_rate=0.01, model=None, value=10, schedule='exponential', cov_scale=1.0):
        super().__init__(target_str, itrnum, dim, learning_rate, model, value, schedule, cov_scale)
        if itrnum % batch_size != 0:
            raise ValueError("itrnum must be a multiple of batch_size")
        self.batch_size = batch_size
        self.n_batch = itrnum // batch_size

    def adpIMH(self):
        X = np.zeros((self.itrnum, self.dim))
        Y = np.zeros((self.itrnum, self.dim))
        alpha = np.zeros(self.itrnum)
        accepted = np.zeros(self.itrnum)
        means = np.zeros((self.n_batch, self.dim))
        covs = np.zeros((self.n_batch, self.dim, self.dim))
        loss = np.zeros(self.n_batch)

        self.init_state()
        self.optimization_init()
        B = self.batch_size
        for b in tqdm(range(self.n_batch), desc="adaptive IM batches", leave=False):
            means[b], covs[b] = self.proposal_params()
            proposals, epsilon = self.IMHsampling(B)
            proposals = np.asarray(proposals)
            Y[b * B:(b + 1) * B] = proposals
            self.IMHsteps(proposals, X, alpha, accepted, b * B)
            self.optimization_collect(epsilon)
            loss[b] = self.cum_obj
            self.optimization_update()

        return dict(X=X, Y=Y, alpha=alpha, accepted=accepted, means=means, covs=covs,
                    batch_idx=np.arange(self.itrnum) // B, loss=loss)


class adp_only(Basic_settings):

    def __init__(self, target_str='multivariate_normal', itrnum=10000, batch_size=50, dim=16,
                 learning_rate=0.01, model=None, value=10, schedule='exponential', cov_scale=1.0):
        super().__init__(target_str, itrnum, dim, learning_rate, model, value, schedule, cov_scale)
        self.batch_size = batch_size
        self.n_batch = itrnum // batch_size

    def adp(self, verbose=True):
        loss = np.zeros(self.n_batch)
        self.init_state()
        self.optimization_init()
        for b in tqdm(range(self.n_batch), desc="burn-in adaptation", disable=None if verbose else True, mininterval=5):
            _, epsilon = self.IMHsampling(self.batch_size)
            self.optimization_collect(epsilon)
            loss[b] = self.cum_obj
            self.optimization_update()
        return loss
