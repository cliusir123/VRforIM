import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, multivariate_normal
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

def im_estimator(F, X):
    fX = F(X)
    return np.mean(fX)

def is_estimator(F, samples, target_densities, proposal_densities):
    w = target_densities / proposal_densities
    Z_hat = np.mean(w)
    return np.mean(F(samples) * w / Z_hat)

def imcv_estimator(F, Fapprox, X, Y, alpha, eq_f):
    l = len(X)
    fX = F(X)
    fY = F(Y)
    fapp = Fapprox(Y)
    b_2 = np.cov(alpha * (fY - fX), (fapp - eq_f))[0,1] / np.var((fapp - eq_f))
    pg = fX + alpha * (fY - fX) - b_2 * (fapp - eq_f)
    numerator = np.mean(fX * (fX + pg)) - np.mean(fX) * np.mean(fX + pg)
    denominator = np.sum((fX[1:] - pg[:-1]) ** 2) / l
    b_1 = numerator / denominator
    delta = b_1* (alpha * (fY - fX) - b_2*(fapp - eq_f))

    return np.mean(fX + delta)

def iscv_estimator(F, Fapp, samples, target_densities, proposal_densities, eq_f):
    w = target_densities / proposal_densities
    Z_hat = np.mean(w)
    corrections = (w / Z_hat)
    f_vals = F(samples)
    fapp = Fapp(samples)
    b_2 = np.cov(f_vals * corrections, (fapp - eq_f))[0,1] / np.var((fapp - eq_f))
    return np.mean(f_vals * corrections - b_2*(fapp-eq_f))


def run_estimators(
    sample_size,
    F,
    Fapp,
    eq_f,
    target_density,
    proposal_density,
    proposal_sampler
):

    def im_step(last_x, last_q, last_pi):
        y = proposal_sampler()
        pi_y = target_density(y)
        q_y = proposal_density(y)

        pi_x = last_pi
        q_x = last_q

        alpha = min(1, (pi_y * q_x) / (pi_x * q_y))
        if np.random.rand() < alpha:
            return y, y, alpha, q_y, pi_y
        else:
            return last_x, y, alpha, q_x, pi_x

    samples = proposal_sampler(size=sample_size)
    target_densities = target_density(samples)
    proposal_densities = proposal_density(samples)

    X, Y, alpha_list = [], [], []

    x = proposal_sampler()
    qx = proposal_density(x)
    px = target_density(x)

    for _ in tqdm(range(sample_size), desc="Metropolis sampling", leave=False):
        x_new, y, alpha, qx_new, px_new = im_step(x, qx, px)
        X.append(x)
        Y.append(y)
        alpha_list.append(alpha)
        x = x_new
        qx = qx_new
        px = px_new

    X = np.array(X)
    Y = np.array(Y)
    alpha_array = np.array(alpha_list)
    results = {
        "IM": im_estimator(F, X),
        "IS": is_estimator(F, samples, target_densities, proposal_densities),
        "IMCV": imcv_estimator(F, Fapp, X, Y, alpha_array, eq_f),
        "ISCV": iscv_estimator(F, Fapp, samples, target_densities, proposal_densities, eq_f),
    }

    return results

def mock_adaptation_estimator(
    num_batches,
    sample_size,
    F,
    Fapp,
    target_density,
    proposal_generator
):
    total_results = {"IM": 0.0, "IS": 0.0, "IMCV": 0.0, "ISCV": 0.0}

    for batch in tqdm(range(num_batches), desc="Processing batches", leave=False):
        proposal_density, proposal_sampler, eq_f = proposal_generator(batch)

        results = run_estimators(
            sample_size=sample_size,
            F=F,
            Fapp=Fapp,
            eq_f=eq_f,
            target_density=target_density,
            proposal_density=proposal_density,
            proposal_sampler=proposal_sampler
        )

        for key in total_results:
            total_results[key] += results[key]

    averaged_results = {key: val / num_batches for key, val in total_results.items()}
    return averaged_results

class OptimizerSettings:
    def __init__(
        self,
        dim: int = 16,
        learning_rate: float = 0.01,
        model=None,
        value: int = 10,
        decay_type: str = "none",
        decay_steps: int = 10000,
        decay_rate: float = 0.9
    ):
        self.dim = dim
        self.lr_l = learning_rate
        self.lr_m = learning_rate * 10
        self.model = model
        self.value = value
        self.decay_type = decay_type
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        self.target = self.model
        self.logtarget = jax.jit(self.target.logpdf)

    def set_distribution_parameters(self, t_mean, t_L, p_mean, p_L):
        self.t_mean = t_mean
        self.t_L = t_L
        self.p_mean = p_mean
        self.p_L = p_L

    def init_state(self):
        self.grad_val = 0
        self.grad_L = 0
        self.cum_obj = 0

    def objective_function(self, data):
        return self.logtarget(data, self.t_mean, self.t_L)

    def _make_schedule(self, init_value):
        if self.decay_type == "exponential":
            return optax.exponential_decay(
                init_value=init_value,
                transition_steps=self.decay_steps,
                decay_rate=self.decay_rate,
                staircase=True
            )
        elif self.decay_type == "linear":
            return optax.linear_schedule(
                init_value=init_value,
                end_value=0.0,
                transition_steps=self.decay_steps
            )
        else:
            return optax.constant_schedule(init_value)

    def optimization_init(self):
        schedule_m = self._make_schedule(self.lr_m)
        schedule_l = self._make_schedule(self.lr_l)
        self.optimizer_mean = optax.adam(schedule_m)
        self.optimizer_L = optax.adam(schedule_l)
        self.opt_state_mean = self.optimizer_mean.init(self.p_mean)
        self.opt_state_L = self.optimizer_L.init(self.p_L)
        self.obj_grad = jax.grad(self.objective_function)

    def optimization_collect(self, sample, epsilon, batch_size):
        cur_grad = self.obj_grad(sample)
        self.grad_val += cur_grad / batch_size
        self.grad_L += (
            cur_grad.reshape((self.dim, 1)) @ epsilon.reshape((1, self.dim)) +
            jnp.diag(1 / jnp.diag(self.p_L))
        ) / batch_size
        self.cum_obj += (
            self.objective_function(sample) + jnp.sum(jnp.log(jnp.diag(self.p_L)))
        ) / batch_size

    def optimization_update(self):
        updates, self.opt_state_mean = self.optimizer_mean.update(-self.grad_val, self.opt_state_mean)
        self.p_mean = optax.apply_updates(self.p_mean, updates)

        updates, self.opt_state_L = self.optimizer_L.update(-self.grad_L, self.opt_state_L)
        self.p_L = optax.apply_updates(self.p_L, updates)

        self.p_L = jnp.tril(self.p_L)
        bounded_diag = jnp.clip(jnp.diag(self.p_L), 0.001, 1000.0)
        self.p_L = self.p_L - jnp.diag(jnp.diag(self.p_L)) + jnp.diag(bounded_diag)

def generate_proposal_list_from_optimizer(
    optimizer: OptimizerSettings,
    num_batches: int,
    batch_size: int,
    kappa: float
):
    proposal_list = []
    dim = optimizer.dim

    for batch_index in tqdm(range(num_batches), desc="Optimizing proposals"):
        epsilons = np.random.normal(size=(batch_size, dim))
        samples = np.array([
            optimizer.p_mean + optimizer.p_L @ eps
            for eps in epsilons
        ])
        samples = jnp.array(samples)

        optimizer.init_state()
        for i in tqdm(range(batch_size), desc=f"Batch {batch_index} gradients", leave=False):
            optimizer.optimization_collect(
                sample=samples[i],
                epsilon=epsilons[i],
                batch_size=batch_size
            )

        optimizer.optimization_update()

        mu = np.array(optimizer.p_mean)
        L = np.array(optimizer.p_L)
        Sigma = L @ L.T

        proposal_density = lambda x, mu=mu, Sigma=Sigma: multivariate_normal.pdf(x, mean=mu, cov=Sigma)
        proposal_sampler = lambda size=None, mu=mu, Sigma=Sigma: np.random.multivariate_normal(mean=mu, cov=Sigma, size=size)

        mu_S = np.sum(mu)
        sigma_S2 = np.sum(Sigma)
        eq_f = norm.cdf((kappa * mu_S) / np.sqrt(1 + kappa**2 * sigma_S2))

        proposal_list.append((proposal_density, proposal_sampler, eq_f))

    return proposal_list

def proposal_generator_from_list(proposal_list):
    def generator(batch_index):
        return proposal_list[batch_index]
    return generator

class StandardGaussian:
    def __init__(self, dim):
        self.dim = dim

    def logpdf(self, x, mean, L):
        solve = jax.scipy.linalg.solve_triangular(L, (x - mean), lower=True)
        return -0.5 * jnp.sum(solve**2) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * self.dim * jnp.log(2 * jnp.pi)

def run_experiment_simple(
    M,
    num_batches,
    sample_size,
    F,
    Fapp,
    mean_true,
    target_density,
    proposal_generator
):
    IM_vals = []
    IS_vals = []
    IMCV_vals = []
    ISCV_vals = []

    for m in tqdm(range(M), desc="Running experiments"):
        res = mock_adaptation_estimator(
            num_batches=num_batches,
            sample_size=sample_size,
            F=F,
            Fapp=Fapp,
            target_density=target_density,
            proposal_generator=proposal_generator
        )
        IM_vals.append(res["IM"])
        IS_vals.append(res["IS"])
        IMCV_vals.append(res["IMCV"])
        ISCV_vals.append(res["ISCV"])

    IM_vals = np.array(IM_vals)
    IS_vals = np.array(IS_vals)
    IMCV_vals = np.array(IMCV_vals)
    ISCV_vals = np.array(ISCV_vals)

    def print_stats(name, vals):
        mean = np.mean(vals)
        bias = mean - mean_true
        var = np.var(vals, ddof=1)
        mse = bias**2 + var
        print(f"{name}:")
        print(f"  Mean    = {mean:.6f}")
        print(f"  Bias    = {bias:.6e}")
        print(f"  Variance= {var:.6e}")
        print(f"  MSE     = {mse:.6e}\n")
        return mse

    mse_im = print_stats("IM", IM_vals)
    mse_is = print_stats("IS", IS_vals)
    mse_imcv = print_stats("IMCV", IMCV_vals)
    mse_iscv = print_stats("ISCV", ISCV_vals)

    return {
        "IM": mse_im,
        "IS": mse_is,
        "IMCV": mse_imcv,
        "ISCV": mse_iscv
    }

mu_target = np.array([0.0, 0.0])
sigma_target = np.eye(2)

mu_proposal = np.array([0.2, 0.2])
sigma_proposal = np.array([[1.2, 0.2],
                           [0.2, 1.2]])

target_density = lambda x: multivariate_normal.pdf(x, mean=mu_target, cov=sigma_target)
proposal_density = lambda x: multivariate_normal.pdf(x, mean=mu_proposal, cov=sigma_proposal)
proposal_sampler = lambda size=None: np.random.multivariate_normal(mean=mu_proposal, cov=sigma_proposal, size=size)

F = lambda x: np.exp(np.sum(x, axis=1)) / (1 + np.exp(np.sum(x, axis=1)))

kappa = np.sqrt(np.pi / 8)
Fapp = lambda x: norm.cdf(kappa * np.sum(x, axis=1))

mean_true = 0.5
mu_S = np.sum(mu_proposal) 
sigma_S2 = np.sum(sigma_proposal)
eq_f = norm.cdf((kappa * mu_S) / np.sqrt(1 + kappa**2 * sigma_S2))


sample_sizes = [50, 100,150, 200,300,400, 500, 600,800, 1000]
num_batches = 100
dim = 2
results = {sample_size: {} for sample_size in sample_sizes}

for sample_size in tqdm(sample_sizes, desc="Running experiments for different sample sizes"):
    print(f"\nRunning experiment with sample size {sample_size}")
    
    model = StandardGaussian(dim=dim)
    optimizer = OptimizerSettings(
        dim=dim,
        learning_rate=0.0001,
        model=model
    )

    optimizer.set_distribution_parameters(
        t_mean=jnp.array(mu_target),
        t_L=jnp.linalg.cholesky(jnp.array(sigma_target)),
        p_mean=jnp.array(mu_proposal),
        p_L=jnp.linalg.cholesky(jnp.array(sigma_proposal))
    )
    optimizer.optimization_init()
    
    proposal_list = generate_proposal_list_from_optimizer(
        optimizer=optimizer,
        num_batches=num_batches,
        batch_size=sample_size,
        kappa=kappa
    )

    proposal_generator = proposal_generator_from_list(proposal_list)

    results[sample_size] = run_experiment_simple(
        M=100,
        num_batches=num_batches,
        sample_size=sample_size,
        F=F,
        Fapp=Fapp,
        mean_true=mean_true,
        target_density=target_density,
        proposal_generator=proposal_generator
    )

plt.figure(figsize=(12, 6))
estimators = ['IM', 'IS', 'IMCV', 'ISCV']
markers = ['+', '*', '^', 'x']
colors = ['blue', 'red', 'green', 'purple']

for estimator, marker, color in zip(estimators, markers, colors):
    mse_values = [results[sample_size][estimator] for sample_size in sample_sizes]
    plt.plot(
        sample_sizes,
        mse_values,
        marker=marker,
        color=color,
        label=estimator,
        linestyle='-',
        markersize=12,
        alpha=0.7
    )

plt.xlabel('Batch Size', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('log')
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

plt.savefig('mse_vs_batch_size.pdf', dpi=300, bbox_inches='tight')
plt.close()

num_batches_list = [10, 20, 50, 75, 100, 125, 150, 175, 200,250,300]
sample_size = 100
dim=2
results = {num_batches: {} for num_batches in num_batches_list}

for num_batches in tqdm(num_batches_list, desc="Running experiments for different numbers of batches"):
    print(f"\nRunning experiment with {num_batches} batches")
    
    model = StandardGaussian(dim=dim)
    optimizer = OptimizerSettings(
        dim=dim,
        learning_rate=0.0001,
        model=model
    )

    optimizer.set_distribution_parameters(
        t_mean=jnp.array(mu_target),
        t_L=jnp.linalg.cholesky(jnp.array(sigma_target)),
        p_mean=jnp.array(mu_proposal),
        p_L=jnp.linalg.cholesky(jnp.array(sigma_proposal))
    )
    optimizer.optimization_init()
    
    proposal_list = generate_proposal_list_from_optimizer(
        optimizer=optimizer,
        num_batches=num_batches,
        batch_size=sample_size,
        kappa=kappa
    )

    proposal_generator = proposal_generator_from_list(proposal_list)

    results[num_batches] = run_experiment_simple(
        M=100,
        num_batches=num_batches,
        sample_size=sample_size,
        F=F,
        Fapp=Fapp,
        mean_true=mean_true,
        target_density=target_density,
        proposal_generator=proposal_generator
    )

plt.figure(figsize=(12, 6))
estimators = ['IM', 'IS', 'IMCV', 'ISCV']
markers = ['+', '*', '^', 'x']
colors = ['blue', 'red', 'green', 'purple']

for estimator, marker, color in zip(estimators, markers, colors):
    mse_values = [results[num_batches][estimator] for num_batches in num_batches_list]
    plt.plot(
        num_batches_list,
        mse_values,
        marker=marker,
        color=color,
        label=estimator,
        linestyle='-',
        markersize=12,
        alpha=0.7
    )

plt.xlabel('Number of Batches', fontsize=14)
plt.ylabel('MSE', fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.yscale('log')
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

plt.savefig('mse_vs_num_batches.pdf', dpi=300, bbox_inches='tight')
plt.close()
