import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.stats import norm, t,multivariate_normal
from statsmodels.graphics.tsaplots import plot_acf
import jax
import jax.numpy as jnp
import optax
from jax.random import PRNGKey, normal
from scipy.stats import multivariate_normal
import scipy
from tqdm import tqdm

def independent_metropolis_step(target_density, proposal_density, proposal_sampler, last_step, last_den):
    """Single step of independent Metropolis algorithm"""
    y = proposal_sampler()
    pi_y = target_density(y)
    q_y = proposal_density(y)

    x = last_step.squeeze()
    pi_x = target_density(x)
    q_x = last_den

    alpha = min(1, (pi_y * q_x) / (pi_x * q_y))
    if np.random.rand() < alpha:
        x = y
        pi_x = pi_y
        q_x = q_y

    return x, y, alpha, q_x

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
    denominator = np.sum((fX[0 : l - 1] - pg[1 : l]) ** 2) / l
    b_1 = numerator / denominator
    delta = b_1* (alpha * (fY - fX) - b_2*(fapp - eq_f))
    # delta = alpha * (fY - fX) - (fapp - eq_f)

    return np.mean(fX + delta)

def iscv_estimator(F, Fapp, samples, target_densities, proposal_densities, eq_f):
    w = target_densities / proposal_densities
    Z_hat = np.mean(w)
    corrections = (w / Z_hat)
    f_vals = F(samples)
    fapp = Fapp(samples)
    b_2 = np.cov(f_vals * corrections, (fapp - eq_f))[0,1] / np.var((fapp - eq_f))
    return np.mean(f_vals * corrections - b_2*(fapp-eq_f))
    # return np.mean(f_vals * corrections - (fapp-eq_f))


def run_estimators(
    sample_size,
    F,
    Fapp,
    eq_f,
    target_density,
    proposal_density,
    proposal_sampler
):
    """
    Run IM, IS, IMCV, ISCV estimators using provided functions and settings.

    Parameters:
        sample_size (int): Number of samples
        F (function): Objective function F(x), accepts (n, d) array
        Fapp (function): Approximation of F(x), same input/output
        eq_f (float): Theoretical expectation of Fapp under proposal
        mean_true (float): True expectation of F under target (for reference)
        target_density (function): Target density, accepts (n, d) or (d,)
        proposal_density (function): Proposal density, accepts (n, d) or (d,)
        proposal_sampler (function): Proposal sampler with size keyword

    Returns:
        dict: Estimator results {'IM', 'IS', 'IMCV', 'ISCV', 'TRUE'}
    """

    # === Helper: one IM step ===
    def im_step(last_x, last_q, last_pi):
        y = proposal_sampler()
        pi_y = target_density(y)
        q_y = proposal_density(y)

        pi_x = last_pi
        q_x = last_q

        alpha = min(1, (pi_y * q_x) / (pi_x * q_y))
        if np.random.rand() < alpha:
            return y, y, alpha, q_y, pi_y  # accepted
        else:
            return last_x, y, alpha, q_x, pi_x  # rejected

    # === Step 1: Sampling from proposal ===
    samples = proposal_sampler(size=sample_size)
    target_densities = target_density(samples)
    proposal_densities = proposal_density(samples)

    # === Step 2: Independent Metropolis sampling ===
    X, Y, alpha_list = [], [], []

    x = proposal_sampler()  # initial
    qx = proposal_density(x)
    px = target_density(x)

    # Main sampling after burn-in
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
    # === Step 3: Estimators ===
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
    """
    Perform mock adaptation and average estimators across improving proposals.
    """
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

def proposal_generator(mu_start, mu_end, Sigma_start, Sigma_end, num_batches, kappa):
    def generator(batch_index):
        # Compute interpolation weight
        alpha = batch_index / num_batches

        # Interpolate mean and covariance
        mu = (1 - alpha) * mu_start + alpha * mu_end
        Sigma = (1 - alpha) * Sigma_start + alpha * Sigma_end
        # Create proposal density and sampler
        proposal_density = lambda x: multivariate_normal.pdf(x, mean=mu, cov=Sigma)
        proposal_sampler = lambda size=None: np.random.multivariate_normal(mean=mu, cov=Sigma, size=size)
        mu_S = np.sum(mu)
        sigma_S2 = np.sum(Sigma)
        eq_f = norm.cdf((kappa * mu_S) / np.sqrt(1 + kappa**2 * sigma_S2))
        # proposal_density = lambda x: norm.pdf(x, loc=mu, scale=Sigma)
        # proposal_sampler = lambda size=None: np.random.normal(loc=mu, scale=Sigma, size=size)
        # eq_f = norm.cdf((kappa * mu) / np.sqrt(1 + kappa**2 * Sigma))

        return proposal_density, proposal_sampler, eq_f
    return generator

class OptimizerSettings:
    def __init__(
        self,
        dim: int = 16,
        learning_rate: float = 0.01,
        model=None,
        value: int = 10,
        decay_type: str = "none",  # 'none', 'exponential', 'linear'
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
        else:  # no decay
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
    """
    Run optimizer over num_batches, collect proposals (density, sampler, eq_f).
    """
    proposal_list = []
    dim = optimizer.dim

    for batch_index in tqdm(range(num_batches), desc="Optimizing proposals"):
        # ---- Step 1: Sample from current proposal ----
        epsilons = np.random.normal(size=(batch_size, dim))
        samples = np.array([
            optimizer.p_mean + optimizer.p_L @ eps
            for eps in epsilons
        ])
        samples = jnp.array(samples)

        # ---- Step 2: Collect gradients ----
        optimizer.init_state()
        for i in tqdm(range(batch_size), desc=f"Batch {batch_index} gradients", leave=False):
            optimizer.optimization_collect(
                sample=samples[i],
                epsilon=epsilons[i],
                batch_size=batch_size
            )

        # ---- Step 3: Update parameters ----
        optimizer.optimization_update()

        # ---- Step 4: Build proposal functions ----
        mu = np.array(optimizer.p_mean)
        L = np.array(optimizer.p_L)
        Sigma = L @ L.T

        proposal_density = lambda x, mu=mu, Sigma=Sigma: multivariate_normal.pdf(x, mean=mu, cov=Sigma)
        proposal_sampler = lambda size=None, mu=mu, Sigma=Sigma: np.random.multivariate_normal(mean=mu, cov=Sigma, size=size)

        # ---- Step 5: Compute eq_f ----
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
    # Store results for each estimator
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

    # Convert to numpy arrays
    IM_vals = np.array(IM_vals)
    IS_vals = np.array(IS_vals)
    IMCV_vals = np.array(IMCV_vals)
    ISCV_vals = np.array(ISCV_vals)

    # Helper to print stats and return MSE
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

    # Get MSE values
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


# mu_target = 0.0
# sigma_target = 1.0

# mu_proposal = 0.1
# sigma_proposal = 1.01

# target_density = lambda x: norm.pdf(x, loc=mu_target, scale=sigma_target)
# proposal_density = lambda x: norm.pdf(x, loc=mu_proposal, scale=sigma_proposal)
# proposal_sampler = lambda size=None: np.random.normal(loc=mu_proposal, scale=sigma_proposal, size=size)

# F = lambda x: np.exp(x) / (1 + np.exp(x))
# kappa = np.sqrt(np.pi / 8)
# Fapp = lambda x: norm.cdf(kappa * x)

# mean_true = 0.5
# eq_f = norm.cdf((kappa * mu_proposal) / np.sqrt(1 + kappa ** 2 * sigma_proposal ** 2))

# Define sample sizes to test
sample_sizes = [50, 100,150, 200,300,400, 500, 600,800, 1000]  # Example sample sizes
num_batches = 100  # Fixed number of batches
dim = 2
# Store results for each sample size
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

# Plot results
plt.figure(figsize=(12, 6))
estimators = ['IM', 'IS', 'IMCV', 'ISCV']
markers = ['+', '*', '^', 'x']  # Different marker styles for each estimator
colors = ['blue', 'red', 'green', 'purple']  # Different colors for each estimator

for estimator, marker, color in zip(estimators, markers, colors):
    mse_values = [results[sample_size][estimator] for sample_size in sample_sizes]
    plt.plot(
        sample_sizes,
        mse_values,
        marker=marker,
        color=color,      # Different color for each estimator
        label=estimator,
        linestyle='-',
        markersize=12,    # Larger symbols
        alpha=0.7         # Make lines lighter (optional, can adjust)
    )

plt.xlabel('Batch Size', fontsize=14)
plt.ylabel('MSE', fontsize=14)  # y-label is just "MSE"
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('log')
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

# Save the figure as a PDF
plt.savefig('mse_vs_batch_size.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Define number of batches to test
num_batches_list = [10, 20, 50, 75, 100, 125, 150, 175, 200,250,300]  # Example number of batches
sample_size = 100  # Fixed sample size
dim=2
# Store results for each number of batches
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

# Plot results
plt.figure(figsize=(12, 6))
estimators = ['IM', 'IS', 'IMCV', 'ISCV']
markers = ['+', '*', '^', 'x']  # Different marker styles for each estimator
colors = ['blue', 'red', 'green', 'purple']  # Different colors for each estimator

for estimator, marker, color in zip(estimators, markers, colors):
    mse_values = [results[num_batches][estimator] for num_batches in num_batches_list]
    plt.plot(
        num_batches_list,
        mse_values,
        marker=marker,
        color=color,      # Different color for each estimator
        label=estimator,
        linestyle='-',
        markersize=12,    # Larger symbols
        alpha=0.7         # Make lines lighter (optional, can adjust)
    )

# Increase font size for axis labels
plt.xlabel('Number of Batches', fontsize=14)
plt.ylabel('MSE', fontsize=14)  # Changed y-axis label to just "MSE"

# Increase font size for axis ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.yscale('log')
plt.legend(fontsize=12)  # Increase font size for legend
plt.grid(True)
plt.tight_layout()

# Save the figure as a PDF
plt.savefig('mse_vs_num_batches.pdf', dpi=300, bbox_inches='tight')
plt.close()