import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t,multivariate_normal
from statsmodels.graphics.tsaplots import plot_acf
from scipy.interpolate import interp1d

def independent_metropolis(n, target_density, proposal_density, proposal_sampler, burnin=100):
    total_n = n + burnin
    Y_proposals = proposal_sampler(size=total_n)
    pi_y = target_density(Y_proposals)
    q_y = proposal_density(Y_proposals)

    x = proposal_sampler(1)
    pi_x = target_density(x)
    q_x = proposal_density(x)

    X_samples = [x]
    alphas = []
    for i in range(total_n):
        y = Y_proposals[i]
        alpha = min(1, (pi_y[i] * q_x) / (pi_x * q_y[i]))
        if np.random.rand() < alpha:
            x = y
            pi_x = pi_y[i]
            q_x = q_y[i]
        X_samples.append(x)
        alphas.append(alpha)

    return np.array(X_samples[-n-1:-1]), Y_proposals[-n:], np.array(alphas[-n:])

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

    # delta = b_1* (alpha * (fY - fX) - b_2*(fapp - eq_f))
    delta = alpha * (fY - fX) - (fapp - eq_f)
    return np.mean(fX + delta)

def cmc_estimator(F, target_sampler, n_samples):
    """Crude Monte Carlo estimator - sample directly from target"""
    X_cmc = target_sampler(size=n_samples)
    fX = F(X_cmc)
    return np.mean(fX)

def estimate(
    n, m, F, Fapp, eq_f,
    target_density, proposal_density, proposal_sampler, target_sampler
):
    imcv_estimates = []
    cmc_estimates = []

    for _ in range(m):
        # IMCV estimation using Metropolis-Hastings
        X, Y, alpha = independent_metropolis(
            n, target_density, proposal_density, proposal_sampler, burnin=1000
        )
        imcv = imcv_estimator(F, Fapp, X, Y, alpha, eq_f)
        imcv_estimates.append(imcv)
        
        # CMC estimation using direct sampling from target
        cmc = cmc_estimator(F, target_sampler, n)
        cmc_estimates.append(cmc)

    imcv_estimates = np.array(imcv_estimates)
    cmc_estimates = np.array(cmc_estimates)

    return imcv_estimates, cmc_estimates

def calculate_vrf(imcv_estimates, cmc_estimates):
    """Calculate Variance Reduction Factor"""
    var_cmc = np.var(cmc_estimates)
    var_imcv = np.var(imcv_estimates)
    vrf = var_cmc / var_imcv if var_imcv > 0 else 1.0
    return vrf

from scipy.integrate import dblquad

mu_target = 0.0
sigma_target = 1.0

target_density = lambda x: norm.pdf(x, loc=mu_target, scale=sigma_target)
target_sampler = lambda size=None: np.random.normal(loc=mu_target, scale=sigma_target, size=size)

F = lambda x: x
Fapp = lambda x: x
eq_f = 0.0

n = 5000
m = 100

sigma_proposal_values = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

# derived from Mathematica
vrf_list = [
    10.121975672868782649,
    2.5146884288958090848,
    1.0901514415033692100,
    0.59214974278451773311,
    0.36406887023521004179,
    0.24239959417065334440
]


# Store results for each sigma proposal value
all_imcv_results = []
all_cmc_results = []
vrf_values = []

for sigma_proposal in sigma_proposal_values:
    print(f"Running simulation for sigma_proposal = {sigma_proposal}")
    
    # Update proposal parameters
    mu_proposal = 0.0
    
    # Update proposal functions
    proposal_density = lambda x: norm.pdf(x, loc=mu_proposal, scale=sigma_proposal)
    proposal_sampler = lambda size=None: np.random.normal(loc=mu_proposal, scale=sigma_proposal, size=size)
    
    # Run estimation
    imcv_results, cmc_results = estimate(n, m, F, Fapp, eq_f, target_density, proposal_density, proposal_sampler, target_sampler)
    
    # Calculate VRF
    vrf = calculate_vrf(imcv_results, cmc_results)
    
    all_imcv_results.append(imcv_results)
    all_cmc_results.append(cmc_results)
    vrf_values.append(vrf)
    
    print(f"VRF for sigma_proposal = {sigma_proposal}: {vrf:.4f}")

# Create VRF plot
plt.figure(figsize=(12, 8))

# Plot the actual sample points
plt.semilogy(sigma_proposal_values, vrf_values, 'ko-', linewidth=2, markersize=8, label='VRF')

# Plot the theoretical bounds
plt.semilogy(sigma_proposal_values, vrf_list, 'rs--', linewidth=2, markersize=8, label='Bound')

# Customize the plot
plt.xlabel('Standard Deviation', fontsize=24)
plt.ylabel('VRF', fontsize=24)
plt.legend(fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True)

plt.tight_layout()
plt.savefig('vrf_plot1.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Print summary
print("\nVariance Reduction Factor Summary:")
for i, sigma in enumerate(sigma_proposal_values):
    print(f"σ = {sigma:.1f}: Simulated VRF = {vrf_values[i]:.4f}, Theoretical VRF = {vrf_list[i]:.4f}")


