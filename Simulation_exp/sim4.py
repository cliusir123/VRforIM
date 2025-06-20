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

df_values = [4, 8, 16, 32, 64, 128]

def alpha(x,y,df):
    ratio = ((df+1)/2) * np.log((df+y**2)/(df+x**2)) -(y**2-x**2)/2
    return np.minimum(0,ratio)

def func(x,y,df):
    f = (np.exp(alpha(x,y,df)) -1 )**2 * (y-x)**2
    return f

from scipy.special import gamma
import numpy.random as nr
def coef(df):
    c = np.sqrt(df/2) * (gamma(df/2)/gamma((df+1)/2)) * (1+1/df)**((df+1)/2) * np.exp(-0.5)
    return c

N0=10000000

def bound(df):
    c = coef(df)
    x = nr.normal(0,1,N0)
    x1 = nr.standard_t(df, N0)
    y = nr.standard_t(df,N0)
    var = 2*c**2 * func(x1,y,df).mean() - func(x,y,df).mean()
    return 1/var

# Calculate theoretical bounds using the bound function
vrf_list = []
for df in df_values:
    theoretical_vrf = bound(df)
    vrf_list.append(theoretical_vrf)
    print(f"Theoretical VRF for df = {df}: {theoretical_vrf:.4f}")

# Store results for each df value
all_imcv_results = []
all_cmc_results = []
vrf_values = []

for df in df_values:
    print(f"Running simulation for Student's t with df = {df}")
    
    # Define Student's t proposal distribution
    mu_proposal = 0.0  # Location parameter
    scale_proposal = 1.0  # Scale parameter
    
    # Update proposal functions for Student's t distribution
    proposal_density = lambda x: t.pdf(x, df=df, loc=mu_proposal, scale=scale_proposal)
    proposal_sampler = lambda size=None: t.rvs(df=df, loc=mu_proposal, scale=scale_proposal, size=size)
    
    # Run estimation
    imcv_results, cmc_results = estimate(n, m, F, Fapp, eq_f, target_density, proposal_density, proposal_sampler, target_sampler)
    
    # Calculate VRF
    vrf = calculate_vrf(imcv_results, cmc_results)
    
    all_imcv_results.append(imcv_results)
    all_cmc_results.append(cmc_results)
    vrf_values.append(vrf)
    
    print(f"VRF for df = {df}: {vrf:.4f}")

# Create VRF plot
plt.figure(figsize=(12, 8))

# Plot the actual sample points
plt.semilogy(df_values, vrf_values, 'ko-', linewidth=2, markersize=8, label='VRF')

# Plot the theoretical bounds
plt.semilogy(df_values, vrf_list, 'rs--', linewidth=2, markersize=8, label='Bound')

# Customize the plot
plt.xlabel('Degrees of Freedom', fontsize=24)
plt.ylabel('VRF', fontsize=24)
plt.legend(fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True)

# Set x-axis to log scale with base 2
plt.xscale('log', base=2)

plt.tight_layout()
plt.savefig('vrf_plot2.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Print summary
print("\nVariance Reduction Factor Summary:")
for i, df in enumerate(df_values):
    print(f"df = {df}: Simulated VRF = {vrf_values[i]:.4f}, Theoretical VRF = {vrf_list[i]:.4f}")


