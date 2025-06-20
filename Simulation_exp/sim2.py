import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t, multivariate_normal
from statsmodels.graphics.tsaplots import plot_acf

def independent_metropolis(n, target_density, proposal_density, proposal_sampler, burnin=100):
    """
    Independent Metropolis-Hastings algorithm
    
    Parameters:
    n: number of samples to generate
    target_density: target distribution density function
    proposal_density: proposal distribution density function
    proposal_sampler: function to sample from proposal distribution
    burnin: number of burn-in samples to discard
    
    Returns:
    X_samples: accepted samples (excluding burn-in)
    Y_proposals: proposal samples (excluding burn-in)
    alphas: acceptance probabilities (excluding burn-in)
    """
    total_n = n + burnin
    # Generate all proposal samples upfront
    Y_proposals = proposal_sampler(size=total_n)
    pi_y = target_density(Y_proposals)
    q_y = proposal_density(Y_proposals)

    # Initialize with a sample from proposal
    x = proposal_sampler(1)
    pi_x = target_density(x)
    q_x = proposal_density(x)

    X_samples = [x]
    alphas = []
    
    # Metropolis-Hastings iterations
    for i in range(total_n):
        y = Y_proposals[i]
        # Calculate acceptance probability
        alpha = min(1, (pi_y[i] * q_x) / (pi_x * q_y[i]))
        
        # Accept or reject based on acceptance probability
        if np.random.rand() < alpha:
            x = y
            pi_x = pi_y[i]
            q_x = q_y[i]
        
        X_samples.append(x)
        alphas.append(alpha)

    # Return samples excluding burn-in period
    return np.array(X_samples[-n-1:-1]), Y_proposals[-n:], np.array(alphas[-n:])

def imcv_estimator(F, Fapprox, X, Y, alpha, eq_f):
    """
    Improved Monte Carlo Variance (IMCV) estimator
    
    Parameters:
    F: target function to estimate expectation of
    Fapprox: approximation function
    X: accepted samples
    Y: proposal samples
    alpha: acceptance probabilities
    eq_f: expected value of Fapprox under target distribution
    
    Returns:
    IMCV estimate of E[F(X)]
    """
    l = len(X)
    fX = F(X)
    fY = F(Y)
    fapp = Fapprox(Y)

    # Calculate control variate coefficient b_2
    b_2 = np.cov(alpha * (fY - fX), (fapp - eq_f))[0,1] / np.var((fapp - eq_f))
    
    # Calculate control variate
    pg = fX + alpha * (fY - fX) - b_2 * (fapp - eq_f)
    
    # Calculate coefficient b_1 for variance reduction
    numerator = np.mean(fX * (fX + pg)) - np.mean(fX) * np.mean(fX + pg)
    denominator = np.sum((fX[0 : l - 1] - pg[1 : l]) ** 2) / l
    b_1 = numerator / denominator

    # Calculate final IMCV estimate
    delta = b_1* (alpha * (fY - fX) - b_2*(fapp - eq_f))
    return np.mean(fX + delta)

def estimate(n, m, F, Fapp, eq_f, target_density, proposal_density, proposal_sampler):
    """
    Run multiple independent IMCV estimations
    
    Parameters:
    n: number of samples per estimation
    m: number of independent estimations
    F: target function
    Fapp: approximation function
    eq_f: expected value of Fapp
    target_density: target distribution
    proposal_density: proposal distribution
    proposal_sampler: proposal sampler
    
    Returns:
    Array of m IMCV estimates
    """
    imcv_estimates = []

    for _ in range(m):
        # Generate samples using Independent Metropolis
        X, Y, alpha = independent_metropolis(
            n, target_density, proposal_density, proposal_sampler, burnin=1000
        )
        
        # Calculate IMCV estimate
        imcv = imcv_estimator(F, Fapp, X, Y, alpha, eq_f)
        imcv_estimates.append(imcv)

    return np.array(imcv_estimates)

# Target distribution parameters (standard normal)
mu_target = 0.0
sigma_target = 1.0

# Define target distribution and sampler
target_density = lambda x: norm.pdf(x, loc=mu_target, scale=sigma_target)
target_sampler = lambda size=None: np.random.normal(loc=mu_target, scale=sigma_target, size=size)

# Define function of interest and its approximation
F = lambda x: x  # Function to estimate E[X]
Fapp = lambda x: x  # Approximation function
eq_f = 0.0  # Expected value of Fapp under target distribution

# Simulation parameters
n = 5000  # Number of samples per estimation
m = 100   # Number of independent estimations

# Degrees of freedom for Student's t proposal distributions
df_values = [4, 8, 16, 32, 64, 128]

# Store results for each degree of freedom
all_results = []

for df in df_values:
    print(f"Running simulation for Student's t with df = {df}")
    
    # Define Student's t proposal distribution
    mu_proposal = 0.0  # Location parameter
    scale_proposal = 1.0  # Scale parameter
    
    # Update proposal functions for Student's t distribution
    proposal_density = lambda x: t.pdf(x, df=df, loc=mu_proposal, scale=scale_proposal)
    proposal_sampler = lambda size=None: t.rvs(df=df, loc=mu_proposal, scale=scale_proposal, size=size)
    
    # Run estimation
    results = estimate(n, m, F, Fapp, eq_f, target_density, proposal_density, proposal_sampler)
    all_results.append(results)

# Create box plot with log scale x-axis
plt.figure(figsize=(12, 8))
# Use the actual df_values as positions instead of arange
positions = df_values

# Calculate proportional widths for log scale to make them appear uniform
# On a log scale, we need to adjust widths to maintain visual consistency
widths = [ pos / df_values[0] for pos in df_values]

box_plot = plt.boxplot(all_results, labels=[f'{df}' for df in df_values], 
                       positions=positions, widths=widths, patch_artist=True, showfliers=False)

# Set all boxes to white color
for patch in box_plot['boxes']:
    patch.set_facecolor('white')
    patch.set_edgecolor('black')  # Keep black edges for visibility

# Add theoretical mean line
theoretical_mean = 0.0  # Since F(x) = x and target mean is 0
plt.axhline(y=theoretical_mean, color='red', linestyle='--', label=f'Theoretical Mean: {theoretical_mean}')

# Customize the plot
plt.xlabel('Degrees of Freedom', fontsize=24)
plt.ylabel('IMCV Estimates', fontsize=24)
plt.legend(fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True)

# Set x-axis to log base 2 scale
plt.xscale('log', base=2)

# Adjust the x-axis limits to ensure consistent box widths
plt.xlim(2, 256)  # Cover the range from 2^1 to 2^8

plt.tight_layout()
plt.savefig('vrf21.pdf', dpi=300, bbox_inches='tight')
plt.close()