import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t,multivariate_normal
from statsmodels.graphics.tsaplots import plot_acf

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

    delta = b_1* (alpha * (fY - fX) - b_2*(fapp - eq_f))
    # delta = alpha * (fY - fX) - (fapp - eq_f)
    return np.mean(fX + delta)

def estimate(
    n, m, F, Fapp, eq_f,
    target_density, proposal_density, proposal_sampler
):
    imcv_estimates = []

    for _ in range(m):
        X, Y, alpha = independent_metropolis(
            n, target_density, proposal_density, proposal_sampler, burnin=1000
        )
        
        imcv = imcv_estimator(F, Fapp, X, Y, alpha, eq_f)
        imcv_estimates.append(imcv)

    imcv_estimates = np.array(imcv_estimates)

    return imcv_estimates


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

# Store results for each sigma proposal value
all_results = []

for sigma_proposal in sigma_proposal_values:
    print(f"Running simulation for sigma_proposal = {sigma_proposal}")
    
    # Update proposal parameters
    mu_proposal = 0.0
    
    # Update proposal functions
    proposal_density = lambda x: norm.pdf(x, loc=mu_proposal, scale=sigma_proposal)
    proposal_sampler = lambda size=None: np.random.normal(loc=mu_proposal, scale=sigma_proposal, size=size)
    
    # Run estimation
    results = estimate(n, m, F, Fapp, eq_f, target_density, proposal_density, proposal_sampler)
    all_results.append(results)

# Create box plot
plt.figure(figsize=(12, 8))
positions = np.arange(1, len(sigma_proposal_values) + 1)
box_plot = plt.boxplot(all_results, labels=[f'{s:.1f}' for s in sigma_proposal_values], 
                       positions=positions, widths=0.3, patch_artist=True, showfliers=False)

# Make boxes white and opaque
for patch in box_plot['boxes']:
    patch.set_facecolor('white')
    patch.set_alpha(1.0)

# Add theoretical mean line
theoretical_mean = 0.0  # Since F(x) = x and target mean is 0
plt.axhline(y=theoretical_mean, color='red', linestyle='--', label=f'Theoretical Mean: {theoretical_mean}')

# Customize the plot
plt.xlabel('Standard Deviation', fontsize=24)
plt.ylabel('IMCV Estimates', fontsize=24)
plt.legend(fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True)

plt.tight_layout()
plt.savefig('vrf11.pdf', dpi=300, bbox_inches='tight')
plt.close()

