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

# -------------------------------
# IMCV Estimator
# -------------------------------
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

def im_estimator(F, X):
    return np.mean(F(X))

# -------------------------------
# Bias-Corrected SNIS Estimator
# -------------------------------
def snis_estimator(F, Fapp, samples, target_density, proposal_density, eq_f):
    w = target_density(samples) / proposal_density(samples)
    Z_hat = np.mean(w)
    corrections = (w / Z_hat)
    f_vals = F(samples)
    fapp = Fapp(samples)
    b_2 = np.cov(f_vals * corrections, (fapp - eq_f))[0,1] / np.var((fapp - eq_f))
    return np.mean(f_vals * corrections - b_2*(fapp-eq_f))
    # return np.mean(f_vals * corrections - (fapp-eq_f))
def is_estimator(F, samples, target_density, proposal_density):
    w = target_density(samples) / proposal_density(samples)
    Z_hat = np.mean(w)
    corrections = (w / Z_hat)
    f_vals = F(samples)
    return np.mean(f_vals * corrections)

def cmc_estimator(F, target_sampler, n):
    samples = target_sampler(size=n)
    return np.mean(F(samples))
# -------------------------------
# Estimation and Comparison
# -------------------------------
def estimate(
    n, m, mean_true, F, Fapp, eq_f,
    target_density, proposal_density, proposal_sampler,
    DO_PRINT=True
):
    imcv_estimates = []
    snis_estimates = []
    im_estimates = []
    is_estimates = []

    for _ in range(m):
        X, Y, alpha = independent_metropolis(
            n, target_density, proposal_density, proposal_sampler
        )
        
        imcv = imcv_estimator(F, Fapp, X, Y, alpha, eq_f)
        imcv_estimates.append(imcv)

        im = im_estimator(F, X)
        im_estimates.append(im)

        x_samples = proposal_sampler(size=n)
        snis = snis_estimator(F, Fapp, x_samples, target_density, proposal_density, eq_f)
        snis_estimates.append(snis)

        is_est = is_estimator(F, x_samples, target_density, proposal_density)
        is_estimates.append(is_est)

    imcv_estimates = np.array(imcv_estimates)
    snis_estimates = np.array(snis_estimates)
    im_estimates = np.array(im_estimates)
    is_estimates = np.array(is_estimates)

    # Calculate statistics for all estimators
    estimates_dict = {
        'IMCV': imcv_estimates,
        'ISCV': snis_estimates,
        'IM': im_estimates,
        'IS': is_estimates
    }

    if DO_PRINT:
        for name, estimates in estimates_dict.items():
            bias = np.mean(estimates - mean_true)
            var = np.var(estimates, ddof=1)
            mse = bias**2 + var
            print(f"Bias {name} = {bias:.6f}")
            print(f"Var  {name} = {var:.6e}")
            print(f"MSE  {name} = {mse:.6e}")
        print()

    return (imcv_estimates - mean_true, 
            snis_estimates - mean_true,
            im_estimates - mean_true,
            is_estimates - mean_true)

# Define parameters
mu_target = 0.0
sigma_target = 1.0

mu_proposal = 0.5
sigma_proposal = 1.01

# Define densities and samplers
target_density = lambda x: norm.pdf(x, loc=mu_target, scale=sigma_target)
target_sampler = lambda size=None: np.random.normal(loc=mu_target, scale=sigma_target, size=size)
proposal_density = lambda x: norm.pdf(x, loc=mu_proposal, scale=sigma_proposal)
proposal_sampler = lambda size=None: np.random.normal(loc=mu_proposal, scale=sigma_proposal, size=size)

# Define F and Fapp
F = lambda x: np.exp(x) / (1 + np.exp(x))  # sigmoid
kappa = np.sqrt(np.pi / 8)
Fapp = lambda x: norm.cdf(kappa * x)

# Assume this is the true value (or you can run a large MC to get it more accurately)
mean_true = 0.5
eq_f = norm.cdf((kappa * mu_proposal) / np.sqrt(1 + kappa ** 2 * sigma_proposal ** 2))

# Estimator function placeholder (you must define this if not done yet)
# Should return (bias_imcv_list, bias_snis_list) of length M
# def estimate(n, m, mean_true, F, Fapp, eq_f, target_density, proposal_density, proposal_sampler):
#     ...

# Estimation parameters
M = 100
N_values = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000]
# N_values = [5000]
avg_mse_imcv = []
avg_mse_snis = []
avg_mse_im = []
avg_mse_is = []

for N in N_values:
    bias_imcv_list, bias_snis_list, bias_im_list, bias_is_list = estimate(
        n=N, m=M,
        mean_true=mean_true, F=F, Fapp=Fapp, eq_f=eq_f,
        target_density=target_density,
        proposal_density=proposal_density,
        proposal_sampler=proposal_sampler,
        DO_PRINT=False
    )
    avg_mse_imcv.append(np.mean(bias_imcv_list**2))
    avg_mse_snis.append(np.mean(bias_snis_list**2))
    avg_mse_im.append(np.mean(bias_im_list**2))
    avg_mse_is.append(np.mean(bias_is_list**2))

estimators = ['IM', 'IS', 'IMCV', 'ISCV']
markers = ['+', '*', '^', 'x']  # Different marker styles for each estimator
colors = ['blue', 'red', 'green', 'purple']  # Different colors for each estimator
# Plotting MSE
plt.figure(figsize=(12, 6))
for i, (mse, name) in enumerate(zip([avg_mse_im, avg_mse_is, avg_mse_imcv, avg_mse_snis], 
                                  estimators)):
    plt.semilogy(
        N_values,
        mse,
        marker=markers[i],
        color=colors[i],
        label=name,
        linestyle='-',
        markersize=12,    # Larger symbols
        alpha=0.7         # Make lines slightly transparent
    )
plt.xlabel('Sample Size', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("mse_plot.pdf")
plt.show()