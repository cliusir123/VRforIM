import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import integrate
from tqdm import tqdm


def independent_metropolis(n, target_density, proposal_density, proposal_sampler,
                           burnin=100):
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


def imcv_estimator_plain(F, Fapprox, X, Y, alpha, eq_f):
    fX = F(X)
    fY = F(Y)
    fapp = Fapprox(Y)
    delta = alpha * (fY - fX) - (fapp - eq_f)
    return np.mean(fX + delta)


def estimate(n, m, F, Fapp, eq_f, target_density, proposal_density,
             proposal_sampler):
    imcv_estimates = []
    for _ in tqdm(range(m), desc="replications", leave=False):
        X, Y, alpha = independent_metropolis(
            n, target_density, proposal_density, proposal_sampler, burnin=1000
        )
        imcv_estimates.append(imcv_estimator_plain(F, Fapp, X, Y, alpha, eq_f))
    return np.array(imcv_estimates)


def v_gaussian(sigma):
    a = 1.0 - 1.0 / sigma**2

    def inner(y):
        f = lambda x: (np.exp(-0.5 * a * (y**2 - x**2)) - 1.0)**2 \
                      * (y - x)**2 \
                      * np.exp(-x**2 / 2.0 - y**2 / (2.0 * sigma**2))
        val, _ = integrate.quad(f, -y, y)
        return val

    I, _ = integrate.quad(inner, 0.0, 60.0, limit=200)
    return I / (np.pi * sigma)


def imcv_var_bound(sigma, sigma_F=1.0):
    v = v_gaussian(sigma)
    w_star = sigma
    bound_A = (1.0 + 2.0 * w_star) * v
    bound_B = v + 2.0 * (w_star - 1.0) * sigma_F * np.sqrt(v)
    return min(bound_A, bound_B), v, bound_A, bound_B


def kl_of_sigma(s):
    s = np.asarray(s, dtype=float)
    return 0.5 * (s**2 - 1.0 - 2.0 * np.log(s))


def sigma_of_kl(k):
    k = np.atleast_1d(np.asarray(k, dtype=float))
    out = np.ones_like(k)
    for i, ki in enumerate(k):
        if ki <= 0:
            out[i] = 1.0
            continue
        lo, hi = 1.0, 10.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if kl_of_sigma(mid) < ki:
                lo = mid
            else:
                hi = mid
        out[i] = 0.5 * (lo + hi)
    return out


mu_target, sigma_target = 0.0, 1.0
target_density = lambda x: norm.pdf(x, loc=mu_target, scale=sigma_target)

F = lambda x: x
Fapp = lambda x: x
eq_f = 0.0
sigma2_F = 1.0

n = 5000
m = 100
sigma_proposal_values = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

all_results = []
vrf_emp = []
vrf_lower_bound = []
bound_info = []

for sigma_proposal in tqdm(sigma_proposal_values, desc="sigma"):
    proposal_density = lambda x, s=sigma_proposal: norm.pdf(x, loc=0.0, scale=s)
    proposal_sampler = lambda size=None, s=sigma_proposal: \
        np.random.normal(0.0, s, size=size)

    results = estimate(n, m, F, Fapp, eq_f, target_density,
                       proposal_density, proposal_sampler)
    all_results.append(results)
    vrf_emp.append(sigma2_F / (n * np.var(results)))

    bound, v, bA, bB = imcv_var_bound(sigma_proposal, sigma_F=np.sqrt(sigma2_F))
    vrf_lower_bound.append(sigma2_F / bound)
    bound_info.append((v, bA, bB))

kl_values = kl_of_sigma(np.array(sigma_proposal_values))

fig, ax = plt.subplots(figsize=(12, 8))
box_plot = ax.boxplot(all_results, positions=sigma_proposal_values,
                      widths=0.04, patch_artist=True, showfliers=False,
                      manage_ticks=False)
for patch in box_plot['boxes']:
    patch.set_facecolor('white')
    patch.set_alpha(1.0)
ax.axhline(y=0.0, color='red', linestyle='--',
           label='Theoretical Mean: 0.0')
ax.set_xticks(sigma_proposal_values)
ax.set_xticklabels([f'{s:.1f}' for s in sigma_proposal_values])
ax.set_xlabel('Standard Deviation', fontsize=24)
ax.set_ylabel('IMCV Estimates', fontsize=24)
ax.legend(fontsize=22)
ax.tick_params(labelsize=22)
ax.grid(True)

fig.tight_layout()
fig.savefig('vrf11.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(sigma_proposal_values, vrf_emp, 'b-s', markersize=8,
        label='Empirical')
ax.plot(sigma_proposal_values, vrf_lower_bound, 'k-o', markersize=8,
        label='Bound')
ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5)
ax.set_yscale('log')
ax.set_xticks(sigma_proposal_values)
ax.set_xticklabels([f'{s:.1f}' for s in sigma_proposal_values])
ax.set_xlabel('Standard Deviation', fontsize=24)
ax.set_ylabel('VRF', fontsize=24)
ax.legend(fontsize=22)
ax.tick_params(labelsize=22)
ax.grid(True)

fig.tight_layout()
fig.savefig('vrf12.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)

for s, k, lb, e, (v, bA, bB) in zip(sigma_proposal_values, kl_values,
                                    vrf_lower_bound, vrf_emp, bound_info):
    which = 'B' if bB < bA else 'A'
    print(f"sigma={s}: KL={k:.4f}  v={v:.5f}  boundA={bA:.4f}  "
          f"boundB={bB:.4f}  (min: {which})  "
          f"guaranteed VRF={lb:.4f}  empirical VRF={e:.4f}")
