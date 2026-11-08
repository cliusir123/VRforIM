import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t as tdist
from scipy.special import gammaln
from scipy import integrate


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
    for _ in range(m):
        X, Y, alpha = independent_metropolis(
            n, target_density, proposal_density, proposal_sampler, burnin=1000
        )
        imcv_estimates.append(imcv_estimator_plain(F, Fapp, X, Y, alpha, eq_f))
    return np.array(imcv_estimates)


def w_star_t(df):
    lg = 0.5 * np.log(df / 2.0) + gammaln(df / 2.0) - gammaln((df + 1) / 2.0)
    return np.exp(lg) * (1.0 + 1.0 / df) ** ((df + 1) / 2.0) * np.exp(-0.5)


def kl_t_vs_norm(df):
    f = lambda x: tdist.pdf(x, df) * (tdist.logpdf(x, df) - norm.logpdf(x))
    val, _ = integrate.quad(f, -np.inf, np.inf, limit=200)
    return val


def v_t(df, N=2_000_000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N)
    y = tdist.rvs(df, size=N, random_state=rng)
    logw = lambda z: norm.logpdf(z) - tdist.logpdf(z, df)
    alpha = np.minimum(1.0, np.exp(logw(y) - logw(x)))
    return np.var((1.0 - alpha) * (x - y))


def imcv_var_bound_t(df, sigma_F=1.0):
    v = v_t(df)
    w_star = w_star_t(df)
    bound_A = (1.0 + 2.0 * w_star) * v
    bound_B = v + 2.0 * (w_star - 1.0) * sigma_F * np.sqrt(v)
    return min(bound_A, bound_B), v, bound_A, bound_B


mu_target, sigma_target = 0.0, 1.0
target_density = lambda x: norm.pdf(x, loc=mu_target, scale=sigma_target)

F = lambda x: x
Fapp = lambda x: x
eq_f = 0.0
sigma2_F = 1.0

n = 5000
m = 100
df_values = [4, 8, 16, 32, 64, 128]

all_results = []
vrf_emp = []
vrf_lower_bound = []
kl_values = []
bound_info = []

for df in df_values:
    print(f"Running simulation for Student's t with df = {df}")
    proposal_density = lambda x, d=df: tdist.pdf(x, df=d, loc=0.0, scale=1.0)
    proposal_sampler = lambda size=None, d=df: \
        tdist.rvs(df=d, loc=0.0, scale=1.0, size=size)

    results = estimate(n, m, F, Fapp, eq_f, target_density,
                       proposal_density, proposal_sampler)
    all_results.append(results)
    vrf_emp.append(sigma2_F / (n * np.var(results)))

    bound, v, bA, bB = imcv_var_bound_t(df, sigma_F=np.sqrt(sigma2_F))
    vrf_lower_bound.append(sigma2_F / bound)
    bound_info.append((v, bA, bB))
    kl_values.append(kl_t_vs_norm(df))


fig, ax = plt.subplots(figsize=(12, 8))
widths = [pos / df_values[0] for pos in df_values]
box_plot = ax.boxplot(all_results, positions=df_values, widths=widths,
                      patch_artist=True, showfliers=False, manage_ticks=False)
for patch in box_plot['boxes']:
    patch.set_facecolor('white')
    patch.set_edgecolor('black')
ax.axhline(y=0.0, color='red', linestyle='--', label='Theoretical Mean: 0.0')
ax.set_xscale('log', base=2)
ax.set_xlim(2, 256)
ax.set_xticks(df_values)
ax.set_xticklabels([f'{d}' for d in df_values])
ax.set_xlabel('Degrees of Freedom', fontsize=24)
ax.set_ylabel('IMCV Estimates', fontsize=24)
ax.legend(fontsize=22)
ax.tick_params(labelsize=22)
ax.grid(True)
fig.tight_layout()
fig.savefig('vrf21.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(df_values, vrf_emp, 'b-s', markersize=8,
        label='Empirical')
ax.plot(df_values, vrf_lower_bound, 'k-o', markersize=8,
        label='Bound')
ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlim(2, 256)
ax.set_xticks(df_values)
ax.set_xticklabels([f'{d}' for d in df_values])
ax.set_xlabel('Degrees of Freedom', fontsize=24)
ax.set_ylabel('VRF', fontsize=24)
ax.legend(fontsize=22)
ax.tick_params(labelsize=22)
ax.grid(True)
fig.tight_layout()
fig.savefig('vrf22.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)

for d, k, lb, e, (v, bA, bB) in zip(df_values, kl_values, vrf_lower_bound,
                                    vrf_emp, bound_info):
    which = 'B' if bB < bA else 'A'
    print(f"df={d}: KL={k:.5f}  w*={w_star_t(d):.4f}  v={v:.5f}  "
          f"boundA={bA:.4f}  boundB={bB:.4f}  (min: {which})  "
          f"guaranteed VRF={lb:.3f}  empirical VRF={e:.3f}")
