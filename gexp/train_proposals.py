import argparse
import os
import sys

import numpy as np
import numpy.random as nr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gdistribution import gprior, log_prior_of_S, mix_normal
from run_gprior import MODELS, N_COMPS, PROPOSAL_DIR, SIGMA, model_label, read_data


def fit_proposal(inputs, output, cov0, restarts, criterion):
    dim = inputs.shape[1]
    prior = gprior(dim)
    prior.set_parameters(cov0, inputs)
    if criterion == 'region':
        beta_hat = cov0 @ inputs.T @ output
        B = beta_hat + SIGMA * nr.standard_normal((2000, dim)) @ np.linalg.cholesky(cov0).T
        S_check = prior.quad(B)
    else:
        S = prior.quad(prior.sampler(100000))
        S_check = np.exp(np.linspace(*np.log(np.quantile(S, [0.001, 0.999])), 400))
    log_prior = log_prior_of_S(S_check, dim)

    best = None
    for _ in range(restarts):
        pro = mix_normal(dim)
        w, g = pro.em_algorithm(prior.sampler(1000), N_COMPS, inputs, 3001)
        pro.set_parameters(w, g, cov0, inputs)
        gap = log_prior - pro.logpdf_of_S(S_check)
        score = np.mean(gap ** 2) if criterion == 'region' else np.max(gap)
        if best is None or score < best[0]:
            best = (score, w, g)
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--restarts', type=int, default=50, help="EM restarts per model")
    p.add_argument('--criterion', choices=['region', 'wstar'], default='region')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    nr.seed(args.seed)
    os.makedirs(PROPOSAL_DIR, exist_ok=True)

    inputs0, output = read_data()
    for cols in MODELS:
        label = model_label(cols)
        inputs = inputs0[:, cols]
        cov0 = np.linalg.inv(inputs.T @ inputs)
        score, w, g = fit_proposal(inputs, output, cov0, args.restarts, args.criterion)
        np.save(os.path.join(PROPOSAL_DIR, f"x{label}.npy"), [w, g])
        print(f"X{label:<5} score {score:8.3f}   weights {np.round(w, 3)}   g {np.round(g, 3)}", flush=True)
    print(f"proposals saved in {PROPOSAL_DIR}")


if __name__ == '__main__':
    main()
