import argparse
import os
import sys

import numpy as np
import numpy.random as nr
import pandas as pd
import scipy.integrate
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from IMH2 import IndMH
from gdistribution import gprior, mix_normal
from gestimators import BF

SIGMA = 2.5
N_COMPS = 4
MODELS = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
          [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]]
PROPOSAL_DIR = os.path.join(HERE, 'proposals')


def model_label(cols):
    return ''.join(str(c + 1) for c in cols)


def read_data():
    df = pd.read_csv(os.path.join(HERE, 'data.csv'))
    return df.values[:, 0:-1], df.values[:, -1]


def numerical_integration(inputs, output, cov0):
    N, dim = inputs.shape
    const = (2 * np.pi) ** (-N / 2) * SIGMA ** (-N) * np.exp(-0.5 * SIGMA ** (-2) * np.sum(output ** 2))
    t = np.sum(output[:, np.newaxis] * inputs, axis=0)
    expcoef = 0.5 * SIGMA ** (-2) * (t.T @ cov0 @ t)

    def integrand(x):
        return const * np.exp(expcoef * x / (x + SIGMA ** 2)) * (1 + x / SIGMA ** 2) ** (-dim / 2) * (1 + x) ** (-2)
    return scipy.integrate.quad(integrand, 0, np.inf)[0]


def load_proposal(label, proposal_dir=PROPOSAL_DIR):
    path = os.path.join(proposal_dir, f"x{label}.npy")
    if not os.path.exists(path):
        sys.exit(f"{path} not found; run train_proposals.py first")
    weights, gs = np.load(path)
    return weights, gs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reps', type=int, default=100, help="independent replications")
    p.add_argument('--n', type=int, default=1000, help="IM / CMC sample size")
    p.add_argument('--burnin', type=int, default=100, help="IM burn-in iterations")
    p.add_argument('--L', type=int, default=1000, help="g draws in the prior density approximation")
    p.add_argument('--models', nargs='*', default=None, help="subset of models to run, e.g. 124 1234")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--proposals', default=PROPOSAL_DIR, help="directory of the saved proposals")
    p.add_argument('--out', default=None, help="also write the table to this markdown file")
    args = p.parse_args()

    inputs0, output = read_data()
    rows = []
    for index, cols in enumerate(MODELS):
        label = model_label(cols)
        if args.models and label not in args.models:
            continue
        nr.seed(args.seed * 1000 + index)
        L = args.L
        inputs = inputs0[:, cols]
        dim = inputs.shape[1]
        cov0 = np.linalg.inv(inputs.T @ inputs)

        truth = numerical_integration(inputs, output, cov0)
        weights, gs = load_proposal(label, args.proposals)
        prior = gprior(dim)
        prior.set_parameters(cov0, inputs, L=L)
        proposal = mix_normal(dim)
        proposal.set_parameters(weights, gs, cov0, inputs)

        imcv, cmc, acc = [], [], []
        for _ in tqdm(range(args.reps), desc=f"X{label}", leave=False):
            X, Y, alpha = IndMH(args.n + args.burnin, prior, proposal).IMH()
            est = BF(args.n, weights, cov0, gs, inputs, output)
            est.sample_loader(X, Y, alpha)
            imcv.append(est.IMCV0()[0])
            acc.append(alpha[-args.n:].mean())
            est_cmc = BF(args.n, weights, cov0, gs, inputs, output)
            est_cmc.cmc_loader(prior.sampler(args.n))
            cmc.append(est_cmc.CMC()[0])

        imcv, cmc = np.array(imcv), np.array(cmc)
        vrf = cmc.var() / imcv.var()
        rows.append((label, -np.log10(imcv.mean()), -np.log10(cmc.mean()), -np.log10(truth), vrf, np.mean(acc), L))
        print(f"X{label:<5} -log10 IMCV {rows[-1][1]:7.2f}   -log10 CMC {rows[-1][2]:7.2f}   "
              f"num. int. {rows[-1][3]:7.2f}   VRF {vrf:9.2f}   AP {np.mean(acc):.2f}   L {L}", flush=True)

    lines = ['# Table 1: marginal likelihood under the mixture of g-priors', '',
             f"n = {args.n}, burn-in {args.burnin}, {args.reps} replications, seed {args.seed}; "
             "proposals from proposals/; VRF = Var(CMC) / Var(IMCV).", '',
             '| Model | -log10(IMCV) | -log10(CMC) | Numerical integration | VRF | AP | L |', '|---|---|---|---|---|---|---|']
    lines += [f"| X{lab} | {mi:.2f} | {mc:.2f} | {ti:.2f} | {vrf:.2f} | {ap:.2f} | {L} |"
              for lab, mi, mc, ti, vrf, ap, L in rows]
    lines += ['', 'LaTeX rows:', '', '```']
    lines += [f"$X_{{{lab}}}$ & {mi:.2f} & {mc:.2f} & {ti:.2f} & {vrf:.2f} \\\\" for lab, mi, mc, ti, vrf, _, _ in rows]
    lines += ['```', '']
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as fh:
            fh.write('\n'.join(lines))
        print(f"\ntable written to {args.out}")


if __name__ == '__main__':
    main()
