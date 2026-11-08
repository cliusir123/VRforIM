import argparse
import json
import os
import re
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ADPIMH import BAIM, IndMH, adp_only
from ADPestimators import EXP_Est, First_Est, Odd_ratio, Second_Est
from Distributions import Bayesian_Logistic_Regression, GP_Regression

ESTIMATORS = {
    'after': [('IMCV', 'IMCV'), ('IMCV(c1,c2)', 'IMCV_coef'), ('RB', 'IMCV_poisson'), ('CIM(c)', 'IMCV_static')],
    'during': [('IMCV', 'BatchIMCV'), ('IMCV(c1,c2)', 'Batch_coef'), ('RB', 'BatchIMCV_poisson'),
               ('CIM(c)', 'BatchIMCV_static')],
}


def build_problem(args):
    if args.model == 'logistic':
        model = Bayesian_Logistic_Regression()
        model.load_data(args.dataset)
        model.inputs = np.concatenate([model.inputs, np.ones((model.inputs.shape[0], 1))], axis=1)
        d = model.inputs.shape[1]
        x_bar = model.inputs.mean(axis=0)
        functions = {
            'beta': lambda n, B, j: First_Est(n, B, j),
            'beta^2': lambda n, B, j: Second_Est(n, B, j),
            'r(x_bar,beta)': lambda n, B, j: Odd_ratio(n, B, x_bar),
        }
        scalar = {'r(x_bar,beta)'}
        return 'model', model, d, None, None, np.zeros(d), np.eye(d), functions, scalar
    if args.model == 'gp':
        model = GP_Regression()
        model.load_data(args.dataset)
        d = model.dim
        functions = {
            'log theta': lambda n, B, j: First_Est(n, B, j),
            'theta': lambda n, B, j: EXP_Est(n, B, j),
        }
        return 'model', model, d, None, None, np.ones(d), np.eye(d), functions, set()
    d = args.dim
    functions = {'x': lambda n, B, j: First_Est(n, B, j)}
    return 'multivariate_normal', None, d, np.zeros(d), np.eye(d), np.ones(d), np.eye(d), functions, set()


def run_replication(args, rep, target_str, model, d, mu_t, L_t, mu_q, L_q):
    seed = args.seed + rep
    np.random.seed(seed)
    if args.mode == 'after':
        itr = args.n + 1 + args.burnin
        sampler = IndMH(target_str, itr, d, 0.0, model, value=seed, cov_scale=args.cov_scale)
        sampler.set_distribution_parameters(mu_t, L_t, mu_q, L_q)
        return sampler.IMH(), args.n + 1, None
    B = args.batch_size
    itr = (args.n_batches + args.burnin_batches) * B
    sampler = BAIM(target_str, itr, B, d, args.lr, model, value=seed, schedule=args.schedule,
                   cov_scale=args.cov_scale)
    sampler.set_distribution_parameters(mu_t, L_t, mu_q, L_q)
    return sampler.adpIMH(), args.n_batches * B, B


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=['logistic', 'gp', 'gaussian'], required=True)
    p.add_argument('--dataset', help="logistic: ripley|pima|heart|german; gp: boston|pendulum")
    p.add_argument('--dim', type=int, default=10, help="dimension of the Gaussian target")
    p.add_argument('--mode', choices=['after', 'during'], default='after')
    p.add_argument('--reps', type=int, default=50, help="independent replications")
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--adapt-iters', type=int, default=5000000)
    p.add_argument('--adapt-batch', type=int, default=100)
    p.add_argument('--adapt-lr', type=float, default=0.01)
    p.add_argument('--schedule', choices=['exponential', 'cosine'], default='cosine',
                   help="learning-rate schedule of the adaptation")
    p.add_argument('--n', type=int, default=5000, help="IM sample size (mode after)")
    p.add_argument('--burnin', type=int, default=100, help="IM burn-in iterations (mode after)")
    p.add_argument('--batch-size', type=int, default=50, help="B (mode during)")
    p.add_argument('--n-batches', type=int, default=20, help="l (mode during)")
    p.add_argument('--burnin-batches', type=int, default=20, help="adaptive batches discarded (mode during)")
    p.add_argument('--lr', type=float, default=1e-5, help="learning rate of Algorithm 2 (mode during)")
    p.add_argument('--cov-scale', type=float, default=1.0,
                   help="IM proposal covariance = cov_scale x adapted covariance (1 = no inflation)")
    p.add_argument('--proposal-from', default=None,
                   help="reuse the adapted proposal (mu_q, L_q) saved in an earlier .npz; skips the burn-in adaptation")
    p.add_argument('--out', default=None, help="where to save the estimates (.npz)")
    args = p.parse_args()

    if args.model in ('logistic', 'gp') and args.dataset is None:
        p.error("--dataset is required for logistic and gp")

    target_str, model, d, mu_t, L_t, mu_q, L_q, functions, scalar = build_problem(args)
    name = args.dataset or f"gaussian{d}"
    print(f"{name}: d = {d}, mode = {args.mode}, replications = {args.reps}")
    print("arguments: " + json.dumps(vars(args)), flush=True)

    if args.proposal_from:
        prev = np.load(args.proposal_from, allow_pickle=True)
        mu_q, L_q = prev['mu_q'], prev['L_q']
        print(f"adapted proposal loaded from {args.proposal_from}")
    else:
        t0 = time.time()
        np.random.seed(args.seed)
        adapter = adp_only(target_str, args.adapt_iters, args.adapt_batch, d, args.adapt_lr, model,
                           value=args.seed, schedule=args.schedule)
        adapter.set_distribution_parameters(mu_t, L_t, mu_q, L_q)
        loss = adapter.adp()
        mu_q, L_q = np.asarray(adapter.p_mean), np.asarray(adapter.p_L)
        print(f"burn-in adaptation done in {time.time() - t0:.0f}s, "
              f"ELBO (mean of last 1% of batches) {np.mean(loss[-max(1, len(loss) // 100):]):.3f}")
    if args.cov_scale != 1.0:
        print(f"IM proposal covariance inflated by {args.cov_scale}")

    est_list = ESTIMATORS[args.mode]
    labels = ['IM'] + [lab for lab, _ in est_list]
    results = {f: np.zeros((len(labels), args.reps, 1 if f in scalar else d)) for f in functions}
    coefs = {f: {lab: [] for lab, _ in est_list} for f in functions}
    acc = np.zeros(args.reps)
    for rep in tqdm(range(args.reps), desc="replications"):
        t0 = time.time()
        run, n, B = run_replication(args, rep, target_str, model, d, mu_t, L_t, mu_q, L_q)
        acc[rep] = run['accepted'][-n:].mean()
        for fname, make in functions.items():
            for j in range(results[fname].shape[2]):
                est = make(n, B, j)
                est.sample_loader(run)
                results[fname][0, rep, j] = est.IM()
                for k, (lab, method) in enumerate(est_list, start=1):
                    out = getattr(est, method)()
                    if isinstance(out, tuple):
                        results[fname][k, rep, j] = out[0]
                        coefs[fname][lab].append(out[1:])
                    else:
                        results[fname][k, rep, j] = out
        tqdm.write(f"replication {rep + 1}/{args.reps}  acceptance {acc[rep]:.3f}  ({time.time() - t0:.0f}s)")

    print(f"\n{name}  AP: {acc.mean():.2f}")
    print(f"{'F':<16}" + ''.join(f"{lab:>18}" for lab in labels[1:]))
    for fname, res in results.items():
        var = res.var(axis=1)
        vrf = var[0] / var[1:]
        cells = [f"{v.min():.1f}" if v.size == 1 else f"{v.min():.1f} - {v.max():.1f}" for v in vrf]
        print(f"{fname:<16}" + ''.join(f"{c:>18}" for c in cells))
        coef_cells = []
        for lab, _ in est_list:
            c = coefs[fname][lab]
            coef_cells.append('(' + ', '.join(f"{x:.3f}" for x in np.mean(c, axis=0)) + ')' if c else '-')
        print(f"{'':<16}" + ''.join(f"{c:>18}" for c in coef_cells))

    out = args.out or f"results_{name}_{args.mode}.npz"
    mean_coefs = {f: {lab: np.mean(c, axis=0).tolist() for lab, c in v.items() if c} for f, v in coefs.items()}
    np.savez(out, labels=labels, acceptance=acc, mu_q=mu_q, L_q=L_q, args=json.dumps(vars(args)),
             coefs=json.dumps(mean_coefs),
             **{"est_" + re.sub(r'\W+', '_', k): v for k, v in results.items()})
    print(f"\nestimates saved to {out}")


if __name__ == '__main__':
    main()
