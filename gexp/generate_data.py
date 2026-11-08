import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def generate(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = rng.normal(4 * X[:, 2] + 4 * X[:, 3], 2.5)
    df = pd.DataFrame(X)
    df['y'] = y
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=50, help="number of observations")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=os.path.join(HERE, 'data_new.csv'))
    args = p.parse_args()
    if os.path.exists(args.out):
        p.error(f"{args.out} exists; choose another --out or delete it first")
    generate(args.n, args.seed).to_csv(args.out, index=False)
    print(f"written {args.out}")


if __name__ == '__main__':
    main()
