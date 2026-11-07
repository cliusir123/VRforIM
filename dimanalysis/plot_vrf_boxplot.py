from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dimanalysis_vrf_results.npy"
OUTPUT_PATH = BASE_DIR / "gaussianbox.pdf"


def main():
    data = np.load(DATA_PATH, allow_pickle=True).item()
    dims = np.asarray(data["dims"], dtype=float)
    vrf_values = [np.asarray(v, dtype=float) for v in data["vrf"]]
    box_width = max(1.0, 0.6 * np.min(np.diff(dims))) if len(dims) > 1 else 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(vrf_values, positions=dims, widths=box_width)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(int(d)) for d in dims])
    ax.set_xlim(left=0)
    ax.set_xlabel("Dimension", fontsize=14)
    ax.set_ylabel("Variance Reduction Factor", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)


if __name__ == "__main__":
    main()
