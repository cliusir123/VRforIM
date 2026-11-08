from adaptiveIM import IndMH, adp_only
from estimator import First_Est
import numpy as np
from tqdm import tqdm


distribution = "multivariate_normal"
proposal = distribution
target = distribution
dim_list = [5, 10, 15, 20, 30, 50, 75, 100]

lr_burnin = 0.001
Itr_burnin = 45000
batch_size_burnin = 100
epoch = 50

Sample_N = 5001
Itr = Sample_N + 100
lr = 0

vrf_results = []
avg_acpt = np.zeros(len(dim_list))

for dim_idx, d in enumerate(tqdm(dim_list, desc="dimensions")):
    mu1 = np.zeros(d)
    L1 = np.eye(d)
    mu2 = np.ones(d)
    L2 = np.eye(d)

    Myadp = adp_only(target, proposal, Itr_burnin, batch_size_burnin, d, lr_burnin)
    Myadp.set_distribution_parameters(mu1, L1, mu2, L2)
    Myadp.adp(True)
    mu2 = np.array(Myadp.p_mean)
    L2 = np.array(Myadp.p_L)

    results = []
    iid_results = []

    for i in tqdm(range(epoch), desc=f"d={d}"):
        MyIMH = IndMH(target, proposal, Itr, d, lr, i + 10)
        MyIMH.set_distribution_parameters(mu1, L1, mu2, L2)
        resul = MyIMH.IMH()
        iid_sample, _ = MyIMH.target.sampler(mu1, L1, Sample_N)
        results.append(resul)
        iid_results.append(np.array(iid_sample))

    Ans = np.zeros((2, epoch, d))
    dim_acpt = 0.0
    for i in range(epoch):
        iid_sample = iid_results[i]
        X_sample = results[i][0]
        Y_sample = results[i][1]
        Acpt = results[i][2]
        pmeans = results[i][3]
        dim_acpt += np.mean(np.asarray(Acpt))
        for j in range(d):
            Esti = First_Est(Sample_N, 1, j)
            Esti.sample_loader(X_sample, Y_sample, Acpt, pmeans)
            Ans[0, i, j] = np.mean(iid_sample[:, j])
            Ans[1, i, j] = Esti.IMCV()

    VRF = Ans[0].var(axis=0) / Ans[1].var(axis=0)
    vrf_results.append(VRF)
    avg_acpt[dim_idx] = dim_acpt / epoch
    print(f"d={d}, avg acceptance={avg_acpt[dim_idx]:.6f}")

results_to_save = {
    "dims": np.array(dim_list),
    "vrf": np.array(vrf_results, dtype=object),
    "avg_acpt": avg_acpt,
}
np.save("dimanalysis_vrf_results.npy", results_to_save, allow_pickle=True)
