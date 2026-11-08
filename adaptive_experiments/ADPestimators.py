import numpy as np


class Estimator():
    def __init__(self, sample_size, batch_size=None) -> None:
        self.sample_size = sample_size
        self.batch_size = sample_size if batch_size is None else batch_size
        if sample_size % self.batch_size != 0:
            raise ValueError("sample_size must be a multiple of batch_size")
        self.n_batch = sample_size // self.batch_size

    def F(self, data):
        raise NotImplementedError

    def F_approx(self, data):
        return self.F(data)

    def analytics(self, means, covs):
        raise NotImplementedError

    def sample_loader(self, run):
        n = self.sample_size
        self.X = run['X'][-n:]
        self.Y = run['Y'][-n:]
        self.Acpt = run['alpha'][-n:]
        self.eq = self.analytics(run['means'], run['covs'])[run['batch_idx'][-n:]]

    def Main(self, start, end):
        return self.F(self.X[start:end])

    def Poisson_cv(self, start, end):
        return self.Acpt[start:end] * (self.F(self.Y[start:end]) - self.F(self.X[start:end]))

    def Static_cv(self, start, end, coupled=False):
        s = slice(start - 1, end - 1) if coupled else slice(start, end)
        return self.F_approx(self.Y[s]) - self.eq[s]

    @staticmethod
    def _imcv_coefs(Main, Poisson_cv, Static_cv):
        coef_2 = np.cov(Poisson_cv, Static_cv)[0, 1] / np.var(Static_cv)
        PG_estimator = Main + Poisson_cv - coef_2 * Static_cv
        numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
        denominator = np.sum((Main[1:] - PG_estimator[:-1]) ** 2) / len(Main)
        return numerator / denominator, coef_2

    def IM(self):
        return self.F(self.X).mean()

    def IMCV(self):
        n = self.sample_size
        return np.mean(self.Main(1, n) + self.Poisson_cv(1, n) - self.Static_cv(1, n))

    def IMCV_coef(self):
        n = self.sample_size
        Main, Poisson_cv, Static_cv = self.Main(1, n), self.Poisson_cv(1, n), self.Static_cv(1, n)
        coef, coef_2 = self._imcv_coefs(Main, Poisson_cv, Static_cv)
        return np.mean(Main + coef * (Poisson_cv - coef_2 * Static_cv)), coef, coef_2

    def IMCV_poisson(self):
        n = self.sample_size
        return np.mean(self.Main(1, n) + self.Poisson_cv(1, n))

    def IMCV_static(self):
        n = self.sample_size
        Main = self.Main(1, n)
        Static_cv = self.Static_cv(1, n, coupled=True)
        PG_estimator = Main - Static_cv
        numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
        denominator = np.sum((Main[:-1] - PG_estimator[1:]) ** 2) / len(Main)
        coef = numerator / denominator
        return np.mean(Main - coef * Static_cv), coef

    def _batches(self, coupled=False):
        B = self.batch_size
        for i in range(self.n_batch):
            start = i * B
            if coupled and i == 0:
                start = 1
            yield start, (i + 1) * B

    def Batch_coef(self):
        est, c1, c2 = [], [], []
        for start, end in self._batches():
            Main, Poisson_cv, Static_cv = self.Main(start, end), self.Poisson_cv(start, end), self.Static_cv(start, end)
            coef, coef_2 = self._imcv_coefs(Main, Poisson_cv, Static_cv)
            est.append(np.mean(Main + coef * (Poisson_cv - coef_2 * Static_cv)))
            c1.append(coef)
            c2.append(coef_2)
        return np.mean(est), np.mean(c1), np.mean(c2)

    def BatchIMCV(self):
        return np.mean([np.mean(self.Main(s, e) + self.Poisson_cv(s, e) - self.Static_cv(s, e))
                        for s, e in self._batches()])

    def BatchIMCV_poisson(self):
        return np.mean([np.mean(self.Main(s, e) + self.Poisson_cv(s, e)) for s, e in self._batches()])

    def BatchIMCV_static(self):
        est, coefs = [], []
        for start, end in self._batches(coupled=True):
            Main = self.Main(start, end)
            Static_cv = self.Static_cv(start, end, coupled=True)
            coef = np.cov(Main, Static_cv)[0, 1] / np.var(Static_cv)
            est.append(np.mean(Main - coef * Static_cv))
            coefs.append(coef)
        return np.mean(est), np.mean(coefs)


class First_Est(Estimator):

    def __init__(self, sample_size, batch_size, pos) -> None:
        super().__init__(sample_size, batch_size)
        self.pos = pos

    def F(self, data):
        return data[:, self.pos]

    def analytics(self, means, covs):
        return means[:, self.pos]


class Second_Est(Estimator):

    def __init__(self, sample_size, batch_size, pos) -> None:
        super().__init__(sample_size, batch_size)
        self.pos = pos

    def F(self, data):
        return data[:, self.pos] ** 2

    def analytics(self, means, covs):
        return covs[:, self.pos, self.pos] + means[:, self.pos] ** 2


class EXP_Est(Estimator):

    def __init__(self, sample_size, batch_size, pos) -> None:
        super().__init__(sample_size, batch_size)
        self.pos = pos

    def F(self, data):
        return np.exp(data[:, self.pos])

    def analytics(self, means, covs):
        return np.exp(means[:, self.pos] + covs[:, self.pos, self.pos] / 2)


class Odd_ratio(Estimator):

    def __init__(self, sample_size, batch_size, x_bar) -> None:
        super().__init__(sample_size, batch_size)
        self.x_bar = np.asarray(x_bar)

    def F(self, data):
        return np.exp(data @ self.x_bar)

    def analytics(self, means, covs):
        quad = np.einsum('i,kij,j->k', self.x_bar, covs, self.x_bar)
        return np.exp(means @ self.x_bar + quad / 2)
