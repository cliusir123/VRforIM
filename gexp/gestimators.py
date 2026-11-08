import numpy as np


class Estimator():
    def __init__(self, sample_size) -> None:
        self.sample_size = sample_size

    def F(self, X):
        raise NotImplementedError

    def sample_loader(self, X, Y, Acpt):
        self.X = X[-self.sample_size:]
        self.Y = Y[-self.sample_size:]
        self.Acpt = Acpt[-self.sample_size:]

    def cmc_loader(self, X):
        self.X = X

    def IMCV0(self):
        Main = self.F(self.X)
        Poisson_cv = self.Acpt * (self.F(self.Y) - Main)
        Static_cv = self.F(self.Y) - self.analy
        est = Main + Poisson_cv - Static_cv
        return est.mean(), est.var()

    def IMCV(self):
        Main = self.F(self.X)
        Poisson_cv = self.Acpt * (self.F(self.Y) - Main)
        Static_cv = self.F(self.Y) - self.analy
        PG_estimator = Main + Poisson_cv - Static_cv
        numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
        denominator = np.sum((Main[1:] - PG_estimator[:-1]) ** 2) / len(Main)
        coef = numerator / denominator
        est = Main + coef * (Poisson_cv - Static_cv)
        return est.mean(), est.var()

    def IM(self):
        Main = self.F(self.X)
        return Main.mean(), Main.var()

    def CMC(self):
        Main = self.F(self.X)
        return Main.mean(), Main.var()


class BF(Estimator):
    SIGMA = 2.5

    def __init__(self, sample_size, coefs, cov0, gs, inputs, outputs) -> None:
        super().__init__(sample_size)
        self.coefs = coefs
        self.gs = gs
        self.cov0 = cov0
        self.inputs = inputs
        self.outputs = outputs
        self.analy = self.analytics()

    def F(self, X):
        s, N = self.SIGMA, self.inputs.shape[0]
        resid = self.outputs[None, :] - np.atleast_2d(X) @ self.inputs.T
        loglik = -0.5 * (N * np.log(2 * np.pi) + N * np.log(s ** 2) + np.sum(resid ** 2, axis=1) / s ** 2)
        return np.exp(loglik)

    def analytics(self):
        s = self.SIGMA
        dim = self.cov0.shape[0]
        N = len(self.outputs)
        const = (2 * np.pi) ** (-N / 2) * s ** (-N) * np.exp(-0.5 * s ** (-2) * np.sum(self.outputs ** 2))
        t = np.sum(self.outputs[:, np.newaxis] * self.inputs, axis=0)
        expcoef = 0.5 * s ** (-2) * (t.T @ self.cov0 @ t)

        def integrand(x):
            return const * np.exp(expcoef * x / (x + s ** 2)) * (1 + x / s ** 2) ** (-dim / 2)
        return sum(self.coefs[i] * integrand(self.gs[i]) for i in range(len(self.coefs)))
