import numpy as np

class Estimator():
    def __init__(self, sample_size, batch_size) -> None:
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.n_batch = int(self.sample_size / self.batch_size)

    def F(self):
        raise NotImplementedError

    def F_approx(self):
        raise NotImplementedError
    
    def analytics(self):
        raise NotImplementedError
    
    def sample_loader(self, X, Y, Acpt, paras):
        self.X = X[-self.sample_size:]
        self.Y = Y[-self.sample_size:]
        self.Acpt = Acpt[-self.sample_size:]
        if type(paras) == tuple:
            self.paras = []
            pMeans = paras[0][-self.sample_size:]
            pCov = paras[1][-self.sample_size:]
            for i in range(self.sample_size):
                self.paras.append((pMeans[i], pCov[i]))
        else:
            self.paras = paras[-self.sample_size:]
    
    def IMCV(self):
        Main = self.F(self.X)
        Poisson_cv = self.Acpt * (self.F(self.Y) - self.F(self.X))
        Static_cv = self.F_approx(self.Y) - self.analytics(self.paras)

        sample_len =  len(self.X)
        PG_estimator = Main + Poisson_cv - Static_cv
        numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
        denominator = np.sum((Main[0 : sample_len - 1] - PG_estimator[1 : sample_len]) ** 2) / sample_len
        coef = numerator / denominator

        IMH_estimator = Main + coef * (Poisson_cv - Static_cv)
        return IMH_estimator.mean(), IMH_estimator.var(), coef

    def IMCV0(self):
        Main = self.F(self.X)
        Poisson_cv = self.Acpt * (self.F(self.Y) - self.F(self.X))
        Static_cv = self.F_approx(self.Y) - self.analytics(self.paras)

        IMH_estimator = Main + Poisson_cv - Static_cv
        return IMH_estimator.mean(), IMH_estimator.var()
    
    def IMCV1(self):
        Main = self.F(self.X)
        Poisson_cv = self.Acpt * (self.F(self.Y) - self.F(self.X))

        sample_len =  len(self.X)
        PG_estimator = Main + Poisson_cv
        numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
        denominator = np.sum((Main[0 : sample_len - 1] - PG_estimator[1 : sample_len]) ** 2) / sample_len
        coef = numerator / denominator

        IMH_estimator = Main + coef * Poisson_cv
        return IMH_estimator.mean(), IMH_estimator.var(), coef
    
    def IMCV2(self):
        Main = self.F(self.X)
        Poisson_cv = self.Acpt * (self.F(self.Y) - self.F(self.X))
        Static_cv = self.F_approx(self.Y) - self.analytics(self.paras)
        sample_len =  len(self.X)
        # coef_2 = np.sum((Static_cv - np.mean(Static_cv))*(Poisson_cv - np.mean(Poisson_cv)))/np.sum((Static_cv - np.mean(Static_cv))**2)
        coef_2 = np.cov(Poisson_cv, Static_cv)[0,1] / np.var(Static_cv)
        PG_estimator = Main + Poisson_cv - coef_2 * Static_cv
        numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
        denominator = np.sum((Main[0 : sample_len - 1] - PG_estimator[1 : sample_len]) ** 2) / sample_len
        coef = numerator / denominator

        IMH_estimator = Main + coef * (Poisson_cv - coef_2 * Static_cv)
        return IMH_estimator.mean(), IMH_estimator.var(), coef, coef_2


    def IM(self):
        Main = self.F(self.X)
        return Main.mean(), Main.var()
    
    def BatchIMCV(self):
        Bimhcv = np.zeros(self.n_batch)
        for i in range(self.n_batch - 1):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            Main = self.F(self.X[start : end])
            Poisson_cv = self.Acpt[start : end] * (self.F(self.Y[start : end]) - self.F(self.X[start : end]))
            Static_cv = self.F_approx(self.Y[start : end]) - self.analytics(self.paras[start : end])

            PG_estimator = Main + Poisson_cv - Static_cv
            numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
            denominator = np.sum((Main[0 : self.batch_size - 1] - PG_estimator[1 : self.batch_size]) ** 2) / self.batch_size
            coef = numerator / denominator
            Bimhcv[i] = np.mean(Main + coef * (Poisson_cv - Static_cv))
        return Bimhcv.mean(), Bimhcv.var()
    
    def BatchIMCV2(self):
        Bimhcv = np.zeros(self.n_batch)
        for i in range(self.n_batch - 1):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            Main = self.F(self.X[start : end])
            Poisson_cv = self.Acpt[start : end] * (self.F(self.Y[start : end]) - self.F(self.X[start : end]))
            Static_cv = self.F_approx(self.Y[start : end]) - self.analytics(self.paras[start : end])
            coef_2 = np.sum((Static_cv - np.mean(Static_cv))*(Poisson_cv - np.mean(Poisson_cv)))/np.sum((Static_cv - np.mean(Static_cv))**2)
            # coef_2 = np.cov(Poisson_cv, Static_cv)[0,1] / np.var(Static_cv)
            PG_estimator = Main + Poisson_cv - coef_2 * Static_cv
            numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
            denominator = np.sum((Main[0 : self.batch_size - 1] - PG_estimator[1 : self.batch_size]) ** 2) / self.batch_size
            coef = numerator / denominator
            Bimhcv[i] = np.mean(Main + coef * (Poisson_cv - coef_2 * Static_cv))
        return Bimhcv.mean(), Bimhcv.var()


class First_Est(Estimator):
    def __init__(self, sample_size, batch_size, pos) -> None:
        super().__init__(sample_size, batch_size)
        self.pos = pos

    def F(self, data):
        return data[:, self.pos]
    
    def F_approx(self, data):
        return data[:, self.pos]
    
    def analytics(self, data):
        return data[:, self.pos]


class Simulation_Est(Estimator):
    def __init__(self, sample_size, batch_size) -> None:
        super().__init__(sample_size, batch_size)
    
    def F(self, data):
        return data
    
    def F_approx(self, data):
        return data
    
    def analytics(self, data):
        return data

    def cmc(self, data):
        return data.mean(), data.var()
    
class Odd_ratio(Estimator):
    def __init__(self, sample_size, batch_size) -> None:
        super().__init__(sample_size, batch_size)
    
    def pre_proc(self, origin_data):
        self.means = np.mean(origin_data,axis = 0)
    
    def F(self, data):
        return np.exp(data @ np.transpose(self.means))

    def F_approx(self, data):
        return np.exp(data @ np.transpose(self.means))
    
    def analytics(self, data):
        length = len(data)
        quad = np.zeros(length)
        lin = np.zeros(length)
        for i in range(length):
          quad[i] = self.means @ data[i][1] @ self.means.T / 2
          lin[i] = data[i][0] @ self.means.T
        return np.exp(lin + quad)

class Second_Est(Estimator):
    def __init__(self, sample_size, batch_size, pos) -> None:
        super().__init__(sample_size, batch_size)
        self.pos = pos
    
    def F(self, data):
        return data[:, self.pos] ** 2
    
    def F_approx(self, data):
        return data[:, self.pos] ** 2
    
    def analytics(self, data):
        length = len(data)
        quad = np.zeros(length)
        for i in range(length):
            quad[i] = data[i][1][self.pos, self.pos] + data[i][0][self.pos] ** 2
        return quad




    