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
            pMeans = paras[0][-self.sample_size:]
            pCov = paras[1][-self.sample_size:]
            self.paras=[pMeans, pCov]
        elif len(paras) == 1:
            self.paras = paras
        else:
            self.paras = paras[-self.sample_size:]
    
    def Main(self,start, end):
        return self.F(self.X[1 + start:end])

    def Poisson_cv(self,start,end):
        return self.Acpt[1+start:end] * (self.F(self.Y[1+start:end]) - self.F(self.X[1+start:end]))
    
    def Static_cv(self, start,end,coupled = False):
        if coupled:
            try:
                return self.F_approx(self.Y[start:end-1]) - self.analytics(self.paras[start:end-1])
            except:
                return self.F_approx(self.Y[start:end-1]) - self.analytics(self.paras[0][start:end-1], self.paras[1][start:end-1])
        else:
            try:
                return self.F_approx(self.Y[1 + start:end]) - self.analytics(self.paras[1 + start:end])
            except:
                return self.F_approx(self.Y[1 + start:end]) - self.analytics(self.paras[0][1 + start:end], self.paras[1][1 + start:end])



    def IMCV_coef(self):
        Main = self.Main(0, self.sample_size+1)
        Poisson_cv = self.Poisson_cv(0, self.sample_size+1)
        Static_cv = self.Static_cv(0,self.sample_size+1,False)
        sample_len =  len(Main)
        coef_2 = np.cov(Poisson_cv, Static_cv)[0,1] / np.var(Static_cv)
        PG_estimator = Main + Poisson_cv - coef_2 * Static_cv
        numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
        denominator = np.sum((Main[0 : sample_len - 1] - PG_estimator[1 : sample_len]) ** 2) / sample_len
        coef = numerator / denominator
        IMH_estimator = Main + coef * (Poisson_cv - coef_2 * Static_cv)
        return IMH_estimator.mean(), coef, coef_2

    def IMCV(self):
        Main = self.Main(0, self.sample_size+1)
        Poisson_cv = self.Poisson_cv(0, self.sample_size+1)
        Static_cv = self.Static_cv(0,self.sample_size+1,False)
        IMH_estimator = Main + Poisson_cv - Static_cv
        return IMH_estimator.mean()
    
    def IMCV_poisson(self):
        Main = self.Main(0, self.sample_size+1)
        Poisson_cv = self.Poisson_cv(0, self.sample_size+1)
        IMH_estimator = Main + Poisson_cv
        return IMH_estimator.mean()
    
    def IMCV_static(self):
        Main = self.Main(0, self.sample_size+1)
        Static_cv = self.Static_cv(0,self.sample_size+1,True)
        sample_len =  len(Main)
        PG_estimator = Main - Static_cv
        numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
        denominator = np.sum((Main[0 : sample_len - 1] - PG_estimator[1 : sample_len]) ** 2) / sample_len
        coef_2 = numerator / denominator
        # coef_2 = np.cov(Main, Static_cv)[0,1] / np.var(Static_cv)
        IMH_estimator = Main - coef_2 * Static_cv
        return IMH_estimator.mean(), coef_2

    def IM(self):
        Main = self.F(self.X)
        return Main.mean()
    
    def Batch_coef(self):
        Bimhcv = np.zeros(self.n_batch)
        for i in range(self.n_batch - 1):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size+1

            Main = self.Main(start, end)
            Poisson_cv = self.Poisson_cv(start, end)
            Static_cv = self.Static_cv(start,end,False)
            coef_2 = np.cov(Poisson_cv, Static_cv)[0,1] / np.var(Static_cv)
            PG_estimator = Main + Poisson_cv - coef_2*Static_cv
            length = len(Main)
            numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
            denominator = np.sum((Main[0:-1] - PG_estimator[1:]) ** 2) / length
            coef = numerator / denominator
            Bimhcv[i] = np.mean(Main + coef * (Poisson_cv - coef_2 * Static_cv))
        return Bimhcv.mean(), coef, coef_2
    
    def BatchIMCV(self):
        Bimhcv = np.zeros(self.n_batch)
        for i in range(self.n_batch - 1):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size+1
            Main = self.Main(start, end)
            Poisson_cv = self.Poisson_cv(start, end)
            Static_cv = self.Static_cv(start,end,False)
            Bimhcv[i] = np.mean(Main + Poisson_cv - Static_cv)
        return Bimhcv.mean()
    
    def BatchIMCV_poisson(self):
        Bimhcv = np.zeros(self.n_batch)
        for i in range(self.n_batch - 1):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size+1
            Main = self.Main(start, end)
            Poisson_cv = self.Poisson_cv(start, end)
            Bimhcv[i] = np.mean(Main + Poisson_cv)
        return Bimhcv.mean()
    
    def BatchIMCV_static(self):
        Bimhcv = np.zeros(self.n_batch)
        for i in range(self.n_batch - 1):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size+1
            Main = self.Main(start, end)
            Static_cv = self.Static_cv(start,end,True)
            coef = np.cov(Main, Static_cv)[0,1] / np.var(Static_cv)
            Bimhcv[i] = np.mean(Main - coef * Static_cv)
        return Bimhcv.mean(), coef


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

class EXP_Est(Estimator):
    def __init__(self, sample_size, batch_size, pos) -> None:
        super().__init__(sample_size, batch_size)
        self.pos = pos

    def F(self, data):
        return np.exp(data[:, self.pos])
    
    def F_approx(self, data):
        return np.exp(data[:, self.pos])
    
    def analytics(self, data1,data2):
        return np.exp(data1[:, self.pos] + data2[:, self.pos] / 2)

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
    
    def analytics(self, data1,data2):
        length = len(data1)
        quad = np.zeros(length)
        lin = np.zeros(length)
        for i in range(length):
          quad[i] = self.means @ data2[i] @ self.means.T / 2
          lin[i] = data1[i] @ self.means.T
        return np.exp(lin + quad)

class Second_Est(Estimator):
    def __init__(self, sample_size, batch_size, pos) -> None:
        super().__init__(sample_size, batch_size)
        self.pos = pos
    
    def F(self, data):
        return data[:, self.pos] ** 2
    
    def F_approx(self, data):
        return data[:, self.pos] ** 2
    
    def analytics(self, data1,data2):
        length = len(data1)
        quad = np.zeros(length)
        for i in range(length):
            quad[i] = np.diag(data2[i])[self.pos] + data1[i][self.pos] ** 2
        return quad

class ex(Estimator):
    def __init__(self, sample_size, batch_size) -> None:
        super().__init__(sample_size, batch_size)
    
    def F(self, data):
        return np.exp(-data[:])
    
    def F_approx(self, data):
        return np.exp(-data[:])
    
    def analytics(self,data):
        return (1/(-1+np.exp(1))+np.sqrt(np.exp(1)))**(1/(1-np.exp(1)))*np.exp(-1/(2*(1-np.exp(1))))



    