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
        self.paras = paras[-self.sample_size:]
    
    def Main(self):
        return self.F(self.X)

    def Poisson_cv(self):
        return self.Acpt * (self.F(self.Y) - self.F(self.X))
    
    def Static_cv(self):
        return self.F_approx(self.Y) - self.analytics(self.paras)

    def IMCV(self):
        Main = self.Main()
        Poisson_cv = self.Poisson_cv()
        Static_cv = self.Static_cv()
        IMH_estimator = Main + Poisson_cv - Static_cv
        return IMH_estimator.mean()
    
    def IM(self):
        Main = self.Main()
        return Main.mean()
    

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



    