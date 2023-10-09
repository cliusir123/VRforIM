from scipy import special
import scipy.stats as ss
import numpy as np
from scipy.optimize import fsolve

class pricing():

    def __init__(
        self,
        K: int = 50,    # strike price of option
        S0: int = 50,   # initial price of underlying asset
        r: int = 0.05,  # risk neutral interest rate
        v: int = 0.1,   # volatility
        T: int = 1,     # total term
        d: int = 16     # number of time steps
        ) -> None:
        self.K = K
        self.S0 = S0
        self.r = r
        self.v = v
        self.T = T
        self.d = d
        self.dt = T / self.d


    def Generative_Matrix(self) -> None:
        L = np.zeros((self.d, self.d))
        for i in range(self.d):
            for j in range(self.d):
                if i >= j:
                    L[i, j] = 1
        self.L = L
        self.A = L @ L.T
        self.default_mean = np.log(self.S0) + (self.r - self.v ** 2 / 2) * self.dt * np.arange(1, self.d + 1, 1)
        self.default_L = self.L * self.v * np.sqrt(self.dt)
        self.default_cov = np.matmul(self.default_L, self.default_L.T)

    def Lognormal_Approx(self) -> None:
        steps = np.arange(1, self.d + 1, 1) * self.r * self.dt
        Splus = np.sum(np.exp(steps))
        S2 = np.reshape(np.exp(steps), (self.d, 1))
        self.VSA = np.sum(np.matmul(S2, S2.T) * self.A) * (self.v ** 2) * self.dt / Splus ** 2
        self.MSA = np.log(Splus) - self.VSA / 2 + np.log(self.S0 / self.d)
        self.proposal_mean = np.ones(self.d) * self.MSA + (self.r - self.v ** 2 / 2)\
             * self.dt * (np.arange(1, self.d + 1, 1) - np.mean(np.arange(1, self.d + 1, 1)))
        self.proposal_L = self.L * self.d * np.sqrt(self.VSA) / np.sqrt(np.sum(self.A))
        self.kappa = np.sqrt(self.VSA) * self.d / np.sqrt(np.sum(self.A))

    def Sj_tilde(self, data):
        new_data = np.exp(data) * self.S0 / np.expand_dims(np.exp(data[:, 0]), axis = 1)
        return new_data

    def backdist(self, data, mean, L):
        ndata = np.zeros(np.shape(data))
        for i in range(np.shape(data)[0]):
            ndata[i] = np.linalg.inv(L) @ (data[i] - mean)
        return ndata
    
    def fordist(self, data, mean, L):
        ndata = np.zeros(np.shape(data))
        for i in range(np.shape(data)[0]):
            ndata[i] = L @ data[i] + mean
        return ndata

    def re_data(self, data):
        rdata = np.zeros(np.shape(data)[0])
        for i in range(np.shape(data)[0]):
            repara_data = np.linalg.inv(self.proposal_L) @ (data[i] - self.proposal_mean)
            rdata[i] = np.arange(1, self.d, 1)[::-1] @ repara_data.reshape(self.d)[1:self.d] / self.d
        return rdata

    def map_data(self, data, mean, L):
        ndata = np.zeros(np.shape(data))
        for i in range(np.shape(data)[0]):
            ndata[i, 0] = data[i, 0]
            ndata[i, 1 : self.d] = L @ data[i][1 : self.d] + mean
        return ndata

    def F(self, *args):
        raise NotImplementedError

    def F_approx(self, *args):
        raise NotImplementedError
    
    def analytics(self):
        raise NotImplementedError

    def F_CMC_Delta(self, data):
        SA = np.mean(np.exp(data), axis = 1)
        const = np.exp( - self.r * self.T) / self.S0
        delta =  const * SA * np.int64(SA > self.K) 
        return np.mean(delta), np.var(delta)

    def F_CMC_Gamma(self, data):
        SA = np.mean(np.exp(data), axis = 1)
        first_repara = (data[:, 0] - self.default_mean[0]) / self.default_L[0, 0]
        const = np.exp( - self.r * self.T) * self.K / (self.S0 ** 2 * self.v * np.sqrt(self.dt))
        gamma = const * first_repara * np.int64(SA > self.K) 
        return np.mean(gamma), np.var(gamma)
    
    def IMCV2(self, *args):
        Main = self.F(args[0])
        Poisson_cv = args[2] * (self.F(args[1]) - self.F(args[0]))
        Static_cv = (self.F_approx(args[1]) - self.analytics()) 
        sample_len =  len(args[0])
        coef_2 = np.sum(Static_cv * Poisson_cv)/np.sum(Static_cv **2)
        # coef_2 = (np.sum(Static_cv * Main)-np.mean(Static_cv)*np.sum(Main))/(np.sum(Static_cv **2)-np.sum(Static_cv)*np.mean(Static_cv))
        PG_estimator = Main + Poisson_cv - coef_2 * Static_cv
        numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
        denominator = np.sum((Main[1 : sample_len] - PG_estimator[0 : sample_len - 1]) ** 2) / sample_len
        coef = numerator / denominator
        IMH_estimator = Main + coef * (Poisson_cv - coef_2 * Static_cv)
        # est2 = Main + Poisson_cv
        # print("corrcoef between F and poiCV :",'\n',np.corrcoef(Main,Poisson_cv))
        # print("corrcoef between poiCV and staticCV :",'\n',np.corrcoef(Poisson_cv,coef_2 * Static_cv))
        # print("corrcoef between F and staticCV :",'\n',np.corrcoef(Main,Static_cv))
        # print("corrcoef between F and poiCV - staticCV :",'\n',np.corrcoef(Main,Poisson_cv - coef_2 * Static_cv))
        # print("mean value of F:" , Main.mean())
        # print("mean value of IMCV:", IMH_estimator.mean())
        # print("mean value of F - poiCV:", est2.mean())
        # print("variance of F:", Main.var())
        # print("variance of IMCV:", IMH_estimator.var())
        # print("variance of F - poiCV:", est2.var())
        # print("static_cv_coef:", coef_2)
        # print("coef:", coef)
        return IMH_estimator.mean(), IMH_estimator.var()
    
    def IMCV(self, *args):
        Main = self.F(args[0])
        Poisson_cv = args[2] * (self.F(args[1]) - self.F(args[0]))
        Static_cv = (self.F_approx(args[1]) - self.analytics()) 

        sample_len =  len(args[0])
        PG_estimator = Main + Poisson_cv - Static_cv
        numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
        denominator = np.sum((Main[1 : sample_len] - PG_estimator[0 : sample_len - 1]) ** 2) / sample_len
        coef = numerator / denominator
        IMH_estimator = Main + coef * (Poisson_cv - Static_cv)
        return IMH_estimator.mean(), IMH_estimator.var()

    def CV(self, *args):
        Main = self.F(args[0])
        Static_cv = (self.F_approx(args[1]) - self.analytics())
        coef_2 = (np.sum(Static_cv * Main)-np.mean(Static_cv)*np.sum(Main))/(np.sum(Static_cv **2)-np.sum(Static_cv)*np.mean(Static_cv))
        cv_estimator = Main - coef_2 * Static_cv
        # print("corrcoef between F and staticCV :",'\n',np.corrcoef(Main,Static_cv))
        # print("mean value of F:" , Main.mean())
        # print("mean value of F + staticCV:", cv_estimator.mean())
        # print("variance of F:", Main.var())
        # print("variance of F + staticCV:", cv_estimator.var())
        # print("cv_coef:", coef_2)
        return cv_estimator.mean(), Main.mean(),Static_cv.mean(), coef_2





class Delta(pricing):
    def __init__(self, K: int = 50, S0: int = 50, r: int = 0.05, v: int = 0.1, T: int = 1, d: int = 16) -> None:
        super().__init__(K, S0, r, v, T, d)
        self.const = np.exp(- self.r * self.T) / self.S0

    def F(self, data):
        SA = np.mean(np.exp(data), axis = 1)
        delta = self.const * SA * np.int64(SA > self.K)
        return delta
    
    def F_approx(self, data):
        SA_approx = np.exp(np.mean(data , axis = 1))
        delta_approx = self.const * SA_approx * np.int64(SA_approx > self.K)
        return delta_approx
    
    def analytics(self):
        b = special.erfc((np.log(self.K) - self.MSA - self.VSA) / (np.sqrt(2 * self.VSA))) * np.exp(self.MSA + self.VSA / 2) / 2
        delta_ana = self.const * b
        return delta_ana


class Delta_con(pricing):
    def __init__(self, K: int = 50, S0: int = 50, r: int = 0.05, v: int = 0.1, T: int = 1, d: int = 16) -> None:
        super().__init__(K, S0, r, v, T, d)
        self.const = np.exp(- self.r * self.T) / self.S0

    def F(self, data):
        SAt = self.Sj_tilde(data)
        SA = np.mean(SAt, axis = 1)
        delta_con = self.const * SA * ss.norm.cdf(self.v * np.sqrt(self.dt) - \
            (np.log(self.K) - np.log(SA) - (self.r - self.v ** 2 / 2) * self.dt) / (self.v * np.sqrt(self.dt)))
        return delta_con
    
    def F_approx(self, data):
        repara_data = self.re_data(data)
        kappa = np.sqrt(self.VSA) * self.d / np.sqrt(np.sum(self.A))
        SA_approx = np.exp(self.MSA + kappa * repara_data + kappa ** 2)
        second_term = special.erfc(((np.log(self.K) - self.MSA) / kappa - repara_data - kappa) / np.sqrt(2))
        delta_approx = self.const * SA_approx * second_term / 2
        return delta_approx
    
    def analytics(self):
        b = special.erfc((np.log(self.K) - self.MSA - self.VSA) / (np.sqrt(2 * self.VSA))) * np.exp(self.MSA + self.VSA / 2) / 2
        delta_ana = self.const * b
        return delta_ana


class Gamma(pricing):
    def __init__(self, K: int = 50, S0: int = 50, r: int = 0.05, v: int = 0.1, T: int = 1, d: int = 16) -> None:
        super().__init__(K, S0, r, v, T, d)
        self.const = np.exp( - self.r * self.T) * self.K / (self.S0 ** 2 * self.v * np.sqrt(self.dt))

    def F(self, data):
        SA = np.mean(np.exp(data), axis = 1)
        first_repara = (data[:, 0] - self.default_mean[0]) / self.default_L[0, 0]
        gamma = self.const * first_repara * np.int64(SA > self.K)
        return gamma
    
    def F_approx(self, data):
        SA_approx = np.exp(np.mean(data , axis = 1))
        first_repara = (data[:, 0] - self.proposal_mean[0]) / self.proposal_L[0, 0]
        gamma_approx = self.const * first_repara * np.int64(SA_approx > self.K)
        return gamma_approx
    
    def analytics(self):
        const2 = self.const / np.sqrt(2 * np.pi)
        var_new = (2 * self.d ** 2 - 3 * self.d + 1) / (6 * self.d)
        gsa = (np.log(self.K) - self.MSA) * np.sqrt(np.sum(self.A)) / (self.d * np.sqrt(self.VSA))
        gamma_ana = const2 * np.exp(- gsa ** 2 / (2 * (var_new + 1))) / np.sqrt(var_new + 1)    
        return gamma_ana


class Gamma_con(pricing):
    def __init__(self, K: int = 50, S0: int = 50, r: int = 0.05, v: int = 0.1, T: int = 1, d: int = 16) -> None:
        super().__init__(K, S0, r, v, T, d)
        self.const = np.exp( - self.r * self.T) * self.K / (self.S0 ** 2 * self.v * np.sqrt(self.dt))

    def F(self, data):
        ndata = self.Sj_tilde(data)
        SA = np.mean(ndata, axis = 1)
        Psi = (np.log(self.K) - np.log(SA) - (self.r - self.v ** 2 / 2) * self.dt) / (self.v * np.sqrt(self.dt))
        gamma_con = self.const * np.exp(- Psi ** 2 / 2) / np.sqrt(2 * np.pi)
        return gamma_con
    
    def F_approx(self, data):
        kappa = np.sqrt(self.VSA) * self.d / np.sqrt(np.sum(self.A))
        Psi = (np.log(self.K) - np.sum(data) / (self.d + 1))
        gamma_approx = self.const * np.exp(- Psi ** 2 / (2 * kappa ** 2)) / np.sqrt(2 * np.pi)
        gamma_approx = self.const * np.exp(- Psi ** 2 / 2) / np.sqrt(2 * np.pi)
        return gamma_approx
    
    def analytics(self):
        const2 = self.const / np.sqrt(2 * np.pi)
        var_new = (2 * self.d ** 2 - 3 * self.d + 1) / (6 * self.d)
        gsa = (np.log(self.K) - self.MSA) * np.sqrt(np.sum(self.A)) / (self.d * np.sqrt(self.VSA))
        gamma_ana = const2 * np.exp(- gsa ** 2 / (2 * (var_new + 1))) / np.sqrt(var_new + 1)    
        return gamma_ana


class Gamma_con_Lap(pricing):
    def __init__(self, K: int = 50, S0: int = 50, r: int = 0.05, v: int = 0.1, T: int = 1, d: int = 16) -> None:
        super().__init__(K, S0, r, v, T, d)
        self.const = np.exp( - self.r * self.T) * self.K / (self.S0 ** 2 * self.v * np.sqrt(self.dt))
        

    def Psid_tar(self, variable):
        psid = np.log(self.K) - np.log((np.sum(np.exp(variable)) + self.S0) / (self.d + 1)) - (self.r - self.v ** 2 / 2) * self.dt
        return psid

    def tar_fun(self, variable):
        eqs = []
        for i in range(self.d):
            eqs.append(((variable - self.default_mean) @ np.linalg.inv(self.default_cov))[i] \
                 - self.Psid_tar(variable) * np.exp(variable[i]) / ((np.sum(np.exp(variable)) + self.S0) * self.v ** 2 * self.dt))
        return eqs
    
    def tar_Hessien(self, opt_mean):
        Hessien = np.zeros((self.d, self.d))
        for i in range(self.d):
            for j in range(self.d):
                Hessien[i, j] = - (1 + self.Psid_tar(opt_mean)) * np.exp(opt_mean[i] + opt_mean[j]) \
                    / (self.v ** 2 * self.dt * (np.sum(np.exp(opt_mean)) + self.S0) ** 2)
        for i in range(self.d) : 
            Hessien[i, i] = Hessien[i, i] + ((opt_mean - self.default_mean) @ np.linalg.inv(self.default_cov))[i]
        return Hessien

    def Psid_pro(self, variable):
        psid = np.log(self.K) - np.log((np.exp(np.mean(variable)) * self.d + self.S0) / (self.d + 1)) - (self.r - self.v ** 2 / 2) * self.dt
        return psid

    def pro_fun(self, variable):
        eqs = []
        for i in range(self.d):
            eqs.append(((variable - self.proposal_mean) @ np.linalg.inv(self.proposal_cov))[i]\
                 - self.Psid_pro(variable) * np.exp(np.mean(variable)) / ((np.exp(np.mean(variable)) * self.d + self.S0) * self.v ** 2 * self.dt))
        return eqs
    
    def pro_Hessien(self, opt_mean):
        entrys = ((opt_mean - self.proposal_mean) @ np.linalg.inv(self.proposal_cov))[0] \
                    - (1 + self.Psid_pro(opt_mean)) * np.exp(2 * np.mean(opt_mean)) \
                    / (self.v ** 2 * self.dt * (np.exp(np.mean(opt_mean)) * self.d + self.S0) ** 2)
        Hessien = np.ones((self.d, self.d)) * entrys
        return Hessien

    # def pro_fun(self, variable):
    #     eqs = []
    #     for i in range(self.d):
    #         eqs.append(((variable - self.proposal_mean) @ np.linalg.inv(self.proposal_cov))[i]\
    #              - (np.sum(variable) / (self.d + 1) - np.log(self.K)) / ((self.d  + 1) * self.kappa ** 2))
    #     return eqs
    
    # def pro_Hessien(self, opt_mean):
    #     entrys = 1 / ((self.d + 1) * self.kappa) ** 2
    #     Hessien = np.ones((self.d, self.d)) * entrys
    #     return Hessien
        
    def Lap(self):
        self.Lap_tar_mean = fsolve(self.tar_fun, np.zeros(self.d))
        self.Lap_pro_mean = fsolve(self.pro_fun, np.zeros(self.d))
        F_Hessien_tar = self.tar_Hessien(self.Lap_tar_mean)
        F_Hessien_pro = self.pro_Hessien(self.Lap_pro_mean)
        self.Lap_tar_cov = np.linalg.inv(np.linalg.inv(self.default_cov) - F_Hessien_tar)
        self.Lap_pro_cov = np.linalg.inv(np.linalg.inv(self.proposal_cov)- F_Hessien_pro)
        self.Lap_tar_L = np.linalg.cholesky(self.Lap_tar_cov)
        self.Lap_pro_L = np.linalg.cholesky(self.Lap_pro_cov)

    def F(self, data):
        SA = np.sum(np.exp(data), axis = 1)
        Psi = (np.log(self.K) - np.log((SA + self.S0) / (self.d + 1)) - (self.r - self.v ** 2 / 2) * self.dt) / (self.v * np.sqrt(self.dt))
        gamma_con = self.const * np.exp(- Psi ** 2 / 2) / np.sqrt(2 * np.pi)
        nratio =  ss.multivariate_normal.pdf(data, self.default_mean, self.default_cov) / ss.multivariate_normal.pdf(data, self.Lap_tar_mean, self.Lap_tar_cov)
        gamma_con_Lap = gamma_con * nratio
        return gamma_con_Lap

    def F_approx(self, data):
        Psi = np.log(self.K) - np.sum(data) / (self.d + 1)
        gamma_approx = self.const * np.exp(- Psi ** 2 / (2 * self.kappa ** 2)) / np.sqrt(2 * np.pi)
        nratio =   ss.multivariate_normal.pdf(data, self.proposal_mean, self.proposal_cov) / ss.multivariate_normal.pdf(data, self.Lap_pro_mean, self.Lap_pro_cov)
        gamma_approx_Lap = gamma_approx * nratio
        return gamma_approx_Lap

    def analytics(self):
        const2 = self.const / np.sqrt(2 * np.pi)
        var_new = (2 * (self.d + 1) ** 2 - 3 * (self.d + 1) + 1) / (6 * (self.d + 1))
        gsa = (np.log(self.K) - self.MSA) * np.sqrt(np.sum(self.A)) / ((self.d + 1) * np.sqrt(self.VSA))
        gamma_ana = const2 * np.exp(- gsa ** 2 / (2 * (var_new + 1))) / np.sqrt(var_new + 1)    
        return gamma_ana