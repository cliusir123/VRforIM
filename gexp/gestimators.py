import jax.numpy as jnp
import jax
import jax.scipy.stats as jss
import numpy as np
import numpy.random as nr
import distrax
import scipy
import scipy.integrate as integrate
from gdistribution import gprior, mix_normal

class Estimator():
    def __init__(self, sample_size) -> None:
        self.sample_size = sample_size

    def F(self):
        raise NotImplementedError

    def F_approx(self):
        raise NotImplementedError
    
    def analytics(self):
        raise NotImplementedError
    
    def sample_loader(self, X, Y, Acpt):
        self.X = X[-self.sample_size:]
        self.Y = Y[-self.sample_size:]
        self.Acpt = Acpt[-self.sample_size:]
    
    def cmc_loader(self, X):
        self.X = X
    
    def IMCV0(self):
        Main = self.F(self.X)
        Poisson_cv = self.Acpt * (self.F(self.Y) - self.F(self.X))
        Static_cv = self.F(self.Y) - self.analy
        IMH_estimator = Main + Poisson_cv - Static_cv
        return IMH_estimator.mean(), IMH_estimator.var()
    
    def IMCV(self):
        Main = self.F(self.X)
        Poisson_cv = self.Acpt * (self.F(self.Y) - self.F(self.X))
        Static_cv = self.F(self.Y) - self.analy
        sample_len =  len(self.X)
        # coef_2 = np.sum((Static_cv - np.mean(Static_cv))*(Poisson_cv - np.mean(Poisson_cv)))/np.sum((Static_cv - np.mean(Static_cv))**2)
        coef_2 = np.cov(Poisson_cv, Static_cv)[0,1] / np.var(Static_cv)
        PG_estimator = Main + Poisson_cv - coef_2 * Static_cv
        numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
        denominator = np.sum((Main[0 : sample_len - 1] - PG_estimator[1 : sample_len]) ** 2) / sample_len
        coef = numerator / denominator

        print(np.cov(Main, Poisson_cv))
        print(np.cov(Poisson_cv, Static_cv))
        print(np.corrcoef(Main, Poisson_cv))
        print(np.corrcoef(Poisson_cv, Static_cv))
        print(coef, coef_2)

        # PG_estimator = Main + Poisson_cv 
        # numerator = np.mean(Main * (Main + PG_estimator)) - np.mean(Main) * np.mean(Main + PG_estimator)
        # denominator = np.sum((Main[0 : sample_len - 1] - PG_estimator[1 : sample_len]) ** 2) / sample_len
        # coef = numerator / denominator

        IMH_estimator = Main + coef * (Poisson_cv - coef_2 * Static_cv)
        return IMH_estimator.mean(), IMH_estimator.var()
    
    def IM(self):  
        Main = self.F(self.X)
        IMH_estimator = Main
        return IMH_estimator.mean(), IMH_estimator.var()
    
    def CMC(self):
        Main = self.F(self.X)
        return Main.mean(), Main.var()

class BF(Estimator):
    def __init__(self, sample_size, coefs, covs, inputs, outputs) -> None:
        super().__init__(sample_size)
        self.coefs = coefs
        self.covs = covs
        self.inputs = inputs
        self.outputs = outputs
        # self.analytics()
        self.analy = 1*10**(-13)
    
    # def F(self, X):
    #     logits = self.inputs @ X.T
    #     lik = np.ones(X.shape[0])
    #     for i in range(X.shape[0]):
    #         logi = logits[:,i]
    #         lik[i] = distrax.Bernoulli(logits=logi).prob(self.outputs).sum()
    #     return lik

    # def F(self, X):
    #     return X[:, 0]

    def F(self, X):
        logits = self.inputs @ X.T
        lik = np.ones(X.shape[0])
        for i in range(X.shape[0]):
            logi = logits[:,i] - self.outputs
            lik[i] = np.exp(-0.5 * np.sum(logi**2) / 2.5**2)
        return lik
    
    def analytics(self):
        mc = mix_normal(self.inputs.shape[1])
        mc.set_parameters(len(self.coefs), self.coefs, self.covs)
        data = mc.sampler(self.coefs, self.covs, 10000)
        return np.mean(self.F(data))
        # def integrand(beta):
        #     beta0 = beta[0]
        #     beta1 = beta[1]
        #     beta2 = beta[2]
        #     fun = 1
        #     for i in range(len(self.outputs)):
        #         fun *= distrax.Bernoulli(logits=self.inputs[i][0]*beta0+self.inputs[i]*beta1+self.inputs[i][2]*beta2).prob(self.outputs[i])
        #     fun = fun * distrax.Normal(loc=0, scale=100).prob(beta2)
        #     for i in range(len(self.coefs)):
        #         scale = jnp.linalg.cholesky(self.covs[i])
        #         fun = fun * distrax.MultivariateNormal(loc=jnp.zeros(len(beta)-1), scale_tril=scale).prob(beta[1:]) * self.coefs[i]
        #     return 
        # self.analy = integrate.quad(integrand, -jnp.inf, jnp.inf)[0]


            