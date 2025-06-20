import jax
import jax.numpy as jnp
import haiku as hk
import numpy.random as nr
import jax.scipy.stats as jss
import numpy as np
import pandas as pd
import distrax
import scipy
import scipy.stats as ss

class DisBase():
    def __init__(self,
                 dim: int = 100,
                 value: int = 10) -> None:
        self.rng_seq = hk.PRNGSequence(value)
        self.dim = dim


class student_t(DisBase):
    def __init__(self, dim: int = 100, value: int = 10, df: int = 3) -> None:
        super().__init__(dim, value)
        self.df = df

    def sampler(self, *args): #args: 0 loc 1 scale 2 number
        epsilon = jax.random.normal(next(self.rng_seq), shape = [self.dim, args[2]])
        y = args[1] @ epsilon
        u = u = jnp.sqrt(self.df / nr.chisquare(self.df))
        x = u * y + args[0].reshape((self.dim,1))
        return x.T, epsilon.T

    def logpdf(self, *args): #args: 0 data 1 loc 2 scale
        v = jax.lax.linalg.triangular_solve(args[2], args[0] - args[1], left_side=True, lower=True)
        logp = - 0.5 * (self.df + self.dim) * jnp.log(1 + jnp.sum(v * v) / self.df)
        return logp


class multivariate_normal(DisBase):
    def __init__(self, dim: int = 100, value: int = 10) -> None:
        super().__init__(dim, value)

    def sampler(self, *args): #args: 0 loc 1 scale 2 number , 3 beta
        epsilon = jax.random.normal(next(self.rng_seq), shape = [self.dim, args[2]])
        x = args[1] @ epsilon + args[0].reshape((self.dim,1))
        return x.T, epsilon.T

    def logpdf(self, *args): #args: 0 data 1 loc 2 scale
        v = jax.lax.linalg.triangular_solve(args[2], args[0] - args[1], left_side=True, lower=True)
        logp = - 0.5 * jnp.sum(v * v)
        return logp

class mixture_normal(DisBase):
    def __init__(self,
                 dim: int = 100,
                 value: int = 10,
                 n_comp: int = 3) -> None:
        super().__init__(dim, value)
        self.n_comp = n_comp

    def sampler(self, *args): #args: 0 loc 1 scale 2 weight 3 number
        z = nr.choice(self.n_comp, args[2])
        epsilon = jax.random.normal(next(self.rng_seq), shape = [self.dim, args[3]])
        x = args[1][z] @ epsilon + args[0][z].reshape((self.dim,1))
        return x.T, epsilon.T

    def logpdf(self, *args): #args: 0 data 1 loc 2 scale 3 weight
        logp = jnp.zeros(self.n_comp)
        for i in range(self.n_comp):
            v = jax.lax.linalg.triangular_solve(args[2][i], args[0] - args[1][i], left_side=True, lower=True)
            logpi = logpi.at[i].set(jnp.log(self.weights[i]) - 0.5 * (jnp.sum(v * v)) - jnp.log(jnp.prod(jnp.diag(self.chol[i]))))
        logp = jax.scipy.special.logsumexp(logp)
        return logp
    

class lognormal(DisBase):
    def __init__(self,
                 dim: int = 1,
                 value: int = 10) -> None:
        super().__init__(dim, value)
    
    def sampler(self, *args):
        return nr.lognormal(args[0], args[1], args[2]), None
    
    def logpdf(self, *args):
        return ss.lognorm.logpdf(args[0], args[1], args[2])
    
class gamma(DisBase):
    def __init__(self,
                 dim: int = 1,
                 value: int = 10) -> None:
        super().__init__(dim, value)
        
    def sampler(self, *args):
        return nr.gamma(args[0], args[1], args[2]), None
        
    def logpdf(self, *args):
        return ss.gamma.logpdf(args[0], args[1], 0, args[2])

# Stochastic Volatility:
# initialization needs parameters: m, phi, s
# load_data: recieve 1 bool argument is_sim = True: simulate data from initialized density || False: load real data
#      save data in specific object of the class
# logpdf:  recieve 1 arguments x : the data input to compute logpdf, notice that args here left blank
#      return log-pdf
class Stochastic_Volatility(DisBase):
    def __init__(self, dim: int = 100,
    m: float = -0.85,
    phi: float = 0.98,
    s: float = 0.15,):
        super().__init__(dim)
        self.m = m
        self.phi = phi
        self.s = s

    def load_data(self, is_sim):
        if is_sim:
            h0 = nr.normal(self.m, self.s/np.sqrt(1-self.phi ** 2), 1)
            h = np.zeros(self.dim - 3)
            r = np.zeros(self.dim - 3)
            eta = nr.normal(0, 1, self.dim - 3)
            epsilon = nr.normal(0, 1, self.dim - 3)
            h[0] = h0
            for i in range(self.dim - 4):
                h[i + 1] = self.m + self.phi * (h[i] - self.m) + self.s * eta[i + 1]
                r[i + 1] = np.exp(h[i] / 2) * epsilon[i + 1]
            self.r = r
            self.h = h
        else:
            data = pd.read_csv("/data.csv",index_col=False)
            data = np.array(data['x'])
            self.r = data[0:self.dim]


    def logpdf(self, x, *args):
        m = x[0]
        s2 = jnp.exp(x[1])
        phi = (jnp.exp(x[2]) - 1) / (jnp.exp(x[2]) + 1)
        h = x[3:self.dim]
        l = len(h)
        logtarget = jss.norm.logpdf(x[0], self.m, 0.1) + \
            jss.norm.logpdf(x[1], jnp.log(self.s ** 2) , 0.1) + \
            jss.norm.logpdf(x[2], jnp.log((1 + self.phi) / (1 - self.phi)), 0.1) + \
            jss.norm.logpdf(h[0], m, jnp.sqrt(s2 / (1 - phi ** 2))) + \
            jnp.sum(jss.norm.logpdf(h[1:l], m * (1 - phi) + phi * h[0 : l - 1], jnp.ones(l - 1) * jnp.sqrt(s2))) + \
            jnp.sum(jss.norm.logpdf(self.r[1:l], jnp.zeros(l - 1), jnp.sqrt(jnp.exp(h[1:l]))))
        return logtarget



# Logistic regression:
# load_data: recieve 1 string argument : the name of dataset
#      save data in specific object of the class
# logpdf:  recieve 2 arguments x : the data input to compute logpdf, prior_var = 1.0 as default, notice that args here left blank
#      return log-pdf without constant term
class Bayesian_Logistic_Regression(DisBase):
    def __init__(self, dim: int = 100) -> None:
        super().__init__(dim)

    def load_data(self, dataset):
        file = "/home/ranbot/VRforIM/data/" + dataset + ".mat"
        mat = scipy.io.loadmat(file)
        data = mat['X']
        if (dataset == 'caravan' or dataset == 'pima' or dataset == 'ripley' or dataset == 'australian'):
            self.inputs = data[:, 0:-1]
            print(self.inputs.shape)
            self.outputs = data[:, -1]
        elif dataset == 'german' or dataset == 'heart':
            inputs = data[:, 0:-1]
            outputs = data[:, -1]
            outputs[np.where(outputs == 1)] = 0
            outputs[np.where(outputs == 2)] = 1
            mean_inputs = np.mean(inputs, axis=0)
            std_inputs = np.std(inputs, axis=0)
            inputs = (inputs - mean_inputs[None, :]) / std_inputs[None, :]
            outputs = outputs[:, None]
            self.inputs = inputs.astype(jnp.float64)
            self.outputs = outputs.astype(jnp.float64)

    def logpdf(self, x, *paras, prior_var = 1.0):
        print(self.inputs.shape)
        logits = self.inputs @ x
        log_lik = distrax.Bernoulli(logits=logits).log_prob(self.outputs).sum()
        log_prior = -(0.5 * x.shape[0]) * jnp.log(2. * jnp.pi) - (0.5 / prior_var) * jnp.sum(x * x)
        return log_lik + log_prior


class GP_Regression(DisBase):
    def __init__(self) -> None:
        super().__init__()

    def load_data(self, dataset):
        file  = "/home/ranbot/VRforIM/data/" + dataset + ".mat"
        mat = scipy.io.loadmat(file)
        if dataset == 'pendulum':
            self.Xtrain = mat['X_tr']
            self.Ytrain = mat['T_tr']
            self.Xtest = mat['X_tst']
            self.Ytest = mat['T_tst']
        elif dataset == 'boston':
            self.Xtrain = mat['X']
            self.Ytrain = mat['y']
            self.Xtest = mat['Xstar']
            self.Ytest = mat['ystar']
        elif dataset == 'bodyfat':
            self.Xtrain = mat['Xtrain'].toarray()
            self.Ytrain = mat['Ytrain']
            self.Xtest = mat['Xtest'].toarray()
            self.Ytest = mat['Ytest']
        mean_inputs = np.mean(np.array(self.Xtrain), axis=0)
        std_inputs = np.std(np.array(self.Xtrain), axis=0)
        self.Xtrain = (self.Xtrain- mean_inputs[None, :]) / std_inputs[None, :]
        self.Ytrain = self.Ytrain - np.mean(self.Ytrain)
        self.dim = 3
        self.n_data = self.Xtrain.shape[0]

    def logpdf(self, x, *paras):
        Xcov = jnp.expand_dims(self.Xtrain, axis=0).repeat(self.n_data, axis=0)
        Xcov2 = jnp.expand_dims(self.Xtrain, axis=0).repeat(self.n_data, axis=0).transpose(1,0,2)
        self.cov = jnp.exp( - jnp.sum((Xcov - Xcov2) ** 2 / jnp.exp(x[0]), axis = 2) / 2 ) * (jnp.exp(x[1]) ) + jnp.exp(x[2]) * jnp.diag(jnp.ones(self.n_data))
        # self.cov = jnp.exp( - jnp.sum((Xcov - Xcov2) ** 2 / jnp.exp(x[:-2]), axis = 2) / 2 ) * (jnp.exp(x[-2]) ) + jnp.exp(x[-1]) * jnp.diag(jnp.ones(self.n_data)) #different lengthscale
        self.L = jnp.linalg.cholesky(self.cov)
        v = jax.lax.linalg.triangular_solve(self.L, self.Ytrain, left_side=True, lower=True)
        log_lik = - 0.5 * jnp.sum(v * v) - 0.5 * self.n_data * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(jnp.diag(self.L)))
        log_prior = -(0.5 * x.shape[0]) * jnp.log(2. * jnp.pi) - (0.5 / 10) * jnp.sum(x * x)
        return log_lik + log_prior


