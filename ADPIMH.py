import numpy.random as nr
import numpy as np
from typing import Tuple, Optional
import jax.numpy as jnp
import jax
import optax
import haiku as hk
from Distributions import student_t, multivariate_normal, Stochastic_Volatility, Bayesian_Logistic_Regression

class Basic_settings():
    def __init__(
        self,
        target_str: str = 'multivariate_normal',      #target distribution
        proposal_str: str = 'multivariate_normal',     #proposal distribution
        itrnum: int = 10000,                #number of MH steps
        dim : int = 16,                  #dimensions
        learning_rate: float = 0.001,            #learning rate
        model: Optional[Stochastic_Volatility] = None,
        df : Optional[float] = 3,             #degree of freedom of student_t
        value : int = 10
        ) -> None:
        self.target_str = target_str
        self.proposal_str = proposal_str
        self.itrnum = itrnum
        self.dim = dim
        self.learning_rate = learning_rate
        self.model = model
        self.df = df
        if self.target_str == 'multivariate_normal':
            self.target = multivariate_normal(self.dim, value)
        elif self.target_str == 'stochastic_volatility' or 'bayesian_logistic_regression':
            self.target = self.model
        else:
            NotImplementedError

        if self.proposal_str == 'multivariate_normal':
            self.proposal = multivariate_normal(self.dim, value)
        elif self.proposal_str == 'student_t':
            self.proposal = student_t(self.dim, self.df)
            self.proposal_adp = multivariate_normal(self.dim)
            self.tcoeff = self.df / (self.df - 2)
        else:
            NotImplementedError
            
        self.logtarget = jax.jit(self.target.logpdf)
        # self.logtarget = self.target.logpdf
        self.logproposal = jax.jit(self.proposal.logpdf)
    
    # t_mean: mean value of target ; t_L: scale of target
    # p_mean: mean value of proposal ; p_L: scale of proposal
    # adp_L: scale of distribution used for adaptation. adp_mean = p_mean due to moment matching
    def set_distribution_parameters(self, *args) -> None:
        self.t_mean = args[0]
        self.t_L = args[1]

        self.p_mean = args[2]
        self.p_L = args[3]

        if self.proposal_str == 'student_t':
            self.adp_L = self.p_L * jnp.sqrt(self.tcoeff) #scale of normal after moment matching
        elif self.proposal_str == 'multivariate_normal':
            self.adp_L = self.p_L

    def init_state(self) -> None:
        self.x0, _ = self.proposal.sampler(self.p_mean, self.p_L, 1)
        self.current = self.x0.reshape((self.dim,))
        self.log_t = self.logtarget(self.current, self.t_mean, self.t_L)
        self.log_p = self.logproposal(self.current, self.p_mean, self.p_L)
        self.grad_val = 0
        self.grad_L = 0
        self.cum_obj = 0


    def IMHsampling(self, num, is_adp = True):
        sample_p, epsilon = self.proposal.sampler(self.p_mean, self.p_L, num)
        if is_adp:
            if self.proposal_str == 'multivariate_normal':   
                sample_adp = sample_p
            if self.proposal_str == 'student_t':
                sample_adp, epsilon = self.proposal_adp.sampler(self.p_mean, self.adp_L, num)
            return sample_p, sample_adp, epsilon
        else:
            return sample_p, epsilon


    def IMHstep(self, p_test):
        is_accept = 0
        log_t_new = self.logtarget(p_test, self.t_mean, self.t_L)
        log_p_new = self.logproposal(p_test, self.p_mean, self.p_L)
        ratio_t = log_t_new - self.log_t
        ratio_p = log_p_new - self.log_p
        accept_ratio = min(1.0, jnp.exp(ratio_t - ratio_p))
        test_uniform = nr.uniform(0.0, 1.0, 1)
        if test_uniform <= accept_ratio:
            self.current = p_test
            is_accept = 1
            self.log_t = log_t_new
            self.log_p = log_p_new
        return accept_ratio, is_accept
    

    def objective_function(self, data):
        obj_fun = self.logtarget(data, self.t_mean, self.t_L)
        return obj_fun
    

    def optimization_init(self, not_fix_mean = False):
        if not_fix_mean:
            self.optimizer_mean = optax.adam(self.learning_rate)
            self.opt_state_mean = self.optimizer_mean.init(self.p_mean)
        self.optimizer_L = optax.adam(self.learning_rate)
        self.opt_state_L = self.optimizer_L.init(self.adp_L)
        self.obj_grad = jax.grad(self.objective_function)

    
    def optimization_collect(self, adp_sample, epsilon):
        self.cur_grad = self.obj_grad(adp_sample)
        self.grad_val += self.cur_grad / self.batch_size
        self.grad_L += (self.cur_grad.reshape((self.dim,1)) @ epsilon.reshape((1,self.dim)) + jnp.diag(1 / jnp.diag(self.adp_L))) / self.batch_size
        self.cum_obj += (self.objective_function(adp_sample) + jnp.sum(jnp.log(jnp.diag(self.adp_L)))) / self.batch_size
        
    
    def optimization_update(self, not_fix_mean = False):
        if not_fix_mean:
            updates, self.opt_state_mean = self.optimizer_mean.update(-self.grad_val, self.opt_state_mean)
            self.p_mean = optax.apply_updates(self.p_mean, updates)
        updates, self.opt_state_L = self.optimizer_L.update(-self.grad_L, self.opt_state_L)
        self.adp_L = optax.apply_updates(self.adp_L, updates)

        self.adp_L = jax.numpy.tril(self.adp_L, k = 0)
        bounded_diag = jnp.clip(jnp.diag(self.adp_L), a_min=0.001, a_max=1000.)
        self.adp_L = self.adp_L - jnp.diag(jnp.diag(self.adp_L)) + jnp.diag(bounded_diag)

        if self.proposal_str == 'student_t': 
            self.p_L = self.adp_L / jnp.sqrt(self.tcoeff)
        elif self.proposal_str == 'multivariate_normal':
            self.p_L = self.adp_L
        self.log_p = self.logproposal(self.current, self.p_mean, self.p_L)


class BAIM(Basic_settings):
    def __init__(self,
        target_str: str = 'multivariate_normal',
        proposal_str: str = 'multivariate_normal',
        itrnum: int = 10000,
        batch_size: int = 50,
        dim : int = 100,
        learning_rate: float = 0.01,
        model: Optional[Stochastic_Volatility] = None,
        df : Optional[float] = 3,
        value: int = 10
        ):
        super().__init__(target_str, proposal_str, itrnum, dim, learning_rate, model, df, value)
        self.batch_size = batch_size
        self.n_batch = int(self.itrnum / self.batch_size)


    def adpIMH(self) -> Tuple[np.ndarray,...]:

        #initialize arrays
        t_sample = np.zeros((self.itrnum, self.dim))
        p_sample = np.zeros((self.itrnum, self.dim))
        acceptance_rate = np.zeros(self.itrnum)
        is_acc = np.zeros(self.itrnum)
        logtar = np.zeros(self.itrnum)
        loss = np.zeros(self.n_batch)
        pmeans = np.zeros((self.itrnum, self.dim))
        pSigs = np.zeros((self.itrnum, self.dim, self.dim))

        self.init_state()
        self.optimization_init(True)

        #IM iterations
        for i in range(self.itrnum):
            if i % self.batch_size == 0:
                print('\r', end = str(int(i / self.batch_size + 1)) + '/'+ str(self.n_batch) + ' batches adapted !')
                pro_sample, adp_sample, epsilon = self.IMHsampling(self.batch_size)

            #IM iteration
            t_sample[i] = self.current         
            pos = i % self.batch_size    
            a, is_a = self.IMHstep(pro_sample[pos])

            self.optimization_collect(adp_sample[pos], epsilon[pos])  
            p_sample[i] = pro_sample[pos]
            acceptance_rate[i] = np.reshape(a, ())
            is_acc[i] = is_a
            logtar[i] = self.log_t     
            pmeans[i] = jnp.reshape(self.p_mean, (self.dim,))
            pSigs[i] = self.p_L @ self.p_L.T

            if (i + 1) % self.batch_size == 0:
                self.optimization_update(True)
                pos = int((i + 1) / self.batch_size - 1)
                loss[pos] = self.cum_obj
                self.grad_val = 0
                self.grad_L = 0
                self.cum_obj = 0
            
        return t_sample, p_sample, acceptance_rate, (pmeans,pSigs), loss,  logtar, is_acc

class IndMH(Basic_settings):
    def __init__(self,
        target_str: str = 'stochastic_volatility',
        proposal_str: str = 'multivariate_normal',
        itrnum: int = 10000,
        dim : int = 16,
        learning_rate: float = 0.01,
        model: Optional[Stochastic_Volatility] = None,
        df : Optional[float] = 3
        ):
        super().__init__(target_str, proposal_str, itrnum, dim,learning_rate, model, df)
    
    
    def IMH(self) -> Tuple[np.ndarray,...]:
       #initialize arrays
        t_sample = np.zeros((self.itrnum, self.dim))
        acceptance_rate = np.zeros(self.itrnum)
        is_acc = np.zeros(self.itrnum)

        self.init_state()
        #IMH iterations
        p_sample , _= self.IMHsampling(self.itrnum, False)
        for i in range(self.itrnum):
                
            a, is_a = self.IMHstep(p_sample[i])          

            #update
            t_sample[i] = self.current.reshape((self.dim,))
            acceptance_rate[i] = np.reshape(a,())
            is_acc[i] = is_a
            print('\r', end = str(i) + '/'+ str(self.itrnum) + ' finished !')
        print('The average acceptance probability is: ', np.mean(is_acc))

        return t_sample, p_sample, acceptance_rate, is_acc
    

class adp_only(Basic_settings):
    def __init__(self, 
                 target_str: str = 'multivariate_normal', 
                 proposal_str: str = 'multivariate_normal', 
                 itrnum: int = 10000,
                 batch_size: int = 50,
                 dim: int = 16, 
                 learning_rate: float = 0.01, 
                 model: Optional[Stochastic_Volatility] = None, 
                 df: Optional[float] = 3) -> None:
        super().__init__(target_str, proposal_str, itrnum, dim, learning_rate, model, df)
        self.batch_size = batch_size
        self.n_batch = int(self.itrnum / self.batch_size)

    def adp(self, nfm):
        loss = np.zeros(self.n_batch)
        self.init_state()
        self.optimization_init(not_fix_mean=nfm)
        for i in range(self.itrnum):
            if i % self.batch_size == 0:
                print('\r', end = str(int(i / self.batch_size + 1)) + '/'+ str(self.n_batch) + ' batches adapted !')
                p_sample, epsilon = self.IMHsampling(self.batch_size, False)

            pos = i % self.batch_size    
            self.optimization_collect(p_sample[pos], epsilon[pos])

            if (i + 1) % self.batch_size == 0:
                self.optimization_update(not_fix_mean=nfm)
                pos = int((i + 1) / self.batch_size - 1)
                loss[pos] = self.cum_obj
                self.grad_val = 0
                self.grad_L = 0
                self.cum_obj = 0
        return loss
    

class pCN(Basic_settings):
    def __init__(self,
        target_str: str = 'multivariate_normal',
        proposal_str: str = 'multivariate_normal',
        itrnum: int = 10000,
        dim : int = 100,
        learning_rate: float = 0.01,
        model: Optional[Stochastic_Volatility] = None,
        df : Optional[float] = 3,
        beta : float = 0.5,
        value : int = 10
        ):
        super().__init__(target_str, proposal_str, itrnum, dim, learning_rate, model, df, value)
        self.beta = beta
    
    def pCNsampling(self):
        sample_p , epsilon = self.proposal.sampler(self.p_mean, self.p_L, 1)
        # sample_p = (1 - self.beta) * self.p_mean + self.beta * self.current + jnp.sqrt(1-self.beta ** 2) * self.p_L @ epsilon.reshape((self.dim,))
        return sample_p

    def pCN_MCMC(self):
        t_sample = np.zeros((self.itrnum, self.dim))
        p_sample = np.zeros((self.itrnum, self.dim))
        acceptance_rate = np.zeros(self.itrnum)
        paras = np.zeros((self.itrnum, self.dim))
        logtar = np.zeros(self.itrnum)
        is_acc = np.zeros(self.itrnum)
        self.init_state()
        #IMH iterations
        for i in range(self.itrnum):
            print(str(i+1) + '/'+ str(self.itrnum) + ' iterations','\r', end = '')
            samplep = self.pCNsampling()
            t_sample[i] = self.current.reshape((self.dim,))
            a, is_a = self.IMHstep(samplep.reshape(self.dim,))     
            #update

            p_sample[i] = samplep.reshape((self.dim,))
            acceptance_rate[i] = np.reshape(a,())
            is_acc[i] = is_a
            paras[i] = (1 - self.beta) * self.p_mean + self.beta * self.current
            logtar[i] = self.log_t
        print('accprob:',is_acc.mean())

        return t_sample, p_sample, acceptance_rate, paras, is_acc, logtar