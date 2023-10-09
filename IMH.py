import numpy.random as nr
import numpy as np
import scipy.stats as ss
from typing import Tuple
from Distributions import multivariate_normal, student_t
import jax
import jax.numpy as jnp

class IndMH():

    def __init__(
        self,
        itrnum: int = 10000,                         #number of MH steps
        sample_num : int = 4096,                     #number of samples
        dim : int = 16,
        value : int = 10
        ) -> None:
        self.dim = dim
        self.target = multivariate_normal(self.dim, value)
        self.proposal = multivariate_normal(self.dim, value)
        self.itrnum = itrnum
        self.sample_num = sample_num
        

        self.logtarget = jax.jit(self.target.logpdf, device=jax.devices('cpu')[0])
        self.logproposal = jax.jit(self.proposal.logpdf, device=jax.devices('cpu')[0])

    def set_distribution_parameters(self, *args) -> None:
        self.t_mean = args[0]
        self.t_L = args[1]

        self.p_mean = args[2]
        self.p_L = args[3]
    
    def init_state(self) -> None:
        self.x0, _ = self.proposal.sampler(self.p_mean, self.p_L, 1)
        self.current = self.x0.reshape((self.dim,))
        self.log_t = self.logtarget(self.current, self.t_mean, self.t_L)
        self.log_p = self.logproposal(self.current, self.p_mean, self.p_L)
    
    def IMHsampling(self, num):
        sample_p, epsilon = self.proposal.sampler(self.p_mean, self.p_L, num)
        return sample_p, epsilon
    
    def IMHstep(self, p_test):
        log_t_new = self.logtarget(p_test, self.t_mean, self.t_L)
        log_p_new = self.logproposal(p_test, self.p_mean, self.p_L)
        ratio_t = log_t_new - self.log_t
        ratio_p = log_p_new - self.log_p
        accept_ratio = min(1.0, jnp.exp(ratio_t - ratio_p))
        test_uniform = nr.uniform(0.0, 1.0, 1)
        if test_uniform <= accept_ratio:
            self.current = p_test
            self.log_t = log_t_new
            self.log_p = log_p_new
        return accept_ratio
    
    
    def IMH(self) -> Tuple[np.ndarray,...]:
       #initialize arrays
        t_sample = np.zeros((self.itrnum, self.dim))
        acceptance_rate = np.zeros(self.itrnum)

        self.init_state()
        #IMH iterations
        p_sample , _= self.IMHsampling(self.itrnum)
        for i in range(self.itrnum):
            t_sample[i] = self.current.reshape((self.dim,))
            a = self.IMHstep(p_sample[i])          
            #update
            acceptance_rate[i] = np.reshape(a,())

        return t_sample, p_sample, acceptance_rate