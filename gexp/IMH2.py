import numpy as np
import jax
import jax.numpy as jnp
import numpy.random as nr
from gdistribution import gprior, mix_normal

class IndMH():

    def __init__(
        self,
        itrnum: int = 10000,                         #number of MH steps
        dim : int = 16,
        target : str = 'gprior',
        proposal : str = 'mix_normal',
        ) -> None:
        self.dim = dim
        self.target = eval(target)(self.dim)
        self.proposal = eval(proposal)(self.dim)
        self.itrnum = itrnum
        
        self.logtarget = self.target.logpdf
        self.logproposal = self.proposal.logpdf
        self.logtarget = jax.jit(self.logtarget)
        self.logproposal = jax.jit(self.logproposal)

    def set_distribution_parameters(self, cov0, n_comps, weights, covs) -> None:
        self.cov = cov0
        self.target.set_parameters(cov0)
        self.proposal.set_parameters(n_comps, weights, covs)
    
    def init_state(self) -> None:
        self.x0 = self.proposal.sampler(self.coefs, self.means, self.covs, 1)
        self.current = self.x0.reshape((self.dim,))
        self.log_t = self.logtarget(self.current, self.cov)
        self.log_p = self.logproposal(self.current, self.coefs, self.means, self.covs)
    
    def IMHsampling(self, num):
        sample_p = self.proposal.sampler(self.coefs,self.means, self.covs, num)
        return sample_p
    
    def IMHstep(self, p_test):
        log_t_new = self.logtarget(p_test, self.cov)
        log_p_new = self.logproposal(p_test, self.coefs, self.means, self.covs)
        ratio_t = log_t_new - self.log_t
        ratio_p = log_p_new - self.log_p
        accept_ratio = min(1.0, jnp.exp(ratio_t - ratio_p))
        test_uniform = nr.uniform(0.0, 1.0, 1)
        if test_uniform <= accept_ratio:
            self.current = p_test
            self.log_t = log_t_new
            self.log_p = log_p_new
        return accept_ratio
    
    def IMH(self):
       #initialize arrays
        t_sample = np.zeros((self.itrnum, self.dim))
        acceptance_rate = np.zeros(self.itrnum)

        self.init_state()
        #IMH iterations
        p_sample = self.IMHsampling(self.itrnum)
        for i in range(self.itrnum):
            t_sample[i] = self.current.reshape((self.dim,))
            a = self.IMHstep(p_sample[i])          
            #update
            acceptance_rate[i] = np.reshape(a,())

        return t_sample, p_sample, acceptance_rate