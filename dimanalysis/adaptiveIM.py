import numpy.random as nr
import numpy as np
from typing import Tuple, Optional
import jax.numpy as jnp
import jax
import optax
import haiku as hk
from distributions import multivariate_normal

class Basic_settings():
    def __init__(
        self,
        target_str: str = 'multivariate_normal',      #target distribution
        proposal_str: str = 'multivariate_normal',     #proposal distribution
        itrnum: int = 10000,                #number of MH steps
        dim : int = 16,                  #dimensions
        learning_rate: float = 0.001,            #learning rate
        value : int = 10
        ) -> None:
        self.target_str = target_str
        self.proposal_str = proposal_str
        self.itrnum = itrnum
        self.dim = dim
        self.lr_l = learning_rate
        self.lr_m = learning_rate * 10
        if self.target_str == 'multivariate_normal':
            self.target = multivariate_normal(self.dim, value)

        if self.proposal_str == 'multivariate_normal':
            self.proposal = multivariate_normal(self.dim, value)

            
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
            return sample_p, sample_adp, epsilon
        else:
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
    

    def objective_function(self, data):
        obj_fun = self.logtarget(data, self.t_mean, self.t_L)
        return obj_fun
    

    def optimization_init(self):
        # The optimizer steps once per batch update, so tie the cosine schedule
        # to the number of batches rather than the raw MH iteration count.
        decay_steps = max(1, getattr(self, "n_batch", self.itrnum))
        schedule_m = optax.cosine_decay_schedule(init_value=self.lr_m, decay_steps=decay_steps)
        schedule_l = optax.cosine_decay_schedule(init_value=self.lr_l, decay_steps=decay_steps)
        self.optimizer_mean = optax.adam(schedule_m)
        self.optimizer_L = optax.adam(schedule_l)
        self.opt_state_mean = self.optimizer_mean.init(self.p_mean)
        self.opt_state_L = self.optimizer_L.init(self.adp_L)
        self.obj_grad = jax.grad(self.objective_function)

    
    def optimization_collect(self, adp_sample, epsilon):
        self.cur_grad = self.obj_grad(adp_sample)
        self.grad_val += self.cur_grad / self.batch_size
        self.grad_L += (self.cur_grad.reshape((self.dim,1)) @ epsilon.reshape((1,self.dim)) + jnp.diag(1 / jnp.diag(self.adp_L))) / self.batch_size
        # v = jax.lax.linalg.triangular_solve(self.adp_L, adp_sample - self.p_mean, left_side=True, lower=True)
        # u = jax.lax.linalg.triangular_solve(self.adp_L.T, v, left_side=True, lower=True)
        # self.grad_L += (self.cur_grad.reshape((self.dim,1)) + u.reshape((self.dim,1))) @ epsilon.reshape((1,self.dim)) / self.batch_size
        self.cum_obj += (self.objective_function(adp_sample) + jnp.sum(jnp.log(jnp.diag(self.adp_L)))) / self.batch_size
        
    
    def optimization_update(self):
        updates, self.opt_state_mean = self.optimizer_mean.update(-self.grad_val, self.opt_state_mean)
        self.p_mean = optax.apply_updates(self.p_mean, updates)
        updates, self.opt_state_L = self.optimizer_L.update(-self.grad_L, self.opt_state_L)
        self.adp_L = optax.apply_updates(self.adp_L, updates)

        self.adp_L = jax.numpy.tril(self.adp_L, k = 0)
        bounded_diag = jnp.clip(jnp.diag(self.adp_L), 0.001, 1000.)
        self.adp_L = self.adp_L - jnp.diag(jnp.diag(self.adp_L)) + jnp.diag(bounded_diag)

        self.p_L = self.adp_L
        self.log_p = self.logproposal(self.current, self.p_mean, self.p_L)


class IndMH(Basic_settings):
    def __init__(self,
        target_str: str = 'multivariate_normal',
        proposal_str: str = 'multivariate_normal',
        itrnum: int = 10000,
        dim : int = 16,
        learning_rate: int=0,
        value: int = 10
        ):
        super().__init__(target_str, proposal_str, itrnum, dim,learning_rate, value)
    
    def IMH(self) -> Tuple[np.ndarray,...]:
       #initialize arrays
        t_sample = np.zeros((self.itrnum, self.dim))
        acceptance_rate = np.zeros(self.itrnum)
        pmeans = np.zeros((self.itrnum, self.dim))

        self.init_state()
        #IMH iterations
        p_sample , _= self.IMHsampling(self.itrnum, False)
        for i in range(self.itrnum):

            t_sample[i] = self.current.reshape((self.dim,))

            a = self.IMHstep(p_sample[i])          
            #update
            
            acceptance_rate[i] = np.reshape(a,())
            pmeans[i] = jnp.reshape(self.p_mean, (self.dim,))


        return t_sample, p_sample, acceptance_rate,pmeans
    

class adp_only(Basic_settings):
    def __init__(self, 
                 target_str: str = 'multivariate_normal', 
                 proposal_str: str = 'multivariate_normal', 
                 itrnum: int = 10000,
                 batch_size: int = 50,
                 dim: int = 16, 
                 learning_rate: float = 0.01, 
                 ) -> None:
        super().__init__(target_str, proposal_str, itrnum, dim, learning_rate)
        self.batch_size = batch_size
        self.n_batch = int(self.itrnum / self.batch_size)

    def adp(self, nfm):
        loss = np.zeros(self.n_batch)
        self.init_state()
        self.optimization_init()
        for i in range(self.itrnum):
            if i % self.batch_size == 0:
                print('\r', end = str(int(i / self.batch_size + 1)) + '/'+ str(self.n_batch) + ' batches adapted !')
                p_sample, epsilon = self.IMHsampling(self.batch_size, False)

            pos = i % self.batch_size    
            self.optimization_collect(p_sample[pos], epsilon[pos])

            if (i + 1) % self.batch_size == 0:
                self.optimization_update()
                pos = int((i + 1) / self.batch_size - 1)
                loss[pos] = self.cum_obj
                self.grad_val = 0
                self.grad_L = 0
                self.cum_obj = 0
        return loss
    
