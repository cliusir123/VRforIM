import numpy as np
import numpy.random as nr


class IndMH():
    def __init__(self, itrnum, prior, proposal) -> None:
        self.itrnum = itrnum
        self.prior = prior
        self.proposal = proposal

    def IMH(self):
        self.prior.new_chain()
        Y = self.proposal.sampler(self.itrnum)
        log_t = self.prior.log_density(Y)
        log_q = self.proposal.logpdf(Y)

        x = self.proposal.sampler(1)[0]
        log_tx = self.prior.log_density(x[None])[0]
        log_qx = self.proposal.logpdf(x[None])[0]

        X = np.zeros_like(Y)
        alpha = np.zeros(self.itrnum)
        for i in range(self.itrnum):
            X[i] = x
            alpha[i] = min(1.0, np.exp((log_t[i] - log_tx) - (log_q[i] - log_qx)))
            if nr.uniform() <= alpha[i]:
                x, log_tx, log_qx = Y[i], log_t[i], log_q[i]
        return X, Y, alpha
