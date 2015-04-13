#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from RunDEMC import RunDEMC, Param, dists
from RunDEMC import DE
from RunDEMC.hierarchy import Hierarchy,HyperParam

import numpy as np
from joblib import Parallel, delayed

# generate the data
n_jobs = 1
nsubj = 4
nobs = 1000
mu=1.0
sigma=.5
dmod_mean = dists.normal(mu,sigma)
alpha=4
beta=10
dmod_std = dists.invgamma(alpha,beta)
data = {}
for s in range(nsubj):
    # draw a mean and std from hypers
    smean = dmod_mean.rvs()
    sstd = np.sqrt(dmod_std.rvs())
    dmod = dists.normal(smean,sstd)
    data[s] = {'mean': smean,
               'std': sstd,
               'obs': dmod.rvs(nobs)}

# set up model evaluation
def eval_prop(indiv, subj_id):
    # get the true pdf for those params
    mod = dists.normal(indiv[0],np.exp(indiv[1]))
    pdf = mod.pdf(data[subj_id]['obs'])
    if np.any(pdf==0):
        return -np.inf
    weight = np.log(pdf).sum()
    return weight
    
def eval_fun(abc, pop, *args):

    res = Parallel(n_jobs=n_jobs)(delayed(eval_prop)(indiv, args[0])
                                  for indiv in pop)

    weights = np.asarray(res)

    if abc._save_posts:
        return weights,None
    else:
        return weights

# set up the parameters
nchains = 25
hyper_mu = HyperParam(name='mu',
                      mu=0.0,
                      sigma=10.0,
                      alpha=4,
                      beta=10,
                      nchains=nchains)
hyper_sd = HyperParam(name='sd',
                      mu=np.log(1.0),
                      sigma=1.0,
                      alpha=4,
                      beta=10,
                      nchains=nchains)
hparams = [hyper_mu,hyper_sd]

params = [Param(name='mu', prior=hyper_mu,),# init_prior=dists.uniform(-20,20)),
          Param(name='sd', prior=hyper_sd,),# init_prior=dists.uniform(0,20)),
          ]

# set up abc
models = [DEMC(params, eval_fun, eval_args=(subj_num,),
                num_groups=1, group_size=nchains,
                proposal_gen=DE(gamma_best=None,rand_base=True),
                migration_prob=0.0, initial_zeros_ok=False,
                use_priors=True, save_posts=False, verbose=False)
          for subj_num in range(nsubj)]

hier = Hierarchy(hparams, models)



burnin = 50
iterations = 500
hier(burnin)
for sm in hier.submodels:
    sm.set_proposal_gen(DE(gamma_best=0.0, rand_base=False))
hier(iterations)

