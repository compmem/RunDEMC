# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from RunDEMC import RunDEMC, Param, dists
from RunDEMC import DE
from RunDEMC import joint_plot, violin_plot
from RunDEMC.density import kdensity
from joblib import Parallel, delayed

import pylab as pl

n_jobs = 2
nsamps = 20000
ndat = 1000
xx = np.linspace(-10, 10, nsamps)
mu = 1.0
sd = 2.0
dmod = dists.normal(mu, sd)
data = dmod.rvs(ndat)


def eval_prop(indiv, do_true=False):
    mod = dists.normal(indiv[0], indiv[1])
    if do_true:
        pdf = mod.pdf(data)
    else:
        mdat = mod.rvs(nsamps)
        pdf, xx = kdensity(mdat, xx=data, nbins=2000,
                           kernel="epanechnikov"
                           )

    if np.any(pdf == 0):
        return -np.inf
    weight = np.log(pdf).sum()
    return weight


def eval_fun(abc, pop, *args):

    res = Parallel(n_jobs=n_jobs)(delayed(eval_prop)(indiv, args[0])
                                  for indiv in pop)

    weights = np.asarray(res)

    if abc._save_posts:
        return weights, None
    else:
        return weights


# set up the parameters
params = [Param(name='mu', prior=dists.uniform(-20, 20)),
          Param(name='sd', prior=dists.uniform(0, 20)),
          ]

burnin = 50
iterations = 500

# set up abc
do_true = True
abc_true = DEMC(params, eval_fun, eval_args=(do_true,),
                num_groups=1, group_size=25,
                proposal_gen=DE(gamma_best=None, rand_base=True),
                migration_prob=0.0, initial_zeros_ok=False,
                use_priors=True, save_posts=False)
abc_true(burnin)
abc_true.set_proposal_gen(DE(gamma_best=0.0, rand_base=False))
abc_true(iterations)

do_true = False
abc = DEMC(params, eval_fun, eval_args=(do_true,),
           num_groups=1, group_size=25,
           proposal_gen=DE(gamma_best=None, rand_base=True),
           migration_prob=0.0, initial_zeros_ok=False,
           use_priors=True, save_posts=False)
abc(burnin)
abc.set_proposal_gen(DE(gamma_best=0.0, rand_base=False))
abc(iterations)

do_true = False
abc_rand = DEMC(params, eval_fun, eval_args=(do_true,),
                num_groups=1, group_size=25,
                proposal_gen=DE(gamma_best=None, rand_base=True),
                migration_prob=0.0, initial_zeros_ok=False,
                use_priors=True, save_posts=False)
abc_rand(burnin)
abc_rand.set_proposal_gen(DE(gamma_best=0.0, rand_base=True))
abc_rand(iterations)


# joint plot
pl.figure(1)
pl.clf()
ax = joint_plot(abc_true.particles, abc_true.weights,
                burnin=burnin,
                names=abc_true.param_display_names,
                rot=45, sep=.02)
pl.show()

pl.figure(2)
pl.clf()
b = pl.hist(abc.particles[(burnin + 100):, :, 0].flat,
            50, alpha=.3, normed=True)
b = pl.hist(abc_true.particles[(burnin + 100):,
                               :, 0].flat, 50, alpha=.3, normed=True)
b = pl.hist(abc_rand.particles[(burnin + 100):,
                               :, 0].flat, 50, alpha=.3, normed=True)
pl.legend(['pda', 'true', 'rand'])
pl.show()

pl.figure(3)
pl.clf()
b = pl.hist(abc.particles[(burnin + 100):, :, 1].flat,
            50, alpha=.3, normed=True)
b = pl.hist(abc_true.particles[(burnin + 100):,
                               :, 1].flat, 50, alpha=.3, normed=True)
b = pl.hist(abc_rand.particles[(burnin + 100):,
                               :, 1].flat, 50, alpha=.3, normed=True)
pl.legend(['pda', 'true', 'rand'])
pl.show()

# show best fit
print("Best fitting params:")
best_ind = abc.weights[burnin:].argmax()
indiv = [abc.particles[burnin:, :, i].ravel()[best_ind]
         for i in range(abc.particles.shape[-1])]
for p, v in zip(abc.param_names, indiv):
    print(('%s: %f' % (p, v)))
