#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from scipy.stats.distributions import norm,expon
from RunDEMC import RunDEMC, Param, dists
from RunDEMC import DE_LOCAL_TO_BEST, DE_LOCAL
from RunDEMC import joint_plot, violin_plot
import pylab as pl

def eval_fun(abc, pop, *args):
    # errors = []
    # for indiv in pop:
    #     # inverse exponential with offset, y = a * exp(b/x) + c
    #     predicted = (indiv[0] * np.exp(indiv[1] / args[0]) + indiv[2])
    #     errors.append(predicted - args[1])

    # evaluate the population with some broadcasting
    pred = (pop[:,0][:,np.newaxis] *
            np.exp(pop[:,1][:,np.newaxis]/args[0][np.newaxis,:]) +
            pop[:,2][:,np.newaxis])
    errors = pred - args[1][np.newaxis,:]

    # sum of squared error
    #errors = np.asarray(errors)
    sse = np.sum(errors*errors,1)
    #sae = np.sum(np.abs(errors),1)

    # calculate the weight with a normal kernel
    weights = np.log(norm.pdf(sse,scale=pop[:,3]))
    #weights = np.log(norm.pdf(sse,scale=.1))

    # see if return both weights and predicted vals
    if abc._save_posts:
        return weights,pred
    else:
        return weights

# set up the data
xData = np.array([5.357, 9.861, 5.457, 5.936, 6.161, 6.731])
yData = np.array([0.376, 7.104, 0.489, 1.049, 1.327, 2.077])

# set up the parameters
params = [Param(name='a',prior=dists.uniform(-100,100)),
          Param(name='b',prior=dists.uniform(-100,100)),
          Param(name='c',prior=dists.uniform(-100,100)),
          Param(name='delta',display_name=r'$\mathbf{\delta}$',
                prior=dists.exp(20),
                init_prior=dists.uniform(0,10),
                ),
          ]

# set up abc
abc = DEMC(params, eval_fun, eval_args = (xData,yData),
            num_groups=4, group_size=30,
            proposal_gen=DE_LOCAL_TO_BEST(),
            migration_prob=0.1, initial_zeros_ok=False,
            use_priors=True, save_posts=True)

# run for a burnin with the local_to_best
burnin = 100
abc(burnin)

# fix the delta error term based on the mean
ind = abc.param_names.index('delta')
abc.fix_delta(ind, abc.particles[-1,:,ind].mean(),
              proposal_gen=DE_LOCAL(),
              recalc_weights=True)

# run for more iterations to map prior
abc(100)

# joint plot
pl.figure(1)
pl.clf()
ax = joint_plot(abc.particles[:,:,:-1],abc.weights,
                burnin=burnin,
                names=abc.param_display_names[:-1],
                rot=45,sep=.02)
pl.show()

# ppd plot
pl.figure(2)
violin_plot([d.ravel() for d in abc.posts[burnin:].T],
            positions=xData)
pl.plot(xData,yData,'xr',markersize=10)
pl.show()

# show best fit
print "Best fitting params:"
best_ind = abc.weights[burnin:].argmax()
indiv = [abc.particles[burnin:,:,i].ravel()[best_ind] for i in range(abc.particles.shape[-1])]
for p,v in zip(abc.param_names,indiv):
    print '%s: %f'%(p,v)
    
