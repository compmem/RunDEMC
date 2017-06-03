# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from RunDEMC import Model, Param, dists
from RunDEMC import joint_plot, violin_plot
import pylab as pl

save_posts = True


def eval_fun(pop, *args):
    # errors = []
    # for indiv in pop:
    #     # inverse exponential with offset, y = a * exp(b/x) + c
    #     predicted = (indiv[0] * np.exp(indiv[1] / args[0]) + indiv[2])
    #     errors.append(predicted - args[1])

    # evaluate the population with some broadcasting
    pred = (pop[:, 0][:, np.newaxis] *
            np.exp(pop[:, 1][:, np.newaxis] / args[0][np.newaxis, :]) +
            pop[:, 2][:, np.newaxis])
    errors = pred - args[1][np.newaxis, :]

    # sum of squared error
    # errors = np.asarray(errors)
    sse = np.sum(errors * errors, 1)
    # sae = np.sum(np.abs(errors),1)

    # calculate the weight with a normal kernel
    weights = np.log(dists.normal(mean=0.0, std=pop[:, 3]).pdf(sse))

    # see if return both weights and predicted vals
    if save_posts:
        return weights, pred
    else:
        return weights


# set up the data
xData = np.array([5.357, 9.861, 5.457, 5.936, 6.161, 6.731])
yData = np.array([0.376, 7.104, 0.489, 1.049, 1.327, 2.077])

# set up the parameters
params = [Param(name='a', prior=dists.uniform(-100, 100)),
          Param(name='b', prior=dists.uniform(-100, 100)),
          Param(name='c', prior=dists.uniform(-100, 100)),
          Param(name='delta', display_name=r'$\mathbf{\delta}$',
                prior=dists.exp(20),
                init_prior=dists.uniform(0, 10),
                ),
          ]

# set up abc
abc = Model(name='fun', params=params,
            like_fun=eval_fun, like_args=(xData, yData),
            initial_zeros_ok=False,
            use_priors=True, verbose=True)

# run for a burnin with the local_to_best
burnin = 400
abc(200, burnin=True)

# fix the delta error term based on the mean
ind = abc.param_names.index('delta')
fixed_delta = abc.particles[-1, :, ind].mean()
abc._particles[-1][:, ind] = fixed_delta
abc._params[ind].prior = fixed_delta
abc._weights[-1][:] = abc._log_likes[-1][:] + \
    abc.calc_log_prior(abc._particles[-1])

# run for more iterations to map posterior
abc(1000, burnin=False)

# joint plot
pl.figure(1)
pl.clf()
ax = joint_plot(abc.particles[burnin:, :, :-1], abc.weights[burnin:],
                burnin=burnin,
                names=abc.param_display_names[:-1],
                rot=45, sep=.02)
pl.show()

# ppd plot
pl.figure(2)
violin_plot([d.ravel() for d in abc.posts[burnin:].T],
            positions=xData)
pl.plot(xData, yData, 'xr', markersize=10)
pl.show()

# show best fit
burnin = 400
print("Best fitting params:")
best_ind = abc.weights[burnin:].argmax()
indiv = [abc.particles[burnin:, :, i].ravel()[best_ind]
         for i in range(abc.particles.shape[-1])]
for p, v in zip(abc.param_names, indiv):
    print('%s: %f' % (p, v))