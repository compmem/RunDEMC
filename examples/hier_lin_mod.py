#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the ABCDE package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

"""
http://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/
"""

import numpy as np
from RunDEMC import Model, Param, dists
from RunDEMC import Hierarchy, HyperPrior
import pylab as pl


###################
# Generate the data
###################

# set up the data
np.random.seed(42)

theta_true = (25, 0.5, 10)


def gen_subj_data(alpha, beta=.5, sigma=10, nx=20):
    xdata = 200 * np.random.random(nx)
    ydata = alpha + beta * xdata

    # add scatter to points
    xdata = np.random.normal(xdata, sigma)
    ydata = np.random.normal(ydata, sigma)

    return {'xdata': xdata, 'ydata': ydata,
            'alpha': alpha, 'beta': beta, 'sigma': sigma}

# generate data
nsubj = 5
alpha_sd = 10

# only across-subject variability is intercept
data = []
for i in range(nsubj):
    alpha = dists.normal(theta_true[0], alpha_sd).rvs()
    data.append(gen_subj_data(alpha))


##################
# Set up the Model
##################

# define the evaluation function
def subj_like(pop, *args):
    alpha, beta, sigma = (pop['alpha'], pop['beta'], pop['sigma'])

    xdata = args[0]['xdata']
    ydata = args[0]['ydata']
    y_model = (alpha + beta*xdata[:, np.newaxis]).T

    # calc the log like for all props
    # this could be implemented more clearly with just a normal
    # distribution, however, this is a faster way to vectorize
    # all proposals
    log_like = -0.5 * np.sum(np.log(2 * np.pi * sigma[:, np.newaxis] ** 2) +
                             (ydata - y_model)**2 / sigma[:, np.newaxis] ** 2,
                             axis=1)

    return log_like

# set up the parameters
gsize = None  # None will let model figure out a good number

# Fixed slope across participants
# Using a custom uniform prior
beta = Param(name='beta', display_name=r'$\beta$',
             prior=dists.CustomDist(pdf=lambda x:
                                    np.exp(-1.5 * np.log(1 + x ** 2)),
                                    rvs=dists.laplace(0, 5).rvs))
# Fixed noise across subjects
# Using a custom Jeffreys' prior
sigma = Param(name='sigma', display_name=r'$\sigma$',
              prior=dists.CustomDist(pdf=lambda x: np.exp(-np.log(np.abs(x))),
                                     rvs=dists.dists.invgamma(1, 1).rvs))

# Hyperprior over intercept using a normal distribution
halpha = HyperPrior('alpha', dists.normal,
                    params=[Param(name='mu',
                                  prior=dists.uniform(-50, 50)),
                            Param(name='sig',
                                  prior=dists.invgamma(1, 1))])

# set up the submodels for each participant
smods = []
for j in range(nsubj):
    # Append a new model, note the use of the hyperprior for setting
    # up the intercept param, and the fixed beta and sigma across
    # participants
    smods.append(Model(name=str(j),
                       params=[Param(name='alpha',
                                     display_name=r'$\alpha$',
                                     prior=halpha),
                               beta, sigma],
                       like_fun=subj_like,
                       like_args=(data[j],),
                       num_chains=gsize,
                       verbose=False))

# put together into a single model
hmod = Hierarchy(smods)


###################
# Run the inference
###################

# burnin
hmod(50, burnin=True)

# sample posterior
hmod(500)


#################
# Some plots
#################

burnin = 200  # allow for relaxation after burnin

lmod = smods[0]

print("Best fitting params:")
best_ind = lmod.weights[burnin:].argmax()
indiv = [lmod.particles[burnin:,:,i].ravel()[best_ind]
         for i in range(lmod.particles.shape[-1])]
for p,v in zip(lmod.param_names,indiv):
    print('%s: %f'%(p,v))


import matplotlib.pyplot as plt

def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]

    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)


def plot_MCMC_trace(ax, xdata, ydata, trace, scatter=False, **kwargs):
    """Plot traces and contours"""
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter:
        ax.plot(trace[0], trace[1], ',k', alpha=0.1)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')


def plot_MCMC_model(ax, xdata, ydata, trace):
    """Plot the linear model and 2sigma contours"""
    ax.plot(xdata, ydata, 'ok')

    alpha, beta = trace[:2]
    xfit = np.linspace(-20, 220, 20)
    yfit = alpha[:, None] + beta[:, None] * xfit
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)

    ax.plot(xfit, mu, '-k')
    ax.fill_between(xfit, mu - sig, mu + sig, color='lightgray')

    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_MCMC_results(xdata, ydata, trace, colors='k'):
    """Plot both the trace and the model together"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_MCMC_trace(ax[0], xdata, ydata, trace, True, colors=colors)
    plot_MCMC_model(ax[1], xdata, ydata, trace)


# get the particles and plot the data
trace = np.vstack([lmod.particles[burnin::7, :, i].flatten()
                   for i in range(len(lmod.param_names))])

plot_MCMC_results(lmod._like_args[0]['xdata'],
                  lmod._like_args[0]['ydata'],
                  trace=trace)
pl.show()
