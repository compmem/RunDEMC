#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from .dists import normal, invgamma

class HyperParam(object):
    """Normal distribution used as a hyperparameter.

    You must specify prior for the mu and sigma of the normal
    distribution of the mean and the alpha and beta of the inverse
    gamma distribution of the std.

    """
    means = property(lambda self: self._mean.copy(),
                     doc="""
                     Get the current mean for each chain.
                     """)
    stds = property(lambda self: self._std.copy(),
                    doc="""
                    Get the current std for each chain.
                    """)

    def __init__(self, name='', display_name=None,
                 mu=0.0, sigma=1.0, alpha=1.0, beta=1.0,
                 nchains=1):
        self.name = name
            
        if display_name is None:
            display_name = self.name
        self.display_name = display_name

        # save the priors
        self._mu = mu
        self._sigma = sigma
        self._alpha = alpha
        self._beta = beta

        # save number of chains
        self._nchains = nchains

        # set initial values
        self._mean = normal(mu,sigma).rvs(nchains)
        self._std = np.sqrt(invgamma(alpha,beta).rvs(nchains))

    def pdf(self, x):
        x = np.atleast_1d(x)
        if len(x) != self._nchains:
            #raise ValueError("You must request pdf for all %d chains simultaneously."%self._nchains)
            # warning, should only happen during initialization
            pass
        pdf = np.zeros(len(x))
        for i in range(len(x)):
            # wrap around as necessary (might only happen at initialization)
            ind = i%self._nchains
            pdf[i] = normal(self._mean[ind], self._std[ind]).pdf(x[i])
        return np.array(pdf)

    def rvs(self, shape):
        rvs = np.zeros(shape)
        for i in range(shape[0]):
            ind = i%self._nchains
            rvs[i] = normal(self._mean[ind], self._std[ind]).rvs([1]+list(shape[1:]))
        return np.array(rvs)

    def update(self, posts):
        """
        Update the mean and std based on the posteriors.
        """
        self._update_mean(posts)
        self._update_std(posts)

    def _update_mean(self, posts):
        """
        Update the current mean.
        """
        if len(posts) != self._nchains:
            raise ValueError("You must update all %d chains simultaneously."%self._nchains)
        for i in range(self._nchains):
            # update based on posterior
            # update.mu=function(x,use.core,use.sigma,prior){
            #     X=use.core[x,]
            X = np.array(posts[i]).flatten()
            #     n=length(X)
            n = len(X)
            #     mu0=prior$mu
            mu0 = self._mu
            #     sigma.sq0=prior$sigma^2
            sigma_sq0 = self._sigma**2
            #     sigma.sq=use.sigma[x]^2
            sigma_sq = self._std[i]**2
            #     num=mu0/sigma.sq0 + sum(X)/sigma.sq
            num = mu0/sigma_sq0 + X.sum()/sigma_sq
            #     den=1/sigma.sq0 + n/sigma.sq
            den = 1./sigma_sq0 + n/sigma_sq
            #     mean=num/den
            mu = num/den
            #     sigma=den^(-1/2)
            sigma = den**(-1/2.)
            #     rnorm(1,mean,sigma)
            self._mean[i] = normal(mu,sigma).rvs()

        return self._mean

    def _update_std(self, posts):
        """
        Update the current std.
        """
        if len(posts) != self._nchains:
            raise ValueError("You must update all %d chains simultaneously."%self._nchains)
        for i in range(self._nchains):
            # update.sigma=function(x,use.core,use.mu,prior){
            #     require(MCMCpack)
            #     X=use.core[x,]
            X = np.array(posts[i]).flatten()
            #     n=length(X)
            n = len(X)
            #     alpha=prior$alpha
            alpha = self._alpha
            #     beta=prior$beta
            beta = self._beta
            #     mu=use.mu[x]
            mu = self._mean[i]
            #     a=alpha+n/2
            a = alpha+n/2.
            #     b=beta+sum((X-mu)**2)/2
            b = beta + ((X-mu)**2).sum()/2.
            #     sqrt(rinvgamma(1,a,b))
            self._std[i] = np.sqrt(invgamma(a,b).rvs())
        return self._std


class Hierarchy(object):
    """
    """

    param_names = property(lambda self:
                           [p.name for p in self._params],
                           doc="""
                           List of hyper-parameter names.
                           """)

    param_display_names = property(lambda self:
                                   [p.display_name for p in self._params],
                                   doc="""
                                   List of hyper-parameter display names.
                                   """)

    submodels = property(lambda self:
                         [m for m in self._submodels],
                         doc="""
                         List of submodels
                         """)

    means = property(lambda self: np.asarray(self._means),
                     doc="""
                     Get the means for each chain for all iterations.
                     """)

    stds = property(lambda self: np.asarray(self._stds),
                    doc="""
                    Get the stds for each chain for all iterations.
                    """)

    def __init__(self, params, submodels, verbose=True):
        # save the input
        self._params = params
        self._submodels = submodels
        self._verbose = verbose

        # init the means and stds
        self._means = []
        self._stds = []

        # save the starting vals
        self._means.append([p.means for p in params])
        self._stds.append([p.stds for p in params])

        # update the mean and std of the hyperparams based on the
        # initialized models
        self._update()

    def _update(self):
        # grab the posts for each matching param
        # the first dimension should be the chains
        for i in range(len(self._params)):
            posts = []
            for sm in self._submodels:
                ind = np.where(np.in1d(sm.param_names, [self._params[i].name]))[0]
                if len(ind) > 0:
                    posts.append(sm.particles[:,:,ind[0]].T)
            posts = np.hstack(posts)
            self._params[i].update(posts)

    def __call__(self, num_iter):
        # loop over iterations
        if self._verbose:
            sys.stdout.write('Iterations (%d): '%(num_iter))
        for i in range(num_iter):
            if self._verbose:
                sys.stdout.write('%d '%(i+1))
                sys.stdout.flush()
            # call each submodel for one iteration
            for sm in self._submodels:
                sm(1)

            # update mean and std for each hyperparam
            self._update()

            # save new hyperparam state
            self._means.append([p.means for p in self._params])
            self._stds.append([p.stds for p in self._params])
        if self._verbose:
            sys.stdout.write('\n')
