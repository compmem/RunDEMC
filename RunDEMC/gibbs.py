# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
import time

from .dists import normal, invgamma
from .param import Param, _apply_param_transform

class NormalHyperPrior(object):
    """Gaussian hyperprior updated via Gibbs sampling."""

    param_names = property(lambda self:
                           [p.name for p in self._params],
                           doc="""
                           List of parameter names.
                           """)

    param_display_names = property(lambda self:
                                   [p.display_name for p in self._params],
                                   doc="""
                                   List of parameter display names.
                                   """)

    particles = property(lambda self:
                         _apply_param_transform(
                             np.asarray(self._particles),
                             self._get_transforms()),
                         doc="""
                         Particles as an array.
                         """)

    def __init__(self, name, mu, sigma, alpha, beta):
        self._name = name
        self._mu = mu
        self._sigma = sigma
        self._alpha = alpha
        self._beta = beta

        self._params = [Param(name='mean',
                              prior=normal(self._mu, self._sigma)),
                        Param(name='std',
                              prior=invgamma(self._alpha,
                                             self._beta),
                              transform=np.sqrt)]

        self._dist = normal

        self._initialized = False

    def _initialize(self, num_chains=None, force=False):
        # start a timer
        stime = time.time()

        if num_chains is None:
            raise ValueError("num_chains must be specified.")
        self._num_chains = num_chains
        
        # set initial values
        means = normal(self._mu,
                       self._sigma).rvs((self._num_chains,1))
        stds = invgamma(self._alpha,
                        self._beta).rvs((self._num_chains,1))
        self._particles = [np.hstack([means, stds])]

        # there are no weights or likes, etc...
        self._weights = []
        self._log_likes = []
        self._times = [time.time() - stime]
        self._accept_rate = []
        
        self._initialized = True

    def sample(self, num_iter, burnin=False, migration_prob=0.0):
        # burnin and migration are ignored
        # make sure we're initialized
        if not self._initialized:
            raise RuntimeError("NormalHyperPrior must be initialized as part of a Hierarchy.")

        times = []
        for i in range(num_iter):
            # start a timer
            stime = time.time()
            
            # grab the current particles
            particles = self._particles[-1].copy()

            # loop over submodels collecting current particles
            sub_parts = []
            for m in self._like_args:
                sub_parts.append(m['model']._particles[-1][:, m['param_ind']])
            sub_parts = np.array(sub_parts)

            # get number of submodels
            n = sub_parts.shape[0]

            # update the means
            # grab the priors
            mu0 = self._mu
            sigma_sq0 = self._sigma**2

            # get the current vars
            sigma_sq = particles[:, 1]

            # calc num and denomenators for new mu and sigma
            num = (mu0/sigma_sq) + (sub_parts.sum(0)/sigma_sq)
            den = (1./sigma_sq0) + (n/sigma_sq)
            mu = num/den
            sigma = den**(-1/2.)

            # generate new means
            particles[:, 0] = normal(mu, sigma).rvs()

            # now update the stds based on these new means
            # get current values
            alpha = self._alpha
            beta = self._beta
            mu = particles[:, 0]

            # calc new a and b for the new stds
            a = alpha + n/2.
            b = beta + ((sub_parts - mu)**2).sum(0)/2.
            particles[:, 1] = invgamma(a, b).rvs()

            # save the new particles
            self._particles.append(particles)

            # save the time
            times.append(time.time() - stime)

        self._times.extend(times)
        return times
        
    def pdf(self, vals, cur_split):
        # self._dist can't be None
        # pick from chains
        vals = np.atleast_1d(vals)

        if cur_split is None:            
            if len(vals) == self._num_chains:
                # have them match
                cur_split = np.arange(self._num_chains)
            else:
                # pick randomly
                cur_split = np.random.randint(0, self._num_chains, len(vals))

        # generate the pdf using the likelihood func
        pop = _apply_param_transform(self._particles[-1][cur_split],
                                     self._get_transforms())
        args = [pop[:, i] for i in range(pop.shape[1])]
        d = self._dist(*args)
        if np.ndim(vals) > 1:
            p = np.vstack([d.pdf(vals[:, i])
                           for i in range(vals.shape[1])]).T
        else:
            p = d.pdf(vals)
        return p

    def logpdf(self, vals, cur_split):
        # self._dist can't be None
        # pick from chains
        vals = np.atleast_1d(vals)

        if cur_split is None:            
            if len(vals) == self._num_chains:
                # have them match
                cur_split = np.arange(self._num_chains)
            else:
                # pick randomly
                cur_split = np.random.randint(0, self._num_chains, len(vals))

        # generate the pdf using the likelihood func
        pop = _apply_param_transform(self._particles[-1][cur_split],
                                     self._get_transforms())
        args = [pop[:, i] for i in range(pop.shape[1])]
        d = self._dist(*args)
        if np.ndim(vals) > 1:
            p = np.vstack([d.logpdf(vals[:, i])
                           for i in range(vals.shape[1])]).T
        else:
            p = d.logpdf(vals)

        return p

    def rvs(self, size, cur_split):
        # randomly pick from chains
        size = np.atleast_1d(size)

        # pick chains
        num_chains = len(cur_split)
        if size[0] == cur_split.sum():
            # have them match
            chains = np.arange(num_chains)[cur_split]
        else:
            # pick randomly
            chains = np.random.randint(0, num_chains, size[0])

        # generate the random vars using the likelihood func
        # pop = self._particles[-1][chains]
        pop = _apply_param_transform(self._particles[-1],
                                     self._get_transforms())
        # r = self._dist(*(pop[:,i] for i in range(pop.shape[1]))).rvs(size[1:])
        r = np.array([self._dist(*pop[ind]).rvs(size[1:])
                      for i, ind in enumerate(chains)])
        return r.reshape(size)

    def _get_transforms(self, inverse=False):
        # loop over priors and get the transforms
        if inverse:
            return [p.inv_transform for p in self._params]
        else:
            return [p.transform for p in self._params]


