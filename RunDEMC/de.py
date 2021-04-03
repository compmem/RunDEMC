# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


# global imports
import numpy as np
import sys
import random
import time


class Proposal(object):
    """
    Generate a new proposal population based on the current population
    and their weights.
    """

    def _generate(self, pop, ref_pop, weights=None):
        raise NotImplemented("You must define this method in a subclass.")

    def generate(self, pop, ref_pop=None, weights=None, fixed=None):
        # process the fixed params
        if fixed is None:
            fixed = np.zeros(pop.shape[1], dtype=np.bool)

        # process the ref_pop
        if ref_pop is None or len(ref_pop) == 0:
            # just use the pop
            ref_pop = pop.copy()

        # allocate for new proposal
        proposal = np.ones_like(pop) * np.nan

        # copy fixed params
        proposal[:, fixed] = pop[:, fixed]

        # generate values for non-fixed params
        proposal[:, ~fixed] = self._generate(pop[:, ~fixed],
                                             ref_pop[:, ~fixed],
                                             weights=weights)

        # return the new proposal
        return proposal

    def __call__(self, pop, ref_pop=None, weights=None, fixed=None):
        return self.generate(pop, ref_pop, weights, fixed)


class DE(Proposal):
    """
    Differential evolution proposal.
    """

    def __init__(self, CR=1.0, gamma=None,
                 gamma_best=0.0, epsilon=.0001,
                 rand_base=False):
        # set the crossover
        self._CR = CR

        # set gamma
        if gamma is None:
            ## based on original DEMC paper
            #gamma = 2.38 / np.sqrt(2*num_params)
            gamma = (0.4, 1.0)
        if np.isscalar(gamma):
            gamma = (gamma, gamma)
        self._gamma = gamma
        
        if gamma_best is None:
            gamma_best = (0.4, 1.0)
        elif np.isscalar(gamma_best):
            gamma_best = (gamma_best, gamma_best)
        self._gamma_best = gamma_best
        self._rand_base = rand_base
        self._epsilon = epsilon

    def _generate(self, pop, ref_pop, weights=None):
        """
        Generate a standard differential evolution proposal.
        """
        # allocate for new proposal
        proposal = np.ones_like(pop) * np.nan

        # process the weights
        if weights is not None:
            # make sure weights not zero
            if hasattr(np, "float128"):
                # np.finfo(weights.dtype).eps
                tweights = np.exp(np.float128(weights)) + .0000001
            else:
                tweights = np.exp(np.float64(weights)) + .0000001

            # zero out nan vals
            tweights[np.isnan(tweights)] = 0.0

            # get cumsum so we can sample probabilistically
            w_sum = tweights.sum()
            if w_sum > 0.0:
                p_w = tweights / w_sum
            else:
                # just use equal prob
                p_w = 1./len(weights)
            cum_w_sum = np.cumsum(p_w)

        # indices for all the particles
        ref_ind = np.arange(len(ref_pop))

        # get the permuted base_inds
        if self._rand_base:
            base_inds = np.random.permutation(len(proposal))
        else:
            base_inds = np.arange(len(proposal))

        # loop generating proposals
        for p in range(len(proposal)):
            # get current gammas
            gamma_best = np.random.uniform(*self._gamma_best)
            gamma = np.random.uniform(*self._gamma)

            # pick best particle probabilistically
            # (works possibly too well)
            if gamma_best > 0.0:
                best_ind = np.nonzero(np.random.rand() < cum_w_sum)[0][0]

            # pick two from ref pop
            ind = random.sample(set(ref_ind), 2)

            # DE_local_to_best
            proposal[p] = (pop[base_inds[p]] +
                           (gamma * (ref_pop[ind[0]] -
                                     ref_pop[ind[1]])) +
                           np.random.randn(pop.shape[1])*self._epsilon)
            if gamma_best > 0.0:
                proposal[p] += (gamma_best *
                                (pop[best_ind] - pop[base_inds[p]]))

        # do crossover
        xold_ind = np.random.rand(*pop.shape) > self._CR
        proposal[xold_ind] = pop[xold_ind]

        return proposal


class Mutate(Proposal):
    """
    """

    def __init__(self, mutate_sd=.1):
        pass

    def generate(self, pop, weights, params):
        pass
