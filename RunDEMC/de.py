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

    def generate(self, pop, weights, params=None):
        raise NotImplemented("You must define this method in a subclass.")

    def __call__(self, pop, weights, params=None):
        return self.generate(pop, weights, params)


class DE(Proposal):
    """
    Differential evolution proposal.
    """

    def __init__(self, CR=1.0, gamma=(0.5, 1.0),
                 gamma_best=(0.0, 0.0), epsilon=.001,
                 rand_base=False):
        self._CR = CR
        if np.isscalar(gamma):
            # turn into tuple range
            gamma = (gamma, gamma)
        self._gamma = gamma
        if gamma_best is None:
            gamma_best = gamma
        elif np.isscalar(gamma_best):
            # turn into tuple range
            gamma_best = (gamma_best, gamma_best)
        self._gamma_best = gamma_best
        self._rand_base = rand_base
        self._epsilon = epsilon

    def generate(self, pop, weights, params=None):
        """
        Generate a standard differential evolution proposal.
        """
        # allocate for new proposal
        proposal = np.ones_like(pop) * np.nan

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
        p_w = tweights / w_sum
        cum_w_sum = np.cumsum(p_w)

        # indices for all the particles
        all_ind = np.arange(len(proposal))

        # see how many fixed params there are
        fixed = np.zeros(proposal.shape[1], dtype=np.bool)
        if params:
            for i, param in enumerate(params):
                if param._fixed or not hasattr(param.prior, "pdf"):
                    fixed[i] = True
        #non_fixed = np.where(~fixed)[0]

        # get the permuted base_inds
        base_inds = np.random.permutation(len(proposal))

        # loop generating proposals
        for p in range(len(proposal)):
            while np.any(np.isnan(proposal[p])):  # or \
                  # np.any([params[i].prior.pdf(proposal[p][i])==0.0
                  #        for i in non_fixed]):
                # generate proposal
                # get current gammas
                gamma_best = np.random.uniform(*self._gamma_best)
                gamma = np.random.uniform(*self._gamma)

                # define which particles to avoid when sampling
                to_avoid = [p]

                # pick best particle probabilistically
                # (works possibly too well)
                if gamma_best > 0.0:
                    best_ind = np.nonzero(np.random.rand() < cum_w_sum)[0][0]
                    to_avoid.append(best_ind)

                # decide the base ind (rand or local)
                if self._rand_base:
                    base_ind = base_inds[p]
                    to_avoid.append(base_ind)
                else:
                    # do local
                    base_ind = p

                # pick two more that are not p or best_ind
                poss_ind = np.ones(len(proposal), dtype=np.bool)
                poss_ind[to_avoid] = False
                ind = random.sample(set(all_ind[poss_ind]), 2)

                # copy the fixed params (might be none)
                proposal[p, fixed] = pop[p, fixed]

                # DE_local_to_best (to not-fixed params)
                proposal[p, ~fixed] = (pop[base_ind] +
                                       (gamma *
                                        (pop[ind[0]] -
                                         pop[ind[1]])))[~fixed]
                if gamma_best > 0.0:
                    proposal[p, ~fixed] += (gamma_best *
                                            (pop[best_ind] -
                                             pop[base_ind]))[~fixed]

                # add in epsilon, but only to nonfixed
                proposal[p, ~fixed] += np.random.uniform(-self._epsilon,
                                                         self._epsilon,
                                                         size=(~fixed).sum())

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
