   
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from KDEpy import FFTKDE
from .density import boxcox, best_boxcox_lambdax


class PDA():
    """Probability Density Approximation"""
    def __init__(self, obs, cat_var=None, cont_var=None,
                 lower=None, upper=None, nbins=2048, min_obs=3,
                 transform=None, shift_start=0.0,
                 kernel='epa', bw='silverman'):
        # save the input vars
        self._obs = obs
        self._min_obs = min_obs
        self._cat_var = cat_var
        self._cont_var = cont_var
        self._boxcox = transform
        self._shift_start = shift_start
        self._lower = lower
        self._upper = upper
        self._nbins = nbins

        # learn boxcox transformation if desired
        if cont_var is not None:
            # set up the kde
            self._kde = FFTKDE(kernel=kernel, bw=bw)

            # set the lower and upper if needed
            if self._lower is None:
                self._lower = obs[self._cont_var].min()
            if self._upper is None:
                self._upper = obs[self._cont_var].max()

            if self._boxcox:
                # learn boxcox param over all data
                self._lambdax, self._shift = best_boxcox_lambdax(obs[self._cont_var],
                                                                 lambdax=0.0,
                                                                 shift=shift_start)
                self._lower = boxcox(np.array([self._lower]),
                                     self._lambdax, self._shift)[0]
                self._upper = boxcox(np.array([self._upper]),
                                     self._lambdax, self._shift)[0]

            # set the xvals for evaluating the kde
            self._xvals = np.linspace(self._lower, self._upper, self._nbins)
          
        # get unique categories/conds
        if cat_var is not None:
            self._ucat = np.unique(obs[self._cat_var])
            self._cat_ind = {c: obs[self._cat_var] == c
                             for c in self._ucat}
        else:
            self._ucat = [None]
            self._cat_ind = {None: np.ones(len(obs), dtype=np.bool)}

    def calc_log_like(self, sims):
        # start with zero like
        log_like = 0.0
        nsims = len(sims)

        # loop over observed categorical vars
        for cat in self._ucat:
            # set the starting index
            if cat is not None:
                cat_ind = sims[self._cat_var] == cat
            else:
                cat_ind = np.ones(len(sims), dtype=np.bool)

            # make sure we have enough obs
            if cat_ind.sum() < self._min_obs:
                # not enough obs for this cond, so 0 like
                log_like = -np.inf
                break

            # process continuous var
            if self._cont_var is not None:
                # apply boxcox if requested
                sdat = sims[self._cont_var][cat_ind]
                odat = self._obs[self._cont_var][self._cat_ind[cat]]
                if self._boxcox:
                    sdat = boxcox(sdat, self._lambdax, self._shift)
                    odat = boxcox(odat, self._lambdax, self._shift)

                # calculate the density
                pp = np.interp(odat, self._xvals,
                               self._kde.fit(sdat).evaluate(self._xvals))
                #pp, xx = kdensity(sdat,
                #                  extrema=(0,2.0),
                #                  xx=odat)

                # scale the density by proportion
                pp *= float(len(sdat))/nsims

                # add to the log likes
                log_like += np.log(pp).sum()
            else:
                # just process as proportion of responses
                log_like += np.log(float(cat_ind.sum())/nsims) * \
                    self._cat_ind[cat].sum()

        # return the log-like
        return log_like
