   
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
    """Probability Density Approximation

    This class helps manage performing probability density
    approximation (PDA) from Turner and Sederberg (2014, PB&R). The
    basic approach is that you initialize the class with your observed
    data and then you pass in simulations in the same format as your
    observed data to perform PDA and calculate your likelihood.

    There is a good bit of flexibility in the types of PDA you can
    perform. Specifically, you can have any combo of conditions,
    categorical variables, and continuous variables. For continuous
    variables, the method will perform a kernel density estimation.

    Parameters
    ----------
    obs : recarray
        Record array containing your observed data.

    cat_var : str or None
        Column name containing the categorical variable.

    cont_var : str or None
        Column name containing the continuous variable.

    cond_var : str or None
        Column name containing the condition variable.

    nbins : int or None
        Number of bins in the kernel density estimation.
        Powers of 2 are good here and large numbers reduce interpolation
        errors. None will let FFTKDE decide.

    min_obs : int
        Minimum number of matching sims required per set of observations.

    transform : bool
        Whether to transform both the observed and simulated continuous
        data via a Box--Cox transformation before applying the kernel
        density estimation. This can help improve KDE performance with
        skewed distributions, such as reaction times.

    shift_start : float or None
        Whether to include a shift param when optimizing the shift and
        lambda parameters for the Box--Cox

    min_shift : float or None
        Whether to include a minimum shift value when estimating the shift
        of the Box--Cox transform.

    kernel : str
        Type of kernel to use for the kernel density estimation of continuous
        data. See options of FFTKDE from KDEpy.
        Default is 'gaussian' for a Gaussian kernel.

    bw : str
        Method fo rcalculating the bandwidth for the kernel density estimation.
        See options for the FFTKDE from KDEpy.
        Default is 'silverman' for Silverman's Rule of Thumb, but it's worth
        trying 'ISJ' for Improved Sheather Jones for multimodal data.

    """
    def __init__(self, obs, cat_var=None, cont_var=None, cond_var=None,
                 nbins=None, min_obs=3,
                 transform=None, shift_start=0.0, min_shift=None,
                 kernel='gaussian', bw='silverman'):
        # save the input vars
        self._obs = obs
        self._min_obs = min_obs
        self._cat_var = cat_var
        self._cont_var = cont_var
        self._cond_var = cond_var
        self._boxcox = transform
        self._shift_start = shift_start
        self._min_shift = min_shift
        self._nbins = nbins
        self._kernel = kernel
        self._bw = bw

        # learn boxcox transformation if desired
        if cont_var is not None:
            if self._boxcox:
                # learn boxcox param over all data
                if shift_start is None:
                    self._lambdax = best_boxcox_lambdax(obs[self._cont_var],
                                                        lambdax=0.0,
                                                        shift=shift_start)
                    self._shift = 0.0
                else:
                    # find optimal shift, too
                    self._lambdax, self._shift = best_boxcox_lambdax(obs[self._cont_var],
                                                                     lambdax=0.0,
                                                                     shift=shift_start,
                                                                     min_shift=min_shift)

        # get unique categories
        if cat_var is not None:
            self._ucat = np.unique(obs[self._cat_var])
            self._cat_ind = {c: obs[self._cat_var] == c
                             for c in self._ucat}
        else:
            self._ucat = [None]
            self._cat_ind = {None: np.ones(len(obs), dtype=np.bool)}

    def calc_log_like(self, sims):
        """Calculate log likelihood based on simulations."""
        # start with zero like
        log_like = 0.0

        # process each condition separately
        if self._cond_var is None:
            # no conds
            ucond = [None]
        else:
            # get the unique conds from the sims
            ucond = np.unique(sims[self._cond_var])

        # loop over each cond passed in (could be none)
        for cond in ucond:
            # get the condition index
            if cond is not None:
                cond_ind = sims[self._cond_var] == cond
                o_cond_ind = self._obs[self._cond_var] == cond
            else:
                cond_ind = np.ones(len(sims), dtype=np.bool)
                o_cond_ind = np.ones(len(self._obs), dtype=np.bool)

            # get the nsims for this cond
            nsims = cond_ind.sum()

            # loop over observed categorical vars
            for cat in self._ucat:
                # get the index for the observed data
                o_cc_ind = self._cat_ind[cat] & o_cond_ind
                if o_cc_ind.sum() == 0:
                    # there are no observations of this cond and cat_ind
                    continue

                # set the starting index
                if cat is not None:
                    cat_ind = sims[self._cat_var] == cat
                else:
                    cat_ind = np.ones(len(sims), dtype=np.bool)

                # combine cat_ind with cond_ind
                cc_ind = cat_ind & cond_ind

                # make sure we have enough obs
                if cc_ind.sum() < self._min_obs:
                    # not enough obs for this cond&cat, so 0 like
                    log_like = -np.inf
                    break

                # process continuous var
                if self._cont_var is not None:
                    # pull the matching sdat and odat
                    odat = self._obs[self._cont_var][o_cc_ind]
                    sdat = sims[self._cont_var][cc_ind]
                    if np.all((np.diff(sdat)==0)):
                        # must have some variability in the cont variable
                        # for the simulated data, or it can't work
                        log_like = -np.inf
                        break

                    # apply boxcox if requested
                    if self._boxcox:
                        sdat = boxcox(sdat, self._lambdax, self._shift)
                        odat = boxcox(odat, self._lambdax, self._shift)

                    # calculate the probability density approx
                    try:
                        # use the autogrid from KDE
                        gps, yval_pdf_e = FFTKDE(kernel=self._kernel,
                                                 bw=self._bw).fit(sdat).evaluate()

                        # will return small values for data not in the grid
                        pp = np.interp(odat, gps, yval_pdf_e)

                        # scale the density by proportion
                        pp *= float(len(sdat))/nsims

                        # add to the log likes
                        log_like += np.log(pp).sum()
                    except ValueError:
                        # likely had an issue with the KDE
                        log_like = -np.inf
                        break
                else:
                    # just process as proportion of responses
                    # log probability of observing that response
                    # multiplied by the number times it was observed
                    log_like += np.log(float(cc_ind.sum())/nsims) * \
                        o_cc_ind.sum()

            # dont bother continuing to loop if we already have zero like
            if log_like == -np.inf:
                break

        # return the log-like
        return log_like
