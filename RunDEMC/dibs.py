    
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from scipy.special import polygamma

from .vbinhist import VBinHist


class DIBS():
    """Data-Driven Discretized Inverse Binomial Sampling"""
    def __init__(self, obs, cat_var=None, cont_var=None,
                 lower=-np.inf, upper=np.inf,
                 adjust_edges=True, range_by_cat=False):
        # save the input vars
        self._obs = obs
        self._cat_var = cat_var
        self._cont_var = cont_var

        # see whether to pick cont range within cat or overall data
        self._range_by_cat = range_by_cat
        
        # get unique categories/conds
        if cat_var is not None:
            self._ucat = np.unique(obs[self._cat_var])

            # calculate proportions for each cat
            self._pcat = {c: (obs[self._cat_var]==c).sum()/len(obs)
                          for c in self._ucat}

            # calculate counts for each cat
            self._ccat = {c: (obs[self._cat_var]==c).sum()
                          for c in self._ucat}
        else:
            self._ucat = [None]
            self._pcat = {None: 1.0}
            self._ccat = {None: len(obs)}

        # create vbin for each cat
        if cont_var is not None:
            if range_by_cat:
                # just pass none so that each cat has a custom range
                # based on the data
                hist_range = None
            else:
                # determine range over entire dataset
                hist_range = (obs[self._cont_var].min(),
                              obs[self._cont_var].max())
            self._vbh = {c: VBinHist(obs[obs[self._cat_var]==c][self._cont_var],
                                     lower=lower, upper=upper,
                                     adjust_edges=adjust_edges,
                                     hist_range=hist_range)
                         for c in self._ucat}
        else:
            self._vbh = {c: None for c in self._ucat}

        # variable to save last counts for each bin
        self.last_counts = None

    def update_bins(self, min_area=0.0, adjust=True):
        # update vbh for each cat
        for c in self._vbh:
            if self._vbh[c] is not None:
                # update the bins
                if adjust:
                    adj_min_area = min(min_area/self._pcat[c], 1.0)
                else:
                    adj_min_area = min_area
                self._vbh[c].calculate(min_area=adj_min_area)

    def calc_log_like(self, sims):
        # start with zero like
        log_like = 0.0
        self.last_counts = []
        
        # loop over observed categorical vars
        for cat in self._ucat:
            # set the starting index
            if cat is not None:
                cat_ind = sims[self._cat_var]==cat
            else:
                cat_ind = np.ones(len(sims), dtype=np.bool)
                
            # grab the related vbh
            vbh = self._vbh[cat]

            # see if it exists
            if vbh is None:
                # use observed counts for cat
                counts = [self._ccat[cat]]
            else:
                # use the vbinhist for counts
                counts = vbh.c
                
            # loop over bins
            for i,c in enumerate(counts):
                if c == 0:
                    # if there are no obs in that bin, skip it
                    continue
                elif vbh is None:
                    # if none, just process with existing cat_ind
                    inds = np.where(cat_ind)[0]
                else:
                    # add in the matches based on bin
                    vals = sims[self._cont_var]
                    if i+1 == len(vbh.b)-1:
                        # must include the rightmost edge
                        bin_ind = (vals>=vbh.b[i]) & (vals<=vbh.b[i+1])
                    else:
                        bin_ind = (vals>=vbh.b[i]) & (vals<vbh.b[i+1])

                    # find where we have matches for this bin
                    inds = np.where(cat_ind & bin_ind)[0]

                # if there are no matches, then we have zero like
                if len(inds) == 0:
                    log_like += -np.inf
                    break

                # pull the first k
                k = np.array([inds[0]+1])

                # test for more k
                if len(inds) > 1:
                    # remaining k can be determined via diff
                    k = np.concatenate([k, np.diff(inds)])

                # convert all k to ibs values
                ibs = polygamma(0,1)-polygamma(0,k)

                # calculate sigma correction
                if len(ibs)>1:
                    sigma = ibs.std()/np.sqrt(len(ibs))
                else:
                    sigma = 0.0

                # save the last_counts
                self.last_counts.append(len(ibs))

                # combine corrected IBS value with count of total observations
                log_like += (ibs.mean()-(sigma**2)/2)*c

        # return the log-like
        return log_like
