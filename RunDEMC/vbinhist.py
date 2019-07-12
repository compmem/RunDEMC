# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np

class VBinHist():
    "Variable-Bin Histogram"
    def __init__(self, x, min_width=None):
        """
        """
        # save input vars
        self.x = np.asarray(x)
        
        if min_width is None:
            # calc min width based on one bin per value if all lined up
            hw, bw = np.histogram(self.x, bins=len(self.x)*3)
            min_width = (bw[1]-bw[0])
        self.min_width = min_width

        # we haven't calculated, yet, so set values to none
        self.h = None
        self.b = None
        self.c = None
        
     def calculate(self, min_area=0.0, lower=-np.inf, upper=np.inf, apply_bounds=True):
        # remove data outside of the bounds if necessary
        if apply_bounds:
            self.x = self.x[self.x>lower]
            self.x = self.x[self.x<upper]
            
        # make sure not greater than 1.0
        if min_area > 1.0:
            min_area = 1.0
        
        # calc initial bins
        self.h, self.b = np.histogram(self.x, bins=3, density=True)
        
        # calc width and area
        self.w = np.diff(self.b)
        self.a = self.h*self.w

        # iterate and split bin pairs
        _ind_to_skip = []    
        for _i in range(len(self.x)**2):  # figure out right value for here    
            ind = self.find_max_bin_pair(_ind_to_skip=_ind_to_skip)
            if ind is not None:
                # we have an ind, so attempt split
                did_split = self.split_bin_pair(ind, min_area=min_area)

                if not did_split:
                    # did not split, so append to skip list
                    _ind_to_skip.append(ind)
                else:
                    # we split, so adjust skip list
                    # first remove inds adjacent to the ind
                    for r in [ind-1, ind+1]:
                        try:
                            _ind_to_skip.remove(r)
                        except ValueError:
                            pass
                    # adjust the remaining ind to skip
                    _ind_to_skip = [i if i < ind else i+1
                                    for i in _ind_to_skip]
            else:
                # done
                break
        
        # iterate and split individual bins
        _ind_to_skip = []        
        for _i in range(len(self.h)**2):  # figure out right value for here
            ind = self.find_max_bin(_ind_to_skip=_ind_to_skip)
            if ind is not None:
                # we have an ind, so attempt split
                did_split = self.split_bin(ind, min_area=min_area)

                if not did_split:
                    # did not split, so append to skip list
                    _ind_to_skip.append(ind)
                else:
                    # we split, so adjust skip list
                    _ind_to_skip = [i if i < ind else i+1
                                    for i in _ind_to_skip]
            else:
                # done
                break
        # extend bins at the edges
        if apply_bounds:
            # find center of mass of the lowest bin
            lowest_vals = self.x[(self.x>=self.b[0]) & (self.x<self.b[1])]
            com_low = np.mean(lowest_vals)
            
            # save the old bin-width to correct h[0]
            bwo_low = self.b[1] - self.b[0]
            
            # adjust left-most bin edge to consider the distance from the center of mass
            # to the second left-most bin edge
            self.b[0]  -= self.b[1] - com_low
            bwn_low = self.b[1] - self.b[0]
            
            # clip lowest bin edge and adjust bin height if necessary
            if self.b[0] < lower:
                self.h[0] = self.h[0]*bwo_low/(self.b[1]-lower)
                self.b[0] = lower
            else:
                self.h[0] = self.h[0]*bwo_low/bwn_low
                
            # find center of mass of the highest bin
            highest_vals = self.x[(self.x<=self.b[-1]) & (self.x>self.b[-2])]
            com_high = np.mean(highest_vals)
            
            # save the old bin-width to correct h[-1]
            bwo_high = self.b[-1] - self.b[-2]
            
            # adjust right-most bin edge
            self.b[-1] += com_high - self.b[-2]
            bwn_high = self.b[-1] - self.b[-2]
            
            # clip highest bin edge and adjust bin height if necessary
            if self.b[-1] > upper:
                self.h[-1] = self.h[-1]*bwo_high/(upper-self.b[-2])
                self.b[-1] = upper
            else:
                self.h[-1] = self.h[-1]*bwo_high/bwn_high
                
        # we're done, so calc counts and return h and b
        self.c, b = np.histogram(self.x, bins=self.b)
        return self.h, self.b

    def find_max_bin_pair(self, _ind_to_skip=None):
        # process args
        if _ind_to_skip is None:
            _ind_to_skip = []

        # pick the max area (sampled at random if necessary)
        pair_area = np.array([self.a[i]+self.a[i+1] 
                              for i in range(len(self.a)-1)])
        good_ind = np.in1d(np.arange(len(pair_area)), _ind_to_skip, 
                           invert=True, assume_unique=True)
        good_ind &= pair_area > 0.0 
        if ~np.any(good_ind):
            # we're done!
            return None

        # pick the max area
        poss_ind = np.where(pair_area==pair_area[good_ind].max())[0]
        poss_ind = poss_ind[np.in1d(poss_ind, _ind_to_skip, 
                                    assume_unique=True, invert=True)]
        if len(poss_ind) == 0:
            # just return what we have
            return None

        # pick one at random if there's more than one
        ind = np.random.choice(poss_ind)

        return ind

    def split_bin_pair(self, ind, min_area=0.0):
        # get the vals
        if ind+2 == len(self.b)-1:
            # we have to include the right-most edge
            vals = self.x[(self.x>=self.b[ind])&(self.x<=self.b[ind+2])]
        else:
            # we can do what makes sense
            vals = self.x[(self.x>=self.b[ind])&(self.x<self.b[ind+2])]
            
        # calc new hist for that range with 3 bins
        h2, b2 = np.histogram(vals, bins=3, density=True)

        # correct the heights based on the width and original area
        w1 = np.diff(np.concatenate([[self.b[ind]], b2[1:-1], [self.b[ind+2]]]))
        if np.any(w1==0.0):
            # can't have a zero-width bin
            return False
        
        # new widths
        w2 = np.diff(b2)

        # scale the heights to unit size after adjusting widths
        h2 *= w2/w1

        # scale the area to match the area of the original bins
        h2 *= (self.a[ind] + self.a[ind+1])

        # calc new area
        a2 = w1*h2
        
        # test to ensure all over min area
        if np.any((a2<min_area)&(a2>0)) or np.any(w1<self.min_width):
            # we're not splitting
            return False

        # we made it to here, so we're going to keep this split
        self.h = np.concatenate([self.h[:ind], h2, self.h[ind+2:]])
        self.b = np.concatenate([self.b[:ind+1], b2[1:-1], self.b[ind+2:]])
        
        # recalc width and areas
        self.w = np.diff(self.b)
        self.a = self.h*self.w

        return True

    def find_max_bin(self, _ind_to_skip=None):
        # process args
        if _ind_to_skip is None:
            _ind_to_skip = []

        # pick the max area (sampled at random if necessary)
        good_ind = np.in1d(np.arange(len(self.a)), _ind_to_skip, 
                           invert=True, assume_unique=True)
        good_ind &= self.a > 0.0         
        if ~np.any(good_ind):
            # we're done!
            return None

        # pick the max area
        poss_ind = np.where(self.a==self.a[good_ind].max())[0]
        poss_ind = poss_ind[np.in1d(poss_ind, _ind_to_skip, 
                                    assume_unique=True, invert=True)]
        if len(poss_ind) == 0:
            # just return what we have
            return None

        # pick one at random if there's more than one
        ind = np.random.choice(poss_ind)

        return ind

    def split_bin(self, ind, min_area=0.0):
        # get the vals
        if ind+1 == len(self.b)-1:
            # we have to include the right-most edge
            vals = self.x[(self.x>=self.b[ind])&(self.x<=self.b[ind+1])]
        else:
            # we can do what makes sense
            vals = self.x[(self.x>=self.b[ind])&(self.x<self.b[ind+1])]
            
        # calc new hist for that range with 2 bins
        h2, b2 = np.histogram(vals, bins=2, density=True)

        # correct the heights based on the width and original area
        w1 = np.diff(np.concatenate([[self.b[ind]], b2[1:-1], [self.b[ind+1]]]))
        if np.any(w1==0.0):
            # can't have a zero-width bin
            return False

        # new widths
        w2 = np.diff(b2)

        # scale the heights to unit size after adjusting widths
        h2 *= w2/w1

        # scale the area to match the area of the original bins
        h2 *= self.a[ind]

        # calc new area
        a2 = w1*h2
        
        # test to ensure all over min area
        if np.any((a2<min_area)&(a2>0)) or np.any(w1<self.min_width):
            # we're not splitting
            return False

        # we made it to here, so we're going to try and keep this split
        self.h = np.concatenate([self.h[:ind], h2, self.h[ind+1:]])
        self.b = np.concatenate([self.b[:ind+1], b2[1:-1], self.b[ind+1:]])

        
        # recalc width and areas
        self.w = np.diff(self.b)
        self.a = self.h*self.w

        return True
