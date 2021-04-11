# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from .dists import invlogit, logit


class Param(object):
    """
    Parameter for use with RunDEMC.
    """

    def __init__(self, name, prior=None, init_prior=None,
                 display_name=None, transform=None, inv_transform=None):
        self.name = name
        self.prior = prior

        if init_prior is None:
            init_prior = self.prior
        self.init_prior = init_prior

        if display_name is None:
            display_name = self.name
        self.display_name = display_name

        self.transform = transform

        # see if can infer inv transform if necessary
        if transform is not None and inv_transform is None:
            if transform == invlogit:
                # we know we can do logit
                inv_transform = logit
            elif transform == np.exp:
                # it's log
                inv_transform = np.log            
        self.inv_transform = inv_transform

        # hidden variable to indicate whether this param is fixed at
        # this level
        self._fixed = False
        self._fixed_info = None

def _apply_param_transform(pop, transforms):
    pop = pop.copy()
    for i, transform in enumerate(transforms):
        if transform:
            # apply the transform
            pop[..., i] = transform(pop[..., i])
    return pop



