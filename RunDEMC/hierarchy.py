#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


import sys
import numpy as np

from demc import Model, HyperPrior, FixedParams

def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if type(item) == type([]):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis

class Hierarchy(object):
    """Collection of Model instances.
    """
    def __init__(self, models, num_chains=None):
        """Make sure to put HyperPriors and FixedParams before the Models
        that use them.
        """
        self._models = flatten(models)
        self._processed = False
        self._num_chains = num_chains

    def _process(self):
        """
        Process the submodels and set up connections between them.
        """
        # make sure all have same number of particles
        if self._num_chains:
            max_num_chains = self._num_chains
        else:
            max_num_chains = np.max([m._num_chains for m in self._models])
        sys.stdout.write('Max Group Size: %d\n'%max_num_chains)
        sys.stdout.flush()        
        # loop over models, seeing if other models use it as a prior
        hyper_priors = []
        fixed_params = []
        all_params = []
        sys.stdout.write('Processing params (%d): '%len(self._models))
        sys.stdout.flush()
        for mi,m in enumerate(self._models):
            sys.stdout.write('%d '%(mi+1))
            sys.stdout.flush()

            # process the params
            for p in m._params:
                # check whether it's fixed
                if p in all_params:
                    if not p in fixed_params:
                        # add it to fixed
                        fixed_params.append(p)
                else:
                    # add it to all
                    all_params.append(p)

                # check for HyperPrior
                if isinstance(p.prior, HyperPrior) and \
                   not p.prior in hyper_priors:
                    hyper_priors.append(p.prior)
                    
        sys.stdout.write('\n')

        # make fixed_params if necessary
        if len(fixed_params) > 0:
            fparams = [FixedParams(fixed_params)]
        else:
            fparams = []

        sys.stdout.write('Linking models (%d): '%(len(fparams)+len(hyper_priors)))
        sys.stdout.flush()

        # process the HyperPriors
        mi = 0
        for m in hyper_priors:
            mi += 1
            sys.stdout.write('%d '%(mi))
            sys.stdout.flush()

            # loop over the models looking for hpriors
            m_args = []
            for n in self._models:
                # loop over that model's params
                for i,p in enumerate(n._params):
                    # see if used as hyper prior
                    if m == p.prior:
                        # we have a match, so add the vals
                        m_args.append({'model':n, 'param_ind':i})
                        
            # set the args for this model
            m._like_args = tuple(m_args)
            
        # loop over fixed params
        for m in fparams:
            mi += 1
            sys.stdout.write('%d '%(mi))
            sys.stdout.flush()

            # loop over the models looking for hpriors
            m_args = []
            for n in self._models:
                # loop over that model's params
                p_inds = []
                for i,p in enumerate(n._params):
                    # see if used as fixed param
                    if p in m._submodel_params:
                        p_inds.append((i,m._submodel_params.index(p)))

                if len(p_inds) > 0:
                    # we have a match, so add the vals
                    m_args.append({'model':n, 'param_ind':p_inds})
                        
            # set the args for this model
            m._like_args = tuple(m_args)
        sys.stdout.write('\n')

        # prepend the hypers and fixed to model list
        self._models = hyper_priors + fparams + self._models
            
        # initialize all models
        sys.stdout.write('Initializing (%d): '%len(self._models))
        sys.stdout.flush()
        for mi,m in enumerate(self._models):
            sys.stdout.write('%d '%(mi+1))
            sys.stdout.flush()        
            # see if must process, must be initialized all with same
            # num_chains
            m._initialize(num_chains=max_num_chains)
        sys.stdout.write('\n')

        # we're done processing
        self._processed = True


    def __call__(self, num_iter=1, burnin=False,
                 migration_prob=0.0): #, reprocess=False):
        # make sure we've processed
        #if reprocess or not self._processed:
        if not self._processed:
            self._process()

        sys.stdout.write('Iterations (%d): '%(num_iter))
        
        # loop over iterations
        for i in xrange(num_iter):
            sys.stdout.write('%d'%(i+1))
            sys.stdout.flush()
            # loop over each model doing some Gibbs action
            for m in self._models:
                sys.stdout.write('.')
                sys.stdout.flush()
                # run for one iteration
                m(1, burnin=burnin, migration_prob=migration_prob)
            
