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
from io import make_dict, gzpickle

# test for scoop
try:
    import scoop
    from scoop import futures
except ImportError:
    scoop = None

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
        """Figures out the HyperPriors and FixedParams from list of submodels.
        """
        self._models = flatten(models)
        self._other_models = []
        self._processed = False
        self._num_chains = num_chains

    def save(self, filename, **kwargs):
        # loop over models adding to a dict of dicts
        all_mod = {}
        for m in self._other_models + self._models:
            all_mod[m._name] = make_dict(m, **kwargs)
        gzpickle(all_mod, filename)

    def _proc_params(self, params):
        # process the params
        for p in params:
            # check whether it's fixed
            if p in self._all_params:
                # we've seen it before, so it's fixed
                if not p in self._fixed_params:
                    # add it to fixed
                    self._fixed_params.append(p)
            else:
                # add it to all
                self._all_params.append(p)

            # check for HyperPrior
            if isinstance(p.prior, HyperPrior) and \
               not p.prior in self._hyper_priors:
                # append to hyper_priors
                self._hyper_priors.append(p.prior)

                # process this prior's params
                self._proc_params(p.prior._params)
        

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
        self._hyper_priors = []
        self._fixed_params = []
        self._all_params = []
        sys.stdout.write('Processing params (%d): '%len(self._models))
        sys.stdout.flush()
        for mi,m in enumerate(self._models):
            sys.stdout.write('%d '%(mi+1))
            sys.stdout.flush()
            # proc this model's params (recusively)
            self._proc_params(m._params)
                    
        sys.stdout.write('\n')

        # make fixed_params if necessary
        if len(self._fixed_params) > 0:
            fparams = [FixedParams('_fixed', self._fixed_params)]
        else:
            fparams = []

        sys.stdout.write('Linking models (%d): '%(len(fparams)+len(self._hyper_priors)))
        sys.stdout.flush()

        # process the HyperPriors
        mi = 0
        for m in self._hyper_priors:
            mi += 1
            sys.stdout.write('%d '%(mi))
            sys.stdout.flush()

            # loop over the models looking for hpriors
            m_args = []
            for n in self._models + self._hyper_priors:
                # make sure not self
                if m == n:
                    # is self, so skip
                    continue
                
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

            # loop over the models looking for fixed params
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
        self._other_models = self._hyper_priors + fparams
            
        # initialize all models
        sys.stdout.write('Initializing (%d): '%len(self._models+self._other_models))
        sys.stdout.flush()
        # make sure to do other_models (fixed and hyper) first
        for mi,m in enumerate(self._other_models + self._models):
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
            
            # loop over each other model (fixed and hyper) doing some Gibbs action
            for m in self._other_models:
                sys.stdout.write('.')
                sys.stdout.flush()
                # run for one iteration
                m(1, burnin=burnin, migration_prob=migration_prob)

            # check if running scoop
            if scoop and scoop.IS_RUNNING:
                # prep each submodel in parallel
                res = []
                for m in self._models:
                    # generate the proposals
                    m._proposal = m._crossover(burnin=burnin)

                    # apply transformation if necessary
                    pop = m.apply_param_transform(m._proposal)

                    # submit the like_fun call in parallel
                    args = [pop] + list(m._like_args)
                    res.append(futures.submit(m._like_fun, *args))

                # collect the results
                for mi,m in enumerate(self._models):
                    # wait for the result
                    #sys.stdout.write('.')
                    out = res[mi].result()
                    if isinstance(out, tuple):
                        # split into likes and posts
                        log_likes,posts = out
                    else:
                        # just likes
                        log_likes = out
                        posts = None
                    m._prop_log_likes = log_likes
                    m._prop_posts = posts
                
            # loop over each model doing some Gibbs action
            # if they have been prepped above this will just do the MH step
            for m in self._models:
                sys.stdout.write('.')
                sys.stdout.flush()
                # run for one iteration
                m(1, burnin=burnin, migration_prob=migration_prob)
            
