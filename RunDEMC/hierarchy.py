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

from .demc import Model, HyperPrior, FixedParams
from .io import make_dict, gzpickle

# test for scoop
try:
    import scoop
    from scoop import futures
except ImportError:
    scoop = None

# test for joblib
try:
    import joblib
    from joblib import Parallel, delayed
except ImportError:
    joblib = None


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
    def __init__(self, models, num_chains=None, partition=None, parallel=None):
        """Figures out the HyperPriors and FixedParams from list of submodels.
        """
        self._models = flatten(models)
        self._other_models = []
        self._processed = False
        self._num_chains = num_chains
        self._partition = partition
        self._parallel = parallel

    def save(self, filename, **kwargs):
        # loop over models adding to a dict of dicts
        all_mod = {}
        for m in self._other_models + self._models:
            all_mod[m._name] = make_dict(m, **kwargs)
        gzpickle(all_mod, filename)

    def _proc_params(self, params, level=0):
        # make sure to set level
        if not len(self._fixed_params) == (level+1):
            self._fixed_params.append([])
            self._hyper_priors.append([])

        # process the params
        for p in params:
            # check whether it's fixed
            if p in self._all_params:
                # we've seen it before, so it's fixed
                if p not in self._fixed_params[level]:
                    # add it to fixed
                    self._fixed_params[level].append(p)
            else:
                # add it to all
                self._all_params.append(p)

            # check for HyperPrior
            if isinstance(p.prior, HyperPrior) and \
               p.prior not in self._hyper_priors[level]:
                # append to hyper_priors
                self._hyper_priors[level].append(p.prior)

                # process this prior's params
                self._proc_params(p.prior._params, level=level+1)

    def _process(self):
        """
        Process the submodels and set up connections between them.
        """
        # make sure all have same number of particles
        if self._num_chains:
            max_num_chains = self._num_chains
        else:
            max_num_chains = np.max([m._num_chains for m in self._models])
        sys.stdout.write('Max Group Size: %d\n' % max_num_chains)
        sys.stdout.flush()
        # loop over models, seeing if other models use it as a prior
        self._hyper_priors = []
        self._fixed_params = []
        self._all_params = []
        sys.stdout.write('Processing params (%d): ' % len(self._models))
        sys.stdout.flush()
        for mi, m in enumerate(self._models):
            sys.stdout.write('%d ' % (mi+1))
            sys.stdout.flush()
            # proc this model's params (recusively)
            self._proc_params(m._params, level=0)

        sys.stdout.write('\n')

        # make fixed_params if necessary
        fparams = []
        for level in range(len(self._fixed_params)):
            if len(self._fixed_params[level]) > 0:
                fparams.append(FixedParams('_fixed_%d' % level,
                                           self._fixed_params[level],
                                           parallel=self._parallel))
            else:
                fparams.append([])

        # flatten the hps
        flattened_hps = flatten(self._hyper_priors)
        sys.stdout.write('Linking models (%d): ' % (len(flatten(fparams)) +
                                                    len(flattened_hps)))
        sys.stdout.flush()

        # process the HyperPriors
        mi = 0
        for m in flattened_hps:
            mi += 1
            sys.stdout.write('%d ' % (mi))
            sys.stdout.flush()

            # loop over the models looking for hpriors
            m_args = []
            for n in self._models + flattened_hps:
                # make sure not self
                if m == n:
                    # is self, so skip
                    continue

                # loop over that model's params
                for i, p in enumerate(n._params):
                    # see if used as hyper prior
                    if m == p.prior:
                        # we have a match, so add the vals
                        m_args.append({'model': n, 'param_ind': i})

            # set the args for this model
            m._like_args = tuple(m_args)

        # loop over fixed params
        for m in fparams:
            if m == []:
                continue
            mi += 1
            sys.stdout.write('%d ' % (mi))
            sys.stdout.flush()

            # loop over the models looking for fixed params
            m_args = []
            for n in self._models + flattened_hps:
                # loop over that model's params
                p_inds = []
                for i, p in enumerate(n._params):
                    # see if used as fixed param
                    if p in m._submodel_params:
                        p_inds.append((i, m._submodel_params.index(p)))

                if len(p_inds) > 0:
                    # we have a match, so add the vals
                    m_args.append({'model': n, 'param_ind': p_inds})

            # set the args for this model
            m._like_args = tuple(m_args)
        sys.stdout.write('\n')

        # prepend the hypers and fixed to model list
        self._other_models = []
        for level in range(len(self._hyper_priors)):
            to_add = []
            if not self._hyper_priors[level] == []:
                to_add.extend(self._hyper_priors[level])
            if not fparams[level] == []:
                to_add.append(fparams[level])
            self._other_models.extend(to_add)
        # reverse the order so the higher-level priors and params init first
        self._other_models.reverse()

        # initialize all models
        sys.stdout.write('Initializing (%d): ' % len(self._models +
                                                     self._other_models))
        sys.stdout.flush()
        # make sure to do other_models (fixed and hyper) first
        for mi, m in enumerate(self._other_models + self._models):
            sys.stdout.write('%d ' % (mi+1))
            sys.stdout.flush()
            # see if must process, must be initialized all with same
            # num_chains
            m._initialize(num_chains=max_num_chains, partition=self._partition)
        sys.stdout.write('\n')

        # we're done processing
        self._processed = True

    def __call__(self, num_iter=1, burnin=False, migration_prob=0.0):
        if not self._processed:
            self._process()

        # put in a try/finally to potentially clean up parallel
        try:
            # see if launch pools
            if not (scoop and scoop.IS_RUNNING) and self._parallel:
                # do what <with> enter would do
                #self._parallel._managed_pool = True
                #self._parallel._initialize_pool()
                self._parallel.__enter__()

            sys.stdout.write('Iterations (%d): ' % (num_iter))

            # loop over iterations
            for i in range(num_iter):
                sys.stdout.write('%d' % (i+1))
                sys.stdout.flush()

                # loop over each other model (fixed and hyper) doing Gibbs
                for m in self._other_models:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    # run for one iteration
                    m(1, burnin=burnin, migration_prob=migration_prob)

                # check if running scoop or joblib
                if (scoop and scoop.IS_RUNNING) or self._parallel:
                    # prep each submodel in parallel
                    res = []
                    jobs = []
                    for m in self._models:
                        # generate the proposals
                        m._proposal = m._crossover(burnin=burnin)

                        # apply transformation if necessary
                        pop = m.apply_param_transform(m._proposal)

                        # submit the like_fun call in parallel
                        if (scoop and scoop.IS_RUNNING):
                            args = [pop] + list(m._like_args)
                            res.append(futures.submit(m._like_fun, *args))
                        else:
                            # do joblib parallel
                            jobs.append(delayed(m._like_fun)(pop,
                                                             *m._like_args))

                    if len(jobs) > 0 and not (scoop and scoop.IS_RUNNING):
                        # submit the joblib jobs
                        res = self._parallel(jobs)

                    # collect the results
                    for mi, m in enumerate(self._models):
                        # wait for the result
                        if (scoop and scoop.IS_RUNNING):
                            # pull results from scoop
                            out = res[mi].result()
                        else:
                            # pull results from joblib
                            out = res[mi]
                        if isinstance(out, tuple):
                            # split into likes and posts
                            log_likes, posts = out
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

        finally:
            # see if clean pools
            if not (scoop and scoop.IS_RUNNING) and self._parallel:
                # do what <with> exit would do
                #self._parallel._terminate_pool()
                #self._parallel._managed_pool = False
                self._parallel.__exit__(None, None, None)
