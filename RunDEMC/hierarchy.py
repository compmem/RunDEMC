# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


import sys
import numpy as np
from fastprogress.fastprogress import progress_bar

from .demc import Model, HyperPrior, FixedParams
from .io import make_dict, gzpickle

# test for joblib
try:
    import joblib
    from joblib import Parallel, delayed
except ImportError:
    joblib = None


def _flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if type(item) == type([]):
            new_lis.extend(_flatten(item))
        else:
            new_lis.append(item)
    return new_lis


class Hierarchy(object):
    """Collection of Model instances.
    """
    model_names = property(lambda self:
                           [m._name for m in self._models],
                           doc="""
                           List of parameter names.
                           """)

    hyper_names = property(lambda self:
                           [h._name for h in self._other_models
                            if isinstance(h, HyperPrior)],
                           doc="""
                           List of parameter names.
                           """)


    def __init__(self, models, num_chains=None, partition=None,
                 delay_hyper_burnin=False):
        """Figures out the HyperPriors and FixedParams from list of submodels.
        """
        self._models = _flatten(models)
        self._other_models = []
        self._processed = False
        self._num_chains = num_chains
        self._partition = partition
        self._parallel = None #parallel
        self._delay_hyper_burnin = delay_hyper_burnin

    def save(self, filename, **kwargs):
        # loop over models adding to a dict of dicts
        all_mod = {}
        for m in self._other_models + self._models:
            all_mod[m._name] = make_dict(m, **kwargs)
        gzpickle(all_mod, filename)

    def _proc_params(self, params, level=0):
        # make sure to set level
        if not len(self._fixed_params) == (level + 1):
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
                self._proc_params(p.prior._params, level=level + 1)

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
            sys.stdout.write('%d ' % (mi + 1))
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
        flattened_hps = _flatten(self._hyper_priors)
        sys.stdout.write('Linking models (%d): ' % (len(_flatten(fparams)) +
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
        progress = progress_bar(self._other_models + self._models)
        for mi, m in enumerate(progress):
            #sys.stdout.write('%d ' % (mi + 1))
            #sys.stdout.flush()
            # see if must process, must be initialized all with same
            # num_chains
            m._initialize(num_chains=max_num_chains, partition=self._partition)
        #sys.stdout.write('\n')

        # we're done processing
        self._processed = True

    def __getitem__(self, name):
        try:
            ind = self.model_names.index(name)
            return self._models[ind]
        except ValueError:
            try:
                ind = self.hyper_names.index(name)
                hmods = [h for h in self._other_models
                         if isinstance(h, HyperPrior)]
                return hmods[ind]
            except ValueError:
                raise ValueError("Model "+str(name)+" not found.")

    def __call__(self, num_iter=1, burnin=False, migration_prob=0.0,
                 reinit_hypers=False):
        self.sample(num_iter=num_iter, burnin=burnin,
                    migration_prob=migration_prob,
                    reinit_hypers=reinit_hypers)

    def sample(self, num_iter=1, burnin=False, migration_prob=0.0,
               reinit_hypers=False):
        if not self._processed:
            self._process()
        
        # put in a try/finally to potentially clean up parallel
        try:
            # see if launch pools
            if self._parallel:
                # do what <with> enter would do
                #self._parallel._managed_pool = True
                # self._parallel._initialize_pool()
                self._parallel.__enter__()

            sys.stdout.write('Iterations (%d): ' % (num_iter))

            # loop over iterations
            progress = progress_bar(range(num_iter))
            for i in progress:
                #sys.stdout.write('%d' % (i + 1))
                #sys.stdout.flush()

                # loop over each other model (fixed and hyper) doing Gibbs
                for m in self._other_models:
                    # run for one iteration
                    #sys.stdout.write('.')
                    #sys.stdout.flush()
                    if self._delay_hyper_burnin and \
                       isinstance(m, HyperPrior) and burnin:
                        # skip hyper during burnin
                        # set proposal to last values
                        #m._proposal = m._particles[-1]
                        continue
                    m.sample(1, burnin=burnin, migration_prob=migration_prob)

                # check if running joblib
                if self._parallel:
                    jobs = []
                    for m in self._models:
                        jobs.append(delayed(m.sample)(1, burnin=burnin,
                                                      migration_prob=migration_prob))
                    if len(jobs) > 0:
                        # submit the joblib jobs
                        res = self._parallel(jobs)

                else:
                    # just run without parallel
                    for m in self._models:
                        m.sample(1, burnin=burnin, migration_prob=migration_prob)

            # if was burnin, then go back and run Hypers
            if self._delay_hyper_burnin and burnin:
                if reinit_hypers:
                    for m in self._other_models:
                        if isinstance(m, HyperPrior):
                            # set init_priors to priors
                            for p in m._params:
                                p.init_prior = p.prior
                            m._initialize(force=True)
                    
                for i in range(num_iter):
                    for m in self._other_models:
                        if isinstance(m, HyperPrior):
                            # sample it
                            m.sample(1, burnin=False,
                                     migration_prob=migration_prob)

        finally:
            # see if clean pools
            if self._parallel:
                # do what <with> exit would do
                # self._parallel._terminate_pool()
                #self._parallel._managed_pool = False
                self._parallel.__exit__(None, None, None)
