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
import copy
from fastprogress.fastprogress import progress_bar

# test for joblib
try:
    import joblib
    from joblib import Parallel, delayed
except ImportError:
    joblib = None

# test for dask
try:
    from dask.distributed import get_client
    has_dask = True
except ImportError:
    has_dask = False

# local imports
from .de import DE
from .io import load_results
from .dists import invlogit, logit
from .param import Param, _apply_param_transform
from .gibbs import NormalHyperPrior

class Model(object):
    """
    Model with params and custom likelihood function.

    Differential Evolution Monte Carlo
    ...and so much more...

    """
    param_names = property(lambda self:
                           [p.name for p in self._params],
                           doc="""
                           List of parameter names.
                           """)

    param_display_names = property(lambda self:
                                   [p.display_name for p in self._params],
                                   doc="""
                                   List of parameter display names.
                                   """)

    particles = property(lambda self:
                         _apply_param_transform(
                             np.asarray(self._particles),
                             self._get_transforms()),
                         doc="""
                         Particles as an array.
                         """)

    log_likes = property(lambda self:
                         np.asarray(self._log_likes),
                         doc="""
                         Log likelihoods as an array.
                         """)

    weights = property(lambda self:
                       np.asarray(self._weights),
                       doc="""
                       Weights as an array.
                       """)

    times = property(lambda self:
                     np.asarray(self._times),
                     doc="""
                     Times as an array.
                     """)

    accept_rate = property(lambda self:
                           np.asarray(self._accept_rate),
                           doc="""
                           Acceptance rate as an array.
                           """)

    def __init__(self, name, params, like_fun,
                 like_args=None,
                 num_chains=None,
                 proposal_gen=None,
                 burnin_proposal_gen=None,
                 initial_zeros_ok=False,
                 #init_multiplier=1,
                 init_file=None,
                 use_priors=True,
                 verbose=False,
                 partition=None,
                 parallel=None,
                 purify_every=0,
                 rand_split=True,
                 pop_recarray=True,
                 pop_parallel=False,
                 n_jobs=-1, backend=None,
                 use_dask=False):
        """
        DEMC
        """
        # set the vars
        self._name = name
        self._params = params  # can't be None
        if num_chains is None:
            num_chains = int(np.min([len(params) * 10, 100]))
        self._num_chains = num_chains
        self._initial_zeros_ok = initial_zeros_ok
        self._init_multiplier = 1 # init_multiplier
        self._init_file = init_file
        self._use_priors = use_priors
        self._verbose = verbose

        # process the partition
        if partition is None:
            partition = len(self._params)
        if partition > len(self._params):
            partition = len(self._params)
        self._partition = partition
        self._parts = np.array([1] * self._partition +
                               [0] * (len(self._params) - self._partition),
                               dtype=np.bool)
        
        self._parallel = parallel
        self._purify_every = purify_every
        self._next_purify = -1 + purify_every
        self._rand_split = rand_split

        # set up proposal generator
        if proposal_gen is None:
            # set gamma based on original DEMC paper
            gamma = 2.38 / np.sqrt(2*len(self._params))
            proposal_gen = DE(gamma=gamma,
                              gamma_best=0.0,
                              rand_base=False)
        self._prop_gen = proposal_gen
        if burnin_proposal_gen is None:
            gamma = 2.38 / np.sqrt(2*len(self._params))
            burnin_proposal_gen = DE(gamma=gamma,
                                     gamma_best=(.4, 1.0),
                                     rand_base=True)
        self._burnin_prop_gen = burnin_proposal_gen

        # set up the like function
        # process the passed in like_fun
        self._like_fun = like_fun
        self._like_args = like_args
        if self._like_args is None:
            self._like_args = ()

        # in some cases we will need to recalc the likes, but the user
        # doesn't need to know about this
        self._recalc_likes = False

        # should pop be passed as a recarray
        self._pop_recarray = pop_recarray

        # pop should be processed in parallel
        self._pop_parallel = pop_parallel

        # use dask?
        self._use_dask = use_dask

        # make a parallel instance if required
        if pop_parallel and self._parallel is None \
           and joblib is not None and (use_dask==False):
            self._parallel = Parallel(n_jobs=n_jobs, backend=backend)

        # see if we're a fixed or hyper
        if isinstance(self, FixedParams) or \
           isinstance(self, HyperPrior) or \
           isinstance(self, NormalHyperPrior):
            self._is_fixed_or_hyper = True
        else:
            self._is_fixed_or_hyper = False

        # we have not initialized
        self._initialized = False
        self._needs_new_generation = True

    def _initialize(self, num_chains=None, force=False):
        if self._initialized and not force:
            # already done
            return

        if self._verbose:
            sys.stdout.write('Initializing: ')
            sys.stdout.flush()

        # initialize starting point, see if from saved file
        if self._init_file:
            # we're loading from file
            # load the file
            m = load_results(self._init_file)

            # fill the variables
            self._num_params = len(m['param_names'])
            self._num_chains = m['particles'].shape[1]

            # pull the particles and apply inverse transform as needed
            particles = _apply_param_transform(m['particles'],
                                               self._get_transforms(inverse=True))

            # must turn everything into lists
            self._particles = [particles[i]
                               for i in range(len(particles))]
            self._log_likes = [m['log_likes'][i]
                               for i in range(len(particles))]
            self._weights = [m['weights'][i]
                             for i in range(len(particles))]
            self._times = [m['times'][i]
                           for i in range(len(particles))]
            if 'accept_rate' in m:
                self._accept_rate = [m['accept_rate'][i]
                                     for i in range(len(particles))]
            else:
                self._accept_rate = []

        else:
            if num_chains is not None:
                self._num_chains = num_chains
            res = _init_chains(_trimmed_init_priors(self._params),
                               self._num_chains,
                               initial_zeros_ok=self._initial_zeros_ok,
                               verbose=self._verbose,
                               like_fun=self._like_fun,
                               like_args=self._like_args,
                               is_fixed_or_hyper=self._is_fixed_or_hyper,
                               pop_recarray=self._pop_recarray,
                               transforms=self._get_transforms(),
                               param_names=self.param_names,
                               use_priors=self._use_priors,
                               parallel=self._parallel,
                               use_dask=self._use_dask)

            # save the return values
            self._particles = [res['particles']]
            self._log_likes = [res['log_likes']]
            self._weights = [res['weights']]
            self._times = [res['times']]
            self._accept_rate = [res['accept_rate']]
            
        # say we've initialized
        self._initialized = True


    def call_calc_log_likes(self, pop, cur_split=None):
        # use this to clean up code later
        return _calc_log_likes(pop,
                               self._like_fun,
                               self._like_args,
                               cur_split=None,
                               is_fixed_or_hyper=self._is_fixed_or_hyper,
                               pop_recarray=self._pop_recarray,
                               transforms=self._get_transforms(),
                               param_names=self.param_names,
                               parallel=self._parallel,
                               use_dask=self._use_dask)


    def _get_part_ind(self):
        # grab the current partition indices
        parts = self._parts.copy()

        # roll them the size of the partition
        self._parts = np.roll(self._parts, self._partition)

        # return the pre-rolled value
        return parts

    def _get_split_ind(self):
        # set all ind
        num_particles = len(self._particles[-1])
        
        if self._rand_split:
            # randomize the indices
            all_ind = np.random.permutation(num_particles)
        else:
            all_ind = np.arange(num_particles)

        # split in half
        split_ind = np.zeros(num_particles, dtype=np.bool)
        split_ind[all_ind[:int(num_particles/2)]] = True
                  
        return split_ind

    def _migrate(self):
        # pick which items will migrate
        num_to_migrate = np.random.random_integers(2, self._num_chains)
        to_migrate = random.sample(
            list(range(self._num_chains)), num_to_migrate)

        # do a circle swap
        keepers = []
        for f_ind in range(len(to_migrate)):
            if f_ind == len(to_migrate) - 1:
                # loop to beg
                t_ind = 0
            else:
                # loop to next
                t_ind = f_ind + 1

            # set the from and to inds
            i = to_migrate[f_ind]
            j = to_migrate[t_ind]

            # do comparison and swap if necessary
            log_diff = np.float64(self._weights[-1][i] -
                                  self._weights[-1][j])

            # now exp so we can get the other probs
            if log_diff > 0.0:
                log_diff = 0.0
            mh_prob = np.exp(log_diff)
            if np.isnan(mh_prob):
                mh_prob = 0.0
            keep = (mh_prob - np.random.rand()) > 0.0
            if keep:
                keepers.append({'ind': j,
                                'particle': self._particles[-1][i],
                                'weight': self._weights[-1][i],
                                'log_like': self._log_likes[-1][i]})

        for k in keepers:
            # do the swap (i.e., replace j with i)
            # replace the particle, weight, log_like
            self._particles[-1][k['ind']] = k['particle']
            self._weights[-1][k['ind']] = k['weight']
            self._log_likes[-1][k['ind']] = k['log_like']

    def _post_evolve(self, pop, cur_split, kept):
        pass

    def __call__(self, num_iter, burnin=False, migration_prob=0.0):
        self.sample(num_iter, burnin=burnin, migration_prob=migration_prob)

    def sample(self, num_iter, burnin=False, migration_prob=0.0):
        """Sample model with DEMC for specified number of iterations."""

        # make sure we've initialized
        self._initialize()

        # loop over iterations
        times = []
        if self._verbose:
            sys.stdout.write('Iterations (%d):\n' % (num_iter))
            progress = progress_bar(range(num_iter))
        else:
            progress = range(num_iter)
        for i in progress:
            if np.random.rand() < migration_prob:
                # migrate, which is deterministic and done in place
                # if self._verbose:
                #     sys.stdout.write('x ')
                self._migrate()

            if self._next_purify == 0:
                # it's time to purify the weights
                # if self._verbose:
                #     sys.stdout.write('= ')
                pure_log_likes = _calc_log_likes(self._particles[-1],
                                                 self._like_fun,
                                                 self._like_args,
                                                 cur_split=None,
                                                 is_fixed_or_hyper=self._is_fixed_or_hyper,
                                                 pop_recarray=self._pop_recarray,
                                                 transforms=self._get_transforms(),
                                                 param_names=self.param_names,
                                                 parallel=self._parallel,
                                                 use_dask=self._use_dask)

                # only keep nonzero
                keep = ~np.isinf(pure_log_likes)
                self._weights[-1][keep] = self._weights[-1][keep] - \
                                          self._log_likes[-1][keep] + \
                                          pure_log_likes[keep]
                self._log_likes[-1][keep] = pure_log_likes[keep]
                self._next_purify += self._purify_every
                
            if self._purify_every > 0:
                # subtract one
                self._next_purify -= 1

            stime = time.time()
            # evolve the population to the next generation
            # get prep for evolution
            evo_args = self._get_evo_args(burnin=burnin)

            # get new state
            evo_res = _evolve(**evo_args)

            # save the state
            self._save_evolution(**evo_res)
            
            times.append(time.time() - stime)

        # if self._verbose:
        #     sys.stdout.write('\n')
        self._times.extend(times)
        return times

    def _get_evo_args(self, burnin=False):
        # determine the args for the evolve call
        if burnin:
            prop_gen = self._burnin_prop_gen
        else:
            prop_gen = self._prop_gen

        # fill in the rest of the dict
        evo_args = dict(
            prop_gen=prop_gen, parts_ind=self._get_part_ind(),
            split_ind=self._get_split_ind(),
            particles=self._particles[-1].copy(),
            param_names=self.param_names,
            transforms=self._get_transforms(),
            priors=_trimmed_priors(self._params),
            fixed=self._get_fixed(),
            log_likes=self._log_likes[-1].copy(),
            weights=self._weights[-1].copy(), use_priors=self._use_priors,
            recalc_likes=isinstance(self, HyperPrior),
            like_fun=self._like_fun, like_args=self._like_args,
            is_fixed_or_hyper=self._is_fixed_or_hyper,
            pop_recarray=self._pop_recarray, parallel=self._parallel,
            use_dask=self._use_dask)

        return evo_args

    def _save_evolution(self, particles=None, log_likes=None,
                        weights=None, kept=None):

        if self._needs_new_generation:
            # append to end
            self._particles.append(particles)
            self._log_likes.append(log_likes)
            self._weights.append(weights)
        else:
            # modify in place
            self._particles[-1][kept] = particles[kept]
            self._log_likes[-1][kept] = log_likes[kept]
            self._weights[-1][kept] = weights[kept]

        # save the accept rate
        self._accept_rate.append(float(kept.sum())/len(kept))

        # call post evolve as needed for FixedParams
        cur_split = np.ones(len(self._particles[-1]), dtype=np.bool)
        self._post_evolve(self._particles[-1][cur_split], cur_split, kept)

        self._needs_new_generation = True

    def _get_transforms(self, inverse=False):
        # loop over priors and get the transforms
        if inverse:
            return [p.inv_transform for p in self._params]
        else:
            return [p.transform for p in self._params]

    def _get_fixed(self):
        fixed = np.zeros(len(self._params), dtype=np.bool)
        for i, param in enumerate(self._params):
            if param._fixed or not hasattr(param.prior, "pdf"):
                fixed[i] = True
        return fixed
    

########
# Functional approach for easier parallel calls
########

def _trimmed_priors(params):
    "Trim HyperPrior to only have most recent info"
    trimmed = []
    for p in params:
        if isinstance(p.prior, HyperPrior) or \
           isinstance(p.prior, NormalHyperPrior):
            tp = _HyperPriorSnapshot(p.prior)
            trimmed.append(tp)
        else:
            trimmed.append(p.prior)
    return trimmed


def _trimmed_init_priors(params):
    "Trim HyperPrior and FixedParam to only have most recent info"
    trimmed = []
    for p in params:
        if isinstance(p.init_prior, HyperPrior) or \
           isinstance(p.init_prior, NormalHyperPrior):
            tp = _HyperPriorSnapshot(p.init_prior)
            trimmed.append(tp)
        elif p._fixed:
            # pull the most recent fixed as the init_prior
            m = p._fixed_info
            trimmed.append(m['model']._particles[-1][:, m['param_ind']][:, np.newaxis].copy())
        else:
            trimmed.append(p.init_prior)
    return trimmed


def _init_chains(priors, num_chains, initial_zeros_ok=False, verbose=False,
                 like_fun=None, like_args=None, is_fixed_or_hyper=False,
                 pop_recarray=True, transforms=None, param_names=None, 
                 use_priors=True, parallel=None, use_dask=False):
    # we're generating ourselves
    # initialize the particles and log_likes
    num_params = len(priors)
    particles = []
    log_likes = []
    weights = []
    times = []
    accept_rate = []

    # time it
    stime = time.time()

    # init the likes for the loop
    log_likes = np.zeros(num_chains) * np.nan
    ind = np.ones(num_chains, dtype=np.bool)
    good_ind = ~ind
    pop = None
    num_attempts = 0
    # keep looping until we have enough non-zero
    while (pop is None) or np.any(np.isnan(pop)) or \
          ((not initial_zeros_ok) and (good_ind.sum() < num_chains)):
        # update attempt count
        num_attempts += ind.sum()

        # provide feedback as requested
        if verbose:
            sys.stdout.write('%d(%d) ' %
                             (ind.sum(),
                              num_chains - good_ind.sum()))
            sys.stdout.flush()

        # generate the new pop
        prop = []
        for i,p in enumerate(priors):
            if hasattr(p, "rvs"):
                # generate with rvs
                if isinstance(p, HyperPrior) or \
                   isinstance(p, _HyperPriorSnapshot) or \
                   isinstance(p, NormalHyperPrior):
                    # must provide the cur_split
                    prop.append(p.rvs((ind.sum(), 1), ind))
                else:
                    prop.append(p.rvs((ind.sum(), 1)))
            elif isinstance(p, np.ndarray):
                # was a fixed that we should just copy into place
                prop.append(p[ind].copy())
            else:
                # copy the scaler value
                prop.append(np.ones((ind.sum(), 1)) * p)

        # stack and check
        npop = np.hstack(prop)
        if npop.ndim < 2:
            npop = npop[:, np.newaxis]
        if pop is None:
            pop = npop
        else:
            # add in these new proposals and test them
            pop[ind, :] = npop

        # calc the log likes for those new pops
        log_likes[ind] = _calc_log_likes(pop[ind],
                                         like_fun,
                                         like_args,
                                         cur_split=ind,
                                         is_fixed_or_hyper=is_fixed_or_hyper,
                                         pop_recarray=pop_recarray,
                                         transforms=transforms,
                                         param_names=param_names,
                                         parallel=parallel,
                                         use_dask=use_dask)

        # update the number of good and bad ind
        ind = np.isinf(log_likes) | np.isnan(log_likes)
        good_ind = ~ind

    # calc the accept_rate
    accept_rate = float(len(pop))/num_attempts

    # save the initial log_likes and particles
    times = time.time() - stime
    if use_priors:
        # calc log_priors
        #log_priors = self.calc_log_prior(pop)
        log_priors = _calc_log_prior(pop,
                                     priors=priors,
                                     cur_split=None)

        weights = log_likes + log_priors
    else:
        weights = log_likes

    return {'particles': pop, 'log_likes': log_likes,
            'weights': weights, 'accept_rate': accept_rate,
            'times': times}


def _calc_log_likes(pop, like_fun, like_args, cur_split=None,
                    is_fixed_or_hyper=False, pop_recarray=True,
                    transforms=None, param_names=None, parallel=None,
                    use_dask=False):
    # check the cur_split
    if cur_split is None:
        # would normally check this
        num_chains = len(pop)
        cur_split = np.ones(num_chains, dtype=np.bool)

    # apply transformation if necessary
    pop = _apply_param_transform(pop, transforms)

    # turn the pop into a rec array with names
    if pop_recarray:
        if param_names is None:
            raise ValueError("param_names required if pop_recarray")
        pop = np.rec.fromarrays(pop.T, names=param_names)

    if use_dask:
        # use dask
        client = get_client()

        if pop_recarray:
            # must make indiv 1d rec array
            if is_fixed_or_hyper:
                # must provide the current split
                out = [client.submit(like_fun, np.atleast_1d(indiv),
                                     cur_split, *(like_args), pure=False)
                       for indiv in pop]
            else:
                out = [client.submit(like_fun, np.atleast_1d(indiv),
                                     *(like_args), pure=False)
                       for indiv in pop]
        else:
            # must make indiv 2d array
            if is_fixed_or_hyper:
                # must provide the current split
                out = [client.submit(like_fun, np.atleast_2d(indiv),
                                     cur_split, *(like_args), pure=False)
                       for indiv in pop]
            else:
                out = [client.submit(like_fun, np.atleast_2d(indiv),
                                     *(like_args), pure=False)
                       for indiv in pop]
                
        # gather the results
        out = client.gather(out)

    elif parallel is not None:
        if pop_recarray:
            # must make indiv 1d rec array
            if is_fixed_or_hyper:
                # must provide the current split
                out = parallel(delayed(like_fun)(np.atleast_1d(indiv),
                                                             cur_split,
                                                             *(like_args))
                               for indiv in pop)
            else:
                out = parallel(delayed(like_fun)(np.atleast_1d(indiv),
                                                             *(like_args))
                               for indiv in pop)
        else:
            # must make indiv 2d array
            if is_fixed_or_hyper:
                # must provide the current split
                out = parallel(delayed(like_fun)(np.atleast_2d(indiv),
                                                             cur_split
                                                             *(like_args))
                               for indiv in pop)
            else:
                out = parallel(delayed(like_fun)(np.atleast_2d(indiv),
                                                             *(like_args))
                               for indiv in pop)

    else:
        # just process all at once in serial
        if is_fixed_or_hyper:
            # must provide the current split
            out = like_fun(pop, cur_split, *(like_args))
        else:
            out = like_fun(pop, *(like_args))

    # process the results
    log_likes = out

    # concatenate as needed
    if isinstance(log_likes, list):
        # concatenate it
        log_likes = np.concatenate(log_likes)

    return log_likes


def _calc_log_prior(*props, priors, cur_split=None):
    # set starting log_priors
    log_priors = [np.zeros(len(p)) for p in props]

    if cur_split is None:
        # pull from first
        cur_split = np.ones(len(props[0]), dtype=np.bool)
        
    # loop over params
    for i, prior in enumerate(priors):
        if hasattr(prior, "logpdf"):
            # it's not a fixed value
            # pick props and make sure to pass all
            # into pdf at the same time
            # to ensure using the same prior dist
            p = np.hstack([props[j][:, i][:, np.newaxis]
                           for j in range(len(props))])

            # ignore divide by zero in log here
            with np.errstate(divide='ignore'):
                if isinstance(prior, _HyperPriorSnapshot) or \
                   isinstance(prior, HyperPrior) or \
                   isinstance(prior, NormalHyperPrior):
                    # must provide cur_split
                    log_pdf = prior.logpdf(p, cur_split)
                else:
                    log_pdf = prior.logpdf(p)

            for j in range(len(props)):
                log_priors[j] += log_pdf[:, j]

    # just pick singular column if there's only one passed in
    if len(log_priors) == 1:
        log_priors = log_priors[0]
    return log_priors


def _evolve(prop_gen=None, parts_ind=None, split_ind=None,
            particles=None, param_names=None,
            transforms=None, priors=None, fixed=None,
            log_likes=None, weights=None, use_priors=True, recalc_likes=False,
            like_fun=None, like_args=None, is_fixed_or_hyper=False,
            pop_recarray=True, parallel=None, use_dask=False):


    # loop over two halves
    kept = np.zeros(len(particles), dtype=np.bool)
    for cur_split in [split_ind, ~split_ind]:
        # skip the split if there are no chains in it
        if cur_split.sum() == 0:
            continue
        
        # generate new proposals
        proposal = prop_gen(particles[cur_split],
                            particles[~cur_split],
                            weights[cur_split],
                            fixed)

        # apply the parts ind
        proposal[:, ~parts_ind] = particles[cur_split][:, ~parts_ind]

        # eval the proposal
        prop_log_likes = _calc_log_likes(proposal,
                                         like_fun, like_args,
                                         cur_split=cur_split,
                                         is_fixed_or_hyper=is_fixed_or_hyper,
                                         pop_recarray=pop_recarray,
                                         transforms=transforms,
                                         param_names=param_names,
                                         parallel=parallel,
                                         use_dask=use_dask)

        # see if recalc previous likes (for HyperPrior)
        if recalc_likes:
            prev_log_likes = _calc_log_likes(particles[cur_split],
                                             like_fun, like_args,
                                             cur_split=cur_split,
                                             is_fixed_or_hyper=is_fixed_or_hyper,
                                             pop_recarray=pop_recarray,
                                             transforms=transforms,
                                             param_names=param_names,
                                             parallel=parallel,
                                             use_dask=use_dask)
        else:
            prev_log_likes = log_likes[cur_split]

        # decide whether to keep the new proposal or not
        # keep with a MH step
        log_diff = np.float64(prop_log_likes - prev_log_likes)

        # next see if we need to include priors for each param
        if use_priors:
            prop_log_prior, prev_log_prior = _calc_log_prior(
                proposal,
                particles[cur_split],
                priors=priors,
                cur_split=cur_split)
            prop_weights = prop_log_likes + prop_log_prior
            log_diff += np.float64(prop_log_prior - prev_log_prior)

            prev_weights = prev_log_likes + prev_log_prior
        else:
            prop_weights = prop_log_likes
            prev_weights = prev_log_likes

        # calc acceptance in log space
        keep = np.log(np.random.rand(len(log_diff))) < log_diff

        # handle much greater than one
        #log_diff[log_diff > 0.0] = 0.0

        # now exp so we can get the other probs
        # mh_prob = np.exp(log_diff)
        # mh_prob[np.isnan(mh_prob)] = 0.0
        # keep = (mh_prob - np.random.rand(len(mh_prob))) > 0.0

        # modify the relevant particles
        # use mask the mask approach
        split_keep_ind = cur_split.copy()
        split_keep_ind[split_keep_ind] = keep
        particles[split_keep_ind] = proposal[keep]
        log_likes[split_keep_ind] = prop_log_likes[keep]
        weights[split_keep_ind] = prop_weights[keep]

        # save the kept info
        kept[cur_split] = keep

    return {'particles': particles, 'log_likes': log_likes,
            'weights': weights, 'kept': kept}
    

class _HyperPriorSnapshot():
    """Wrapper for HyperPrior to avoid parallelizing large objects"""
    def __init__(self, hp):
        self._pop = _apply_param_transform(hp._particles[-1],
                                           hp._get_transforms())
        self._dist = hp._dist
        self._num_chains = len(self._pop)
        
    def pdf(self, vals, cur_split):
        # self._dist can't be None
        # pick from chains
        vals = np.atleast_1d(vals)

        # generate the pdf using the likelihood func
        pop = self._pop[cur_split]
        args = [pop[:, i] for i in range(pop.shape[1])]
        d = self._dist(*args)
        # p = np.hstack([d.pdf(vals[:,i]) for i in range(vals.shape[1])])
        if np.ndim(vals) > 1:
            p = np.vstack([d.pdf(vals[:, i])
                           for i in range(vals.shape[1])]).T
        else:
            p = d.pdf(vals)

        return p

    def logpdf(self, vals, cur_split):
        # self._dist can't be None
        # pick from chains
        vals = np.atleast_1d(vals)

        # generate the pdf using the likelihood func
        pop = self._pop[cur_split]
        args = [pop[:, i] for i in range(pop.shape[1])]
        d = self._dist(*args)
        # p = np.hstack([d.pdf(vals[:,i]) for i in range(vals.shape[1])])
        if np.ndim(vals) > 1:
            p = np.vstack([d.logpdf(vals[:, i])
                           for i in range(vals.shape[1])]).T
        else:
            p = d.logpdf(vals)

        return p

    def rvs(self, size, cur_split):
        # randomly pick from chains
        size = np.atleast_1d(size)

        # pick chains
        if size[0] == self._num_chains:
            # have them match
            chains = np.arange(self._num_chains)
        else:
            # pick randomly
            chains = np.random.randint(0, self._num_chains, size[0])

        # generate the random vars using the likelihood func
        # pop = self._particles[-1][chains]
        # r = self._dist(*(pop[:,i] for i in range(pop.shape[1]))).rvs(size[1:])
        r = np.array([self._dist(*self._pop[ind]).rvs(size[1:])
                      for i, ind in enumerate(chains)])
        return r.reshape(size)


class HyperPrior(Model):
    """Model that acts as a prior for lower-level models
    """

    def __init__(self, name, dist, params,
                 num_chains=None,
                 proposal_gen=None,
                 burnin_proposal_gen=None,
                 use_priors=True,
                 verbose=False):

        # handle the dist
        self._dist = dist

        # like_args will get filled in later
        super(HyperPrior, self).__init__(name=name,
                                         params=params,
                                         like_fun=self._dist_like,
                                         num_chains=num_chains,
                                         proposal_gen=proposal_gen,
                                         burnin_proposal_gen=burnin_proposal_gen,
                                         initial_zeros_ok=True,
                                         use_priors=use_priors,
                                         verbose=verbose,
                                         pop_recarray=False)

        # We need to recalc likes of prev iteration in case of hyperprior
        self._recalc_likes = True

    def _dist_like(self, pop, cur_split, *args):
        # the args will contain the list of params in the other models
        # that use this model as a prior
        if len(args) == 0:
            # we are not likely
            return -np.ones(len(pop)) * np.inf

        if cur_split is None:
            raise ValueError("Current split can not be empty.")

        # default to not like
        log_like = np.zeros(len(pop))

        # loop over population (eventually parallelize this)
        d_args = [pop[:, i] for i in range(pop.shape[1])]
        d = self._dist(*d_args)
        for m in args:
            if not hasattr(m['model'], '_particles'):
                return -np.ones(len(pop)) * np.inf
            # ignore divide by zero in log here
            with np.errstate(divide='ignore'):
                log_like += d.logpdf(m['model']._particles[-1][cur_split,
                                                               m['param_ind']])

        return log_like

    def pdf(self, vals, cur_split):
        # self._dist can't be None
        # pick from chains
        vals = np.atleast_1d(vals)

        if cur_split is None:            
            if len(vals) == self._num_chains:
                # have them match
                cur_split = np.arange(self._num_chains)
            else:
                # pick randomly
                cur_split = np.random.randint(0, self._num_chains, len(vals))

        # generate the pdf using the likelihood func
        pop = _apply_param_transform(self._particles[-1][cur_split],
                                     self._get_transforms())
        args = [pop[:, i] for i in range(pop.shape[1])]
        d = self._dist(*args)
        # p = np.hstack([d.pdf(vals[:,i]) for i in range(vals.shape[1])])
        if np.ndim(vals) > 1:
            p = np.vstack([d.pdf(vals[:, i])
                           for i in range(vals.shape[1])]).T
        else:
            p = d.pdf(vals)
        # p = d.pdf(vals.T).T
        # p = self._dist(*self._particles[-1][chains]).pdf(vals)
        # p = np.array([self._dist(*self._particles[-1][ind]).pdf(vals[i])
        #               for i,ind in enumerate(chains)])
        return p

    def logpdf(self, vals, cur_split):
        # self._dist can't be None
        # pick from chains
        vals = np.atleast_1d(vals)

        if cur_split is None:            
            if len(vals) == self._num_chains:
                # have them match
                cur_split = np.arange(self._num_chains)
            else:
                # pick randomly
                cur_split = np.random.randint(0, self._num_chains, len(vals))

        # generate the pdf using the likelihood func
        pop = _apply_param_transform(self._particles[-1][cur_split],
                                     self._get_transforms())
        args = [pop[:, i] for i in range(pop.shape[1])]
        d = self._dist(*args)
        # p = np.hstack([d.logpdf(vals[:,i]) for i in range(vals.shape[1])])
        if np.ndim(vals) > 1:
            p = np.vstack([d.logpdf(vals[:, i])
                           for i in range(vals.shape[1])]).T
        else:
            p = d.logpdf(vals)
        # p = d.pdf(vals.T).T
        # p = self._dist(*self._particles[-1][chains]).pdf(vals)
        # p = np.array([self._dist(*self._particles[-1][ind]).pdf(vals[i])
        #               for i,ind in enumerate(chains)])
        return p

    def rvs(self, size, cur_split):
        # randomly pick from chains
        size = np.atleast_1d(size)

        # pick chains
        num_chains = len(cur_split)
        if size[0] == cur_split.sum():
            # have them match
            chains = np.arange(num_chains)[cur_split]
        else:
            # pick randomly
            chains = np.random.randint(0, num_chains, size[0])

        # generate the random vars using the likelihood func
        # pop = self._particles[-1][chains]
        pop = _apply_param_transform(self._particles[-1],
                                     self._get_transforms())
        
        # r = self._dist(*(pop[:,i] for i in range(pop.shape[1]))).rvs(size[1:])
        r = np.array([self._dist(*pop[ind]).rvs(size[1:])
                      for i, ind in enumerate(chains)])
        return r.reshape(size)


class FixedParams(Model):
    """Modeled parameter that is fixed across lower-level models

    sigma = Param('sigma', prior=np.dists.invgamma(1,1))
    params = [Param('mu', prior=dists.normal(0,1)),
              sigma]
    """

    def __init__(self, name, params,
                 num_chains=None,
                 proposal_gen=None,
                 burnin_proposal_gen=None,
                 use_priors=True,
                 verbose=False,
                 parallel=None):

        # set each param to be fixed
        for i in range(len(params)):
            params[i]._fixed = True
            params[i]._fixed_info = {'model': self,
                                     'param_ind': i}

        # save the input params
        self._submodel_params = params

        # make new params based on these that are not fixed
        new_params = []
        for p in params:
            new_params.append(Param(name=p.name,
                                    prior=p.prior,
                                    init_prior=p.init_prior,
                                    display_name=p.display_name,
                                    transform=p.transform))

        # init the Model parent
        Model.__init__(self, name, new_params,
                       like_fun=self._fixed_like,
                       num_chains=num_chains,
                       proposal_gen=proposal_gen,
                       burnin_proposal_gen=burnin_proposal_gen,
                       initial_zeros_ok=True,
                       use_priors=use_priors,
                       verbose=verbose,
                       parallel=parallel,
                       pop_recarray=False)

        # We need to recalc likes of prev iteration in case of FixedParams
        self._recalc_likes = True

        # mechanism for saving temporary log likes for the models
        # this fixed param is in
        self._mprop_log_likes = {}
        self._mprop_log_prior = {}

        pass

    def _fixed_like(self, pop, cur_split, *args):
        # the args will contain the list of params in the other models
        # that use this param
        if len(args) == 0:
            # we are not likely
            return -np.ones(len(pop)) * np.inf

        # init like to zero
        log_like = np.zeros(len(pop))

        # loop over models, calculating their likes with the proposed value
        # of this fixed param
        for m in args:
            #from IPython.core.debugger import Tracer ; Tracer()()
            # make sure the submodel has initialized
            if not hasattr(m['model'], '_particles'):
                return -np.ones(len(pop)) * np.inf

            # get the current population and replace with this proposal
            mpop = m['model']._particles[-1].copy()[cur_split]

            # make sure there are corresponding mprop values of the correct shape
            if not m['model'] in self._mprop_log_likes:
                # add both log likes and log prior
                self._mprop_log_likes[m['model']] = np.zeros(len(m['model']._particles[-1]))
                self._mprop_log_prior[m['model']] = np.zeros(len(m['model']._particles[-1]))

            # set all the fixed params and transforms
            transforms = m['model']._get_transforms()
            for i, j in m['param_ind']:
                # set the param
                mpop[:, i] = pop[:, j]

                # fixed param has already been transformed
                transforms[i] = None

            # see if we're just updating log_like for updated children
            if np.all((mpop - m['model']._particles[-1][cur_split]) == 0.0):
                # it's the same params, so just pull the likes
                mprop_log_likes = m['model']._log_likes[-1][cur_split]
                log_like += mprop_log_likes
                self._mprop_log_likes[m['model']][cur_split] = mprop_log_likes

                if m['model']._use_priors:
                   mprop_log_prior = m['model']._weights[-1][cur_split] - \
                       m['model']._log_likes[-1][cur_split]
                   self._mprop_log_prior[m['model']][cur_split] = mprop_log_prior
            else:
                # calc the log-likes from all the models using these params
                # must provide current split
                #mprop_log_likes, mprop_posts = m['model']._calc_log_likes(mpop,
                #                                                          cur_split)
                mprop_log_likes = _calc_log_likes(mpop,
                                                  m['model']._like_fun,
                                                  m['model']._like_args,
                                                  cur_split=cur_split,
                                                  is_fixed_or_hyper=m['model']._is_fixed_or_hyper,
                                                  pop_recarray=m['model']._pop_recarray,
                                                  transforms=transforms,
                                                  param_names=m['model'].param_names,
                                                  parallel=m['model']._parallel,
                                                  use_dask=m['model']._use_dask)

                # save these model likes for updating the model with those
                # that were kept when we call _post_evolve
                self._mprop_log_likes[m['model']][cur_split] = mprop_log_likes

                # aggregate log_likes for each particle
                log_like += mprop_log_likes

                if m['model']._use_priors:
                   #mprop_log_prior = m['model'].calc_log_prior(mpop,
                   #                                            cur_split=cur_split)
                   mprop_log_prior = _calc_log_prior(mpop,
                                                     priors=_trimmed_priors(m['model']._params),
                                                     cur_split=cur_split)

                   self._mprop_log_prior[m['model']][cur_split] = mprop_log_prior

        return log_like

    def _post_evolve(self, pop, cur_split, kept):
        # for any particle we kept, go back through the submodels and
        # update their params and likelihoods for the most recent pop
        # loop over the models
        split_kept_ind = cur_split.copy()
        split_kept_ind[split_kept_ind] = kept
        for m in self._like_args:
            # copy the prev state
            if m['model']._needs_new_generation:
                m['model']._particles.append(m['model']._particles[-1].copy())
                m['model']._log_likes.append(m['model']._log_likes[-1].copy())
                m['model']._weights.append(m['model']._weights[-1].copy())
                m['model']._needs_new_generation = False

            # update most recent particles
            for i, j in m['param_ind']:
                m['model']._particles[-1][split_kept_ind, i] = pop[kept, j]

            # update most recent weights
            m['model']._weights[-1][split_kept_ind] = self._mprop_log_likes[m['model']][kept]
            if m['model']._use_priors:
                # add the prior
                m['model']._weights[-1][split_kept_ind] += self._mprop_log_prior[m['model']][kept]

            # update most recent log_likes
            m['model']._log_likes[-1][split_kept_ind] = self._mprop_log_likes[m['model']][kept]

        # from IPython.core.debugger import Tracer ; Tracer()()
        pass

    def _all_submodels_hyperpriors(self):
        for m in self._like_args:
            if not isinstance(m['model'], HyperPrior):
                return False
        return True

if __name__ == "__main__":

    pass
