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
from fastprogress.fastprogress import progress_bar

# test for joblib
try:
    import joblib
    from joblib import Parallel, delayed
except ImportError:
    joblib = None

# local imports
from .de import DE
from .io import load_results
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
        self.inv_transform = inv_transform

        # hidden variable to indicate whether this param is fixed at
        # this level
        self._fixed = False
        self._fixed_info = None


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
                         self.apply_param_transform(
                             np.asarray(self._particles)),
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

    posts = property(lambda self:
                     np.asarray(self._posts),
                     doc="""
                     Posts as an array.
                     """)

    accept_rate = property(lambda self:
                           np.asarray(self._accept_rate),
                           doc="""
                           Acceptance rate as an array.
                           """)

    default_prop_gen = DE(gamma_best=0.0, rand_base=False)
    default_burnin_prop_gen = DE(gamma_best=None, rand_base=True)

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
                 n_jobs=-1, backend=None):
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
        if partition is None:
            partition = len(self._params)
        if partition > len(self._params):
            partition = len(self._params)
        self._partition = partition
        self._parallel = parallel
        self._purify_every = purify_every
        self._next_purify = -1 + purify_every
        self._rand_split = rand_split

        # set up proposal generator
        if proposal_gen is None:
            proposal_gen = self.default_prop_gen
        self._prop_gen = proposal_gen
        if burnin_proposal_gen is None:
            burnin_proposal_gen = self.default_burnin_prop_gen
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

        # see if we need to apply a transform to any params
        if np.any([p.transform for p in self._params]):
            self._transform_needed = True
        else:
            self._transform_needed = False

        # should pop be passed as a recarray
        self._pop_recarray = pop_recarray

        # pop should be processed in parallel
        self._pop_parallel = pop_parallel

        # make a parallel instance if required
        if pop_parallel and self._parallel is None \
           and joblib is not None:
            self._parallel = Parallel(n_jobs=n_jobs, backend=backend)

        # we have not initialized
        self._initialized = False

    def _initialize(self, force=False, num_chains=None, partition=None):
        if self._initialized and not force:
            # already done
            return

        if self._verbose:
            sys.stdout.write('Initializing: ')
            sys.stdout.flush()

        # time it
        stime = time.time()

        # set the partition
        # see if we're modifying it
        if partition is not None:
            if partition > len(self._params):
                partition = len(self._params)
            self._partition = partition

        self._parts = np.array([1] * self._partition +
                               [0] * (len(self._params) - self._partition),
                               dtype=np.bool)

        # initialize starting point, see if from saved file
        if self._init_file:
            # we're loading from file
            # load the file
            m = load_results(self._init_file)

            # fill the variables
            self._num_params = len(m['param_names'])
            self._num_chains = m['particles'].shape[1]

            # pull the particles and apply inverse transform as needed
            particles = self.apply_param_transform(m['particles'],
                                                   inverse=True)
            # must turn everything into lists
            self._particles = [particles[i]
                               for i in range(len(particles))]
            self._log_likes = [m['log_likes'][i]
                               for i in range(len(particles))]
            self._weights = [m['weights'][i]
                             for i in range(len(particles))]
            self._times = [m['times'][i]
                           for i in range(len(particles))]
            if len(m['posts']) > 0:
                self._posts = [m['posts'][i]
                               for i in range(len(particles))]
            else:
                self._posts = []
            if has_key(m, 'accept_rate'):
                self._accept_rate = [m['accept_rate'][i]
                                     for i in range(len(particles))]
            else:
                self._accept_rate = []

        else:
            # we're generating ourselves
            # initialize the particles and log_likes
            self._num_params = len(self._params)
            if num_chains is not None:
                self._num_chains = num_chains
            self._particles = []
            self._log_likes = []
            self._weights = []
            self._times = []
            self._posts = []
            self._accept_rate = []

            # fill using init priors (for initialization)
            init_parts = self._num_chains * self._init_multiplier
            ind = np.ones(init_parts, dtype=np.bool)
            prop = []
            for p in self._params:
                if p._fixed:
                    # copy from fixed
                    m = p._fixed_info
                    prop.append(m['model']._particles[-1][ind, m['param_ind']][:, np.newaxis])
                elif hasattr(p.init_prior, "rvs"):
                    # generate with rvs
                    prop.append(p.init_prior.rvs((init_parts, 1)))
                else:
                    # copy the scaler value
                    prop.append(np.ones((init_parts, 1)) * p.init_prior)

            # stack it up and check dims
            pop = np.hstack(prop)
            if pop.ndim < 2:
                pop = pop[:, np.newaxis]

            # get the initial log_likes
            if self._verbose:
                sys.stdout.write('%d(%d) ' % (init_parts, self._num_chains))
                sys.stdout.flush()
            log_likes, posts = self._calc_log_likes(pop)

            # keep track of total proposals
            num_attempts = len(pop)

            # make sure not zero
            if not self._initial_zeros_ok:
                # get indices of bad and good likes
                ind = np.isinf(log_likes) | np.isnan(log_likes)
                good_ind = ~ind

                # keep looping until we have enough non-zero
                while good_ind.sum() < self._num_chains:
                    # update attempt count
                    num_attempts += ind.sum()

                    # provide feedback as requested
                    if self._verbose:
                        sys.stdout.write('%d(%d) ' %
                                         (ind.sum(),
                                          self._num_chains - good_ind.sum()))
                        sys.stdout.flush()

                    # generate the new pop
                    prop = []
                    for p in self._params:
                        if p._fixed:
                            # copy from fixed
                            m = p._fixed_info
                            prop.append(m['model']._particles[-1][ind, m['param_ind']][:, np.newaxis])
                        elif hasattr(p.init_prior, "rvs"):
                            # generate with rvs
                            prop.append(p.init_prior.rvs((ind.sum(), 1)))
                        else:
                            # copy the scaler value
                            prop.append(np.ones((ind.sum(), 1)) * p.init_prior)

                    # stack and check
                    npop = np.hstack(prop)
                    if npop.ndim < 2:
                        npop = npop[:, np.newaxis]
                    
                    # add in these new proposals and test them
                    pop[ind, :] = npop

                    # calc the log likes for those new pops
                    log_likes[ind], temp_posts = self._calc_log_likes(pop[ind],
                                                                      ind)
                    if temp_posts is not None:
                        posts[ind] = temp_posts

                    # update the number of good and bad ind
                    ind = np.isinf(log_likes) | np.isnan(log_likes)
                    good_ind = ~ind

                # # save the good pop
                # good_ind = ~ind
                # pop = pop[good_ind]
                # if pop.ndim < 2:
                #     pop = pop[:, np.newaxis]
                # log_likes = log_likes[good_ind]
                # if posts is not None:
                #     posts = posts[good_ind]

            # calc the accept_rate
            self._accept_rate.append(float(len(pop))/num_attempts)

            # if len(pop) > self._num_chains:
            #     pop = pop[:self._num_chains]
            #     log_likes = log_likes[:self._num_chains]
            #     if posts is not None:
            #         posts = posts[:self._num_chains]

            # append the initial log_likes and particles
            self._times.append(time.time() - stime)
            self._log_likes.append(log_likes)
            if self._use_priors:
                # calc log_priors
                log_priors = self.calc_log_prior(pop)
                self._weights.append(log_likes + log_priors)
            else:
                self._weights.append(log_likes)
            self._particles.append(pop)
            if posts is not None:
                self._posts.append(posts)

        # say we've initialized
        self._initialized = True
        self._needs_new_generation = True

    def apply_param_transform(self, pop, inverse=False):
        if self._transform_needed:
            pop = pop.copy()
            for i, p in enumerate(self._params):
                if p.transform:
                    if not inverse:
                        # just go in forward direction
                        pop[..., i] = p.transform(pop[..., i])
                    else:
                        # going in inverse direction
                        if p.inv_transform:
                            invtrans = p.inv_transform
                        else:
                            # see if figure out
                            if p.transform == invlogit:
                                # we know we can do logit
                                invtrans = logit
                            else:
                                raise ValueError("Could not infer inverse transform.")
                        # apply the inverse transform
                        pop[..., i] = invtrans(pop[..., i])
        return pop

    def _calc_log_likes(self, pop, cur_split=None):
        # check the cur_split
        if cur_split is None:
            if len(pop) != self._num_chains:
                raise ValueError("Proposal must be same size as num chains if no split is specified.")
            else:
                cur_split = np.ones(self._num_chains, dtype=np.bool)
                
        # apply transformation if necessary
        pop = self.apply_param_transform(pop)

        # turn the pop into a rec array with names
        if self._pop_recarray:
            pop = np.rec.fromarrays(pop.T, names=self.param_names)

        if self._pop_parallel and self._parallel is not None:
            if self._pop_recarray:
                # must make indiv 1d rec array
                if isinstance(self, HyperPrior) or isinstance(self, FixedParams):
                    # must provide the current split
                    out = self._parallel(delayed(self._like_fun)(np.atleast_1d(indiv),
                                                                 cur_split,
                                                                 *(self._like_args))
                                         for indiv in pop)
                else:
                    out = self._parallel(delayed(self._like_fun)(np.atleast_1d(indiv),
                                                                 *(self._like_args))
                                         for indiv in pop)
            else:
                # must make indiv 2d array
                if isinstance(self, HyperPrior) or isinstance(self, FixedParams):
                    # must provide the current split
                    out = self._parallel(delayed(self._like_fun)(np.atleast_2d(indiv),
                                                                 cur_split
                                                                 *(self._like_args))
                                         for indiv in pop)
                else:
                    out = self._parallel(delayed(self._like_fun)(np.atleast_2d(indiv),
                                                                 *(self._like_args))
                                         for indiv in pop)

        else:
            # just process all at once in serial
            if isinstance(self, HyperPrior) or isinstance(self, FixedParams):
                # must provide the current split
                out = self._like_fun(pop, cur_split, *(self._like_args))
            else:
                out = self._like_fun(pop, *(self._like_args))

        # process the results
        if isinstance(out, tuple):
            # split into likes and posts
            log_likes, posts = out
        else:
            # just likes
            log_likes = out
            posts = None

        # concatenate as needed
        if isinstance(log_likes, list):
            # concatenate it
            log_likes = np.concatenate(log_likes)
        if posts is not None and isinstance(posts, list):
            # concatenate it
            posts = np.concatenate(posts)
            
        return log_likes, posts

    def _get_part_ind(self):
        # grab the current partition indices
        parts = self._parts.copy()

        # roll them the size of the partition
        self._parts = np.roll(self._parts, self._partition)

        # return the pre-rolled value
        return parts

    def _get_split_ind(self):
        # set all ind
        num_particles = len(self.particles[-1])
        
        if self._rand_split:
            # randomize the indices
            all_ind = np.random.permutation(num_particles)
        else:
            all_ind = np.arange(num_particles)

        # split in half
        split_ind = np.zeros(num_particles, dtype=np.bool)
        split_ind[all_ind[:int(num_particles/2)]] = True
                  
        return split_ind

    def _crossover(self, pop_ind, ref_pop_ind, parts_ind, burnin=False):
        if burnin:
            prop_gen = self._burnin_prop_gen
        else:
            prop_gen = self._prop_gen

        # always pass params, though no longer using priors
        proposal = prop_gen(self._particles[-1][pop_ind],
                            self._particles[-1][ref_pop_ind],
                            self._weights[-1][pop_ind],
                            self._params)

        # apply the parts ind
        proposal[:, ~parts_ind] = self._particles[-1][pop_ind][:, ~parts_ind]

        return proposal

    def _purify(self):
        # we're going to purify, so just copy last state
        cur_split = np.ones(self._num_chains, dtype=np.bool)
        proposal = self._particles[-1]

        # eval the population
        pure_log_likes, temp_posts = self._calc_log_likes(proposal,
                                                          cur_split)

        # no MH step, just set the log_likes and weights
        # only keep non-inf
        keep = ~np.isinf(pure_log_likes)
        self._weights[-1][keep] = self._weights[-1][keep] - \
                                  self._log_likes[-1][keep] + \
                                  pure_log_likes[keep]
        self._log_likes[-1][keep] = pure_log_likes[keep]

        # reset the purification
        self._next_purify += self._purify_every

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

    def _evolve(self, burnin=False):
        # grab the parts ind for this evolution
        parts_ind = self._get_part_ind()

        # get the split
        split_ind = self._get_split_ind()

        # copy the prev state
        if self._needs_new_generation:
            self._particles.append(self._particles[-1].copy())
            self._log_likes.append(self._log_likes[-1].copy())
            self._weights.append(self._weights[-1].copy())
            if len(self._posts) > 0:
                self._posts.append(self._posts[-1].copy())
            self._needs_new_generation = False
        
        # loop over two halves
        kept = np.zeros(len(self._particles[-1]), dtype=np.bool)
        for cur_split in [split_ind, ~split_ind]:
            # generate new proposals
            proposal = self._crossover(cur_split, ~cur_split,
                                       parts_ind, burnin=burnin)

            # eval the proposal
            prop_log_likes, prop_posts = self._calc_log_likes(proposal,
                                                              cur_split)

            # see if recalc previous likes (for HyperPrior)
            if self._recalc_likes:
                prev_log_likes, prev_posts = self._calc_log_likes(
                    self._particles[-1][cur_split],
                    cur_split)
            else:
                prev_log_likes = self._log_likes[-1][cur_split]

            # decide whether to keep the new proposal or not
            # keep with a MH step
            log_diff = np.float64(prop_log_likes - prev_log_likes)

            # next see if we need to include priors for each param
            if self._use_priors:
                prop_log_prior, prev_log_prior = self.calc_log_prior(
                    proposal,
                    self._particles[-1][cur_split],
                    cur_split=cur_split)
                weights = prop_log_likes + prop_log_prior
                log_diff += np.float64(prop_log_prior - prev_log_prior)

                prev_weights = prev_log_likes + prev_log_prior
            else:
                weights = prop_log_likes
                prev_weights = prev_log_likes

            # handle much greater than one
            log_diff[log_diff > 0.0] = 0.0

            # now exp so we can get the other probs
            mh_prob = np.exp(log_diff)
            mh_prob[np.isnan(mh_prob)] = 0.0
            keep = (mh_prob - np.random.rand(len(mh_prob))) > 0.0

            # modify the relevant particles
            # use mask the mask approach
            split_keep_ind = cur_split.copy()
            split_keep_ind[split_keep_ind] = keep
            self._particles[-1][split_keep_ind] = proposal[keep]
            self._log_likes[-1][split_keep_ind] = prop_log_likes[keep]
            self._weights[-1][split_keep_ind] = weights[keep]
            if prop_posts is not None:
                self._posts[-1][split_keep_ind] = prop_posts[keep]

            # call post_evolve hook
            self._post_evolve(self._particles[-1][cur_split], cur_split, keep)

            # save the kept info
            kept[cur_split] = keep

        # save the acceptance rate
        self._accept_rate.append(float(kept.sum())/len(kept))
        self._needs_new_generation = True

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
                self._purify()
            if self._purify_every > 0:
                # subtract one
                self._next_purify -= 1
            # if self._verbose:
            #     sys.stdout.write('%d ' % (i + 1))
            #     sys.stdout.flush()
            stime = time.time()
            # evolve the population to the next generation
            self._evolve(burnin=burnin)
            times.append(time.time() - stime)
        # if self._verbose:
        #     sys.stdout.write('\n')
        self._times.extend(times)
        return times

    def calc_log_prior(self, *props, cur_split=None):
        # set starting log_priors
        log_priors = [np.zeros(len(p)) for p in props]

        # loop over params
        for i, param in enumerate(self._params):
            if hasattr(param.prior, "pdf"):
                # it's not a fixed value
                # pick props and make sure to pass all
                # into pdf at the same time
                # to ensure using the same prior dist
                p = np.hstack([props[j][:, i][:, np.newaxis]
                               for j in range(len(props))])

                # ignore divide by zero in log here
                with np.errstate(divide='ignore'):
                    if isinstance(param.prior, HyperPrior):
                        # must provide cur_split
                        log_pdf = np.log(param.prior.pdf(p, cur_split))
                    else:
                        log_pdf = np.log(param.prior.pdf(p))

                for j in range(len(props)):
                    log_priors[j] += log_pdf[:, j]

        # just pick singular column if there's only one passed in
        if len(log_priors) == 1:
            log_priors = log_priors[0]
        return log_priors


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
                log_like += np.log(d.pdf(m['model']._particles[-1][cur_split,
                                                                   m['param_ind']]))

        # for i,p in enumerate(pop):
        #     # set up the distribution
        #     d = self._dist(*p)

        #     # loop over all the mod/param using this as prior
        #     vals = []
        #     #c = None
        #     for j,m in enumerate(args):
        #         # grab the most recent vals from
        #         # that DEMC model and param index
        #         # pick a chain at random (keep same for each model)
        #         #if c is None:
        #         #c = np.random.randint(0,m['model']._num_chains)
        #         # extract that val
        #         #c = self._cur_ind[i][j]
        #         vals.append(m['model']._particles[-1][i,m['param_ind']])

        #     # add the sum log like for that pop
        #     log_like[i] += np.log(d.pdf(vals)).sum()

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
        pop = self._particles[-1][cur_split]
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

    def rvs(self, size):
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
        r = np.array([self._dist(*self._particles[-1][ind]).rvs(size[1:])
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

            # set all the fixed params
            for i, j in m['param_ind']:
                mpop[:, i] = pop[:, j]

            # see if we're just updating log_like for updated children
            if np.all((mpop - m['model']._particles[-1][cur_split]) == 0.0):
                # it's the same params, so just pull the likes
                mprop_log_likes = m['model']._log_likes[-1][cur_split]
                log_like += mprop_log_likes
                self._mprop_log_likes[m['model']] = mprop_log_likes

                if m['model']._use_priors:
                   mprop_log_prior = m['model']._weights[-1][cur_split] - \
                       m['model']._log_likes[-1][cur_split]
                   self._mprop_log_prior[m['model']] = mprop_log_prior
            else:
                # calc the log-likes from all the models using these params
                # must provide current split
                mprop_log_likes, mprop_posts = m['model']._calc_log_likes(mpop,
                                                                          cur_split)

                # save these model likes for updating the model with those
                # that were kept when we call _post_evolve
                self._mprop_log_likes[m['model']] = mprop_log_likes

                # aggregate log_likes for each particle
                log_like += mprop_log_likes

                if m['model']._use_priors:
                   mprop_log_prior = m['model'].calc_log_prior(mpop,
                                                               cur_split=cur_split)
                   self._mprop_log_prior[m['model']] = mprop_log_prior

        return log_like

    def _post_evolve(self, pop, cur_split, kept):
        # for any particle we keep, go back through the submodels and
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
                if len(m['model']._posts) > 0:
                    m['model']._posts.append(m['model']._posts[-1].copy())
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


if __name__ == "__main__":

    pass
