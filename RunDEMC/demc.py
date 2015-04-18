#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
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

# local imports
from de import DE


class Param(object):
    """
    Parameter for use with RunDEMC.
    """
    def __init__(self, name='', prior=None, init_prior=None,
                 display_name=None, transform=None):
        self.name = name
        self.prior = prior

        if init_prior is None:
            init_prior = self.prior
        self.init_prior = init_prior
            
        if display_name is None:
            display_name = self.name
        self.display_name = display_name

        self.transform = transform

        # hidden variable to indicate whether this param is fixed at
        # this level
        self._fixed = False

        
class Model(object):
    """
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
                         np.asarray(self._particles),
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

    default_prop_gen = DE(gamma_best=0.0,rand_base=False)
    default_burnin_prop_gen = DE(gamma_best=None,rand_base=True)
    
    def __init__(self, params,
                 like_fun, 
                 like_args = None,
                 num_chains=None,
                 proposal_gen=None,
                 burnin_proposal_gen=None,
                 initial_zeros_ok=False,
                 init_multiplier=1,
                 use_priors=True,
                 save_posts=False,
                 verbose=False):
        """
        DEMC
        """
        # set the vars
        self._params = params  # can't be None
        if num_chains is None:
            num_chains = int(np.min([len(params)*10,50]))
        self._num_chains = num_chains
        self._initial_zeros_ok = initial_zeros_ok
        self._init_multiplier = init_multiplier
        self._use_priors = use_priors
        self._save_posts = save_posts
        self._verbose = verbose

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
            

        # we have not initialized
        self._initialized = False
        
    def _initialize(self, force=False, num_chains=None):
        if self._initialized and not force:
            # already done
            return
        
        if self._verbose:
            sys.stdout.write('Initializing: ')
            sys.stdout.flush()

        # time it
        stime = time.time()

        # initialize the particles and log_likes
        self._num_params = len(self._params)
        if not num_chains is None:
            self._num_chains = num_chains
        self._particles = []
        self._log_likes = []
        self._weights = []
        self._times = []
        self._posts = []

        # fill using init priors (for initialization)
        init_parts = self._num_chains*self._init_multiplier
        pop = np.hstack([p.init_prior.rvs((init_parts,1))
                         if hasattr(p.init_prior,"rvs")
                         else np.ones((init_parts,1))*p.init_prior
                         for p in self._params])

        # get the initial log_likes
        if self._verbose:
            sys.stdout.write('%d(%d) '%(init_parts, self._num_chains))
            sys.stdout.flush()
        log_likes,posts = self._calc_log_likes(pop)

        # make sure not zero
        if not self._initial_zeros_ok:
            ind = np.isinf(log_likes) | np.isnan(log_likes)
            good_ind = ~ind
            while good_ind.sum() < self._num_chains:
                if self._verbose:
                    sys.stdout.write('%d(%d) '%(ind.sum(),self._num_chains-good_ind.sum()))
                    sys.stdout.flush()
                pop[ind,:] = np.hstack([p.init_prior.rvs((ind.sum(),1))
                                        if hasattr(p.init_prior,"rvs")
                                        else np.ones((ind.sum(),1))*p.init_prior
                                        for p in self._params])
                log_likes[ind],temp_posts = self._calc_log_likes(pop[ind])
                if self._save_posts:
                    posts[ind] = temp_posts
                ind = np.isinf(log_likes)|np.isnan(log_likes)
                good_ind = ~ind

            # save the good pop
            good_ind = ~ind
            pop = pop[good_ind]
            log_likes = log_likes[good_ind]
            if self._save_posts:
                posts = posts[good_ind]
        
        if len(pop) > self._num_chains:
            pop = pop[:self._num_chains]
            log_likes = log_likes[:self._num_chains]
            if self._save_posts:
                posts = posts[:self._num_chains]
                
        # append the initial log_likes and particles
        self._times.append(time.time()-stime)
        self._log_likes.append(log_likes)
        if self._use_priors:
            # calc log_priors
            log_priors = self.calc_log_prior(pop)
            self._weights.append(log_likes+log_priors)
        else:
            self._weights.append(log_likes)
        self._particles.append(pop)
        if self._save_posts:
            self._posts.append(posts)

        # say we've initialized
        self._initialized = True

    def apply_param_transform(self, pop):
        if self._transform_needed:
            pop = pop.copy()
            for i,p in enumerate(self._params):
                if p.transform:
                    pop[...,i] = p.transform(pop[...,i])
        return pop
        
    def _calc_log_likes(self, pop):
        # apply transformation if necessary
        pop = self.apply_param_transform(pop)
        
        # first get the log likelihood for the pop
        if self._save_posts:
            log_likes,posts = self._like_fun(self, pop, *(self._like_args))
        else:
            log_likes = self._like_fun(self, pop, *(self._like_args))
            posts = None

        return log_likes,posts

    def _crossover(self, burnin=False):
        if burnin:
            prop_gen = self._burnin_prop_gen
        else:
            prop_gen = self._prop_gen

        # always pass params, though no longer using priors
        proposal = prop_gen(self._particles[-1],
                            self._weights[-1],
                            self._params)
        return proposal

    def _migrate(self):
        # pick which items will migrate        
        num_to_migrate = np.random.random_integers(2,self._num_chains)
        to_migrate = random.sample(range(self._num_chains), num_to_migrate)

        # do a circle swap
        keepers = []
        for f_ind in range(len(to_migrate)):
            if f_ind == len(to_migrate)-1:
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
            keep = (mh_prob-np.random.rand())>0.0
            if keep:
                keepers.append({'ind':j,
                                'particle':self._particles[-1][i],
                                'weight':self._weights[-1][i],
                                'log_like':self._log_likes[-1][i]})

        for k in keepers:
            # do the swap (i.e., replace j with i)
            # replace the particle, weight, log_like
            self._particles[-1][k['ind']] = k['particle']
            self._weights[-1][k['ind']] = k['weight']
            self._log_likes[-1][k['ind']] = k['log_like']

    def _post_evolve(self, pop, kept):
        pass

    def _evolve(self, burnin=False):
        # first generate new proposals
        # loop over groups, making new proposal pops via mutation
        # or crossover
        proposal = self._crossover(burnin=burnin)

        # eval the population (this is separate from the proposals so
        # that we can parallelize the entire operation)
        prop_log_likes,prop_posts = self._calc_log_likes(proposal)

        # see if recalc prev_likes in case of HyperPrior
        if self._recalc_likes:
            prev_log_likes,prev_posts = self._calc_log_likes(self._particles[-1])
        else:
            prev_log_likes = self._log_likes[-1]

        # decide whether to keep the new proposal or not
        # keep with a MH step
        log_diff = np.float64(prop_log_likes - prev_log_likes)

        # next see if we need to include priors for each param
        if self._use_priors:
            prop_log_prior,prev_log_prior = self.calc_log_prior(proposal,
                                                                self._particles[-1])
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
        keep = (mh_prob-np.random.rand(len(mh_prob)))>0.0

        # set the not keepers from previous population
        proposal[~keep] = self._particles[-1][~keep]
        prop_log_likes[~keep] = prev_log_likes[~keep]
        weights[~keep] = prev_weights[~keep]
        #if self._use_priors:
        #    weights[~keep] += prev_log_prior[~keep]
        if self._save_posts:
            prop_posts[~keep] = self._posts[-1][~keep]

        # append the new proposal
        self._particles.append(proposal)
        self._log_likes.append(prop_log_likes)
        self._weights.append(weights)
        if self._save_posts:
            self._posts.append(prop_posts)

        # call post_evolve hook
        self._post_evolve(proposal, keep)
        pass

    def __call__(self, num_iter, burnin=False, migration_prob=0.0):
        # make sure we've initialized
        self._initialize()
        
        # loop over iterations
        if self._verbose:
            sys.stdout.write('Iterations (%d): '%(num_iter))
        times = []
        for i in xrange(num_iter):
            if np.random.rand() < migration_prob:
                # migrate, which is deterministic and done in place
                if self._verbose:
                    sys.stdout.write('x ')
                self._migrate()
            if self._verbose:
                sys.stdout.write('%d '%(i+1))
                sys.stdout.flush()
            stime = time.time()
            # evolve the population to the next generation
            self._evolve(burnin=burnin)
            times.append(time.time()-stime)
        if self._verbose:
            sys.stdout.write('\n')
        self._times.extend(times)
        return times

    def calc_log_prior(self, *props):
        # set starting log_priors
        log_priors = [np.zeros(len(p)) for p in props]
        
        # loop over params
        for i,param in enumerate(self._params):
            if hasattr(param.prior,"pdf"):
                # it's not a fixed value
                # pick props and make sure to pass all into pdf at the same time
                # to ensure using the same prior dist
                p = np.hstack([props[j][:,i][:,np.newaxis]
                               for j in range(len(props))])
                log_pdf = np.log(param.prior.pdf(p))

                for j in range(len(props)):
                    log_priors[j] += log_pdf[:,j]

        # just pick singular column if there's only one passed in
        if len(log_priors) == 1:
            log_priors = log_priors[0]
        return log_priors

    
class HyperPrior(Model):
    """Model that acts as a prior for lower-level models
    """
    def __init__(self, dist, params,
                 num_chains=None,
                 proposal_gen=None,
                 burnin_proposal_gen=None,
                 use_priors=True,
                 verbose=False):
        
        # handle the dist
        self._dist = dist

        # like_args will get filled in later
        super(HyperPrior, self).__init__(params=params,
                                         like_fun=self._dist_like, 
                                         num_chains=num_chains,
                                         proposal_gen=proposal_gen,
                                         burnin_proposal_gen=burnin_proposal_gen,
                                         initial_zeros_ok=True,
                                         use_priors=use_priors,
                                         save_posts=False,
                                         verbose=verbose)

        # We need to recalc likes of prev iteration in case of hyperprior
        self._recalc_likes = True

        # keep track of current pop
        self._cur_iter = -1
        pass

    def _dist_like(self, demc, pop, *args):
        # the args will contain the list of params in the other models
        # that use this model as a prior
        if len(args) == 0:
            # we are not likely
            return -np.ones(len(pop))*np.inf

        # default to not like
        log_like = np.zeros(len(pop))

        # # see if pick random inds for sub models
        # if self._cur_iter != len(self._particles):
        #     # pick new indices
        #     self._cur_iter = len(self._particles)
        #     self._cur_ind = [[np.random.randint(0,m['model']._num_chains)
        #                       for m in args]
        #                      for p in pop]
            
        # loop over population (eventually parallelize this)
        d_args = [pop[:,i] for i in range(pop.shape[1])]
        d = self._dist(*d_args)
        for m in args:
            if not hasattr(m['model'],'_particles'):
                return -np.ones(len(pop))*np.inf
            log_like += np.log(d.pdf(m['model']._particles[-1][:,m['param_ind']]))
            
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

    def pdf(self, vals):
        # self._dist can't be None
        # pick from chains
        vals = np.atleast_1d(vals)
        if len(vals) == self._num_chains:
            # have them match
            chains = np.arange(self._num_chains)
        else:
            # pick randomly
            chains = np.random.randint(0,self._num_chains,len(vals))

        # generate the pdf using the likelihood func
        pop = self._particles[-1][chains]
        args = [pop[:,i] for i in range(pop.shape[1])]
        d = self._dist(*args)
        #p = np.hstack([d.pdf(vals[:,i]) for i in range(vals.shape[1])])
        if np.ndim(vals) > 1:
            p = np.vstack([d.pdf(vals[:,i]) for i in range(vals.shape[1])]).T
        else:
            p = d.pdf(vals)
        #p = d.pdf(vals.T).T
        #p = self._dist(*self._particles[-1][chains]).pdf(vals)        
        #p = np.array([self._dist(*self._particles[-1][ind]).pdf(vals[i])
        #              for i,ind in enumerate(chains)])
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
            chains = np.random.randint(0,self._num_chains,size[0])

        # generate the random vars using the likelihood func
        #pop = self._particles[-1][chains]
        #r = self._dist(*(pop[:,i] for i in range(pop.shape[1]))).rvs(size[1:])
        r = np.array([self._dist(*self._particles[-1][ind]).rvs(size[1:])
                      for i,ind in enumerate(chains)])
        return r.reshape(size)


class FixedParams(Model):
    """Modeled parameter that is fixed across lower-level models

    sigma = FixedParam('sigma', prior=np.dists.invgamma(1,1))
    params = [Param('mu', prior=dists.normal(0,1)),
              sigma]
    """
    def __init__(self, params,
                 num_chains=None,
                 proposal_gen=None,
                 burnin_proposal_gen=None,
                 use_priors=True,
                 verbose=False):

        # set each param to be fixed
        for i in range(len(params)):
            params[i]._fixed = True

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
        Model.__init__(self, new_params,
                       like_fun=self._fixed_like, 
                       num_chains=num_chains,
                       proposal_gen=proposal_gen,
                       burnin_proposal_gen=burnin_proposal_gen,
                       initial_zeros_ok=True,
                       use_priors=use_priors,
                       save_posts=False,
                       verbose=verbose)

        # mechanism for saving temporary log likes for the models
        # this fixed param is in
        self._mprop_log_likes = {}
        self._mprop_log_prior = {}

        pass

    def _fixed_like(self, demc, pop, *args):
        # the args will contain the list of params in the other models
        # that use this model as a prior
        if len(args) == 0:
            # we are not likely
            return -np.ones(len(pop))*np.inf

        # init like to zero
        log_like = np.zeros(len(pop))

        # loop over models, calculating their likes with the proposed value
        # of this fixed param
        for m in args:
            #from IPython.core.debugger import Tracer ; Tracer()()
            if not hasattr(m['model'],'_particles'):
                return -np.ones(len(pop))*np.inf

            # get the current population and replace with this proposal
            mpop = m['model']._particles[-1].copy()

            # set all the fixed params
            for i,j in m['param_ind']:
                mpop[:,i] = pop[:,j]

            # calc the log-likes from all the models using this param
            mprop_log_likes,mprop_posts = m['model']._calc_log_likes(mpop)
            if m['model']._use_priors:
                mprop_log_prior = m['model'].calc_log_prior(mpop)

            # aggregate log_likes for each particle
            log_like += mprop_log_likes

            # save these model likes for updating the model with those
            # that were kept when we call _post_evolve
            self._mprop_log_likes[m['model']] = mprop_log_likes
            if m['model']._use_priors:
                self._mprop_log_prior[m['model']] = mprop_log_prior
                
        return log_like

    def _post_evolve(self, pop, kept):
        # for any particle we keep, go back through the submodels and
        # update their params and likelihoods for the most recent pop
        # loop over the models
        for m in self._like_args:
            # update most recent particles
            for i,j in m['param_ind']:            
                m['model']._particles[-1][kept,i] = pop[kept,j]
            
            # update most recent weights
            m['model']._weights[-1][kept] = self._mprop_log_likes[m['model']][kept]
            if m['model']._use_priors:
                # add the prior
                m['model']._weights[-1][kept] += self._mprop_log_prior[m['model']][kept]
                
            # update most recent log_likes
            m['model']._log_likes[-1][kept] = self._mprop_log_likes[m['model']][kept]

        #from IPython.core.debugger import Tracer ; Tracer()()
        pass


    
if __name__ == "__main__":

    pass
