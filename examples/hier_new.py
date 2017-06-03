# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

"""For hyperparam the rvs pulls values from supplied distribution,
picking a chain at random.

like_fun will search the model for anywhere the hyperparam is used and will
calc like based on current params and dist for the thetas in use.

pdf will randomly pick from hyperparam chains and provide the pdf from
the supplied distribution.

log_prior: convenience method for looping over params for a proposal
and calling each param's pdf to get the log_like for that param.


"""

from RunDEMC import Model, HyperPrior, Hierarchy, Param, dists


# set up the hyper priors
h_alpha = HyperPrior(name='h_alpha',
                     dist=dists.normal,
                     params=[Param(name='mu', prior=dists.normal(1, .5)),
                             Param(name='sigma', prior=dists.invgamma(4, 10))])

h_beta = HyperPrior(name='h_beta',
                    dist=dists.normal,
                    params=[Param(name='mu', prior=dists.normal(1, .5)),
                            Param(name='sigma', prior=dists.invgamma(4, 10))])

# set up lower level (i.e., subject)


def subj_like(pop, *args):
    return np.log(dists.beta(pop[:, 0], pop[:, 1]).pdf(args[0]))


submods = [Model(name='subj_%s' % s,
                 params=[Param(name='alpha', prior=h_alpha),
                         Param(name='beta', prior=h_beta)],
                 like_fun=subj_like,
                 like_args=(sdat[s],),
                 verbose=False)
           for s in subjs]

hier = Hiearachy(submods)

hier(50, burnin=True)
hier(500)
