# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import scipy.stats.distributions
import scipy.stats.distributions as dists
from scipy.stats._distn_infrastructure import argsreduce
from scipy.special import gammaln, betaln, xlogy, xlog1py, log1p
from numpy.random import default_rng
import numpy as np


def logit(x):
    """Returns the logit transform of x."""
    return np.log(x / (1. - x))


def invlogit(x):
    """Return the inverse logit transform of x."""
    return 1. / (1. + np.exp(-x))


def log_factorial(x):
    """Returns the logarithm of x!
    Also accepts lists and NumPy arrays in place of x."""
    return gammaln(np.array(x) + 1)


def multinomial(xs, ps):
    """
    Calculate multinomial probability.

    xs = vector of observations in each choice
    ps = probabilities of observing each choice
    """
    xs, ps = np.array(xs), np.array(ps)
    n = xs.sum()
    result = log_factorial(n) - np.sum(log_factorial(xs)) + \
        np.sum(xs * np.log(ps))
    return np.exp(result)


_log_half = np.log(.5)
_norm_pdf_C = np.sqrt(2*np.pi)
_norm_pdf_logC = np.log(_norm_pdf_C)
class normal:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
        self.rng = default_rng()

        self._second_term = -_norm_pdf_logC - log(std)

    def logpdf(self, x):
        y = (x - self.mean) / self.std
        return -y**2 / 2.0 + self._second_term

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        if random_state is None:
            random_state = self.rng

        return random_state.normal(self.mean, self.std, size)


_norm_cdf = scipy.stats.distributions._continuous_distns._norm_cdf
_norm_sf = scipy.stats.distributions._continuous_distns._norm_sf
_norm_isf = scipy.stats.distributions._continuous_distns._norm_isf
_norm_ppf = scipy.stats.distributions._continuous_distns._norm_ppf
np = scipy.stats.distributions._continuous_distns.np
log = np.log
my_tn_gen = scipy.stats.distributions._continuous_distns.truncnorm_gen


def _argcheck_fixed(self, a, b):
    self.a = a
    self.b = b
    self._nb = _norm_cdf(b)
    self._na = _norm_cdf(a)
    self._sb = _norm_sf(b)
    self._sa = _norm_sf(a)
    if np.ndim(self.a) == 0:
        if self.a > 0:
            self._delta = -(self._sb - self._sa)
        else:
            self._delta = self._nb - self._na
    else:
        self._delta = np.zeros_like(self._sa)
        self._delta[self.a > 0] = - \
            (self._sb[self.a > 0] - self._sa[self.a > 0])
        self._delta[self.a <= 0] = self._nb[self.a <= 0] - \
            self._na[self.a <= 0]
    self._logdelta = log(self._delta)
    return np.all((a - b) != 0.0)


def _ppf_fixed(self, q, a, b):
    if np.ndim(self.a) == 0:
        if self.a > 0:
            return _norm_isf(q * self._sb + self._sa * (1.0 - q))
        else:
            return _norm_ppf(q * self._nb + self._na * (1.0 - q))
    else:
        out = np.zeros_like(self._sa)
        ind = self.a > 0
        out[ind] = _norm_isf(q * self._sb[ind] + self._sa[ind] * (1.0 - q))
        out[~ind] = _norm_ppf(q * self._nb[~ind] + self._na[~ind] * (1.0 - q))
        return out


def _pdf_fixed(self, x, *args, **kwds):
    """
    Probability density function at x of the given RV.
    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)
    Returns
    -------
    pdf : ndarray
        Probability density function evaluated at x
    """
    args, loc, scale = self._parse_args(*args, **kwds)
    x, loc, scale = list(map(np.asarray, (x, loc, scale)))
    args = tuple(map(np.asarray, args))
    x = np.asarray((x - loc) * 1.0 / scale)
    cond0 = self._argcheck(*args) & (scale > 0)
    cond1 = (scale > 0) & (x >= self.a) & (x <= self.b)
    cond = cond0 & cond1
    output = np.zeros(np.shape(cond), 'd')
    np.putmask(output, (1 - cond0) + np.isnan(x), self.badvalue)
    if any(cond.flatten()):
        goodargs = argsreduce(cond | ~cond, *((x,) + args + (scale,)))
        scale, goodargs = goodargs[-1], goodargs[:-1]
        ccond = cond.copy()
        ccond.shape = goodargs[0].shape
        output[cond] = (self._pdf(*goodargs) / scale)[ccond]
        # place(output, cond, self._pdf(*goodargs) / scale)
    if output.ndim == 0:
        return output[()]
    return output


my_tn_gen._argcheck = _argcheck_fixed
my_tn_gen._ppf = _ppf_fixed
my_tn_gen.pdf = _pdf_fixed
my_tn = my_tn_gen(name='truncnorm')


def trunc_normal(mean=0.0, std=1.0, lower=0.0, upper=1.0):
    a = (np.array(lower) - np.array(mean)) / np.array(std)
    b = (np.array(upper) - np.array(mean)) / np.array(std)
    return my_tn(a, b, loc=mean, scale=std)


class uniform:
    def __init__(self, lower=0.0, upper=1.0):
        self.lower = lower
        self.upper = upper
        self.rng = default_rng()

        self._size = upper - lower
        self._inv_size = 1 / self._size
        self._log_size = log(self._size)

    def logpdf(self, x):
        return np.where((x >= self.lower) & (x <= self.upper), -self._log_size, np.NINF)

    def pdf(self, x):
        return np.where((x >= self.lower) & (x <= self.upper), self._inv_size, 0)

    def rvs(self, size=1, random_state=None):
        if random_state is None:
            random_state = self.rng
        return random_state.uniform(self.lower, self.upper, size)


class beta:
    def __init__(self, alpha=.5, beta=.5):
        self.alpha = alpha
        self.beta = beta
        self.rng = default_rng()

        self._neg_betaln = -betaln(self.alpha, self.beta)
    
    def logpdf(self, x):
        return np.where((x >= 0) & (x <= 1), xlog1py(self.beta - 1.0, -x) + xlogy(self.alpha - 1.0, x) + self._neg_betaln, np.NINF)

    def pdf(self, x):
        return np.where((x >= 0) & (x <= 1), np.exp(self.logpdf(x)), 0)

    def rvs(self, size=1, random_state=None):
        if random_state is None:
            random_state = self.rng

        return random_state.beta(self.alpha, self.beta, size)


class gamma:
    """
    alpha = k
    beta = 1/theta
    """

    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.rng = default_rng()

        self._second_term = -gammaln(self.alpha) + log(beta)
        self._theta = 1 / beta

    def logpdf(self, x):
        y = x*self.beta
        return np.where(x >= 0, xlogy(self.alpha-1.0, y) - y + self._second_term, np.NINF)

    def pdf(self, x):
        return np.where(x >= 0, np.exp(self.logpdf(x)), 0)

    def rvs(self, size=1, random_state=None):
        if random_state is None:
            random_state = self.rng
        return random_state.gamma(self.alpha, self._theta, size)


class invgamma:
    """
    alpha = k
    beta = 1/theta
    """

    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.rng = default_rng()

        self._last_term = log(beta) - gammaln(self.alpha)
        self._theta = 1 / beta

    def logpdf(self, x):
        y = x*self.beta
        with np.errstate(divide='warn', invalid='warn'):
            return np.where(x > 0, -(self.alpha+1) * np.log(y) - 
                            np.divide(1.0, y, out=np.zeros_like(y, dtype=np.float64), where=(y!=0.))
                            + self._last_term, np.NINF)

    def pdf(self, x):
        return np.where(x > 0, np.exp(self.logpdf(x)), 0)

    def rvs(self, size=1, random_state=None):
        if random_state is None:
            random_state = self.rng
        return 1. / random_state.gamma(self.alpha, self._theta, size)


class exp:
    def __init__(self, lam=1.0):
        self.lam = lam
        self.rng = default_rng()
        self._log_lam = log(self.lam)

    def logpdf(self, x):
        return np.where(x >= 0, self._log_lam - x*self.lam, np.NINF)

    def pdf(self, x):
        return np.where(x >= 0, np.exp(self.logpdf(x)), 0)

    def rvs(self, size=1, random_state=None):
        if random_state is None:
            random_state = self.rng

        return random_state.exponential(self.lam, size)

class poisson:
    def __init__(self, lam=1.0):
        self.lam = lam
        self.rng = default_rng()
        
    def logpmf(self, x):
        return np.where((x >= 0) & (x == np.round(x)), xlogy(x, self.lam) - gammaln(x + 1) - self.lam, np.NINF)
        
    def pmf(self, x):
        return np.where((x >= 0) & (x == np.round(x)), np.exp(self.logpmf(x)), 0)
    
    def rvs(self, size=1, random_state=None):
        if random_state is None:
            random_state = self.rng
        
        return random_state.poisson(self.lam, size)

class laplace:
    def __init__(self, loc=0.0, diversity=1.0):
        self.loc = loc
        self.diversity = diversity
        self.rng = default_rng()

        self._log_diversity = np.log(diversity)

    def logpdf(self, x):
        y = (x - self.loc) / self.diversity
        return _log_half - np.abs(y) - self._log_diversity

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        if random_state is None:
            random_state = self.rng
        
        return random_state.laplace(self.loc, self.diversity, size=size)        

class students_t:
    def __init__(self, mean=0, std=1.0, df=1.0):
        self.mean = mean
        self.std = std
        self.df = df
        self.rng = default_rng()

        self._half_df_plus_one = (self.df+1)/2
        self._first_term = gammaln(self._half_df_plus_one) - gammaln(self.df/2) - 0.5*np.log(self.df*np.pi) - log(self.std)

    def logpdf(self, x):
        y = (x - self.mean) / self.std
        return self._first_term - self._half_df_plus_one*np.log(1+(y**2)/self.df)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        if random_state is None:
            random_state = self.rng
        
        return random_state.standard_t(self.df, size)*self.std + self.mean     

def noncentral_t(mean=0, std=1.0, df=1.0, nc=0.0):
    return dists.nct(df=df, nc=nc, loc=mean, scale=std)


class halfcauchy:
    def __init__(self, scale=1.0, loc=0.0):
        self.scale = scale
        self.loc = loc
        self.rng = default_rng()

        self._first_term = np.log(2.0/np.pi) - np.log(self.scale)

    def logpdf(self, x):
        y = (x - self.loc) / self.scale
        return np.where(x >= self.loc, np.log(2.0/np.pi) - log1p(y**2) - np.log(self.scale), np.NINF)

    def pdf(self, x):
        return np.where(x >= self.loc, np.exp(self.logpdf(x)), 0)

    def rvs(self, size=1, random_state=None):
        if random_state is None:
            random_state = self.rng
        
        return np.abs(random_state.standard_cauchy(size))*self.scale + self.loc


def epa_kernel(x, delta):
    """
    Epanechnikov kernel.
    """
    # make sure 1d
    x = np.atleast_1d(x)
    delta = np.atleast_1d(delta)

    # make sure we have matching deltas
    if len(delta) == 1:
        delta = delta.repeat(len(x))

    # allocate for the weights
    w = np.zeros_like(x)

    # determine
    ind = (delta - np.abs(x)) > 0.0  # np.abs(x)<delta
    w[ind] = (1. - (x[ind] / delta[ind])**2) * 1. / delta[ind]
    return w


class CustomDist(object):
    """
    """

    def __init__(self, pdf=None, rvs=None):
        self.pdf = pdf
        self.rvs = rvs


def uniform_weights(n, nd):
    rv = np.random.rand(n, nd - 1)
    rv.sort(axis=1)
    uv = np.hstack([np.zeros((n, 1)),
                    rv,
                    np.ones((n, 1))])
    return np.diff(uv, axis=1)


class Mixture(object):
    """Class to handle a mixture of distributions.
    """

    def __init__(self, dist_list):
        self._dist_list = dist_list
        self._weights = None

    def __call__(self, *weights):
        weights = np.atleast_2d(weights)  # [w for w in weights]
        # weights = np.array([np.atleast_1d(w) for w in weights])
        if len(weights) == (len(self._dist_list) - 1):
            weights = weights.T
        wnew = np.atleast_2d(1 - np.sum(weights, axis=1)).T
        weights = np.hstack([weights, wnew])
        self._weights = weights
        return self

    def pdf(self, *args, **kwargs):
        # calc weighted combo of pdfs
        pdfs = np.array([d.pdf(*args, **kwargs)
                         for d in self._dist_list]).T

        # handle the weights
        if self._weights is None:
            pdfs = pdfs.sum(0) * 0.0
        else:
            pdfs = (pdfs * self._weights).T.sum(0)

            # set any pdf with bad weights to zero
            bad_ind = ((self._weights > 1.0) |
                       (self._weights < 0.0)).sum(1) > 0
            pdfs[bad_ind] = 0.0

        return pdfs

    def rvs(self, size):
        # calc the rvs
        rv = np.array([d.rvs(size)
                       for d in self._dist_list])

        # handle the weights
        inds = np.array([np.random.choice(list(range(len(self._dist_list))),
                                          size=rv.shape[2:],
                                          p=self._weights[i])
                         for i in range(len(self._weights))])

        return rv[inds, np.arange(len(inds))]
