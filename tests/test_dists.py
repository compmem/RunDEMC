# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import unittest
from numpy.testing import *

from RunDEMC import dists
import scipy.stats as stats
import numpy as np
from numpy.random import default_rng


class TestDists(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reals = np.linspace(-10, 10, 100)
        naturals = np.arange(0, 10)

        cls.x_vals = np.concatenate((reals, naturals))
        cls.seed = default_rng().integers(low=0, high=10000, size=1)
        cls.rv_size = 10

    def dist_assertions(self, scipy, demc, pdf=True, test_rvs=True):
        if pdf:
            assert_allclose(scipy.pdf(self.x_vals), demc.pdf(self.x_vals))
            assert_allclose(scipy.logpdf(self.x_vals), demc.logpdf(self.x_vals))
        else:
            assert_allclose(scipy.pmf(self.x_vals), demc.pmf(self.x_vals))
            assert_allclose(scipy.logpmf(self.x_vals), demc.logpmf(self.x_vals)) 
        
        if test_rvs:
            assert_allclose(scipy.rvs(self.rv_size, default_rng(self.seed)),
                            demc.rvs(self.rv_size, default_rng(self.seed)))

    def test_normal(self):
        means = [0, 1, 2]
        stds = [1, 3, 5]

        for mean, std in zip(means, stds):
            scipy = stats.norm(loc=mean, scale=std)
            demc = dists.normal(mean, std)
            self.dist_assertions(scipy, demc)

    def test_uniform(self):
        lowers = [0, 1, 2]
        uppers = [1, 3, 5]

        for lower, upper in zip(lowers, uppers):
            scipy = stats.uniform(loc=lower, scale=upper-lower)
            demc = dists.uniform(lower, upper)
            self.dist_assertions(scipy, demc)

    def test_beta(self):
        alphas = [1, 2, 3]
        betas = [3, 2, 1]

        for alpha, beta in zip(alphas, betas):
            scipy = stats.beta(alpha, beta)
            demc = dists.beta(alpha, beta)
            self.dist_assertions(scipy, demc)

    def test_gamma(self):
        alphas = [1, 2, 3]
        betas = [3, 2, 1]
        
        for alpha, beta in zip(alphas, betas):
            scipy = stats.gamma(alpha, scale=1. / beta)
            demc = dists.gamma(alpha, beta)
            self.dist_assertions(scipy, demc)

    def test_invgamma(self):
        alphas = [1, 2, 3]
        betas = [3, 2, 1]
        
        for alpha, beta in zip(alphas, betas):
            scipy = stats.invgamma(alpha, scale=1/beta)
            demc = dists.invgamma(alpha, beta)
            self.dist_assertions(scipy, demc, test_rvs=False)

    def test_exp(self):
        lams = [1, 2, 3]

        for lam in lams:
            scipy = stats.expon(scale=1. / lam)
            demc = dists.exp(lam)
            self.dist_assertions(scipy, demc, test_rvs=False)

    def test_poisson(self):
        lams = [1, 2, 3]

        for lam in lams:
            scipy = stats.poisson(mu=lam)
            demc = dists.poisson(lam)
            self.dist_assertions(scipy, demc, pdf=False)

    def test_laplace(self):
        locs = [0, 1, 2]
        diversities = [1, 2, 3]

        for loc, diversity in zip(locs, diversities):
            scipy = stats.laplace(loc=loc, scale=diversity)
            demc = dists.laplace(loc, diversity)
            self.dist_assertions(scipy, demc)

    def test_students_t(self):
        dfs = [1, 2, 3]
        means = [0, 1, 2]
        stds = [1, 2, 3]

        for df, mean, std, in zip(dfs, means, stds):
            scipy = stats.t(df=df, loc=mean, scale=std)
            demc = dists.students_t(mean, std, df)
            self.dist_assertions(scipy, demc)

    @unittest.skip("not implemented")
    def test_noncentral_t(self):
        dfs = [1, 2, 3]
        ncs = [1, 2, 3]
        means = [0, 1, 2]
        stds = [1, 2, 3]

        for df, nc, mean, std in zip(dfs, ncs, means, stds):
            scipy = stats.nct(df=df, nc=nc, loc=mean, scale=std)
            demc = dists.noncentral_t(mean, std, df, nc)
            self.dist_assertions(scipy, demc)

    def test_halfcauchy(self):
        scales = [1, 2, 3]
        locs = [0, 1, 2]

        for scale, loc in zip(scales, locs):
            scipy = stats.halfcauchy(loc=loc, scale=scale)
            demc = dists.halfcauchy(scale, loc)
            self.dist_assertions(scipy, demc, test_rvs=False)

if __name__ == '__main__':
    unittest.main()
