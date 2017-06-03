# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


import gzip
import pickle as pickle


def make_dict(abc, burnin=None, **kwargs):
    return dict(name=abc._name,
                particles=abc.particles,
                param_names=abc.param_names,
                param_display_names=abc.param_display_names,
                weights=abc.weights,
                log_likes=abc.log_likes,
                times=abc.times,
                posts=abc.posts, burnin=burnin,
                **kwargs)


def gzpickle(obj, filename):
    gf = gzip.open(filename, 'wb')
    gf.write(pickle.dumps(obj, 2))
    gf.close()


def save_results(filename, abc, burnin=None, **kwargs):
    """
    Save an simulation as a .pickle.gz:

    save_results('myrun.pickle.gz',burnin=50)

    """
    gzpickle(make_dict(abc, burnin=burnin, **kwargs), filename)


def load_results(filename):
    """
    Load in a simulation that was saved to a pickle.gz.
    """
    gf = gzip.open(filename, 'rb')
    res = pickle.loads(gf.read())
    gf.close()
    return res
