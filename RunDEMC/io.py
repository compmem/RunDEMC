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


def make_dict(model, burnin=None, **kwargs):
    return dict(name=model._name,
                particles=model.particles,
                param_names=model.param_names,
                param_display_names=model.param_display_names,
                weights=model.weights,
                log_likes=model.log_likes,
                times=model.times,
                posts=model.posts, burnin=burnin,
                accept_rate=model.accept_rate,
                **kwargs)


def gzpickle(obj, filename):
    gf = gzip.open(filename, 'wb')
    gf.write(pickle.dumps(obj, 2))
    gf.close()


def save_results(filename, model, burnin=None, **kwargs):
    """
    Save an simulation as a .pickle.gz:

    save_results('myrun.pickle.gz',burnin=50)

    """
    gzpickle(make_dict(model, burnin=burnin, **kwargs), filename)


def load_results(filename):
    """
    Load in a simulation that was saved to a pickle.gz.
    """
    gf = gzip.open(filename, 'rb')
    res = pickle.loads(gf.read())
    gf.close()
    return res
