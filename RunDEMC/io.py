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
    """Generate a dict from a model instance."""
    return dict(name=model._name,
                particles=model.particles,
                param_names=model.param_names,
                param_display_names=model.param_display_names,
                weights=model.weights,
                log_likes=model.log_likes,
                times=model.times,
                burnin=burnin,
                accept_rate=model.accept_rate,
                **kwargs)


def arviz_dict(model, burnin=0, step=1):
    """Generate a dict that can be used to create an ArviZ data structure."""
    return dict(posterior={model.param_names[i]: model.particles[burnin::step,:,i].T 
                           for i in range(len(model.param_names))}, 
                sample_stats={'acceptance_rate': model.accept_rate[burnin::step].T,
                              'times': model.times[burnin::step].T,
                },
                log_likelihood={model._name: model.weights[burnin::step].T},
                prior={'log_prior':(model.weights-model.log_likes)[burnin:].T}
                )


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
