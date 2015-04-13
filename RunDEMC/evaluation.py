#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


def calc_bpic(weights):
    """
    get.dic=function(weight){
    term1=apply(weight,c(1,2),sum)
    d.theta=-2*term1
    dbar=mean(d.theta)
    dhat=min(d.theta)
    pd=dbar-dhat
    dic=pd+dbar
    ic=-2*mean(term1)+2*pd
    list(dic=dic,ic=ic,pd=pd,dbar=dbar,dhat=dhat)
    }
    """
    loglike = weights
    d_theta = -2*loglike
    dbar = d_theta.mean()
    dhat = d_theta.min()
    pd = dbar-dhat
    dic= pd + dbar
    ic = -2*loglike.mean() + 2*pd
    return {'bpic':ic,'dic':dic,'dbar':dbar,'dhat':dhat,'pd':pd}
